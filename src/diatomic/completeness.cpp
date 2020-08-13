/*
 *                This source code is part of
 *
 *                          HelFEM
 *                             -
 * Finite element methods for electronic structure calculations on small systems
 *
 * Written by Susi Lehtola, 2018-
 * Copyright (c) 2018- Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */
#include "../general/cmdline.h"
#include "../general/checkpoint.h"
#include "../general/constants.h"
#include "../general/timer.h"
#include "../general/lcao.h"
#include "basis.h"
#include "twodquadrature.h"
#include <cfloat>
#include <climits>

using namespace helfem;

int main(int argc, char **argv) {
  cmdline::parser parser;

  // full option name, no short option, description, argument required
  parser.add<int>("ldft", 0, "theta rule for quadrature (0 for auto)", false, 0);
  parser.add<std::string>("load", 0, "load guess from checkpoint", false, "");
  parser.add<int>("completeness", 0, "perform completeness scan up to lmax", false, -1);
  parser.add<double>("minexp", 0, "minimum exponent", false, 1e-5);
  parser.add<double>("maxexp", 0, "minimum exponent", false, 1e10);
  parser.add<int>("nexp", 0, "number of points in exponent scan", false, 151);
  parser.add<int>("iprobe", 0, "probe to use: 0 for gto, 1 for sto", false, 0);
  parser.parse_check(argc, argv);

  // Get parameters
  int ldft(parser.get<int>("ldft"));
  int completeness=parser.get<int>("completeness");
  std::string load(parser.get<std::string>("load"));
  double minexp(parser.get<double>("minexp"));
  double maxexp(parser.get<double>("maxexp"));
  int nexp(parser.get<int>("nexp"));
  int iprobe(parser.get<int>("iprobe"));
  
  // Load checkpoint
  Checkpoint loadchk(load,false);
  // Basis set
  diatomic::basis::TwoDBasis basis;
  loadchk.read(basis);
  // Sinvh
  arma::mat Sinvh;
  loadchk.read("Sinvh",Sinvh);
  arma::mat Sinv(Sinvh*arma::trans(Sinvh));
  // Orbitals
  arma::mat Ca, Cb;
  loadchk.read("Ca",Ca);
  loadchk.read("Cb",Cb);
  // Number of occupied orbitals
  int nela, nelb;
  loadchk.read("nela",nela);
  loadchk.read("nelb",nelb);

  // Completeness probe
  int lquad = (ldft>0) ? ldft : 4*arma::max(basis.get_lval())+12;
  helfem::diatomic::twodquad::TwoDGrid qgrid;
  qgrid=helfem::diatomic::twodquad::TwoDGrid(&basis,lquad);
  
  // Unique m values
  arma::ivec muni(basis.get_mval());
  muni=arma::sort(muni(arma::find_unique(muni)),"ascend");
    
  // Exponents
  arma::vec expn(arma::exp10(arma::linspace<arma::vec>(log10(minexp),log10(maxexp),nexp)));

  // Compute radial functions
  arma::vec r(arma::linspace<arma::vec>(0.0,100.0,1000));

  for(size_t im=0;im<muni.size();im++) {
    int m=muni(im);
    for(int l=std::abs(m);l<=completeness;l++) {
      arma::mat Plcao;

      static const std::string indices[]={"lh", "mid", "rh"};
      static const helfem::diatomic::twodquad::probe_t probes[]={helfem::diatomic::twodquad::PROBE_LEFT, helfem::diatomic::twodquad::PROBE_MIDDLE, helfem::diatomic::twodquad::PROBE_RIGHT};
      for(int icen=0;icen<3;icen++) {
        // LCAO projection wrt <\alpha|FEM> in terms of orthonormal orbitals
        std::string lcao;
        if(iprobe==0) {
          lcao="gto";
          Plcao=qgrid.gto_projection(l, m, expn, probes[icen]);
        } else if(iprobe==1) {
          lcao="sto";
          Plcao=qgrid.sto_projection(l, m, expn, probes[icen]);
        } else
          throw std::logic_error("Unknown probe\n");

        // FEM completeness profile
        arma::mat Y(arma::diagvec(Plcao*Sinv*arma::trans(Plcao)));
        Y.insert_cols(0,expn);
        
        std::ostringstream oss;
        oss << "fem_basis_" << lcao << "cpl_" << indices[icen] << "_" << l << "_" << m << ".dat";
        Y.save(oss.str(),arma::raw_ascii);
        
        oss.str("");
        
        // Project GTO projection onto occupied orbitals
        arma::mat Pa(Plcao*Ca.cols(0,nela-1));
        Y.zeros(Y.n_rows,nela+2);
        Y.col(0)=expn;
        for(int io=0;io<nela;io++)
          Y.col(io+1)=arma::diagvec(Pa.col(io)*arma::trans(Pa.col(io)));
        Y.col(nela+1)=arma::sum(Y.cols(1,nela),1);
        oss.str("");
        oss << "fem_aocc_" << lcao << "cpl_" << indices[icen] << "_" << l << "_" << m << ".dat";
        Y.save(oss.str(),arma::raw_ascii);
      
        arma::mat Pb(Plcao*Cb.cols(0,nelb-1));
        Y.zeros(Y.n_rows,nelb+2);
        Y.col(0)=expn;
        for(int io=0;io<nelb;io++)
          Y.col(io+1)=arma::diagvec(Pb.col(io)*arma::trans(Pb.col(io)));
        Y.col(nelb+1)=arma::sum(Y.cols(1,nelb),1);
        oss.str("");
        oss << "fem_bocc_" << lcao << "cpl_" << indices[icen] << "_" << l << "_" << m << ".dat";
        Y.save(oss.str(),arma::raw_ascii);

        /*
        printf("*** l=%i m=%i ***\n",l,m);        
        Slcao=qgrid.gto_overlap(l, m, expn, diatomic::twodquad::PROBE_LEFT);
        Slcao.print("lh gto overlap");
        Slcao=qgrid.gto_overlap(l, m, expn, diatomic::twodquad::PROBE_RIGHT);
        Slcao.print("rh gto overlap");
        
        Slcao=qgrid.sto_overlap(l, m, expn, diatomic::twodquad::PROBE_LEFT);
        Slcao.print("lh sto overlap");
        Slcao=qgrid.sto_overlap(l, m, expn, diatomic::twodquad::PROBE_RIGHT);
        Slcao.print("rh sto overlap");
        printf("\n");
        */
      }
    }
  }

  return 0;
}
