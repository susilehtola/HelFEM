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
  parser.add<int>("nexp", 0, "number of points in exponent scan", false, 501);
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
  // Nuclear charges
  int Z1, Z2;
  loadchk.read("Z1",Z1);
  loadchk.read("Z2",Z2);

  // Completeness probe
  int lquad = (ldft>0) ? ldft : 4*arma::max(basis.get_lval())+12;
  helfem::diatomic::twodquad::TwoDGrid qgrid;
  qgrid=helfem::diatomic::twodquad::TwoDGrid(&basis,lquad);
  qgrid.compute_atoms(Z1,Z2);

  // Unique m values
  arma::ivec muni(basis.get_mval());
  muni=arma::sort(muni(arma::find_unique(muni)),"ascend");

  // Exponents
  arma::vec expn(arma::exp10(arma::linspace<arma::vec>(log10(minexp),log10(maxexp),nexp)));

  // Compute radial functions
  arma::vec r(arma::linspace<arma::vec>(0.0,100.0,1000));

  for(size_t im=0;im<muni.size();im++) {
    int m=muni(im);

    // Compute the projection of the atomic orbitals on the grid
    arma::mat minimal_basis_orbitals;
    size_t nminimal = 0;
    {
      std::vector<arma::mat> minimal_basis;
      for(int l=std::abs(m);l<=completeness;l++) {
        arma::mat proj = qgrid.atomic_projection(l, m, helfem::diatomic::twodquad::PROBE_LEFT);
        if(proj.n_elem) {
          minimal_basis.push_back(proj);
          nminimal += proj.n_rows;
          printf("Left-hand atom has %i orbitals for m = %i from l = %i\n",proj.n_rows,m,l);
          fflush(stdout);
        }
      }
      for(int l=std::abs(m);l<=completeness;l++) {
        arma::mat proj = qgrid.atomic_projection(l, m, helfem::diatomic::twodquad::PROBE_RIGHT);
        if(proj.n_elem) {
          minimal_basis.push_back(proj);
          nminimal += proj.n_rows;
          printf("Right-hand atom has %i orbitals for m = %i from l = %i\n",proj.n_rows,m,l);
          fflush(stdout);
        }
      }

      printf("We have %i minimal basis functions for m = %i\n",nminimal, m);
      fflush(stdout);
      if(nminimal>0) {
        minimal_basis_orbitals.zeros(nminimal, minimal_basis[0].n_cols);
        size_t ioff=0;
        for(size_t i = 0; i < minimal_basis.size();i++) {
          minimal_basis_orbitals.rows(ioff, ioff+minimal_basis[i].n_rows-1) = minimal_basis[i];
          ioff += minimal_basis[i].n_rows;
        }
      }
    }

    // Inverse overlap matrix
    std::function<arma::mat(const arma::mat & S)> form_Sinv = [](const arma::mat & S) {
      arma::vec Sval;
      arma::mat Svec;
      arma::eig_sym(Sval, Svec, S);
      // Find well-conditioned part
      arma::uvec idx(arma::find(Sval>=1e-6));
      arma::mat result =  Svec.cols(idx) * arma::diagmat(arma::pow(Sval(idx), -1.0)) * Svec.cols(idx).t();
      return result;
    };

    // Minimal-basis overlap and inverse overlap matrices
    arma::mat Smin, Smin_inv;
    if(nminimal>0) {
      Smin = minimal_basis_orbitals*Sinv*minimal_basis_orbitals.t();

      arma::mat dSmin = Smin - arma::eye<arma::mat>(Smin.n_rows, Smin.n_cols);
      printf("Minimal basis overlap matrix for m = %i, difference from orthonormality\n",m);
      dSmin.print();
      fflush(stdout);

      Smin_inv = form_Sinv(Smin);
    }

    // Compute the projection of the wave function on the minimal basis
    double alpha_proj = 0.0, beta_proj = 0.0;

    arma::mat alpha_minbas;
    if(nminimal>0) {
      alpha_minbas = ((minimal_basis_orbitals * Ca.cols(0,nela-1)).t());
      alpha_proj = arma::trace(alpha_minbas * Smin_inv * alpha_minbas.t());
      printf("Projection of m = %i alpha orbitals on minimal basis is % .6f\n", m, alpha_proj);
      fflush(stdout);
    }
    arma::mat beta_minbas;
    if(nelb>0 && nminimal>0) {
      beta_minbas = ((minimal_basis_orbitals * Cb.cols(0,nelb-1)).t());
      beta_proj = arma::trace(beta_minbas * Smin_inv * beta_minbas.t());
      printf("Projection of m = %i beta  orbitals on minimal basis is % .6f\n", m, beta_proj);
      fflush(stdout);
    }

    for(int l=std::abs(m);l<=completeness;l++) {
      static const std::string indices[]={"lh", "mid", "rh"};
      static const helfem::diatomic::twodquad::probe_t probes[]={helfem::diatomic::twodquad::PROBE_LEFT, helfem::diatomic::twodquad::PROBE_MIDDLE, helfem::diatomic::twodquad::PROBE_RIGHT};
      for(int icen=0;icen<3;icen++) {
        // Importance profile (no minimal basis): expn, alpha, beta
        arma::mat I0(3, expn.n_elem);
        // Importance profile: expn, alpha, beta
        arma::mat I(3, expn.n_elem);
        // Finite element basis set completeness: expn, Y
        arma::mat Y(2, expn.n_elem, arma::fill::zeros);

        std::string lcao;
        if(iprobe==0) {
          lcao="gto";
        } else if(iprobe==1) {
          lcao="sto";
        } else
          throw std::logic_error("Unknown probe\n");

        // Loop over batches of exponents
        arma::uword exponent_batch_size = 200;
        for(arma::uword exponent_batch = 0; exponent_batch <= expn.n_elem/exponent_batch_size; exponent_batch++) {
          arma::uword istart = exponent_batch_size*exponent_batch;
          arma::uword iend = std::min(exponent_batch_size*(exponent_batch+1), expn.n_elem) - 1;
          if(istart>iend)
            break;
          arma::vec expbatch(expn.subvec(istart, iend));

          // LCAO projection <\alpha|FEM>
          arma::mat Plcao;
          if(iprobe==0) {
            Plcao=qgrid.gto_projection(l, m, expbatch, probes[icen]);
          } else if(iprobe==1) {
            Plcao=qgrid.sto_projection(l, m, expbatch, probes[icen]);
          } else
            throw std::logic_error("Unknown probe\n");

          // Completeness profile
          arma::vec Ysub = arma::diagvec(Plcao*Sinv*arma::trans(Plcao));

          arma::mat Plcao_t(Plcao.t());

          // Projections onto occupied orbitals
          arma::mat Pa(Ca.cols(0,nela-1).t()*Plcao_t);
          arma::mat Pa_t = Pa.t();
          arma::mat Pb;
          if(nelb)
            Pb = Cb.cols(0,nelb-1).t()*Plcao_t;
          arma::mat Pb_t = Pb.t();

          // Compute <\alpha|minimal basis>
          arma::mat lcao_overlap_minbas;
          if(nminimal>0)
            lcao_overlap_minbas = minimal_basis_orbitals*Sinv*arma::trans(Plcao);

          // Compute the importance profile
#pragma omp parallel for
          for(size_t ix=0;ix<expbatch.n_elem; ix++) {
            // Output index
            size_t iout = istart+ix;

            // Importance profile without minimal basis
            I0(0, iout) = expbatch(ix);
            I0(1, iout) = arma::dot(Pa.col(ix), Pa.col(ix));
            if(nelb)
              I0(2, iout) = arma::dot(Pb.col(ix), Pb.col(ix));

            // Inverse of padded overlap matrix
            arma::mat Spad_inv;
            if(nminimal>0) {
              // Construct padded overlap matrix:
              arma::mat Spad(Smin.n_rows+1, Smin.n_cols+1, arma::fill::zeros);
              // minimal basis
              Spad.submat(0,0,Smin.n_rows-1,Smin.n_cols-1) = Smin;
              // minimal basis - added function
              Spad.submat(0,Smin.n_cols,Smin.n_rows-1,Smin.n_cols) = lcao_overlap_minbas.col(ix);
              Spad.submat(Smin.n_cols,0,Smin.n_cols,Smin.n_rows-1) = lcao_overlap_minbas.col(ix).t();
              // added function - added function
              Spad(Smin.n_rows, Smin.n_cols) = arma::as_scalar(Plcao_t.col(ix).t() * Sinv * Plcao_t.col(ix));
              // Inverse
              Spad_inv = form_Sinv(Spad);

              // Overlap of the orbitals with the scanning function
              arma::mat alpha_pad(alpha_minbas.n_rows, alpha_minbas.n_cols+1);
              alpha_pad.submat(0,0,alpha_minbas.n_rows-1,alpha_minbas.n_cols-1) = alpha_minbas;
              alpha_pad.col(alpha_minbas.n_cols) = Pa.col(ix);

              double alpha_padproj = arma::trace(alpha_pad * Spad_inv * alpha_pad.t());
              I(0, iout) = expbatch(ix);
              I(1, iout) = alpha_padproj - alpha_proj;
            } else {
              for(size_t ir=0;ir<I.n_rows;ir++)
                I(ir, iout) = I0(ir, iout);
            }

            //printf("Projection of alpha orbitals on padded basis %e is % .6f\n", expbatch(ix), alpha_padproj);
            //printf("Projection increment is  % .e\n", alpha_padproj - alpha_proj);
            if(nelb>0 && nminimal>0) {
              arma::mat beta_pad(beta_minbas.n_rows, beta_minbas.n_cols+1);
              beta_pad.submat(0,0,beta_minbas.n_rows-1,beta_minbas.n_cols-1) = beta_minbas;
              beta_pad.col(beta_minbas.n_cols) = Pb.col(ix);

              double beta_padproj = arma::trace(beta_pad * Spad_inv * beta_pad.t());
              I(2, iout) = beta_padproj - beta_proj;
              //printf("Projection of beta orbitals on padded basis %e is % .6f\n", expbatch(ix), beta_padproj);
              //printf("Projection increment is  %e\n", beta_padproj-beta_proj);
            }

            // Save completeness profile
            Y(0, iout) = expbatch(ix);
            Y(1, iout) = Ysub(ix);
          }
        }

        // Save importance profile
        {
          std::ostringstream oss;
          oss << "importance0_" << lcao << "_" << indices[icen] << "_" << l << "_" << m << ".dat";
          I0 = I0.t();
          I0.save(oss.str(),arma::raw_ascii);
        }
        {
          std::ostringstream oss;
          oss << "importance_" << lcao << "_" << indices[icen] << "_" << l << "_" << m << ".dat";
          I = I.t();
          I.save(oss.str(),arma::raw_ascii);
        }

        // Save completeness profile
        {
          std::ostringstream oss;
          oss << "completeness_" << lcao << "_" << indices[icen] << "_" << l << "_" << m << ".dat";
          Y = Y.t();
          Y.save(oss.str(),arma::raw_ascii);
        }

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
