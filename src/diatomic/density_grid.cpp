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
#include "../general/spherical_harmonics.h"
#include "../general/timer.h"
#include "utils.h"
#include "basis.h"
#include <cfloat>
#include <climits>

// Angular quadrature
#include "../general/angular.h"

using namespace helfem;

int main(int argc, char **argv) {
  cmdline::parser parser;

  // full option name, no short option, description, argument required
  parser.add<std::string>("load", 0, "load guess from checkpoint", false, "");
  parser.add<int>("nquad", 0, "number of quadrature points in mu (automatic default)", false, -1);
  parser.add<int>("lang", 0, "number of quadrature points in nu (automatic default)", false, -1);
  parser.add<int>("mang", 0, "number of quadrature points in phi (automatic default)", false, -1);
  parser.add<std::string>("output", 0, "save density to file", false, "density.hdf5");
  parser.parse_check(argc, argv);

  // Get parameters
  std::string load(parser.get<std::string>("load"));

  int nquad(parser.get<int>("nquad"));

  // Angular grid
  int lang(parser.get<int>("lang"));
  int mang(parser.get<int>("mang"));

  std::string output(parser.get<std::string>("output"));

  // Load checkpoint
  Checkpoint loadchk(load,false);
  // Basis set
  diatomic::basis::TwoDBasis basis;
  loadchk.read(basis);

  // Process defaults
  if(nquad<=0)
    nquad=5*basis.get_poly_nnodes();
  if(lang<=0)
    lang=4*arma::max(basis.get_lval())+12;
  if(mang<=0)
    mang=4*arma::max(arma::abs(basis.get_mval()))+5;

  arma::vec cth, phi, wang;
  helfem::angular::angular_chebyshev(lang,mang,cth,phi,wang);
  printf("Using angular quadrature grid with L=%i M=%i with %i points\n",lang,mang,(int) cth.n_elem);

  // Density matrix
  arma::mat Ca, Cb;
  loadchk.read("Ca",Ca);
  loadchk.read("Cb",Cb);
  int nela, nelb;
  loadchk.read("nela",nela);
  loadchk.read("nelb",nelb);

  // mu array
  std::vector<arma::vec> mu(basis.get_rad_Nel()), wmu(basis.get_rad_Nel());
  for(size_t iel=0;iel<mu.size();iel++) {
    mu[iel]=basis.get_r(iel);
    wmu[iel]=basis.get_wrad(iel);
  }

  size_t Nradpts=mu.size()*mu[0].n_elem;
  printf("Using radial quadrature grid with %i points\n",(int) Nradpts);

  // Size of total matrix
  size_t Ngrid = Nradpts*wang.n_elem;

  // Pretabulate basis function data
  arma::ivec lval(basis.get_lval());
  arma::ivec mval(basis.get_mval());
  arma::cx_mat sph(wang.n_elem,lval.n_elem);
  for(size_t il=0;il<lval.n_elem;il++)
    for(size_t iang=0;iang<wang.n_elem;iang++)
      sph(iang,il)=::spherical_harmonics(lval(il),mval(il),cth(iang),phi(iang));

  // Evaluate radial functions
  std::vector<arma::mat> radbf(basis.get_rad_Nel());
  for(size_t iel=0;iel<mu.size();iel++) {
    radbf[iel] = basis.get_rad_bf(iel);
  }

  // Density arrays
  arma::vec dena(Ngrid,arma::fill::zeros), denb(Ngrid,arma::fill::zeros), dV(Ngrid,arma::fill::zeros);
  arma::mat orba(Ngrid,nela,arma::fill::zeros), orbb(Ngrid,nelb,arma::fill::zeros);

  arma::vec mugrid(Ngrid,arma::fill::zeros);
  arma::vec cthgrid(Ngrid,arma::fill::zeros);
  arma::vec phigrid(Ngrid,arma::fill::zeros);
  arma::vec wquad(Ngrid,arma::fill::zeros);
  arma::cx_mat orbagrid(Ngrid,nela,arma::fill::zeros);
  arma::cx_mat orbbgrid;
  if(nelb)
    orbbgrid.zeros(Ngrid,nelb);

  arma::cx_mat Sa(nela,nela,arma::fill::zeros), Sb;
  if(nelb>0)
    Sb.zeros(nelb,nelb);

  size_t igrid=0;
  // Loop over radial elements
  for(size_t iel=0;iel<mu.size();iel++) {
    // Get the list of basis functions in the element in the dummy
    // indexing
    arma::uvec bidx=basis.bf_list(iel);

    // Orbital submatrices
    arma::mat Casub(Ca.cols(0,nela-1));
    Casub = Casub.rows(bidx);
    arma::mat Cbsub;
    if(nelb) {
      Cbsub=Cb.cols(0,nelb-1);
      Cbsub=Cbsub.rows(bidx);
    }

    // Radial values
    arma::vec r(mu[iel]);
    arma::vec wr(wmu[iel]);

    // Loop over angular output grid
    for(size_t iang=0;iang<wang.n_elem;iang++) {
      // Loop over radial points
      for(size_t irad=0;irad<radbf[iel].n_rows;irad++) {
        // Form matrix of basis function values
        arma::cx_rowvec bf(bidx.n_elem);
        {
          size_t ioff=0;
          // Loop over angular basis
          for(size_t il=0;il<lval.n_elem;il++) {
            // Loop over in-element radial functions
            size_t firstfun = ((iel==0) && (mval(il)!=0)) ? 1 : 0;
            for(size_t ifun=firstfun;ifun<radbf[iel].n_cols;ifun++) {
              bf(ioff++) = sph(iang,il)*radbf[iel](irad,ifun);
            }
          }
          if(ioff != bidx.n_elem) {
            printf("iel=%i iang=%i ioff=%i bidx.n_elem=%i\n",(int) iel,(int) iang,(int) ioff,(int) bidx.n_elem);
            fflush(stdout);
            throw std::logic_error("Indexing problem!\n");
          }
        }

        // Compute orbital values
        arma::cx_rowvec orbaval = bf*Casub;
        arma::cx_rowvec orbbval;
        if(nelb)
          orbbval = bf*Cbsub;

        // Store result
        dena(igrid)=std::real(arma::cdot(orbaval,orbaval));
        denb(igrid)=nelb ? std::real(arma::cdot(orbbval,orbbval)) : 0.0;

        // Store grid values
        mugrid(igrid) = r(irad);
        cthgrid(igrid) = cth(iang);
        phigrid(igrid) = phi(iang);

        // Volume element is
        double shmu(sinh(r(irad)));
        double chmu(cosh(r(irad)));
        wquad(igrid)=wr(irad)*wang(iang);
        dV(igrid)=std::pow(basis.get_Rhalf(),3)*shmu*(chmu*chmu - cth(iang)*cth(iang))*wr(irad)*wang(iang);

        // Orbital values
        orbagrid.row(igrid) = orbaval;
        if(nelb)
          orbbgrid.row(igrid) = orbbval;

        Sa += arma::trans(orbaval)*dV(igrid)*orbaval;
        if(nelb)
          Sb += arma::trans(orbbval)*dV(igrid)*orbbval;

        igrid++;
      }
    }
  }
  if(igrid != dena.n_elem)
    throw std::logic_error("Indexing error!\n");

  // Total density
  arma::vec den(dena+denb);

  printf("Norm of Pa on grid is %e\n",arma::sum(dena%dV));
  printf("Norm of Pb on grid is %e\n",arma::sum(denb%dV));
  printf("Norm of P on grid is %e\n",arma::sum(den%dV));

  printf("Alpha-alpha orbital non-orthonormality %e\n",arma::norm(Sa-arma::eye<arma::cx_mat>(Sa.n_rows,Sa.n_cols),"fro"));
  if(nelb)
    printf("Beta-beta   orbital non-orthonormality %e\n",arma::norm(Sb-arma::eye<arma::cx_mat>(Sb.n_rows,Sb.n_cols),"fro"));

  Checkpoint savechk(output,true);
  savechk.write("mu",mugrid);
  savechk.write("dV",dV);
  savechk.write("wquad",wquad);
  savechk.write("cth",cthgrid);
  savechk.write("phi",phigrid);
  savechk.write("P",den);
  savechk.write("Pa",dena);
  savechk.write("Pb",denb);
  savechk.write("orba.re",arma::real(orbagrid));
  savechk.write("orba.im",arma::imag(orbagrid));
  if(nelb) {
    savechk.write("orbb.re",arma::real(orbbgrid));
    savechk.write("orbb.im",arma::imag(orbbgrid));
  }
  savechk.write("Rh",basis.get_Rhalf());
  savechk.write("Z1",basis.get_Z1());
  savechk.write("Z2",basis.get_Z2());
  int mmax = arma::max(basis.get_mval());
  savechk.write("mmax",mmax);
  printf("Saved density to file %s\n",output.c_str());

  return 0;
}
