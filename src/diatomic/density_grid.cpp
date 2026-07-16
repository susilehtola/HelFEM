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
 * SPDX-License-Identifier: BSD-3-Clause
 * See the LICENSE file at the root of this source distribution
 * for the full license text.
 */

#include "../general/cmdline.h"
#include "../general/checkpoint.h"
#include "Matrix.h"
#include "../general/constants.h"
#include "../general/spherical_harmonics.h"
#include "../general/timer.h"
#include "utils.h"
#include "basis.h"
#include <cfloat>
#include <climits>
#include <complex>

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
    lang=4*basis.get_lval().maxCoeff()+12;
  if(mang<=0)
    mang=4*basis.get_mval().cwiseAbs().maxCoeff()+5;

  helfem::Vector cth, phi, wang;
  helfem::angular::angular_chebyshev(lang,mang,cth,phi,wang);
  printf("Using angular quadrature grid with L=%i M=%i with %i points\n",lang,mang,(int) cth.size());

  // Density matrix
  helfem::Matrix Ca, Cb;
  loadchk.read("Ca",Ca);
  loadchk.read("Cb",Cb);
  int nela, nelb;
  loadchk.read("nela",nela);
  loadchk.read("nelb",nelb);

  // Inverse overlap
  helfem::Matrix Sinvh;
  loadchk.read("Sinvh", Sinvh);
  helfem::Matrix Sinv(Sinvh*Sinvh.transpose());

  // mu array
  std::vector<helfem::Vector> mu(basis.get_rad_Nel()), wmu(basis.get_rad_Nel());
  for(size_t iel=0;iel<mu.size();iel++) {
    mu[iel]=basis.get_r(iel);
    wmu[iel]=basis.get_wrad(iel);
  }

  size_t Nradpts=mu.size()*mu[0].size();
  printf("Using radial quadrature grid with %i points\n",(int) Nradpts);

  // Size of total matrix
  size_t Ngrid = Nradpts*wang.size();

  // Pretabulate basis function data
  Eigen::VectorXi lval(basis.get_lval());
  Eigen::VectorXi mval(basis.get_mval());
  Eigen::MatrixXcd sph(wang.size(),lval.size());
  for(Eigen::Index il=0;il<lval.size();il++)
    for(Eigen::Index iang=0;iang<wang.size();iang++)
      sph(iang,il)=::spherical_harmonics(lval(il),mval(il),cth(iang),phi(iang));

  // Evaluate radial functions
  std::vector<helfem::Matrix> radbf(basis.get_rad_Nel());
  for(size_t iel=0;iel<mu.size();iel++) {
    radbf[iel] = basis.get_rad_bf(iel);
  }

  // Density arrays
  helfem::Vector dena(helfem::Vector::Zero(Ngrid)), denb(helfem::Vector::Zero(Ngrid)), dV(helfem::Vector::Zero(Ngrid));

  helfem::Vector mugrid(helfem::Vector::Zero(Ngrid));
  helfem::Vector cthgrid(helfem::Vector::Zero(Ngrid));
  helfem::Vector phigrid(helfem::Vector::Zero(Ngrid));
  helfem::Vector wquad(helfem::Vector::Zero(Ngrid));
  Eigen::MatrixXcd orbagrid(Eigen::MatrixXcd::Zero(Ngrid,nela));
  Eigen::MatrixXcd orbbgrid;
  if(nelb)
    orbbgrid=Eigen::MatrixXcd::Zero(Ngrid,nelb);
  Eigen::MatrixXcd Torbagrid(Eigen::MatrixXcd::Zero(Ngrid,nela));
  Eigen::MatrixXcd Torbbgrid;
  if(nelb)
    Torbbgrid=Eigen::MatrixXcd::Zero(Ngrid,nelb);

  Eigen::MatrixXcd Sa(Eigen::MatrixXcd::Zero(nela,nela)), Sb;
  if(nelb>0)
    Sb=Eigen::MatrixXcd::Zero(nelb,nelb);

  helfem::Matrix T(basis.kinetic());
  helfem::Matrix STCa = Sinv*T*Ca;
  helfem::Matrix STCb = Sinv*T*Cb;

  size_t igrid=0;
  // Loop over radial elements
  for(size_t iel=0;iel<mu.size();iel++) {
    // Get the list of basis functions in the element in the dummy indexing.
    std::vector<Eigen::Index> bidx=basis.bf_list(iel);

    // Orbital submatrices (cast to complex once per element for the
    // complex-valued basis function products below)
    Eigen::MatrixXcd Casub = Ca.leftCols(nela)(bidx, Eigen::all).cast<std::complex<double>>();
    Eigen::MatrixXcd Cbsub;
    if(nelb)
      Cbsub = Cb.leftCols(nelb)(bidx, Eigen::all).cast<std::complex<double>>();

    // Kinetic energy submatrix
    Eigen::MatrixXcd STCasub = STCa.leftCols(nela)(bidx, Eigen::all).cast<std::complex<double>>();
    Eigen::MatrixXcd STCbsub;
    if(nelb) {
	STCbsub = STCb(bidx, Eigen::all).cast<std::complex<double>>();
    }

    // Radial values
    helfem::Vector r(mu[iel]);
    helfem::Vector wr(wmu[iel]);

    // Loop over angular output grid
    for(Eigen::Index iang=0;iang<wang.size();iang++) {
      // Loop over radial points
      for(Eigen::Index irad=0;irad<radbf[iel].rows();irad++) {
        // Form matrix of basis function values
        Eigen::RowVectorXcd bf(bidx.size());
        {
          Eigen::Index ioff=0;
          // Loop over angular basis
          for(Eigen::Index il=0;il<lval.size();il++) {
            // Loop over in-element radial functions
            Eigen::Index firstfun = ((iel==0) && (mval(il)!=0)) ? 1 : 0;
            for(Eigen::Index ifun=firstfun;ifun<radbf[iel].cols();ifun++) {
              bf(ioff++) = sph(iang,il)*radbf[iel](irad,ifun);
            }
          }
          if(ioff != (Eigen::Index) bidx.size()) {
            printf("iel=%i iang=%i ioff=%i bidx.n_elem=%i\n",(int) iel,(int) iang,(int) ioff,(int) bidx.size());
            fflush(stdout);
            throw std::logic_error("Indexing problem!\n");
          }
        }

        // Compute orbital values
        Eigen::RowVectorXcd orbaval = bf*Casub;
        Eigen::RowVectorXcd orbbval;
        if(nelb)
          orbbval = bf*Cbsub;

	// Compute action of kinetic energy operator on orbitals
	Eigen::RowVectorXcd Torbaval = bf*STCasub;
        Eigen::RowVectorXcd Torbbval;
        if(nelb)
	    Torbbval = bf*STCbsub;

        // Store result
        dena(igrid)=orbaval.squaredNorm();
        denb(igrid)=nelb ? orbbval.squaredNorm() : 0.0;

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

	Torbagrid.row(igrid) = Torbaval;
	if(nelb)
	    Torbbgrid.row(igrid) = Torbbval;

        Sa += orbaval.adjoint()*orbaval*std::complex<double>(dV(igrid));
        if(nelb)
          Sb += orbbval.adjoint()*orbbval*std::complex<double>(dV(igrid));

        igrid++;
      }
    }
  }
  if(igrid != (size_t) dena.size())
    throw std::logic_error("Indexing error!\n");

  // Total density
  helfem::Vector den(dena+denb);

  printf("Norm of Pa on grid is %e\n",(dena.array()*dV.array()).sum());
  printf("Norm of Pb on grid is %e\n",(denb.array()*dV.array()).sum());
  printf("Norm of P on grid is %e\n",(den.array()*dV.array()).sum());

  printf("Alpha-alpha orbital non-orthonormality %e\n",(Sa-Eigen::MatrixXcd::Identity(Sa.rows(),Sa.cols())).norm());
  if(nelb)
    printf("Beta-beta   orbital non-orthonormality %e\n",(Sb-Eigen::MatrixXcd::Identity(Sb.rows(),Sb.cols())).norm());

  Checkpoint savechk(output,true);
  savechk.write("mu",mugrid);
  savechk.write("dV",dV);
  savechk.write("wquad",wquad);
  savechk.write("cth",cthgrid);
  savechk.write("phi",phigrid);
  savechk.write("P",den);
  savechk.write("Pa",dena);
  savechk.write("Pb",denb);
  savechk.write("orba.re",helfem::Matrix(orbagrid.real()));
  savechk.write("orba.im",helfem::Matrix(orbagrid.imag()));
  if(nelb) {
    savechk.write("orbb.re",helfem::Matrix(orbbgrid.real()));
    savechk.write("orbb.im",helfem::Matrix(orbbgrid.imag()));
  }
  savechk.write("Torba.re",helfem::Matrix(Torbagrid.real()));
  savechk.write("Torba.im",helfem::Matrix(Torbagrid.imag()));
  if(nelb) {
    savechk.write("Torbb.re",helfem::Matrix(Torbbgrid.real()));
    savechk.write("Torbb.im",helfem::Matrix(Torbbgrid.imag()));
  }
  savechk.write("Rh",basis.get_Rhalf());
  savechk.write("Z1",basis.get_Z1());
  savechk.write("Z2",basis.get_Z2());
  int mmax = basis.get_mval().maxCoeff();
  savechk.write("mmax",mmax);

  printf("Saved density to file %s\n",output.c_str());

  return 0;
}
