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
#include "../general/constants.h"
#include "../general/dftfuncs.h"
#include "../general/timer.h"
#include "utils.h"
#include "../general/elements.h"
#include "../general/scf_helpers.h"
#include "../general/model_potential.h"
#include "basis.h"
#include "dftgrid.h"
#include "twodquadrature.h"
#include <cfloat>
#include <climits>

using namespace helfem;

int main(int argc, char **argv) {
  cmdline::parser parser;

  // full option name, no short option, description, argument required
  parser.add<std::string>("Z1", 0, "first nuclear charge", true);
  parser.add<std::string>("Z2", 0, "second nuclear charge", true);
  parser.add<double>("Rbond", 0, "internuclear distance", true);
  parser.add<bool>("angstrom", 0, "input distances in angstrom", false, false);
  parser.add<std::string>("lmax", 0, "maximum l quantum number", true, "");
  parser.add<int>("mmax", 0, "maximum m quantum number", false, -1);
  parser.add<double>("Rmax", 0, "practical infinity in au", false, 40.0);
  parser.add<int>("grid", 0, "type of grid: 1 for linear, 2 for quadratic, 3 for polynomial, 4 for exponential", false, 4);
  parser.add<double>("zexp", 0, "parameter in radial grid", false, 1.0);
  parser.add<int>("nelem", 0, "number of elements", true);
  parser.add<int>("nnodes", 0, "number of nodes per element", false, 15);
  parser.add<int>("nquad", 0, "number of quadrature points", false, 0);
  parser.add<int>("primbas", 0, "primitive radial basis", false, 4);
  parser.add<std::string>("save", 0, "save calculation to checkpoint", false, "helfem.chk");
  parser.parse_check(argc, argv);

  // Get parameters
  double Rmax(parser.get<double>("Rmax"));
  int igrid(parser.get<int>("grid"));
  double zexp(parser.get<double>("zexp"));

  int primbas(parser.get<int>("primbas"));
  // Number of elements
  int Nelem(parser.get<int>("nelem"));
  // Number of nodes
  int Nnodes(parser.get<int>("nnodes"));
  // Order of quadrature rule
  int Nquad(parser.get<int>("nquad"));
  // Angular grid
  std::string lmax(parser.get<std::string>("lmax"));
  int mmax(parser.get<int>("mmax"));

  // Nuclear charge
  int Z1(get_Z(parser.get<std::string>("Z1")));
  int Z2(get_Z(parser.get<std::string>("Z2")));
  double Rbond(parser.get<double>("Rbond"));

  std::string save(parser.get<std::string>("save"));

  // Open checkpoint in save mode
  Checkpoint chkpt(save,true);

  if(parser.get<bool>("angstrom")) {
    // Convert to atomic units
    Rbond*=ANGSTROMINBOHR;
  }

  chkpt.write("nela",1);
  chkpt.write("nelb",0);

  // Get primitive basis
  auto poly(std::shared_ptr<const polynomial_basis::PolynomialBasis>(polynomial_basis::get_basis(primbas,Nnodes)));

  if(Nquad==0)
    // Set default value
    Nquad=5*poly->get_nbf();
  else if(Nquad<2*poly->get_nbf())
    throw std::logic_error("Insufficient radial quadrature.\n");

  printf("Using %i point quadrature rule.\n",Nquad);

  arma::ivec lmmax;
  if(mmax>=0) {
    lmmax.ones(mmax+1);
    lmmax*=atoi(lmax.c_str());
  } else {
    // Parse list of l values
    std::vector<arma::uword> lmmaxv;
    std::stringstream ss(lmax);
    while( ss.good() ) {
      std::string substr;
      getline( ss, substr, ',' );
      lmmaxv.push_back(atoi(substr.c_str()));
    }
    lmmax=arma::conv_to<arma::ivec>::from(lmmaxv);
  }
  // l and m values
  arma::ivec lval, mval;
  diatomic::basis::lm_to_l_m(lmmax,lval,mval);

  double Rhalf(0.5*Rbond);
  double mumax(utils::arcosh(Rmax/Rhalf));
  arma::vec bval(atomic::basis::normal_grid(Nelem, mumax, igrid, zexp));

  diatomic::basis::TwoDBasis basis(Z1, Z2, Rhalf, poly, Nquad, bval, lval, mval, 0);
  chkpt.write(basis);
  printf("Basis set consists of %i angular shells composed of %i radial functions, totaling %i basis functions\n",(int) basis.Nang(), (int) basis.Nrad(), (int) basis.Nbf());

  double Enucr=Z1*Z2/Rbond;
  printf("Left- and right-hand nuclear charges are %i and %i at distance % .3f\n",Z1,Z2,Rbond);

  Timer timer;

  // Form overlap matrix
  arma::mat S(helfem::to_arma(basis.overlap()));
  chkpt.write("S",S);
  // Form kinetic energy matrix
  arma::mat T(helfem::to_arma(basis.kinetic()));
  chkpt.write("T",T);

  // Get half-inverse
  timer.set();
  bool diag=true;
  bool symm=false;
  arma::mat Sinvh(helfem::to_arma(basis.Sinvh(!diag,symm)));
  chkpt.write("Sinvh",Sinvh);
  printf("Half-inverse formed in %.6f\n",timer.get());
  {
    arma::mat Smo(Sinvh.t()*S*Sinvh);
    Smo-=arma::eye<arma::mat>(Smo.n_rows,Smo.n_cols);
    printf("Orbital orthonormality deviation is %e\n",arma::norm(Smo,"fro"));
  }

  // Form nuclear attraction energy matrix
  Timer tnuc;
  arma::mat Vnuc=helfem::to_arma(basis.nuclear());
  chkpt.write("Vnuc",Vnuc);

  // Form Hamiltonian
  const arma::mat H0(T+Vnuc);
  chkpt.write("H0",H0);

  printf("One-electron matrices formed in %.6f\n",timer.get());

  arma::vec E;
  arma::mat C;
  { // Phase 5.11 bridge: eig_gsym is Eigen.
    helfem::Vector E_e; helfem::Matrix C_e;
    scf::eig_gsym(E_e, C_e, helfem::to_eigen(H0), helfem::to_eigen(Sinvh));
    E = helfem::to_arma(E_e);
    C = helfem::to_arma(C_e);
  }

  double Ekin=arma::as_scalar(C.col(0).t()*T*C.col(0));
  double Enuc=arma::as_scalar(C.col(0).t()*Vnuc*C.col(0));
  double Etot=Ekin+Enuc+Enucr;

  chkpt.write("Ekin",Ekin);
  chkpt.write("Enuc",Enuc);
  chkpt.write("Enucr",Enucr);
  chkpt.write("Etot",Etot);

  chkpt.write("nela",1);
  chkpt.write("nelb",0);
  chkpt.write("Ca",C);
  chkpt.write("Cb",C);
  chkpt.write("Ea",E);
  chkpt.write("Eb",E);

  return 0;
}
