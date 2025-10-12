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
#include "PolynomialBasis.h"
#include "FiniteElementBasis.h"
#include "chebyshev.h"

using namespace helfem;

arma::mat overlap(const helfem::polynomial_basis::FiniteElementBasis & fem, const arma::vec & x, const arma::vec & wx) {
  return fem.matrix_element(false, false, x, wx, nullptr);
}

arma::mat potential(const helfem::polynomial_basis::FiniteElementBasis & fem, const arma::vec & x, const arma::vec & wx, double De, double a, double re) {
  std::function<double(double)> V = [&](double r) {
    return De*std::pow(1-exp(-a*(r-re)),2);
  };
  return fem.matrix_element(false, false, x, wx, V);
}

arma::mat kinetic(const helfem::polynomial_basis::FiniteElementBasis & fem, const arma::vec & x, const arma::vec & wx) {
  return 0.5*fem.matrix_element(true, true, x, wx, nullptr);
}

int main(int argc, char **argv) {
  cmdline::parser parser;

  // full option name, no short option, description, argument required
  parser.add<double>("rmax", 0, "practical infinity in au", false, 40.0);
  parser.add<int>("nelem", 0, "number of elements", false, 5);
  parser.add<int>("nnodes", 0, "number of elements", false, 15);
  parser.add<int>("primbas", 0, "primitive basis", false, 4);
  parser.add<int>("nquad", 0, "primitive basis", false, -1);
  parser.add<double>("De", 0, "dissociation energy", false, 1.0);
  parser.add<double>("a", 0, "inverse length scale", false, 1.0);
  parser.add<double>("re", 0, "equilibrium bond length", false, 1.0);
  parser.add<double>("m", 0, "mass", false, 1.0);
  parser.add<bool>("deuteron", 0, "deuteron?", false, false);
  parser.add<std::string>("save", 0, "Checkpoint to save results to", false, "morse.chk");

  parser.parse_check(argc, argv);
  double rmax = parser.get<double>("rmax");
  int Nelem = parser.get<int>("nelem");
  int Nnodes = parser.get<int>("nnodes");
  int primbas = parser.get<int>("primbas");
  int Nquad = parser.get<int>("nquad");
  double De = parser.get<double>("De");
  double a = parser.get<double>("a");
  double re = parser.get<double>("re");
  double m = parser.get<double>("m");
  bool deuteron = parser.get<bool>("deuteron");
  std::string save = parser.get<std::string>("save");

  // Get polynomial basis
  auto poly(std::shared_ptr<const helfem::polynomial_basis::PolynomialBasis>(helfem::polynomial_basis::get_basis(primbas, Nnodes)));
  if(Nquad<0)
    Nquad=5*poly->get_nbf();

  // Radial grid
  arma::vec r(arma::linspace<arma::vec>(0,rmax,Nelem+1));

  // Finite element basis
  bool zero_func_left=true;
  bool zero_deriv_left=false;
  bool zero_func_right=true;
  bool zero_deriv_right=false;
  helfem::polynomial_basis::FiniteElementBasis fem(poly, r, zero_func_left, zero_deriv_left, zero_func_right, zero_deriv_right);

  // Quadrature rule
  arma::vec xq, wq;
  chebyshev::chebyshev(Nquad,xq,wq);

  size_t Nbf(fem.get_nbf());
  printf("Basis set contains %i functions\n",(int) Nbf);

  // Form overlap matrix
  arma::mat S(overlap(fem, xq, wq));
  // Form potential matrix
  arma::mat V(potential(fem, xq, wq, De, a, re));
  // Form kinetic energy matrix
  arma::mat T(kinetic(fem, xq, wq));
  if(deuteron)
    T/=2.0;

  // Form Hamiltonian
  arma::mat H(T+V);

  //S.print("Overlap");
  //T.print("Kinetic");
  //V.print("Potential");
  //H.print("Hamiltonian");

  // Form orthonormal basis
  arma::vec Sval;
  arma::mat Svec;
  arma::eig_sym(Sval,Svec,S);

  //Sval.print("S eigenvalues");
  printf("Smallest value of overlap matrix is % e, condition number is %e\n",Sval(0),Sval(Sval.n_elem-1)/Sval(0));
  printf("Smallest and largest bf norms are %e and %e\n",arma::min(arma::abs(arma::diagvec(S))),arma::max(arma::abs(arma::diagvec(S))));

  // Form half-inverse
  arma::mat Sinvh(Svec * arma::diagmat(arma::pow(Sval, -0.5)) * arma::trans(Svec));

  // Form orthonormal Hamiltonian
  arma::mat Horth(arma::trans(Sinvh)*H*Sinvh);

  // Diagonalize Hamiltonian
  arma::vec E;
  arma::mat C;
  arma::eig_sym(E,C,Horth);

  // Go back to non-orthonormal basis
  C=Sinvh*C;

  for(size_t i=0;i<10;i++)
    printf("E[%i] = % .15e\n",(int) i, E[i]);

  // Test orthonormality
  arma::mat Smo(C.t()*S*C);
  Smo-=arma::eye<arma::mat>(Smo.n_rows,Smo.n_cols);
  printf("Orbital orthonormality devation is %e\n",arma::norm(Smo,"fro"));

  // Evaluate the basis set: 0th derivative
  arma::mat bfval(fem.eval_dnf(xq, 0));
  arma::mat phival(bfval*C);
  arma::mat coords(fem.eval_coord(xq));
  arma::mat weights(fem.eval_weights(wq));

  // Test orbitals are still orthonormal
  arma::mat Sgrid(phival.t()*arma::diagmat(weights)*phival);
  Sgrid-=arma::eye<arma::mat>(Sgrid.n_rows,Sgrid.n_cols);
  printf("Orbital orthonormality devation on grid is %e\n",arma::norm(Sgrid,"fro"));

  Checkpoint chkpt(save, true);
  chkpt.write("bf",bfval);
  chkpt.write("C",C);
  chkpt.write("phi",phival);
  chkpt.write("coords",coords);
  chkpt.write("weights",weights);

  return 0;
}
