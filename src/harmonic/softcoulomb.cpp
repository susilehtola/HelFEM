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

arma::mat potential(const helfem::polynomial_basis::FiniteElementBasis & fem, const arma::vec & x, const arma::vec & wx, double z, double x0, double alpha, bool abs) {
  std::function<double(double)> soft_coulomb;
  if(abs) {
    soft_coulomb = [z, x0, alpha](double x) {
      return -z/(std::abs(x-x0)+alpha);
    };
  } else {
    soft_coulomb = [z, x0, alpha](double x) {
      return -z/sqrt((x-x0)*(x-x0)+alpha*alpha);
    };
  }
  return fem.matrix_element(false, false, x, wx, soft_coulomb);
}

arma::mat kinetic(const helfem::polynomial_basis::FiniteElementBasis & fem, const arma::vec & x, const arma::vec & wx) {
  return 0.5*fem.matrix_element(true, true, x, wx, nullptr);
}

int main(int argc, char **argv) {
  cmdline::parser parser;

  // full option name, no short option, description, argument required
  parser.add<double>("xmax", 0, "practical infinity in au", false, 40.0);
  parser.add<int>("nelem", 0, "number of elements", false, 5);
  parser.add<int>("nnodes", 0, "number of elements", false, 15);
  parser.add<int>("primbas", 0, "primitive basis", false, 4);
  parser.add<int>("nquad", 0, "primitive basis", false, -1);
  parser.add<int>("Z1", 0, "primitive basis", true);
  parser.add<int>("Z2", 0, "primitive basis", true);
  parser.add<double>("R", 0, "Bond length", true);
  parser.add<double>("alpha", 0, "Coulomb regularization parameter", true);
  parser.add<bool>("abs", 0, "Use 1/(|x-x0|+alpha) instead of 1/sqrt( (x-x0)^2 + alpha^2 ) as potential", false, 0);
  parser.add<std::string>("save", 0, "Checkpoint to save results to", false, "softcoulomb.chk");

  parser.parse_check(argc, argv);
  double xmax = parser.get<double>("xmax");
  int Nelem = parser.get<int>("nelem");
  int Nnodes = parser.get<int>("nnodes");
  int primbas = parser.get<int>("primbas");
  int Nquad = parser.get<int>("nquad");
  int Z1 = parser.get<int>("Z1");
  int Z2 = parser.get<int>("Z2");
  double R = parser.get<double>("R");
  double alpha = parser.get<double>("alpha");
  bool abs = parser.get<bool>("abs");
  std::string save = parser.get<std::string>("save");

  if(abs)
    printf("Using potential V(x) = -Z / ( |x-x0| + alpha )\n");
  else
    printf("Using potential V(x) = -Z / sqrt( (x-x0)^2 + alpha^2 )\n");

  // Get polynomial basis
  auto poly(std::shared_ptr<const helfem::polynomial_basis::PolynomialBasis>(helfem::polynomial_basis::get_basis(primbas, Nnodes)));
  if(Nquad<0)
    Nquad=5*poly->get_nbf();

  // Radial grid
  arma::vec x(arma::linspace<arma::vec>(-xmax,xmax,Nelem+1));

  // Finite element basis
  bool zero_func_left=true;
  bool zero_deriv_left=true;
  bool zero_func_right=true;
  bool zero_deriv_right=true;
  helfem::polynomial_basis::FiniteElementBasis fem(poly, x, zero_func_left, zero_deriv_left, zero_func_right, zero_deriv_right);

  // Quadrature rule
  arma::vec xq, wq;
  chebyshev::chebyshev(Nquad,xq,wq);

  size_t Nbf(fem.get_nbf());
  printf("Basis set contains %i functions\n",(int) Nbf);

  // Form overlap matrix
  arma::mat S(overlap(fem, xq, wq));
  // Form potential matrix
  arma::mat V(potential(fem, xq, wq, Z1, -0.5*R, alpha, abs)+potential(fem, xq, wq, Z2, 0.5*R, alpha, abs));
  // Form kinetic energy matrix
  arma::mat T(kinetic(fem, xq, wq));

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
