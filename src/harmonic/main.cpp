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

#include "PolynomialBasis.h"
#include "FiniteElementBasis.h"
#include "chebyshev.h"
#include "quadrature.h"

using namespace helfem;

arma::mat overlap(const helfem::polynomial_basis::FiniteElementBasis & fem, const arma::vec & x, const arma::vec & wx) {
  return fem.matrix_element(false, false, x, wx, nullptr);
}

double square_potential(double r) {
  return r*r;
}

arma::mat potential(const helfem::polynomial_basis::FiniteElementBasis & fem, const arma::vec & x, const arma::vec & wx) {
  return fem.matrix_element(false, false, x, wx, square_potential);
}

arma::mat kinetic(const helfem::polynomial_basis::FiniteElementBasis & fem, const arma::vec & x, const arma::vec & wx) {
  return fem.matrix_element(true, true, x, wx, nullptr);
}

int main(int argc, char **argv) {
  if(argc!=6) {
    printf("Usage: %s xmax Nel Nnode primbas Nquad\n",argv[0]);
    return 1;
  }

  // Maximum R
  double xmax=atof(argv[1]);
  // Number of elements
  int Nelem=atoi(argv[2]);

  // Number of nodes
  int Nnodes=atoi(argv[3]);
  // Derivative order
  int primbas=atoi(argv[4]);
  // Order of quadrature rule
  int Nquad=atoi(argv[5]);

  printf("Running calculation with xmax=%e and %i elements.\n",xmax,Nelem);
  printf("Using %i point quadrature rule.\n",Nquad);

  // Get polynomial basis
  auto poly(std::shared_ptr<const helfem::polynomial_basis::PolynomialBasis>(helfem::polynomial_basis::get_basis(primbas, Nnodes)));

  // Radial grid
  arma::vec r(arma::linspace<arma::vec>(-xmax,xmax,Nelem+1));

  // Finite element basis
  bool zero_func_left=true;
  bool zero_deriv_left=true;
  bool zero_func_right=true;
  bool zero_deriv_right=true;
  helfem::polynomial_basis::FiniteElementBasis fem(poly, r, zero_func_left, zero_deriv_left, zero_func_right, zero_deriv_right);

  // Quadrature rule
  arma::vec xq, wq;
  chebyshev::chebyshev(Nquad,xq,wq);

  // Evaluate polynomials at quadrature points
  arma::mat bf(poly->eval_dnf(xq, 0, 1.0));
  arma::mat dbf(poly->eval_dnf(xq, 1, 1.0));

  xq.save("x.dat",arma::raw_ascii);
  bf.save("bf.dat",arma::raw_ascii);
  dbf.save("dbf.dat",arma::raw_ascii);

  size_t Nbf(fem.get_nbf());
  printf("Basis set contains %i functions\n",(int) Nbf);

  // Form overlap matrix
  arma::mat S(overlap(fem, xq, wq));
  // Form potential matrix
  arma::mat V(potential(fem, xq, wq));
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

  printf("Eigenvalues\n");
  size_t neig=std::min(E.n_elem,(arma::uword) 8);
  for(size_t i=0;i<neig;i++)
    printf("%i % 10.6f % 10.6f\n",(int) i, E(i),E(i)-(2*i+1));

  // Test orthonormality
  arma::mat Smo(C.t()*S*C);
  Smo-=arma::eye<arma::mat>(Smo.n_rows,Smo.n_cols);
  printf("Orbital orthonormality devation is %e\n",arma::norm(Smo,"fro"));

  return 0;
}
