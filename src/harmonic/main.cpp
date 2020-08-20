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
#include "polynomial.h"
#include "../general/polynomial_basis.h"
#include "chebyshev.h"
#include "quadrature.h"

using namespace helfem;

size_t get_Nbf(const arma::vec & r, const arma::mat & bf, int noverlap) {
  // Number of elements is
  size_t Nelem=r.n_elem-1;
  // Count the number of basis functions
  size_t Nbf=bf.n_cols*Nelem;
  // .. minus the number of functions that overlap between elements
  Nbf-=(Nelem-1)*noverlap;

  return Nbf;
}

void get_functions(size_t iel, const arma::mat & bf, int noverlap, size_t & ifirst, size_t & ilast) {
  // The first function will be
  ifirst=iel*(bf.n_cols-noverlap);
  // and the last one
  ilast=ifirst+bf.n_cols-1;
}

/*
arma::mat l2n2_element_overlap(double xmin, double h) {
  arma::mat S(2,2);
  S(0,0)=S(1,1)=h/3;
  S(0,1)=S(1,0)=h/6;
  return S;
}

arma::mat l2n2_element_kinetic(double xmin, double h) {
  arma::mat T(2,2);
  T(0,0)=T(1,1)=1/h;
  T(0,1)=T(1,0)=-1/h;
  return T;
}

arma::mat l2n2_element_potential(double xmin, double h) {
  arma::mat V0(2,2);
  V0(0,0)=V0(1,1)=1.0/3.0;
  V0(0,1)=V0(1,0)=1.0/6.0;

  arma::mat V1(2,2);
  V1(0,0)=V1(0,1)=V1(1,0)=1.0/6.0;
  V1(1,1)=1.0/2.0;

  arma::mat V2(2,2);
  V2(0,0)=1.0/30.0;
  V2(0,1)=V2(1,0)=1.0/20.0;
  V2(1,1)=1.0/5.0;

  return V0*h*xmin*xmin + V1*h*h*xmin + V2*h*h*h;
}
*/

arma::mat overlap(const arma::vec & r, const arma::vec & x, const arma::vec & wx, const arma::mat & bf, int noverlap) {
  // Build overlap matrix
  size_t Nbf(get_Nbf(r,bf,noverlap));
  arma::mat S(Nbf,Nbf);
  S.zeros();

  // Loop over elements
  for(size_t iel=0;iel<r.n_elem-1;iel++) {
    // Get the primitive overlap matrix
    arma::mat Sel(quadrature::radial_integral(r(iel),r(iel+1),0,x,wx,bf));

    // Where are we in the matrix?
    size_t ifirst, ilast;
    get_functions(iel,bf,noverlap,ifirst,ilast);
    S.submat(ifirst,ifirst,ilast,ilast)+=Sel;

    //printf("Element %i: functions %i - %i\n",(int) iel,(int) ifirst, (int) ilast);
  }

  return S;
}

arma::mat potential(const arma::vec & r, const arma::vec & x, const arma::vec & wx, const arma::mat & bf, int noverlap) {
  // Build nuclear attraction matrix
  size_t Nbf(get_Nbf(r,bf,noverlap));
  arma::mat V(Nbf,Nbf);
  V.zeros();

  // Loop over elements
  for(size_t iel=0;iel<r.n_elem-1;iel++) {
    // Get the primitive overlap matrix
    arma::mat Vel(quadrature::radial_integral(r(iel),r(iel+1),2,x,wx,bf));

    // Where are we in the matrix?
    size_t ifirst, ilast;
    get_functions(iel,bf,noverlap,ifirst,ilast);
    V.submat(ifirst,ifirst,ilast,ilast)+=Vel;
  }

  return V;
}

arma::mat kinetic(const arma::vec & r, const arma::vec & x, const arma::vec & wx, const arma::mat & bf, const arma::mat & dbf, int noverlap) {
  // Build kinetic energy matrix
  size_t Nbf(get_Nbf(r,bf,noverlap));
  arma::mat T(Nbf,Nbf);
  T.zeros();

  // Loop over elements
  for(size_t iel=0;iel<r.n_elem-1;iel++) {
    // Get the primitive overlap matrix
    arma::mat Tel(quadrature::derivative_integral(r(iel),r(iel+1),x,wx,dbf));

    // Where are we in the matrix?
    size_t ifirst, ilast;
    get_functions(iel,bf,noverlap,ifirst,ilast);
    T.submat(ifirst,ifirst,ilast,ilast)+=Tel;
  }

  return T;
}

arma::mat remove_edges(const arma::mat & M, int noverlap) {
  // Full list of functions is
  arma::uvec idx(arma::linspace<arma::uvec>(0,M.n_cols-1,M.n_cols));

  // Drop first function
  idx=idx.subvec(1,idx.n_elem-1);
  // Drop last function
  arma::uvec newidx(idx.n_elem-1);
  newidx.subvec(0,idx.n_elem-noverlap-1)=idx.subvec(0,idx.n_elem-noverlap-1);
  if(noverlap>1)
    newidx.subvec(idx.n_elem-noverlap,newidx.n_elem-1)=idx.subvec(idx.n_elem-noverlap+1,idx.n_elem-1);
  idx=newidx;

  // Return submatrix
  return M(idx,idx);
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

  // Get primitive basis
  polynomial_basis::PolynomialBasis *poly(polynomial_basis::get_basis(primbas,Nnodes));
  int noverlap=poly->get_noverlap();

  printf("Using %i point quadrature rule.\n",Nquad);
  // Radial grid
  arma::vec r(arma::linspace<arma::vec>(-xmax,xmax,Nelem+1));

  // Quadrature rule
  arma::vec x, wx;
  chebyshev::chebyshev(Nquad,x,wx);

  // Evaluate polynomials at quadrature points
  arma::mat bf, dbf;
  poly->eval(x,bf,dbf);

  x.save("x.dat",arma::raw_ascii);
  bf.save("bf.dat",arma::raw_ascii);
  dbf.save("dbf.dat",arma::raw_ascii);

  size_t Nbf(get_Nbf(r,bf,noverlap));
  printf("Basis set contains %i functions\n",(int) Nbf);

  // Form overlap matrix
  arma::mat S(overlap(r,x,wx,bf,noverlap));
  // Form potential matrix
  arma::mat V(potential(r,x,wx,bf,noverlap));
  // Form kinetic energy matrix
  arma::mat T(kinetic(r,x,wx,bf,dbf,noverlap));

  // Form Hamiltonian
  arma::mat H(T+V);

  // Get rid of the spurious trail elements
  S=remove_edges(S,noverlap);
  T=remove_edges(T,noverlap);
  V=remove_edges(V,noverlap);
  H=remove_edges(H,noverlap);

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
