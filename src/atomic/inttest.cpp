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
#include "quadrature.h"
#include "polynomial_basis.h"
#include "polynomial.h"
#include "chebyshev.h"
#include "lobatto.h"

using namespace helfem;

void run(double R, int n_quad) {
  // Basis functions on [0, R]: x/R, (R-x)/R.

  // Get primitive polynomial representation for LIP
  arma::vec x, w;
  ::lobatto_compute(2,x,w);
  polynomial_basis::LIPBasis pbas(x,0);

  // Get quadrature rule
  arma::vec xq, wq;
  chebyshev::chebyshev(n_quad,xq,wq);

  // Get inner integral by quadrature
  arma::mat teiinner(quadrature::twoe_inner_integral(0,R,xq,wq,&pbas,0));

  // Test against analytical integrals. r values are
  arma::vec r(0.5*R*arma::ones<arma::vec>(xq.n_elem)+0.5*R*xq);

  // The inner integral should give the following
  arma::mat teiishould(r.n_elem,4);
  teiishould.col(0)=(arma::ones<arma::vec>(r.n_elem) - r/R + 1.0/3.0*arma::square(r)/(R*R));
  teiishould.col(1)=r%(-2.0*r + 3*R*arma::ones<arma::vec>(r.n_elem))/(6.0*R*R);
  teiishould.col(2)=teiishould.col(1);
  teiishould.col(3)=arma::square(r)/(3.0*R*R);

  r.save("r.dat",arma::raw_ascii);
  teiinner.save("teii_q.dat",arma::raw_ascii);
  teiishould.save("teii.dat",arma::raw_ascii);

  teiishould-=teiinner;
  printf("Error in inner integral is %e\n",arma::norm(teiishould,"fro"));

  arma::mat teiq(quadrature::twoe_integral(0,R,xq,wq,&pbas,0));

  arma::mat tei(4,4);
  // Maple gives the following integrals for L=0, in units of R

  // 1111
  tei(0,0) = 47.0/180.0;
  // 1112
  tei(0,1) = 11/360.0;
  // 1121
  tei(0,2) = tei(0,1);
  // 1122
  tei(0,3) = 1.0/90.0;

  // 1211
  tei(1,0) = 1.0/10.0;
  // 1212
  tei(1,1) = 1.0/40.0;
  // 1221
  tei(1,2) = tei(1,1);
  // 1222
  tei(1,3) = 1.0/60.0;

  // 2111
  tei(2,0) = tei(1,0);
  // 2112
  tei(2,1) = tei(1,1);
  // 2121
  tei(2,2) = tei(1,2);
  // 2122
  tei(2,3) = tei(1,3);

  // 2211
  tei(3,0) = 3.0/20.0;
  // 2212
  tei(3,1) = 7.0/120.0;
  // 2221
  tei(3,2) = tei(3,1);
  // 2222
  tei(3,3) = 1.0/15.0;

  // Symmetrization and coefficient
  tei=4.0*M_PI*(tei+tei.t())*R;

  tei.print("Analytical");
  teiq.print("Quadrature");
  teiq-=tei;
  teiq.print("Difference");
}

int main(int argc, char **argv) {
  if(argc!=3) {
    printf("Usage: %s nquad R\n",argv[0]);
    return 1;
  }

  int nquad(atoi(argv[1]));
  double R(atof(argv[2]));
  run(R,nquad);
  return 0;
}
