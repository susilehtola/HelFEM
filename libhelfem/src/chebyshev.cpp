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
#include "chebyshev.h"

namespace helfem {
  namespace chebyshev {
    // Modified Gauss-Chebyshev quadrature of the second kind for calculating
    // \int_{-1}^{1} f(x) dx
    void chebyshev(int n, arma::vec & x, arma::vec & w) {
      // Resize vectors to correct size
      x.zeros(n);
      w.zeros(n);

      // 1/(n+1)
      double oonpp=1.0/(n+1.0);

      // cos ( i*pi / (n+1))
      double cosine;
      // sin ( i*pi / (n+1))
      double sine;
      double sinesq;

      // Fill elements
      for(int i=1;i<=n;i++) {
        // Compute value of sine and cosine
        sine=sin(i*M_PI*oonpp);
        sinesq=sine*sine;
        cosine=cos(i*M_PI*oonpp);

        // Weight is
        w(i-1)=16.0/3.0/(n+1.0)*sinesq*sinesq;

        // Node is
        x(i-1)=1.0 - 2.0*i*oonpp + M_2_PI*(1.0 + 2.0/3.0*sinesq)*cosine*sine;
      }

      // Reverse order
      x=reverse(x);
      w=reverse(w);
    }

    void radial_chebyshev(int nrad, arma::vec & rad, arma::vec & wrad) {
      // Get Chebyshev nodes and weights for radial part
      arma::vec xc, wc;
      chebyshev(nrad,xc,wc);

      // Compute radii
      rad.zeros(nrad);
      wrad.zeros(nrad);
      for(int ir=0;ir<nrad;ir++) {
        // Calculate value of radius
        double ixc=xc.n_elem-1-ir;
        double r=1.0/M_LN2*log(2.0/(1.0-xc(ixc)));

        // Jacobian of transformation is
        double jac=1.0/M_LN2/(1.0-xc(ixc));
        // so total quadrature weight (excluding r^2!) is
        double w=wc[ixc]*jac;

        rad(ixc) = r;
        wrad(ixc) = w;
      }
    }
  }
}
