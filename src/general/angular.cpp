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
#include "angular.h"
#include "chebyshev.h"
#include "lobatto.h"

namespace helfem {
  namespace angular {
    void compound_rule(const arma::vec & xcth0, const arma::vec & wcth0, int nphi, arma::vec & cth, arma::vec & phi, arma::vec & w) {
      // Form compound rule
      cth.zeros(xcth0.n_elem*nphi);
      phi.zeros(xcth0.n_elem*nphi);
      w.zeros(xcth0.n_elem*nphi);

      /*
	An evenly spaced phi grid of n points integrates cos (m phi)
	and sin (m phi) exactly for m = 0, 1, ..., n-1. [Murray,
	Handy, Laming, Molecular Physics 78, 997 (1993).

	Phi spacing is
      */
      double dphi=2.0*M_PI/nphi;
      
      for(size_t i=0;i<xcth0.n_elem;i++)
        for(int j=0;j<nphi;j++) {
          // Global index
          size_t idx=i*nphi+j;
          cth(idx)=xcth0(i);
          phi(idx)=j*dphi;
          w(idx)=wcth0(i)*dphi;
        }
    }

    void angular_lobatto(int l, arma::vec & cth, arma::vec & phi, arma::vec & wang) {
      angular_lobatto(l,l,cth,phi,wang);
    }

    void angular_lobatto(int l, int m, arma::vec & cth, arma::vec & phi, arma::vec & wang) {
      // Get input quadrature: l part
      arma::vec xl, wl;
      ::lobatto_compute(l,xl,wl);

      // Form compound rule
      compound_rule(xl,wl,m,cth,phi,wang);
    }

    void angular_chebyshev(int l, arma::vec & cth, arma::vec & phi, arma::vec & wang) {
      angular_chebyshev(l,l,cth,phi,wang);
    }

    void angular_chebyshev(int l, int m, arma::vec & cth, arma::vec & phi, arma::vec & wang) {
      // Get input quadrature: l part
      arma::vec xl, wl;
      chebyshev::chebyshev(l,xl,wl);

      // Form compound rule
      compound_rule(xl,wl,m,cth,phi,wang);
    }
  }
}
