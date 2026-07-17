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
#include "angular.h"
#include "chebyshev.h"
#include "lobatto.h"

namespace helfem {
  namespace angular {
    void compound_rule(const helfem::Vector & xcth0, const helfem::Vector & wcth0, int nphi, helfem::Vector & cth, helfem::Vector & phi, helfem::Vector & w) {
      // Form compound rule
      cth=helfem::Vector::Zero(xcth0.size()*nphi);
      phi=helfem::Vector::Zero(xcth0.size()*nphi);
      w=helfem::Vector::Zero(xcth0.size()*nphi);

      /*
	An evenly spaced phi grid of n points integrates cos (m phi)
	and sin (m phi) exactly for m = 0, 1, ..., n-1. [Murray,
	Handy, Laming, Molecular Physics 78, 997 (1993).

	Phi spacing is
      */
      double dphi=2.0*M_PI/nphi;

      for(size_t i=0;i<(size_t) xcth0.size();i++)
        for(int j=0;j<nphi;j++) {
          // Global index
          size_t idx=i*nphi+j;
          cth(idx)=xcth0(i);
          phi(idx)=j*dphi;
          w(idx)=wcth0(i)*dphi;
        }
    }

    void angular_lobatto(int l, helfem::Vector & cth, helfem::Vector & phi, helfem::Vector & wang) {
      angular_lobatto(l,l,cth,phi,wang);
    }

    void angular_lobatto(int l, int m, helfem::Vector & cth, helfem::Vector & phi, helfem::Vector & wang) {
      // Get input quadrature: l part
      helfem::Vector xl, wl;
      helfem::lobatto::lobatto_compute<double>(l,xl,wl);

      // Form compound rule
      compound_rule(xl,wl,m,cth,phi,wang);
    }

    void angular_chebyshev(int l, helfem::Vector & cth, helfem::Vector & phi, helfem::Vector & wang) {
      angular_chebyshev(l,l,cth,phi,wang);
    }

    void angular_chebyshev(int l, int m, helfem::Vector & cth, helfem::Vector & phi, helfem::Vector & wang) {
      // Get input quadrature: l part
      helfem::Vector xl, wl;
      chebyshev::chebyshev(l,xl,wl);

      // Form compound rule
      compound_rule(xl,wl,m,cth,phi,wang);
    }
  }
}
