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
#ifndef ATOMIC_ERFC_EXPN_H
#define ATOMIC_ERFC_EXPN_H

namespace helfem {
  namespace atomic {
    namespace erfc_expn {
      /// Damping functions
      double Dnk(int n, int k, double Xi);
      /// Short-range helper
      double Phi_short(unsigned int n, unsigned int k, double Xi, double xi);
      /// General expansion, unstable in short range
      double Phi_general(unsigned int n, double Xi, double xi);

      /**
       * Computes the complementary error function expansion as
       * described in J. G. Ángyán, I. Gerber and M. Marsman, "Spherical
       * harmonic expansion of short-range screened Coulomb
       * interactions", J. Phys. A: Math. Gen. 39, 8613 (2006).
       */
      double Phi(unsigned int n, double Xi, double xi);
    }
  }
}

#endif
