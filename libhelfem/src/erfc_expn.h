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
  namespace erfc_expn {
      /**
       * Computes the complementary error function expansion as
       * described in J. G. Ángyán, I. Gerber and M. Marsman, "Spherical
       * harmonic expansion of short-range screened Coulomb
       * interactions", J. Phys. A: Math. Gen. 39, 8613 (2006).
       *
       * Templated on the scalar type T and explicitly instantiated for
       * double, long double and (under HELFEM_HAVE_FLOAT128) _Float128.
       * The whole erfc/Legendre special-function chain runs at T, so the
       * range-separated (erfc) two-electron integrals carry T's precision
       * rather than being capped at double.
       */
      template <typename T> T Phi(unsigned int n, T Xi, T xi);
  }
}

#endif
