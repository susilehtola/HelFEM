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
#ifndef ATOMIC_ERFC_EXPN_H
#define ATOMIC_ERFC_EXPN_H

namespace helfem {
  namespace atomic {
    namespace erfc_expn {
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
