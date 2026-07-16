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
#ifndef ANGULAR_H
#define ANGULAR_H

#include <Matrix.h>

namespace helfem {
  namespace angular {
    /// Form compound rule
    void compound_rule(const helfem::Vector & xth0, const helfem::Vector & wth0, int nphi, helfem::Vector & th, helfem::Vector & phi, helfem::Vector & w);

    /// Angular quadrature rule of order (l,l)
    void angular_lobatto(int l, helfem::Vector & cth, helfem::Vector & phi, helfem::Vector & w);
    /// Angular quadrature rule of order (l,m)
    void angular_lobatto(int l, int m, helfem::Vector & cth, helfem::Vector & phi, helfem::Vector & w);

    /// Angular quadrature rule of order (l,l)
    void angular_chebyshev(int l, helfem::Vector & cth, helfem::Vector & phi, helfem::Vector & w);
    /// Angular quadrature rule of order (l,m)
    void angular_chebyshev(int l, int m, helfem::Vector & cth, helfem::Vector & phi, helfem::Vector & w);
  }
}

#endif
