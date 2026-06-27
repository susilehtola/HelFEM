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
#ifndef POLYNOMIAL_BASIS_POLYNOMIALBASIS_H
#define POLYNOMIAL_BASIS_POLYNOMIALBASIS_H

// v2 refactor (Phase 1): the abstract PolynomialBasis class is now a
// template living in lib1dfem. libhelfem is compiled at T = double, so
// this header exposes the double-only instantiation under the original
// helfem::polynomial_basis::PolynomialBasis name -- existing concrete
// subclasses (LIPBasis, HIPBasis, GeneralHIPBasis, LegendreBasis) and
// downstream callers continue to compile without source changes.

#include <lib1dfem/PolynomialBasis.h>

namespace helfem {
  namespace polynomial_basis {
    /// Alias for the lib1dfem template at T = double.
    using PolynomialBasis = helfem::lib1dfem::polynomial_basis::PolynomialBasis<double>;

    /// Factory: construct a primitive polynomial basis by ID.
    PolynomialBasis * get_basis(int primbas, int Nnodes);
  }
}

#endif
