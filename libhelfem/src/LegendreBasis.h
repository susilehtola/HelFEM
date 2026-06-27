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
#ifndef POLYNOMIAL_BASIS_LEGENDREBASIS_H
#define POLYNOMIAL_BASIS_LEGENDREBASIS_H

// v2 refactor (Phase 1): the Legendre spectral-element shape-function class
// now lives in lib1dfem as a templated header. libhelfem exposes the
// double-only instantiation under the original
// helfem::polynomial_basis::LegendreBasis name so existing callers and the
// get_basis() factory continue to compile without source changes.

#include "PolynomialBasis.h"
#include <lib1dfem/LegendreBasis.h>

namespace helfem {
  namespace polynomial_basis {
    using LegendreBasis = helfem::lib1dfem::polynomial_basis::LegendreBasis<double>;
  }
}
#endif
