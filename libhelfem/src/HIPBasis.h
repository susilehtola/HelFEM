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
#ifndef POLYNOMIAL_BASIS_HIPBASIS_H
#define POLYNOMIAL_BASIS_HIPBASIS_H

// v2 refactor (Phase 1): the Hermite interpolating polynomial basis class
// now lives in lib1dfem as a templated header. libhelfem exposes the
// double-only instantiation under the original
// helfem::polynomial_basis::HIPBasis name so existing callers compile
// unchanged.

#include "PolynomialBasis.h"
#include "LIPBasis.h"
#include <lib1dfem/HIPBasis.h>

namespace helfem {
  namespace polynomial_basis {
    using HIPBasis = helfem::lib1dfem::polynomial_basis::HIPBasis<double>;
  }
}
#endif
