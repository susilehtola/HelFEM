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
#ifndef POLYNOMIAL_BASIS_LIPBASIS_H
#define POLYNOMIAL_BASIS_LIPBASIS_H

// v2 refactor (Phase 1): the Lagrange-interpolating-polynomial basis class
// now lives in lib1dfem as a templated header (and the auto-generated
// eval body is an instantiable header template too). libhelfem exposes the
// double-only instantiation under the original
// helfem::polynomial_basis::LIPBasis name so existing callers (HIPBasis,
// GeneralHIPBasis, get_basis() factory, ...) compile unchanged.

#include "PolynomialBasis.h"
#include <lib1dfem/LIPBasis.h>

namespace helfem {
  namespace polynomial_basis {
    using LIPBasis = helfem::lib1dfem::polynomial_basis::LIPBasis<double>;
  }
}
#endif
