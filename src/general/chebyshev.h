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
#ifndef CHEBYSHEV_H
#define CHEBYSHEV_H

// v2 refactor (Phase 1): the actual Gauss-Chebyshev quadrature now lives in
// lib1dfem and is templated on the scalar type. This header is a thin
// double-only compatibility shim that forwards to the templated lib1dfem
// implementation at T = double, spelling the output vectors as the
// Eigen-typed helfem::Vector. lib1dfem::Vec<double> IS helfem::Vector
// (both are Eigen::Matrix<double, Dynamic, 1>), so the forward is a direct
// call with no copy.

#include <lib1dfem/chebyshev.h>
#include <Matrix.h>

namespace helfem {
namespace chebyshev {

inline void chebyshev(int n, helfem::Vector & x, helfem::Vector & w) {
  lib1dfem::chebyshev::chebyshev<double>(n, x, w);
}

inline void radial_chebyshev(int n, helfem::Vector & r, helfem::Vector & wr) {
  lib1dfem::chebyshev::radial_chebyshev<double>(n, r, wr);
}

} // namespace chebyshev
} // namespace helfem

#endif
