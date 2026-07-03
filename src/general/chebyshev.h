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
// lib1dfem and is templated on the scalar type. libhelfem is still compiled
// against T = double, so this header is a thin double-only compatibility
// shim that forwards to the templated lib1dfem implementation. The shim
// keeps libhelfem's internal callers (and any external consumer using the
// libhelfem headers) source-compatible during the migration.

#include <lib1dfem/chebyshev.h>
#include <armadillo>
#include <cstring>

namespace helfem {
namespace chebyshev {

// Phase 5.1: lib1dfem chebyshev is now Eigen-typed. These libhelfem
// shims preserve the legacy arma::vec API used throughout the rest of
// the codebase by bridging once at the boundary -- one Eigen Vector
// build + one arma::vec copy per call. Both quadrature routines are
// called a small constant number of times per SCF setup, so the bridge
// cost is negligible.
inline void chebyshev(int n, arma::vec & x, arma::vec & w) {
  Eigen::Matrix<double, Eigen::Dynamic, 1> xe, we;
  lib1dfem::chebyshev::chebyshev<double>(n, xe, we);
  x.set_size(xe.size()); w.set_size(we.size());
  std::memcpy(x.memptr(), xe.data(), sizeof(double) * (size_t) xe.size());
  std::memcpy(w.memptr(), we.data(), sizeof(double) * (size_t) we.size());
}

inline void radial_chebyshev(int n, arma::vec & r, arma::vec & wr) {
  Eigen::Matrix<double, Eigen::Dynamic, 1> re, wre;
  lib1dfem::chebyshev::radial_chebyshev<double>(n, re, wre);
  r.set_size(re.size()); wr.set_size(wre.size());
  std::memcpy(r.memptr(),  re.data(),  sizeof(double) * (size_t) re.size());
  std::memcpy(wr.memptr(), wre.data(), sizeof(double) * (size_t) wre.size());
}

} // namespace chebyshev
} // namespace helfem

#endif
