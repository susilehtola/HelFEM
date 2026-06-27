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
#ifndef LOBATTO_H
#define LOBATTO_H

// v2 refactor (Phase 1): the Gauss-Lobatto quadrature implementation now
// lives in lib1dfem and is templated on the scalar type. libhelfem is
// still compiled against T = double, so this header is a thin double-only
// compatibility shim that forwards to the templated lib1dfem version. The
// shim keeps existing callers (libhelfem-internal and downstream) source-
// compatible during the migration.

#include <lib1dfem/lobatto.h>

/// Compute a Gauss-Lobatto quadrature rule for
///     integral_{-1}^{1} f(x) dx
///         ~ 2/(n*(n-1)) * (f(-1) + f(1)) + sum_{i=2}^{n-1} w_i f(x_i)
inline void lobatto_compute(int n, arma::vec & x, arma::vec & w) {
  helfem::lib1dfem::lobatto::lobatto_compute<double>(n, x, w);
}

#endif
