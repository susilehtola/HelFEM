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
#ifndef LIB1DFEM_LOBATTO_H
#define LIB1DFEM_LOBATTO_H

#include <lib1dfem/types.h>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace helfem {
namespace lib1dfem {
namespace lobatto {

/// Compute a Gauss-Lobatto quadrature rule for
///     integral_{-1}^{1} f(x) dx
///         ~ 2/(n*(n-1)) * (f(-1) + f(1)) + sum_{i=2}^{n-1} w_i f(x_i)
///
/// Templated on the scalar type T. Interior nodes are roots of
/// P'_{n-1}(x), found by Newton iteration starting from the
/// Chebyshev-Gauss-Lobatto guess. Weights are
/// w_i = 2 / (n * (n - 1) * P_{n-1}(x_i)^2).
template <typename T>
void lobatto_compute(int n, Vec<T> & x, Vec<T> & w) {
  if (n < 2) {
    std::ostringstream oss;
    oss << "Lobatto called with n=" << n << ", but n>=2 is required.\n";
    throw std::runtime_error(oss.str());
  }

  x.resize(n);
  w.resize(n);

  const T pi        = std::acos(T(-1));
  const T tolerance = T(100) * std::numeric_limits<T>::epsilon();

  // Initial guess: Chebyshev-Gauss-Lobatto nodes.
  for (int i = 0; i < n; ++i) {
    x(i) = std::cos(pi * T(i) / T(n - 1));
  }

  Vec<T> xold(n);
  // p(i, j) holds P_j(x_i) for j in [0, n-1].
  Mat<T> p(n, n);

  for (;;) {
    xold = x;

    // Recurrence for Legendre polynomials:
    //   P_0(x) = 1, P_1(x) = x
    //   j P_j(x) = (2j - 1) x P_{j-1}(x) - (j - 1) P_{j-2}(x)
    for (int i = 0; i < n; ++i) {
      p(i, 0) = T(1);
      p(i, 1) = x(i);
    }
    for (int j = 2; j <= n - 1; ++j) {
      const T two_jm1 = T(2 * j - 1);
      const T jm1     = T(j - 1);
      const T inv_j   = T(1) / T(j);
      for (int i = 0; i < n; ++i) {
        p(i, j) = (two_jm1 * x(i) * p(i, j - 1) - jm1 * p(i, j - 2)) * inv_j;
      }
    }

    // Newton step: roots of P'_{n-1}(x). Endpoints stay at +/-1.
    for (int i = 0; i < n; ++i) {
      x(i) = xold(i)
           - (x(i) * p(i, n - 1) - p(i, n - 2)) / (T(n) * p(i, n - 1));
    }

    // Max change for convergence check.
    T error = T(0);
    for (int i = 0; i < n; ++i) {
      const T test = std::abs(x(i) - xold(i));
      if (test > error) error = test;
    }
    if (error <= tolerance) break;
  }

  // Reverse order so the result runs from -1 to +1.
  x = x.reverse().eval();

  // Weights use P_{n-1}(x_i) at the (reversed) x values to keep w aligned with x.
  for (int i = 0; i < n; ++i) {
    T pn = T(1), pnm1 = T(1);
    T pnm2;
    pn = x(i);   // P_1
    for (int j = 2; j <= n - 1; ++j) {
      pnm2 = pnm1;
      pnm1 = pn;
      pn   = (T(2 * j - 1) * x(i) * pnm1 - T(j - 1) * pnm2) / T(j);
    }
    // pn = P_{n-1}(x_i)
    w(i) = T(2) / (T(n - 1) * T(n) * pn * pn);
  }
}

} // namespace lobatto
} // namespace lib1dfem
} // namespace helfem

#endif
