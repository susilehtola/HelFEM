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
#ifndef LIB1DFEM_CHEBYSHEV_H
#define LIB1DFEM_CHEBYSHEV_H

#include <lib1dfem/types.h>
#include <cmath>

namespace helfem {
namespace lib1dfem {
namespace chebyshev {

// Templated Gauss-Chebyshev quadrature primitives. libhelfem keeps a
// thin double-only shim at libhelfem/src/chebyshev.h for callers that
// still spell the output vectors as arma::vec.

/// Modified Gauss-Chebyshev quadrature of the second kind for
///     integral_{-1}^{1} f(x) dx
/// Templated on the scalar type T.
template <typename T>
void chebyshev(int n, Vec<T> & x, Vec<T> & w) {
  x = Vec<T>::Zero(n);
  w = Vec<T>::Zero(n);

  const T pi          = std::acos(T(-1));
  const T oonpp       = T(1) / T(n + 1);
  const T two_over_pi = T(2) / pi;

  for (int i = 1; i <= n; ++i) {
    const T angle  = T(i) * pi * oonpp;
    const T sine   = std::sin(angle);
    const T sinesq = sine * sine;
    const T cosine = std::cos(angle);

    // Weight
    w(i - 1) = T(16) / T(3) / T(n + 1) * sinesq * sinesq;
    // Node: 1 - 2 i/(n+1) + (2/pi) * (1 + (2/3) sin^2) * sin*cos
    x(i - 1) = T(1) - T(2) * T(i) * oonpp
             + two_over_pi * (T(1) + T(2) / T(3) * sinesq) * cosine * sine;
  }

  x = x.reverse().eval();
  w = w.reverse().eval();
}

/// Modified Gauss-Chebyshev quadrature of the second kind for
///     integral_{0}^{infty} f(r) dr
/// (For integration in spherical coordinates the caller must still
/// multiply by r^2.) Templated on the scalar type T.
template <typename T>
void radial_chebyshev(int nrad, Vec<T> & rad, Vec<T> & wrad) {
  Vec<T> xc, wc;
  chebyshev<T>(nrad, xc, wc);

  rad  = Vec<T>::Zero(nrad);
  wrad = Vec<T>::Zero(nrad);
  const T inv_ln2 = T(1) / std::log(T(2));

  for (int ir = 0; ir < nrad; ++ir) {
    const Eigen::Index ixc = xc.size() - 1 - ir;
    // r = (1 / ln 2) * log( 2 / (1 - x) )
    const T one_minus_x = T(1) - xc(ixc);
    const T r           = inv_ln2 * std::log(T(2) / one_minus_x);
    // Jacobian = 1 / (ln 2 * (1 - x))
    const T jac         = inv_ln2 / one_minus_x;
    rad(ixc)  = r;
    wrad(ixc) = wc(ixc) * jac;
  }
}

} // namespace chebyshev
} // namespace lib1dfem
} // namespace helfem

#endif
