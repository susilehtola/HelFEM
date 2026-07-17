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
#ifndef HELFEM_FEM_LEGENDRE_POLY_H
#define HELFEM_FEM_LEGENDRE_POLY_H

#include <types.h>

namespace helfem {
namespace legendre {

/// Evaluate Legendre polynomials P_l(x) for l = 0, 1, ..., lmax at every
/// point x in `x`. Returns a (n_points x (lmax+1)) matrix with P_l(x_i) at
/// (i, l). Templated on the scalar type T.
///
/// Recurrence: P_0(x) = 1, P_1(x) = x;
///             (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x).
template <typename T>
Mat<T> legendre_batch(int lmax, const Vec<T> & x) {
  Mat<T> P(x.size(), lmax + 1);
  for (Eigen::Index i = 0; i < x.size(); ++i) {
    P(i, 0) = T(1);
    if (lmax >= 1) P(i, 1) = x(i);
    for (int n = 1; n < lmax; ++n) {
      P(i, n + 1) = (T(2*n + 1) * x(i) * P(i, n) - T(n) * P(i, n - 1)) / T(n + 1);
    }
  }
  return P;
}

/// First derivatives P_l'(x), same layout.
template <typename T>
Mat<T> dlegendre_batch(int lmax, const Vec<T> & x) {
  Mat<T> P = legendre_batch<T>(lmax, x);
  Mat<T> dP(x.size(), lmax + 1);
  for (Eigen::Index i = 0; i < x.size(); ++i) {
    dP(i, 0) = T(0);
    if (lmax >= 1) dP(i, 1) = T(1);
    for (int n = 1; n < lmax; ++n) {
      dP(i, n + 1) = (T(2*n + 1) * P(i, n)
                    + T(2*n + 1) * x(i) * dP(i, n)
                    - T(n) * dP(i, n - 1)) / T(n + 1);
    }
  }
  return dP;
}

/// Second derivatives P_l''(x), same layout.
template <typename T>
Mat<T> d2legendre_batch(int lmax, const Vec<T> & x) {
  Mat<T> dP = dlegendre_batch<T>(lmax, x);
  Mat<T> d2P(x.size(), lmax + 1);
  for (Eigen::Index i = 0; i < x.size(); ++i) {
    d2P(i, 0) = T(0);
    if (lmax >= 1) d2P(i, 1) = T(0);
    for (int n = 1; n < lmax; ++n) {
      d2P(i, n + 1) = (T(2 * (2*n + 1)) * dP(i, n)
                     + T(2*n + 1) * x(i) * d2P(i, n)
                     - T(n) * d2P(i, n - 1)) / T(n + 1);
    }
  }
  return d2P;
}

} // namespace legendre
} // namespace helfem

#endif
