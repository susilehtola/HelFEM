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
#ifndef LIB1DFEM_MATH_H
#define LIB1DFEM_MATH_H

#include <armadillo>
#include <cmath>
#include <algorithm>
#include <limits>

namespace helfem {
namespace lib1dfem {
namespace math {

/// Inverse cosh. x is mathematically >= 1 but is clamped to 1 because the
/// caller may pass values that round to slightly below 1 (e.g. cosh(mu) for
/// very small mu).
template <typename T>
T arcosh(T x) {
  return std::acosh(std::max(x, T(1)));
}

template <typename T>
arma::Col<T> arcosh(const arma::Col<T> & x) {
  arma::Col<T> y(x.n_elem);
  for (size_t i = 0; i < x.n_elem; ++i) y(i) = arcosh<T>(x(i));
  return y;
}

/// Inverse sinh.
template <typename T>
T arsinh(T x) {
  return std::asinh(x);
}

template <typename T>
arma::Col<T> arsinh(const arma::Col<T> & x) {
  arma::Col<T> y(x.n_elem);
  for (size_t i = 0; i < x.n_elem; ++i) y(i) = arsinh<T>(x(i));
  return y;
}

/// Modified Bessel function i_L(x), where
///     i_L(x) = sqrt(pi/(2|x|)) * I_{L+1/2}(|x|),  i_L(-x) = (-1)^L i_L(x).
///
/// For small |x| we evaluate the Taylor series
///     i_L(x) = x^L/(2L+1)!! * sum_{k>=0} (x^2/2)^k / (k! prod_{j=1..k}(2L+2j+1))
/// to dodge the sqrt(pi/(2x))*I_{L+1/2}(x) ~ 0/0 limit at x=0; the standard
/// library's small-argument behaviour for cyl_bessel_i is also more lossy
/// than the explicit series.
///
/// For larger |x| we fall back to the C++17 std::cyl_bessel_i, which
/// overloads on float/double/long double. For __float128 (or any other T
/// without a std::cyl_bessel_i overload) a user-supplied specialisation
/// is required.
template <typename T>
T bessel_il(T r, int L) {
  const T absr = std::abs(r);
  T val;
  if (absr < T(0.5)) {
    // (2L+1)!!
    T dfac = T(1);
    for (int j = 3; j <= 2*L + 1; j += 2) dfac *= T(j);
    T term = std::pow(absr, L) / dfac;
    val = term;
    const T r2half = T(0.5) * absr * absr;
    const T tol    = std::numeric_limits<T>::epsilon() * T(0.01);
    for (int k = 1; k < 256; ++k) {
      term *= r2half / (T(k) * T(2*L + 2*k + 1));
      val += term;
      if (std::abs(term) <= tol * std::abs(val)) break;
    }
  } else {
    const T pi = std::acos(T(-1));
    val = std::cyl_bessel_i(T(L) + T(0.5), absr) * std::sqrt(pi / (T(2) * absr));
  }
  return (r < T(0) && (L & 1)) ? -val : val;
}

template <typename T>
arma::Col<T> bessel_il(const arma::Col<T> & r, int L) {
  arma::Col<T> ret(r.n_elem);
  for (size_t i = 0; i < ret.n_elem; ++i) ret(i) = bessel_il<T>(r(i), L);
  return ret;
}

/// Modified Bessel function k_L(r), normalised as
///     k_L(r) = sqrt(2/(pi r)) * K_{L+1/2}(r).
/// Singular at r=0.
///
/// Uses C++17 std::cyl_bessel_k under the hood; same precision-portability
/// caveat as bessel_il (user-supplied specialisation needed for T without
/// a std::cyl_bessel_k overload).
template <typename T>
T bessel_kl(T r, int L) {
  const T pi = std::acos(T(-1));
  return std::cyl_bessel_k(T(L) + T(0.5), r) * std::sqrt(T(2) / (pi * r));
}

template <typename T>
arma::Col<T> bessel_kl(const arma::Col<T> & r, int L) {
  arma::Col<T> ret(r.n_elem);
  for (size_t i = 0; i < ret.n_elem; ++i) ret(i) = bessel_kl<T>(r(i), L);
  return ret;
}

} // namespace math
} // namespace lib1dfem
} // namespace helfem

#endif
