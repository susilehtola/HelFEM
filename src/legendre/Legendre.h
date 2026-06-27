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

// Associated Legendre functions of the first and second kinds, P_l^m(x) and
// Q_l^m(x), for arbitrary real x.
//
// Convention. The Hobson form (no Condon-Shortley phase) is used outside the
// cut, |x| > 1; on the cut |x| < 1 the standard three-term recurrence is
// seeded with P_m^m = -s (2m-1) sqrt(s (1-x^2)) P_(m-1)^(m-1) where s =
// sign(1 - x^2), which leaves the (-1)^m sign on the cut.
//
// Storage. Output arrays are column-major:
//
//     out[m * (lmax+1) + l]  holds the (l, m) entry, 0 <= l <= lmax, 0 <= m <= mmax.
//
// Entries with m > l are physical zeros and are written as 0.
//
// Templating. The implementation is parameterised on the floating-point type
// T. Explicit instantiations for `double` (and `long double`) ship with the
// library; users may instantiate other types (e.g. boost::multiprecision)
// just by including this header. The unqualified math functions (log, sqrt,
// abs, cosh) resolve via ADL so no `using std::...` is needed for non-std
// types.
//
// Threading. Each call allocates from a thread-local workspace, so successive
// calls within a thread allocate only when (lmax, mmax) grow. The functions
// are reentrant and may be called concurrently from multiple threads.

#ifndef HELFEM_LEGENDRE_H
#define HELFEM_LEGENDRE_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace helfem {
namespace legendre {

namespace detail {

// Column-major index helper.
inline std::size_t idx(int l, int m, int lmax) {
  return static_cast<std::size_t>(m) * static_cast<std::size_t>(lmax + 1) +
         static_cast<std::size_t>(l);
}

// Per-type math wrappers. The default goes via std:: which works for any
// floating type with std overloads (double, long double, boost::multiprecision
// types, ...). Users wanting __float128 (or any other type whose std
// overload set is ambiguous) include the matching adapter header before
// instantiating; that adapter specialises these to whatever the type's
// libm-equivalent is (e.g. logq for __float128).
template <typename T> inline T m_log (T x) { return std::log (x); }
template <typename T> inline T m_sqrt(T x) { return std::sqrt(x); }
template <typename T> inline T m_cosh(T x) { return std::cosh(x); }
template <typename T> inline T m_abs (T x) { return std::abs (x); }

template <typename T>
inline T sign_factor(T x) {
  // s = sign(1 - x^2): +1 on the cut, -1 outside it.
  return (m_abs(x) <= T(1)) ? T(1) : T(-1);
}

template <typename T>
inline T sqrt_clamped(T y) {
  // sqrt of a quantity that should be >= 0 but may be slightly negative due
  // to round-off.
  return m_sqrt(std::max(y, T(0)));
}

// 1/2 ln((x+1)/(x-1)). Stable away from x = +/- 1.
template <typename T>
inline T q00(T x) {
  return T(0.5) * (m_log(m_abs(x + T(1))) - m_log(m_abs(x - T(1))));
}

// Seed P_m^m and P_(m+1)^m for every m needed.
//   P_0^0 = 1
//   P_m^m = -s (2m-1) w P_(m-1)^(m-1)            (w = sqrt(s (1-x^2)))
//   P_(m+1)^m = (2m+1) x P_m^m
template <typename T>
void seed_plm(T *P, int lmax, int mmax, T x) {
  const T s = sign_factor(x);
  const T w = sqrt_clamped(s * (T(1) - x * x));

  P[idx(0, 0, lmax)] = T(1);
  const int diag_max = std::min(mmax, lmax);
  for(int m = 1; m <= diag_max; ++m)
    P[idx(m, m, lmax)] = -s * T(2 * m - 1) * w * P[idx(m - 1, m - 1, lmax)];
  // Sub-diagonal P_(m+1)^m exists only when m+1 <= lmax.
  const int sub_max = std::min(mmax, lmax - 1);
  for(int m = 0; m <= sub_max; ++m)
    P[idx(m + 1, m, lmax)] = T(2 * m + 1) * x * P[idx(m, m, lmax)];
}

// Forward L recurrence at fixed m: (l+1-m) P_(l+1)^m = (2l+1) x P_l^m - (l+m) P_(l-1)^m.
template <typename T>
void recur_plm_in_l(T *P, int lmax, int m, T x) {
  for(int l = m + 1; l < lmax; ++l) {
    const T a = T(2 * l + 1) * x;
    const T b = T(l + m);
    const T c = T(l + 1 - m);
    P[idx(l + 1, m, lmax)] =
        (a * P[idx(l, m, lmax)] - b * P[idx(l - 1, m, lmax)]) / c;
  }
}

template <typename T>
void compute_plm(T *P, int lmax, int mmax, T x) {
  std::fill_n(P, static_cast<std::size_t>(lmax + 1) *
                     static_cast<std::size_t>(mmax + 1),
              T(0));
  seed_plm(P, lmax, mmax, x);
  for(int m = 0; m <= std::min(mmax, lmax); ++m)
    recur_plm_in_l(P, lmax, m, x);
}

// Seed Q_0^0, Q_1^0, Q_0^1, Q_1^1.
template <typename T>
void seed_qlm(T *Q, int lmax, int mmax, T x) {
  const T s = sign_factor(x);
  const T w = sqrt_clamped(s * (T(1) - x * x));
  const T Q00 = q00(x);

  Q[idx(0, 0, lmax)] = Q00;
  if(lmax >= 1) Q[idx(1, 0, lmax)] = x * Q00 - T(1);

  if(mmax >= 1) {
    if(w == T(0)) {
      // Reachable only at x = +/- 1, which q00() already would have made
      // infinite; the public entry point throws first.
      Q[idx(0, 1, lmax)] = std::numeric_limits<T>::infinity();
      if(lmax >= 1)
        Q[idx(1, 1, lmax)] = std::numeric_limits<T>::infinity();
    } else {
      Q[idx(0, 1, lmax)] = T(-1) / w;
      if(lmax >= 1)
        Q[idx(1, 1, lmax)] = -s * w * (Q00 + x / (T(1) - x * x));
    }
  }
}

// Lentz-Thompson modified continued fraction for Q_lmax/Q_(lmax-1) at x>1.
// Tolerance defaults to a few times machine epsilon for the type T.
template <typename T>
T cf_legendre_ratio(T x, int lmax, int m) {
  const T tiny = std::numeric_limits<T>::min() * T(1e4);
  const T tol = T(8) * std::numeric_limits<T>::epsilon();
  constexpr int max_iter = 1000000;

  T f = tiny;
  T C = f;
  T D = T(0);
  T a = T(1);
  T b = T(0);
  int n = lmax;

  for(int i = 0; i < max_iter; ++i) {
    b = T(2 * n + 1) * x / T(n + m);
    D = b + a * D;
    if(D == T(0)) D = tiny;
    C = b + a / C;
    if(C == T(0)) C = tiny;
    D = T(1) / D;
    const T delta = C * D;
    f *= delta;
    if(m_abs(delta - T(1)) < tol) return f;
    a = -T(n - m + 1) / T(n + m);
    ++n;
  }
  std::ostringstream oss;
  oss.precision(17);
  oss << "helfem::legendre: continued fraction for Q ratio at x="
      << static_cast<long double>(x) << " l=" << lmax << " m=" << m
      << " did not converge in " << max_iter << " iterations.\n";
  throw std::runtime_error(oss.str());
}

// Upward L for one m column, seeded at l=0,1.
template <typename T>
void recur_qlm_in_l_up(T *Q, int lmax, int m, T x) {
  for(int l = 1; l < lmax; ++l) {
    const T a = T(2 * l + 1) * x;
    const T b = T(l + m);
    const T c = T(l + 1 - m);
    Q[idx(l + 1, m, lmax)] =
        (a * Q[idx(l, m, lmax)] - b * Q[idx(l - 1, m, lmax)]) / c;
  }
}

// Downward L for one m column. Reads (lmax, m) and (lmax-1, m) seeds; writes
// l = lmax-2 .. 0.
template <typename T>
void recur_qlm_in_l_down(T *Q, int lmax, int m, T x) {
  for(int l = lmax - 1; l >= 1; --l) {
    const T a = T(2 * l + 1) * x;
    const T b = T(l + 1 - m);
    const T c = T(l + m);
    Q[idx(l - 1, m, lmax)] =
        (a * Q[idx(l, m, lmax)] - b * Q[idx(l + 1, m, lmax)]) / c;
  }
}

// Upward m: Q_l^(m+1) = -2m x / w * Q_l^m - s (l+m)(l-m+1) Q_l^(m-1).
template <typename T>
void recur_qlm_in_m_up(T *Q, int lmax, int mmax, T x) {
  if(mmax < 2) return;
  const T s = sign_factor(x);
  const T w = sqrt_clamped(s * (T(1) - x * x));
  if(w == T(0))
    throw std::domain_error(
        "helfem::legendre::qlm: w = 0 in m-recurrence; this only happens "
        "at x = +/- 1, which the public entry point should reject.");

  for(int l = 0; l <= lmax; ++l) {
    for(int m = 1; m < mmax; ++m) {
      const T next =
          -T(2 * m) * x / w * Q[idx(l, m, lmax)] -
          s * T(l + m) * T(l - m + 1) * Q[idx(l, m - 1, lmax)];
      Q[idx(l, m + 1, lmax)] = next;
    }
  }
}

// Upward L path: stable on the cut |x| <= 1, and just above the cut where
// Miller's CF doesn't converge fast.
template <typename T>
void compute_qlm_upward(T *Q, int lmax, int mmax, T x) {
  if(lmax >= 2) recur_qlm_in_l_up(Q, lmax, 0, x);
  if(mmax >= 1 && lmax >= 2) recur_qlm_in_l_up(Q, lmax, 1, x);
  recur_qlm_in_m_up(Q, lmax, mmax, x);
}

// Christoffel/Bonnet decomposition path. Stable to machine precision for x
// close to 1+. Cancellation grows like (x + sqrt(x^2-1))^(2l) so the caller
// switches to Miller's once that grows too large.
//
// Identity:   Q_l(x)    = P_l(x) * Q_0(x) - W_l(x),
//             Q_l^1(x)  = w * P_l'(x) * Q_0(x) - P_l(x)/w - w * W_l'(x).
//
// W_l satisfies the same three-term recurrence as P_l, seeded W_0 = 0,
// W_1 = 1. Both derivatives satisfy the differentiated form
//   (n+1) f_(n+1)' = (2n+1) (f_n + x f_n') - n f_(n-1)'
// with seeds P'_0 = 0, P'_1 = 1, W'_0 = 0, W'_1 = 0.
template <typename T>
void compute_qlm_christoffel(T *Q, int lmax, int mmax, T x) {
  static thread_local std::vector<T> P, W, Pp, Wp;
  P.assign(lmax + 1, T(0));
  W.assign(lmax + 1, T(0));
  Pp.assign(lmax + 1, T(0));
  Wp.assign(lmax + 1, T(0));

  P[0] = T(1);
  if(lmax >= 1) {
    P[1]  = x;
    W[1]  = T(1);
    Pp[1] = T(1);
  }
  for(int n = 1; n < lmax; ++n) {
    const T inv = T(1) / T(n + 1);
    const T a = T(2 * n + 1);
    const T b = T(n);
    P[n + 1]  = (a * x * P[n]  - b * P[n - 1])  * inv;
    W[n + 1]  = (a * x * W[n]  - b * W[n - 1])  * inv;
    Pp[n + 1] = (a * (P[n] + x * Pp[n]) - b * Pp[n - 1]) * inv;
    Wp[n + 1] = (a * (W[n] + x * Wp[n]) - b * Wp[n - 1]) * inv;
  }

  const T Q00 = q00(x);

  // m = 0 column
  for(int l = 0; l <= lmax; ++l)
    Q[idx(l, 0, lmax)] = P[l] * Q00 - W[l];

  // m = 1 column. Reachable only for x > 1 (|x| <= 1 routes through
  // compute_qlm_upward), so x*x - 1 > 0.
  if(mmax >= 1) {
    const T w = m_sqrt(std::max(x * x - T(1), T(0)));
    Q[idx(0, 1, lmax)] = T(-1) / w;
    for(int l = 1; l <= lmax; ++l)
      Q[idx(l, 1, lmax)] = w * Pp[l] * Q00 - P[l] / w - w * Wp[l];
  }

  // m >= 2 columns from the standard m-recurrence
  recur_qlm_in_m_up(Q, lmax, mmax, x);
}

// Miller's algorithm: stable for x sufficiently above 1, but the CF can't
// converge to a fixed tolerance as x -> 1+.
template <typename T>
void compute_qlm_miller(T *Q, int lmax, int mmax, T x) {
  const T seed_low = std::numeric_limits<T>::min() * T(1e4);

  if(lmax >= 1) {
    const T ratio0 = cf_legendre_ratio(x, lmax, 0);
    Q[idx(lmax - 1, 0, lmax)] = seed_low;
    Q[idx(lmax, 0, lmax)] = seed_low * ratio0;
    recur_qlm_in_l_down(Q, lmax, 0, x);
    const T scale = q00(x) / Q[idx(0, 0, lmax)];
    for(int l = 0; l <= lmax; ++l) Q[idx(l, 0, lmax)] *= scale;
  }
  if(mmax >= 1 && lmax >= 1) {
    const T ratio1 = cf_legendre_ratio(x, lmax, 1);
    Q[idx(lmax - 1, 1, lmax)] = seed_low;
    Q[idx(lmax, 1, lmax)] = seed_low * ratio1;
    recur_qlm_in_l_down(Q, lmax, 1, x);
    const T w = sqrt_clamped(sign_factor(x) * (T(1) - x * x));
    const T scale = (T(-1) / w) / Q[idx(0, 1, lmax)];
    for(int l = 0; l <= lmax; ++l) Q[idx(l, 1, lmax)] *= scale;
  }
  recur_qlm_in_m_up(Q, lmax, mmax, x);
}

}  // namespace detail

// ===== public API ==========================================================

template <typename T>
void plm(T *out, int lmax, int mmax, T x) {
  if(lmax < 0 || mmax < 0)
    throw std::invalid_argument(
        "helfem::legendre::plm: lmax, mmax must be non-negative");
  detail::compute_plm(out, lmax, mmax, x);
}

template <typename T>
void qlm(T *out, int lmax, int mmax, T x) {
  if(lmax < 0 || mmax < 0)
    throw std::invalid_argument(
        "helfem::legendre::qlm: lmax, mmax must be non-negative");
  if(detail::m_abs(x) == T(1)) {
    std::ostringstream oss;
    oss << "helfem::legendre::qlm: Q is logarithmically singular at x = "
        << static_cast<long double>(x);
    throw std::domain_error(oss.str());
  }

  std::fill_n(out,
              static_cast<std::size_t>(lmax + 1) *
                  static_cast<std::size_t>(mmax + 1),
              T(0));
  detail::seed_qlm(out, lmax, mmax, x);

  // Three machine-precision regimes:
  //   1. |x| <= 1: upward L recurrence on the cut.
  //   2. 1 < x < christoffel_xmax(lmax): Christoffel decomposition.
  //   3. x >= christoffel_xmax: Miller's downward + Lentz-Thompson CF.
  if(detail::m_abs(x) <= T(1)) {
    detail::compute_qlm_upward(out, lmax, mmax, x);
    return;
  }
  // (x + sqrt(x^2-1))^(2*lmax) capped near a few thousand keeps Christoffel
  // cancellation below ~3 lost digits.
  const T christoffel_xmax =
      (lmax > 0)
          ? detail::m_cosh(detail::m_log(T(1e3)) / T(2 * lmax))
          : std::numeric_limits<T>::infinity();
  if(detail::m_abs(x) < christoffel_xmax)
    detail::compute_qlm_christoffel(out, lmax, mmax, x);
  else
    detail::compute_qlm_miller(out, lmax, mmax, x);
}

// Explicit instantiations live in Legendre.cpp; declare them extern here so
// translation units that only use these instances don't redo template work.
extern template void plm<double>(double *, int, int, double);
extern template void qlm<double>(double *, int, int, double);
extern template void plm<long double>(long double *, int, int, long double);
extern template void qlm<long double>(long double *, int, int, long double);

}  // namespace legendre
}  // namespace helfem

#endif
