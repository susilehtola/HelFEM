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
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

// Compares helfem::legendre::plm/qlm against a Maple reference computed at
// 35 digits. Reference values are stored as decimal strings so each test
// precision parses its own copy without being clipped by the C++ literal
// lexer. Runs at double, long double, and (if available) __float128.

#include "Legendre.h"
#include "legendre_reference_data.h"

#ifdef HELFEM_LEGENDRE_TEST_FLOAT128
#  include "Legendre_float128.h"
#  include <quadmath.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

// Per-type string -> floating parser, named to avoid collision with
// std::strtod / std::strtold prototypes in some headers.
template <typename T> T parse_value(const char *s);

template <> double parse_value<double>(const char *s) {
  return std::strtod(s, nullptr);
}
template <> long double parse_value<long double>(const char *s) {
  return std::strtold(s, nullptr);
}
#ifdef HELFEM_LEGENDRE_TEST_FLOAT128
template <> __float128 parse_value<__float128>(const char *s) {
  return strtoflt128(s, nullptr);
}
#endif

template <typename T> const char *type_name();
template <> const char *type_name<double>()       { return "double"; }
template <> const char *type_name<long double>()  { return "long double"; }
#ifdef HELFEM_LEGENDRE_TEST_FLOAT128
template <> const char *type_name<__float128>()   { return "__float128"; }
#endif

// std::numeric_limits<T>::epsilon() returns T; cast to double for printing
// and for the (rtol, atol) thresholds we use to compare errors.
template <typename T>
double machine_eps_d() {
  return static_cast<double>(std::numeric_limits<T>::epsilon());
}

template <typename T>
double to_double(T v) { return static_cast<double>(v); }

template <typename T>
T t_abs(T v) {
  using std::abs;
  return abs(v);
}

template <typename T>
struct Failure {
  int l, m;
  double x;
  char kind;
  T got, want;
};

// Bucket reference entries by x so we hit each (lmax, mmax) sweep just once.
struct Bucket {
  int lmax = 0;
  int mmax = 0;
  std::vector<int> idx;
};

template <typename T>
int run_tests() {
  using helfem_test::legendre_reference;
  using helfem_test::legendre_reference_count;

  // The recurrence chain accumulates a few ULP per step. The longest path is
  // (lmax + mmax) steps for the Christoffel + m-recurrence regime, plus a
  // factor for Miller's normalization. ~1e5 ULP is generous enough to absorb
  // worst-case (l=60, m=20) chains without tolerating real algorithm bugs.
  const double type_eps = machine_eps_d<T>();
  const double rtol = 1.0e5 * type_eps;
  const double atol = 1.0e5 * type_eps;

  // Group by exact x string (so the same x value always lands in the same
  // bucket regardless of how many digits the parser keeps).
  std::vector<std::pair<std::string, Bucket>> buckets;
  for(int i = 0; i < legendre_reference_count; ++i) {
    const auto &r = legendre_reference[i];
    auto it = std::find_if(buckets.begin(), buckets.end(),
                           [&](const auto &b) { return b.first == r.x; });
    if(it == buckets.end()) {
      buckets.push_back({std::string(r.x), {r.l, r.m, {i}}});
    } else {
      it->second.lmax = std::max(it->second.lmax, r.l);
      it->second.mmax = std::max(it->second.mmax, r.m);
      it->second.idx.push_back(i);
    }
  }

  std::vector<Failure<T>> failures;
  std::size_t checked = 0;

  for(const auto &kv : buckets) {
    const auto &b = kv.second;
    const T x = parse_value<T>(kv.first.c_str());
    const std::size_t stride =
        static_cast<std::size_t>(b.lmax + 1) *
        static_cast<std::size_t>(b.mmax + 1);
    std::vector<T> P(stride);
    std::vector<T> Q(stride);

    helfem::legendre::plm<T>(P.data(), b.lmax, b.mmax, x);
    helfem::legendre::qlm<T>(Q.data(), b.lmax, b.mmax, x);

    for(int i : b.idx) {
      const auto &r = legendre_reference[i];
      const std::size_t off = static_cast<std::size_t>(r.m) *
                                  static_cast<std::size_t>(b.lmax + 1) +
                              static_cast<std::size_t>(r.l);

      const T want_P = parse_value<T>(r.P);
      const T want_Q = parse_value<T>(r.Q);
      const T err_P = t_abs(P[off] - want_P);
      const T err_Q = t_abs(Q[off] - want_Q);
      const T tol_P = T(rtol) * t_abs(want_P) + T(atol);
      const T tol_Q = T(rtol) * t_abs(want_Q) + T(atol);

      if(err_P > tol_P)
        failures.push_back({r.l, r.m, std::strtod(r.x, nullptr), 'P',
                            P[off], want_P});
      ++checked;
      if(err_Q > tol_Q)
        failures.push_back({r.l, r.m, std::strtod(r.x, nullptr), 'Q',
                            Q[off], want_Q});
      ++checked;
    }
  }

  std::printf("[%-12s] %zu/%zu values matched (rtol=%.1e, atol=%.1e)\n",
              type_name<T>(), checked - failures.size(), checked, rtol, atol);

  if(failures.empty()) return 0;

  // Worst failure per (x, kind) bucket.
  std::printf("  Worst by (x, fn):\n");
  std::sort(failures.begin(), failures.end(),
            [](const Failure<T> &a, const Failure<T> &b) {
              if(a.x != b.x) return a.x < b.x;
              if(a.kind != b.kind) return a.kind < b.kind;
              return to_double(t_abs(a.got - a.want)) >
                     to_double(t_abs(b.got - b.want));
            });
  double prev_x = -1e300;
  char prev_kind = 0;
  for(const auto &f : failures) {
    if(f.x == prev_x && f.kind == prev_kind) continue;
    prev_x = f.x; prev_kind = f.kind;
    const double rel = to_double(t_abs(f.got - f.want)) /
                       std::max(to_double(t_abs(f.want)), atol);
    std::printf("    l=%-3d m=%-3d x=%-9.4g %c got=%.16e want=%.16e rel=%.3e\n",
                f.l, f.m, f.x, f.kind,
                to_double(f.got), to_double(f.want), rel);
  }
  return 1;
}

}  // namespace

int main() {
  int rc = 0;
  rc |= run_tests<double>();
  rc |= run_tests<long double>();
#ifdef HELFEM_LEGENDRE_TEST_FLOAT128
  rc |= run_tests<__float128>();
#endif
  return rc;
}
