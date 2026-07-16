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

// Round-trip test for the precision-generic HDF5 checkpoint I/O.
//
// For every supported scalar type T (double, long double, and _Float128
// when built with HELFEM_HAVE_FLOAT128) it writes a MatT<T> and a Vec<T>
// filled with values that need MORE than double precision to represent
// (1/3, sqrt(2), pi), reads them back into fresh MatT<T>/Vec<T>, and
// requires the reload to be BIT-IDENTICAL to the original (max|Δ| == 0).
//
// A double-capped I/O path (memory type stuck at H5T_NATIVE_DOUBLE) would
// truncate long double / _Float128 to 53 significant bits and show a
// ~1e-16..1e-19 error at those types; a genuine per-T path shows exactly 0.
//
// It additionally checks that a value written through the DOUBLE path
// (on-disk H5T_IEEE_F64LE) loads correctly through the _Float128 memory
// type -- i.e. HDF5's F64LE -> binary128 conversion is exact -- which is
// what pins down the binary128 field parameters in h5_native_float.

#include "checkpoint.h"
#include "Matrix.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>

namespace {

// High-precision reference constants, evaluated at the scalar type T so the
// low-order bits genuinely differ between double / long double / _Float128.
template <typename T> T one_third() { return T(1) / T(3); }
template <typename T> T root_two()  { return std::sqrt(T(2)); }
template <typename T>
T my_pi() {
  // 4*atan(1) evaluated at T.
  return T(4) * std::atan(T(1));
}

// Print a scalar difference as long double; exact-zero prints as 0.
long double as_ld(double v)      { return (long double) v; }
long double as_ld(long double v) { return v; }
#ifdef HELFEM_HAVE_FLOAT128
long double as_ld(_Float128 v)   { return (long double) v; }
#endif

template <typename T>
int roundtrip(const char * tname) {
  const int n = 5;

  // Build a matrix and a vector whose entries need >double precision.
  helfem::Mat<T> M(n, n);
  helfem::Vec<T> v(n);
  for (int j = 0; j < n; j++) {
    v(j) = one_third<T>() + T(j) * root_two<T>();
    for (int i = 0; i < n; i++)
      M(i, j) = my_pi<T>() * T(i + 1) + root_two<T>() / T(j + 1) + one_third<T>();
  }

  // Round-trip through a temporary checkpoint.
  std::string fname = tempname();
  {
    Checkpoint chk(fname, /*write=*/true);
    chk.write("M", M);
    chk.write("v", v);
    chk.write("s", one_third<T>());
  }

  helfem::Mat<T> M2;
  helfem::Vec<T> v2;
  T s2;
  {
    Checkpoint chk(fname, /*write=*/false);
    chk.read("M", M2);
    chk.read("v", v2);
    chk.read("s", s2);
  }
  remove(fname.c_str());

  // Bit-identity: max|Δ| must be exactly 0, and the raw buffers must match.
  const T sref = one_third<T>();
  T dM = (M2 - M).cwiseAbs().maxCoeff();
  T dv = (v2 - v).cwiseAbs().maxCoeff();
  T ds = (s2 > sref) ? (s2 - sref) : (sref - s2);

  bool bits_ok =
      (M2.rows() == M.rows() && M2.cols() == M.cols() &&
       std::memcmp(M2.data(), M.data(), sizeof(T) * n * n) == 0) &&
      (v2.size() == v.size() &&
       std::memcmp(v2.data(), v.data(), sizeof(T) * n) == 0) &&
      (std::memcmp(&s2, &sref, sizeof(T)) == 0);

  long double maxd = as_ld(dM);
  if (as_ld(dv) > maxd) maxd = as_ld(dv);
  if (as_ld(ds) > maxd) maxd = as_ld(ds);

  bool ok = (dM == T(0)) && (dv == T(0)) && (ds == T(0)) && bits_ok;
  printf("  %-12s sizeof=%2zu  max|delta| = %.3Le   %s\n",
         tname, sizeof(T), maxd, ok ? "OK" : "FAIL");
  return ok ? 0 : 1;
}

#ifdef HELFEM_HAVE_FLOAT128
// Cross-type conversion: a value written through the DOUBLE path (F64LE on
// disk) must reload exactly through the _Float128 memory type. Verifies the
// binary128 field parameters -- a wrong exponent bias / field layout would
// corrupt the converted value.
int cross_double_to_quad() {
  double d = 1.0 / 3.0;             // exact double bit pattern
  std::string fname = tempname();
  {
    Checkpoint chk(fname, /*write=*/true);
    chk.write("d", d);             // T=double -> on-disk H5T_IEEE_F64LE
  }
  _Float128 q;
  {
    Checkpoint chk(fname, /*write=*/false);
    chk.read("d", q);             // T=_Float128 memory type; HDF5 converts
  }
  remove(fname.c_str());

  _Float128 expect = (_Float128) d; // exact widening of the double value
  _Float128 diff = (q > expect) ? (q - expect) : (expect - q);
  bool ok = (diff == (_Float128) 0);
  printf("  %-12s               F64LE->binary128 delta = %.3Le   %s\n",
         "double->quad", (long double) diff, ok ? "OK" : "FAIL");
  return ok ? 0 : 1;
}
#endif

} // namespace

int main() {
  printf("checkpoint precision round-trip:\n");
  int fail = 0;
  fail += roundtrip<double>("double");
  fail += roundtrip<long double>("long double");
#ifdef HELFEM_HAVE_FLOAT128
  fail += roundtrip<_Float128>("_Float128");
  fail += cross_double_to_quad();
#endif
  if (fail) {
    printf("FAILED: %d check(s) did not round-trip bit-identically.\n", fail);
    return 1;
  }
  printf("All checkpoint precision round-trips are bit-identical.\n");
  return 0;
}
