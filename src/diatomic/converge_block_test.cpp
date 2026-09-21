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

// What does converge_block REPORT when the adaptive quadrature hits its order
// cap?
//
// For a long time the answer was "a relative change of zero, always": the cap
// warning computed (cur - prev) after prev had been overwritten with cur, so
// every warning read "relative change still 0.000e+00" -- which says
// "converged perfectly" in the same breath as the sentence above it says the
// opposite. A test that merely checks the warning FIRES passes on that bug.
// So these drive converge_block with probes whose block sequence is known in
// closed form, and assert on the NUMBERS it reports.

#include "converge_block.h"

#include <cmath>
#include <cstdio>

using helfem::diatomic::basis::detail::CapReport;
using helfem::diatomic::basis::detail::converge_block;
using helfem::diatomic::basis::detail::twoe_nmax;

namespace {

  int nfail = 0;

  void report(const char * what, bool ok, const char * detail = "") {
    if(!ok) nfail++;
    printf("  %-52s %s %s\n", what, ok ? "ok" : "FAIL", detail);
  }

} // namespace

int main() {
  printf("order cap = %d\n\n", twoe_nmax);

  printf("1  a block that never converges\n");
  {
    // Every entry is c * n, so the orders visited from nstart = 5 are
    // 5, 10, 20, ..., 320 and then 512 (the doubling is clamped at the cap).
    // The last comparison is therefore 512 against 320, and the relative
    // change the convergence test saw there is exactly (512 - 320) / 512.
    const double c = 3.0;
    CapReport r;
    converge_block([&](int n) { return helfem::Matrix::Constant(4, 4, c * n); },
                   5, "test(never converges)", nullptr, false, 0, &r);
    const double expect = (512.0 - 320.0) / 512.0;
    char d[96];
    snprintf(d, sizeof(d), "-- rel = %.6f, expected %.6f", r.rel, expect);
    report("hits the cap", r.capped);
    report("reports the relative change it actually saw", std::abs(r.rel - expect) < 1e-14, d);
    // The regression proper. Under the old code this was identically zero.
    report("... which is NOT zero", r.rel > 0.0);
    report("reports the block magnitude", std::abs(r.scale - c * 512.0) < 1e-12);
  }

  printf("\n2  the cap reached on the very first probe\n");
  {
    // Seeded at or above the cap, the loop never makes a comparison. That is
    // a different statement from "the change was small", and the old code
    // could not tell them apart: both came out as 0. Now it is a sentinel.
    CapReport r;
    converge_block([&](int n) { return helfem::Matrix::Constant(3, 3, 1.0 * n); },
                   twoe_nmax + 88, "test(seeded past cap)", nullptr, false, 0, &r);
    report("hits the cap", r.capped);
    report("says no comparison was made (rel == -1)", r.rel == -1.0);
  }

  printf("\n3  a block that converges\n");
  {
    CapReport r;
    converge_block([&](int) { return helfem::Matrix::Constant(3, 3, 2.5); },
                   5, "test(converges)", nullptr, false, 0, &r);
    report("does not hit the cap", !r.capped);
    report("stops at the first comparison", r.n == 10);
  }

  printf("\n4  nskip: a divergent boundary row that is later discarded\n");
  {
    // The shape of radial_integral(-1, 0): 1/sinh(mu) diverges at mu = 0, so
    // the row and column of the one basis function non-zero there grow with
    // the quadrature order and never converge, while the rest of the block is
    // fine. remove_boundaries drops that function, so the block should be
    // judged without it.
    auto probe = [&](int n) {
      helfem::Matrix M = helfem::Matrix::Constant(4, 4, 1.0);
      M.row(0).setConstant(std::log((double) n));
      M.col(0).setConstant(std::log((double) n));
      return M;
    };
    CapReport all, skip;
    converge_block(probe, 5, "test(nskip=0)", nullptr, false, 0, &all);
    converge_block(probe, 5, "test(nskip=1)", nullptr, false, 1, &skip);
    report("judged whole, the divergent row caps it", all.capped);
    report("judged without it, the block converges", !skip.capped);
  }

  printf("\n%s\n", nfail ? "CONVERGE_BLOCK TEST FAILED" : "CONVERGE_BLOCK TEST PASSED");
  return nfail ? 1 : 0;
}
