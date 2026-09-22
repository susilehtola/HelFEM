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

// helfem::adaptive::refine -- the one refinement loop behind every one-electron
// matrix element, the atomic two-electron primitives and the diatomic radial
// integrals.
//
// Each probe below is built so its block sequence is known in closed form and
// reaches exactly ONE of the four exits. The ratios are chosen to clear each
// threshold by a margin rather than sit on it: a ratio of exactly 1/2 against
// a test of "diff > 0.5 * prevdiff" would make the outcome a matter of rounding.

#include "adaptive_quadrature.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <string>

using helfem::adaptive::CapReport;
using helfem::adaptive::Options;
using helfem::adaptive::refine;

namespace {

  int nfail = 0;

  void report(const char * what, bool ok, const char * detail = "") {
    if(!ok) nfail++;
    printf("  %-56s %s %s\n", what, ok ? "ok" : "FAIL", detail);
  }

  /// Doublings from nstart: n = nstart * 2^k.
  int doublings(int n, int nstart) {
    return (int) std::lround(std::log2((double) n / nstart));
  }

  std::string label() { return "test"; }

} // namespace

int main() {
  const int nstart = 4;
  Options opt;
  opt.nstart = nstart;

  printf("1  eps convergence\n");
  {
    std::atomic<bool> w{false};
    CapReport r;
    refine<double>([](int) { return helfem::Matrix::Constant(3, 3, 2.5); },
                   opt, label, w, &r);
    report("a constant block stops at exit (1)", !r.capped && r.exit == 1);
    report("... on the first comparison", r.n == 2 * nstart);
  }

  printf("\n2  machine floor\n");
  {
    // The deviation shrinks 10x per doubling -- geometric, so neither stall
    // test fires -- and passes into (8 eps, 256 eps] before it reaches 8 eps.
    std::atomic<bool> w{false};
    CapReport r;
    refine<double>([&](int n) {
                     const double d = 1e-11 * std::pow(10.0, -doublings(n, nstart));
                     return helfem::Matrix::Constant(3, 3, 1.0 + d);
                   }, opt, label, w, &r);
    char d[64]; snprintf(d, sizeof(d), "-- exit %d at n=%d", r.exit, r.n);
    report("stops at exit (2), still converging geometrically", !r.capped && r.exit == 2, d);
  }

  printf("\n3  one-doubling stall\n");
  {
    // Roundoff noise: the block flips by a fixed amount, so the difference
    // never shrinks at all.
    std::atomic<bool> w{false};
    CapReport r;
    refine<double>([&](int n) {
                     const double d = (doublings(n, nstart) % 2 ? -1e-12 : 1e-12);
                     return helfem::Matrix::Constant(3, 3, 1.0 + d);
                   }, opt, label, w, &r);
    char d[64]; snprintf(d, sizeof(d), "-- exit %d at n=%d", r.exit, r.n);
    report("stops at exit (3)", !r.capped && r.exit == 3, d);
  }

  printf("\n4  two-doubling stall\n");
  {
    // Shrinking 0.45x per doubling: fast enough to dodge (3), which asks for
    // at least a halving, yet slower than the 8x over two doublings that (4)
    // requires of genuine convergence. This is also the regime to be wary of:
    // a block converging this slowly is stopped here at ~sqrt(eps), not at
    // eps. That is why the report says which exit fired.
    std::atomic<bool> w{false};
    CapReport r;
    refine<double>([&](int n) {
                     const double d = 1e-9 * std::pow(0.45, doublings(n, nstart));
                     return helfem::Matrix::Constant(3, 3, 1.0 + d);
                   }, opt, label, w, &r);
    char d[64]; snprintf(d, sizeof(d), "-- exit %d at n=%d", r.exit, r.n);
    report("stops at exit (4)", !r.capped && r.exit == 4, d);
  }

  printf("\n5  the order cap\n");
  {
    // Every entry c * n: from nstart = 4 the orders are 4, 8, ..., 256, 512,
    // so the last comparison is 512 against 256.
    const double c = 3.0;
    std::atomic<bool> w{false};
    CapReport r;
    refine<double>([&](int n) { return helfem::Matrix::Constant(4, 4, c * n); },
                   opt, label, w, &r);
    report("hits the cap", r.capped && r.exit == 0);
    report("reports the change it saw, 256/512", std::abs(r.rel - 0.5) < 1e-14);
    report("reports the magnitude", std::abs(r.scale - c * 512.0) < 1e-12);
    report("the first cap prints", r.printed);

    CapReport again;
    refine<double>([&](int n) { return helfem::Matrix::Constant(4, 4, c * n); },
                   opt, label, w, &again);
    report("a second cap on the same flag reports but does not print",
           again.capped && !again.printed && std::abs(again.rel - 0.5) < 1e-14);
  }

  printf("\n6  exactly one warning, from many threads at once\n");
  {
    // The warn-once flag used to be a plain bool written from inside OpenMP
    // parallel regions. Count how many of 64 concurrent capping calls report
    // having printed. Threads are forced, since the test harness pins
    // OMP_NUM_THREADS=1 and a serial loop would prove nothing.
    std::atomic<bool> w{false};
    int printed = 0, capped = 0;
#ifdef _OPENMP
#pragma omp parallel for num_threads(8) reduction(+:printed,capped)
#endif
    for (int i = 0; i < 64; i++) {
      CapReport r;
      refine<double>([](int n) { return helfem::Matrix::Constant(2, 2, 1.0 * n); },
                     opt, label, w, &r);
      printed += r.printed ? 1 : 0;
      capped += r.capped ? 1 : 0;
    }
    char d[64]; snprintf(d, sizeof(d), "-- %d of %d capped calls printed", printed, capped);
    report("exactly one of them printed", capped == 64 && printed == 1, d);
  }

  printf("\n7  extended precision\n");
  {
    // refine is templated on the scalar: the tolerances follow eps(T), so a
    // long double block is held to long double's eps, not double's.
    std::atomic<bool> w{false};
    CapReport r;
    refine<long double>([](int) {
                          return helfem::Mat<long double>::Constant(2, 2, 1.25L);
                        }, opt, label, w, &r);
    report("instantiates and converges for long double", !r.capped && r.exit == 1);
  }

  printf("\n%s\n", nfail ? "ADAPTIVE QUADRATURE TEST FAILED" : "ADAPTIVE QUADRATURE TEST PASSED");
  return nfail ? 1 : 0;
}
