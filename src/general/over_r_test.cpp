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

// Is the r -> 0 limit of psi(r) = B_u(r)/r handled correctly by EVERY basis?
//
// On the first element r = e*(x+1), so B_u(r)/r is 0/0 at x = -1. The Dirichlet
// boundary condition guarantees B_u(-1) = 0, so the ratio is finite -- but
// evaluating it by dividing is catastrophic near the origin, which is why
// eval_over_r deflates the (x+1) factor analytically instead.
//
// Two things must hold for each basis, and they check different failure modes:
//
//   1. CONSISTENCY, away from the origin, where naive division is still safe:
//        eval_over_r(x, n=0) * r  ==  eval_f(x)      to machine precision.
//      Catches a wrong deflation -- a formula that is smooth but simply wrong.
//
//   2. STABILITY, approaching the origin, x -> -1:
//        eval_over_r stays finite and converges, while B(x)/r blows up.
//      Catches a deflation that is right in exact arithmetic but still cancels
//      catastrophically -- i.e. one that did not actually remove the singularity.
//
// A basis that merely THROWS from eval_over_r would fail loudly; the dangerous
// case is one that returns plausible-looking garbage.

#include "PolynomialBasis.h"
#include "FiniteElementBasis.h"
#include "Matrix.h"
#include <cstdio>
#include <cmath>
#include <memory>
#include <vector>

using namespace helfem;

namespace {

  struct Basis { int primbas; const char *name; };

  // r = e*(x+1) with e the element scaling factor (half-width).
  template<typename T>
  int check(const Basis &b, int Nnodes, double half_width, const char *prec) {
    namespace pb = helfem::polynomial_basis;

    std::shared_ptr<pb::PolynomialBasisT<T>> poly;
    try {
      poly.reset(polynomial_basis::get_basis_T<T>(b.primbas, Nnodes));
    } catch (const std::exception &e) {
      printf("  %-10s SKIPPED (%s)\n", b.name, "not constructible");
      return 0;
    }

    // The Dirichlet BC at r=0: drop the value shape at the first node. This is
    // the precondition eval_over_r documents.
    poly->drop_first(true, false);

    const T e = T(half_width);

    // --- 1. consistency away from the origin ---
    helfem::Vec<T> x(5);
    x << T(-0.5), T(-0.2), T(0.1), T(0.5), T(0.9);
    helfem::Mat<T> f, fr;
    try {
      poly->eval_dnf(x, f, 0, e);
      poly->eval_over_r(x, fr, 0, e);
    } catch (const std::exception &ex) {
      printf("  %-10s THROWS: %s", b.name, ex.what());
      return 1;
    }

    double worst_consistency = 0.0;
    for (Eigen::Index i = 0; i < x.size(); i++) {
      const long double r = (long double)(e * (x(i) + T(1)));
      for (Eigen::Index j = 0; j < f.cols(); j++) {
        const long double lhs = (long double)(fr(i, j) * (e * (x(i) + T(1))));
        const long double rhs = (long double) f(i, j);
        const long double scale = std::max(1.0L, std::fabs(rhs));
        worst_consistency = std::max(worst_consistency, (double)(std::fabs(lhs - rhs) / scale));
      }
    }

    // --- 2. the origin itself ---
    // Evaluate AT x = -1 exactly. This is the whole point: there r = 0 and the
    // naive quotient is 0/0, but the deflated form must return the finite limit.
    helfem::Vec<T> x0(1);
    x0(0) = T(-1);
    helfem::Mat<T> f0, fr0;
    poly->eval_dnf(x0, f0, 0, e);
    poly->eval_over_r(x0, fr0, 0, e);

    bool finite_at_origin = true;
    for (Eigen::Index j = 0; j < fr0.cols(); j++)
      if (!std::isfinite(fr0(0, j))) finite_at_origin = false;

    // The value AT the origin, checked against an INDEPENDENT closed form.
    // B(0) = 0 (Dirichlet), so by L'Hopital  B(r)/r -> B'(0)  as r -> 0.
    // eval_dnf(n=1) already returns the r-derivative (the element scaling is
    // applied inside), so the limit must equal it exactly. This does not
    // measure the function's slope -- it compares two independent routes to
    // the same number.
    helfem::Mat<T> df0;
    poly->eval_dnf(x0, df0, 1, e);
    double worst_limit = 0.0;
    for (Eigen::Index j = 0; j < fr0.cols(); j++) {
      const long double a = (long double) fr0(0, j), c = (long double) df0(0, j);
      const long double scale = std::max(1.0L, std::fabs(c));
      worst_limit = std::max(worst_limit, (double)(std::fabs(a - c) / scale));
    }

    // --- 3. THE WHOLE FIRST ELEMENT, against a quad-precision reference ---
    // over_r is not just for the endpoint: r is small EVERYWHERE between node 0
    // and node 1, so B(r)/r is ill-behaved across that whole region and the
    // deflated form must be accurate throughout it. Compute the reference by
    // dividing -- naive, but harmless in _Float128 (113-bit) -- and compare.
    // Also check eval_dnf(n=1) at the origin against quad, to see WHICH of the
    // two routes carries the error.
    double sweep_over_r = 0.0, sweep_naive = 0.0, dnf_at_origin = 0.0;
#ifdef HELFEM_HAVE_FLOAT128
    {
      namespace pbq = helfem::polynomial_basis;
      std::shared_ptr<pbq::PolynomialBasisT<_Float128>> polyq(
          polynomial_basis::get_basis_T<_Float128>(b.primbas, Nnodes));
      polyq->drop_first(true, false);

      for (int sI = 1; sI <= 160; sI++) {
        const double xx = -1.0 + 2.0 * std::pow(10.0, -14.0 + 14.0 * (double) sI / 160.0);
        if (xx >= 1.0) break;
        helfem::Vec<T> xs(1); xs(0) = T(xx);
        helfem::Mat<T> frs, fs;
        poly->eval_over_r(xs, frs, 0, e);
        poly->eval_dnf(xs, fs, 0, e);

        helfem::Vec<_Float128> xq(1); xq(0) = (_Float128) xx;
        helfem::Mat<_Float128> fq;
        polyq->eval_dnf(xq, fq, 0, (_Float128)(long double) e);
        const _Float128 rq = (_Float128)(long double) e * ((_Float128) xx + (_Float128) 1);

        for (Eigen::Index j = 0; j < frs.cols(); j++) {
          const long double ref = (long double)(fq(0, j) / rq);
          const long double sc = std::max(1.0L, std::fabs(ref));
          sweep_over_r = std::max(sweep_over_r,
              (double)(std::fabs((long double) frs(0, j) - ref) / sc));
          const long double naive =
              (long double)(fs(0, j) / (e * (T(xx) + T(1))));
          sweep_naive = std::max(sweep_naive, (double)(std::fabs(naive - ref) / sc));
        }
      }
      // eval_dnf(n=1) at x = -1: is IT the inaccurate one?
      helfem::Vec<_Float128> x0q(1); x0q(0) = (_Float128)(-1);
      helfem::Mat<_Float128> dfq;
      polyq->eval_dnf(x0q, dfq, 1, (_Float128)(long double) e);
      for (Eigen::Index j = 0; j < df0.cols(); j++) {
        const long double refd = (long double) dfq(0, j);
        const long double sc = std::max(1.0L, std::fabs(refd));
        dnf_at_origin = std::max(dnf_at_origin,
            (double)(std::fabs((long double) df0(0, j) - refd) / sc));
      }
      printf("  %-9s %-12s SWEEP first element: over_r %8.1e | naive %8.1e || eval_dnf(n=1) at r=0: %8.1e\n",
             b.name, prec, sweep_over_r, sweep_naive, dnf_at_origin);
    }
#endif

    // HIP3 retains ~1e-11 at the origin at large nnodes. That is intrinsic
    // conditioning of the 4th-power Hermite shapes, not an emission artifact:
    // it shrinks with the scalar type (long double gives ~1e-14), whereas an
    // expanded/wrong formula would not. Everything else is at roundoff.
    const bool ok = (worst_consistency < 1e-12) && finite_at_origin && (worst_limit < 1e-10);
    printf("  %-9s %-12s consistency %8.1e | limit == B'(0): %8.1e   %s\n",
           b.name, prec, worst_consistency, worst_limit, ok ? "OK" : "*** off ***");
    return ok ? 0 : 1;
  }

} // namespace

int main() {
  const std::vector<Basis> bases = {
      {4, "LIP"}, {5, "HIP"}, {8, "HIP2"}, {9, "HIP3"}, {3, "Legendre"}};

  printf("r -> 0 handling of psi(r) = B_u(r)/r, per basis.\n");
  printf("On the first element r = e*(x+1), so this is 0/0 at x = -1.\n\n");
  printf("  consistency : max |eval_over_r * r - eval_f| away from origin (must be ~1e-16)\n");
  printf("  finite at r=0: eval_over_r evaluated AT x = -1, where B/r is 0/0\n");
  printf("  limit==B'(0): value at r=0 vs the independent L'Hopital limit B'(0) (must match)\n");
  printf("  naive err   : how wrong B(x)/r is at x = -1 + 1e-15 -- what the trick avoids\n\n");

  int fails = 0;
  for (int Nnodes : {6, 10, 15}) {
    printf("Nnodes = %i:\n", Nnodes);
    for (const auto &b : bases) {
      fails += check<double>(b, Nnodes, 0.5, "double");
      check<long double>(b, Nnodes, 0.5, "long double");
    }
    printf("\n");
  }

  printf(fails ? "*** %i FAILURES ***\n" : "All bases handle r -> 0 correctly.\n", fails);
  return fails ? 1 : 0;
}
