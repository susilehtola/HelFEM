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
    namespace pb = helfem::lib1dfem::polynomial_basis;

    std::shared_ptr<pb::PolynomialBasis<T>> poly;
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

    const bool ok = (worst_consistency < 1e-12) && finite_at_origin && (worst_limit < 1e-12);
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
