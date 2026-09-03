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

// Accuracy of the Lagrange interpolating polynomial evaluation.
//
// The test rests on identities that hold ANALYTICALLY, so every deviation
// measured here is pure evaluation round-off rather than approximation
// error:
//
//   sum_j l_j(x)      = 1     (the interpolant reproduces a constant)
//   sum_j l_j^(k)(x)  = 0     for k >= 1 (its derivatives annihilate it)
//
// The sums run over a full LIP set on the reference element, sampled at
// points that are deliberately NOT the nodes -- at a node the second
// barycentric form takes its exact Kronecker-delta branch and would flatter
// the result.
//
// Scaling with the node count is the point: a naive product evaluation of
// l_j^(k) loses accuracy quickly with order, which is what bounds the usable
// node count per element.

#include <PolynomialBasis.h>
#include <lobatto.h>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace helfem;

static int failures = 0;

/// Sample points inside the element, offset off the nodes.
static Vector sample_points(int npt) {
  Vector x(npt);
  for (int i = 0; i < npt; i++)
    // irrational offset, so no sample lands on a Gauss-Lobatto node
    x(i) = -1.0 + 2.0 * (i + 0.31830988618379067) / (npt - 1 + 0.6366197723675813);
  return x;
}

int main() {
  const std::vector<int> nodecounts = {10, 15, 20, 25, 30, 40, 50};
  const int npt = 97;
  const Vector x = sample_points(npt);

  printf("Worst deviation from the exact identities, over %d interior points.\n", npt);
  printf("k=0 measures |sum_j l_j(x) - 1|; k>=1 measures |sum_j l_j^(k)(x)|.\n");
  printf("The relative column divides by max_j |l_j^(k)(x)|, which is what says\n");
  printf("how many digits were lost: the derivatives themselves grow like n^(2k)\n");
  printf("on Gauss-Lobatto nodes, so a growing absolute deviation need not mean a\n");
  printf("less accurate evaluation, while a growing relative one does.\n\n");

  for (int k = 0; k <= 3; k++) {
    printf("  k=%d\n", k);
    printf("    %-8s %-14s %-14s %-14s\n", "nodes", "absolute", "max term", "relative");
    for (int n : nodecounts) {
      Vector x0, w0;
      lobatto::lobatto_compute<double>(n, x0, w0);
      polynomial_basis::LIPBasis basis(x0, 4);

      Matrix dnf;
      // element_length 2 keeps the reference element unscaled, so what is
      // measured is the primitive evaluation and not the chain rule.
      basis.eval_prim_dnf(x, dnf, k, 2.0);

      double wabs = 0.0, wrel = 0.0, biggest = 0.0;
      for (Eigen::Index ix = 0; ix < dnf.rows(); ix++) {
        double s = 0.0, big = 0.0;
        for (Eigen::Index j = 0; j < dnf.cols(); j++) {
          s += dnf(ix, j);
          big = std::max(big, std::abs(dnf(ix, j)));
        }
        const double dev = std::abs(s - (k == 0 ? 1.0 : 0.0));
        wabs = std::max(wabs, dev);
        biggest = std::max(biggest, big);
        if (big > 0.0)
          wrel = std::max(wrel, dev / big);
      }
      printf("    %-8d %-14.3e %-14.3e %-14.3e\n", n, wabs, biggest, wrel);

      // The evaluation is required to stay at round-off RELATIVE to the size
      // of the terms being summed. That is the property a backward stable
      // evaluation has and a naive product form loses with rising order.
      if (!(wrel < 1e-13)) {
        printf("      FAILED: %d nodes, k=%d, lost accuracy relative to the terms\n", n, k);
        failures++;
      }
    }
    printf("\n");
  }

  // A second, independent probe of the same question. The identity above is
  // a cancellation test; this one compares each l_j^(k) against the SAME
  // algorithm run in long double (64-bit mantissa against 53), which
  // measures directly how much precision the double evaluation gives up.
  // An algorithm that is losing digits with order shows it here even if the
  // errors happened to cancel in the sum.
  printf("Relative error of each l_j^(k) against a long double evaluation.\n\n");
  printf("  %-8s %-14s %-14s %-14s %-14s\n", "nodes", "k=0", "k=1", "k=2", "k=3");
  for (int n : nodecounts) {
    Vector x0d, w0d;
    lobatto::lobatto_compute<double>(n, x0d, w0d);
    polynomial_basis::LIPBasisT<double> bd(x0d, 4);

    Vec<long double> x0l, w0l;
    lobatto::lobatto_compute<long double>(n, x0l, w0l);
    polynomial_basis::LIPBasisT<long double> bl(x0l, 4);
    Vec<long double> xl(x.size());
    for (Eigen::Index i = 0; i < x.size(); i++)
      xl(i) = (long double) x(i);

    printf("  %-8d", n);
    for (int k = 0; k <= 3; k++) {
      Matrix dd;
      Mat<long double> dl;
      bd.eval_prim_dnf(x, dd, k, 2.0);
      bl.eval_prim_dnf(xl, dl, k, 2.0L);

      // Scale by the largest term at each point: an individual l_j^(k) can
      // pass through zero, where a per-element relative error is meaningless.
      double worst = 0.0;
      for (Eigen::Index ix = 0; ix < dd.rows(); ix++) {
        long double big = 0.0L;
        for (Eigen::Index j = 0; j < dd.cols(); j++)
          big = std::max(big, std::abs(dl(ix, j)));
        if (!(big > 0.0L)) continue;
        for (Eigen::Index j = 0; j < dd.cols(); j++)
          worst = std::max(worst,
                           (double) (std::abs((long double) dd(ix, j) - dl(ix, j)) / big));
      }
      printf(" %-14.3e", worst);
      if (!(worst < 1e-12)) {
        printf("\n    FAILED: %d nodes, k=%d differs from long double by %.3e\n",
               n, k, worst);
        failures++;
      }
    }
    printf("\n");
  }
  printf("\n");

  printf("\n%s\n", failures ? "FAILURES" : "All tests passed.");
  return failures ? 1 : 0;
}
