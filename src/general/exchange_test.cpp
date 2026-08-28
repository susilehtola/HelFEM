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

// Tests for FEMRadialBasis::exchange, the cross-basis exchange.
//
// Checked against a brute-force evaluation of the defining double integral
//
//   e = int int u(r1) (r_<^k / r_>^(k+1)) u(r2) dr1 dr2,   u = g * phi,
//
// on a composite grid. The two routes share no code: the implementation
// works panel-wise with prefix-summed moments and an in-panel potential,
// the reference just integrates. The functions are evaluated from their FE
// expansions, not from their analytic forms, so the two are integrating the
// same object and any disagreement is the implementation's.

#include "lcao_projection.h"
#include <lobatto.h>
#include <cmath>
#include <cstdio>

using namespace helfem;
using FEMRad = atomic::basis::FEMRadialBasis;

static int failures = 0;
static void check(bool ok, const char *what, double val) {
  printf("  %-52s %-12.3e %s\n", what, val, ok ? "ok" : "FAILED");
  if (!ok)
    failures++;
}

/// Value of the expansion sum_u c_u B_u(r) at an arbitrary radius.
static double eval_expansion(const FEMRad &b, const Vector &c, double r) {
  const Vector bv = b.bval();
  if (r <= bv(0) || r >= bv(bv.size() - 1))
    return 0.0;
  size_t iel = 0;
  while (iel + 1 < (size_t)bv.size() - 1 && r > bv(iel + 1))
    iel++;
  Vector xv(1);
  xv(0) = (2 * r - (bv(iel) + bv(iel + 1))) / (bv(iel + 1) - bv(iel));
  const Matrix B = b.fem().eval_dnf(xv, 0, iel);
  size_t f, l;
  b.idx(iel, f, l);
  double s = 0.0;
  for (Eigen::Index j = 0; j < B.cols(); j++)
    s += c((Eigen::Index)f + j) * B(0, j);
  return s;
}

/// Brute-force  int int u(r1) g_k(r1,r2) u(r2)  on [0,R], by symmetry
///   = 2 int u(r1) r1^-(k+1) [ int_0^r1 r2^k u(r2) dr2 ] dr1.
static double brute_force(const std::function<double(double)> &u, double R,
                          int k, int npanel, int nq) {
  Vector x, w;
  lobatto::lobatto_compute<double>(nq, x, w);
  const double h = R / npanel;
  // Cumulative inner integral at each panel edge, plus the partial pieces.
  double outer = 0.0, cum = 0.0;
  for (int p = 0; p < npanel; p++) {
    const double a = p * h, c = a + h, mid = 0.5 * (a + c), hf = 0.5 * h;
    // Contribution of this panel to the outer integral, with the inner
    // integral split into "whole panels below" plus "inside this panel".
    for (Eigen::Index g = 0; g < x.size(); g++) {
      const double r1 = mid + hf * x(g);
      if (r1 <= 0.0)
        continue;
      // inside-panel part of int_0^r1, integrated from a to r1
      double inpart = 0.0;
      const double m2 = 0.5 * (a + r1), h2 = 0.5 * (r1 - a);
      for (Eigen::Index j = 0; j < x.size(); j++) {
        const double r2 = m2 + h2 * x(j);
        inpart += w(j) * h2 * std::pow(r2, k) * u(r2);
      }
      outer += w(g) * hf * u(r1) * std::pow(r1, -(k + 1)) * (cum + inpart);
    }
    // Advance the cumulative inner integral past this whole panel.
    for (Eigen::Index j = 0; j < x.size(); j++) {
      const double r2 = mid + hf * x(j);
      cum += w(j) * hf * std::pow(r2, k) * u(r2);
    }
  }
  return 2.0 * outer;
}

int main() {
  printf("Cross-basis exchange tests\n");

  // Two DIFFERENT meshes, so the common refinement is non-trivial: the
  // element boundaries of one fall inside the elements of the other.
  const double zeta = 1.4, alpha = 0.6;
  const lcao::AOBasis B = lcao::make_ao_basis(12.0, 4, 15, 7);   // orbitals
  const lcao::AOBasis A = lcao::make_ao_basis(10.0, 4, 15, 11);  // "auxiliary"

  auto eval_sto = [](double r, int l, double z) {
    return lcao::radial_STO(r, l, z);
  };
  auto eval_gto = [](double r, int l, double a) {
    return lcao::radial_GTO(r, l, a);
  };

  // One occupied orbital on B, one trial function on A.
  const Vector cphi = lcao::ao_coefficients(B, 0, zeta, eval_sto);
  const Vector cg = lcao::ao_coefficients(A, 0, alpha, eval_gto);
  Matrix C_occ(cphi.size(), 1);
  C_occ.col(0) = cphi;
  Vector occ(1);
  occ(0) = 1.0;

  for (int k : {0, 1, 2}) {
    const Matrix K = A.rad.exchange(B.rad, C_occ, occ, k);
    const double got = cg.dot(K * cg);

    auto u = [&](double r) {
      return eval_expansion(A.rad, cg, r) * eval_expansion(B.rad, cphi, r);
    };
    // Two resolutions, so the reference itself is shown to be converged.
    const double R = 10.0;
    const double ref1 = brute_force(u, R, k, 160, 14);
    const double ref2 = brute_force(u, R, k, 320, 14);
    const double refconv = std::abs(ref2 - ref1) / std::abs(ref2);
    const double dev = std::abs(got - ref2) / std::abs(ref2);
    printf("  k=%d  implementation %.12e   reference %.12e\n", k, got, ref2);
    check(refconv < 1e-10, "  brute-force reference is itself converged", refconv);
    check(dev < 1e-8, "  exchange matches the brute-force double integral", dev);
  }

  // Symmetry is structural, but cheap to assert.
  {
    const Matrix K = A.rad.exchange(B.rad, C_occ, occ, 0);
    const double asym = (K - K.transpose()).cwiseAbs().maxCoeff() /
                        std::max(1e-30, K.cwiseAbs().maxCoeff());
    check(asym < 1e-12, "K is symmetric", asym);
  }

  // Occupations scale linearly, and an empty orbital set gives nothing.
  {
    Vector occ2(1);
    occ2(0) = 2.5;
    const Matrix K1 = A.rad.exchange(B.rad, C_occ, occ, 0);
    const Matrix K2 = A.rad.exchange(B.rad, C_occ, occ2, 0);
    const double dev = (K2 - 2.5 * K1).cwiseAbs().maxCoeff() /
                       std::max(1e-30, K2.cwiseAbs().maxCoeff());
    check(dev < 1e-12, "K is linear in the occupations", dev);
  }

  // The case the cross-basis route exists for: a primitive so tight that
  // the orbital basis cannot represent it at all. Projecting it onto B
  // would destroy its norm; here it never touches B's grid.
  printf("  tight primitives, which projection cannot reach:\n");
  for (double a : {1e2, 1e4, 1e6, 1e8}) {
    const lcao::AOBasis At = lcao::make_ao_basis(lcao::gto_rmax(a));
    const Vector ct = lcao::ao_coefficients(At, 0, a, eval_gto);
    const Matrix K = At.rad.exchange(B.rad, C_occ, occ, 0);
    const double got = ct.dot(K * ct);
    auto ut = [&](double r) {
      return eval_expansion(At.rad, ct, r) * eval_expansion(B.rad, cphi, r);
    };
    // Brute force over the primitive's own support, which is all it has.
    const double Rt = lcao::gto_rmax(a);
    const double ref = brute_force(ut, Rt, 0, 200, 14);
    const double dev = std::abs(got - ref) / std::abs(ref);
    printf("    alpha=%-9.1e  K=%.12e  ref=%.12e  dev=%.2e\n", a, got, ref, dev);
    if (!(dev < 1e-7)) {
      printf("      FAILED\n");
      failures++;
    }
  }

  printf("%s\n", failures ? "FAILURES" : "All tests passed.");
  return failures ? 1 : 0;
}
