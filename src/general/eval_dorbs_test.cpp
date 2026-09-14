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

// Does RadialBasis::eval_dorbs return the true derivatives of the PHYSICAL
// radial function R(r) = u(r)/r at an arbitrary radius?
//
// The obvious test -- finite-difference eval_orbs and compare -- is worthless
// here, because differencing a tabulated R is exactly the thing eval_dorbs
// exists to replace. It would confirm the derivative against the tool it is
// meant to retire, and it would pass just as happily on a formula that is
// consistently wrong.
//
// So the check is against PHYSICS instead. Solve the isotropic harmonic trap
// in this basis, where the eigenvalues are analytic: with m = hbar = 1 and
// V = w^2 r^2 / 2,
//
//     E = (2 n_r + l + 3/2) w.
//
// Any converged eigenpair must then satisfy the radial Schroedinger equation
// POINTWISE,
//
//     -1/2 [ R'' + (2/r) R' - l(l+1) R / r^2 ] + V(r) R  ==  E R,
//
// which exercises n = 1 and n = 2 simultaneously, against an equation neither
// of them was derived from. It is also the sharpest available probe of the
// first element: R = u/r is 0/0 at the origin and eval_psi_dnf deflates the
// (x+1) factor analytically, so a deflation that is wrong -- or merely
// cancelling catastrophically -- puts a large residual exactly where the
// centrifugal and 2/r terms are largest.
//
// What the residual does NOT measure is worth stating, because it decides how
// the test has to be written. A variational solution satisfies the equation
// only within the span of the basis, so the POINTWISE residual is not machine
// zero even for exact derivatives: it is the basis truncation error, and it
// grows with n_r and l (measured 2e-10 for the l=0 ground state, 3e-8 for
// l=1, n_r=2 on the grid below). Any fixed tolerance here would therefore be
// a statement about the basis, not about eval_dorbs.
//
// So the test is made self-calibrating instead: REFINE the basis and require
// the residual to fall. A correct derivative converges with the basis; a
// wrong formula leaves an O(1) residual that refinement cannot touch. The
// absolute bar is kept only as a loose sanity floor.

#include "PolynomialBasis.h"
#include "FiniteElementBasis.h"
#include "RadialBasis.h"
#include "Matrix.h"

#include <Eigen/Eigenvalues>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace helfem;

namespace {

  int nfail = 0;

  void report(const char * what, bool ok, const char * detail = "") {
    if(!ok) nfail++;
    printf("  %-46s %s %s\n", what, ok ? "ok" : "FAIL", detail);
  }

  void below(const char * what, double got, double tol) {
    const bool ok = (got < tol) && std::isfinite(got);
    if(!ok) nfail++;
    printf("  %-46s %9.2e  (tol %7.1e)  %s\n", what, got, tol, ok ? "ok" : "FAIL");
  }

  const double omega = 1.0;

  /// A RadialBasisT that implements only the pure virtuals, to check that the
  /// defaulted eval_dorbs keeps such a class compiling and that it fails only
  /// when a derivative is actually requested.
  class BareRadialBasis : public atomic::basis::RadialBasisT<double> {
   public:
    size_t Nbf() const override { return 1; }
    helfem::Matrix overlap() const override { return helfem::Matrix::Identity(1,1); }
    helfem::Matrix kinetic() const override { return helfem::Matrix::Zero(1,1); }
    helfem::Matrix kinetic_l() const override { return helfem::Matrix::Zero(1,1); }
    helfem::Matrix nuclear() const override { return helfem::Matrix::Zero(1,1); }
    helfem::Vector eval_orbs(const helfem::Matrix & C, double r) const override {
      (void) r;
      return helfem::Vector::Constant(C.cols(), 7.0);
    }
  };

  /// Worst relative eigenvalue error and worst relative pointwise
  /// Schroedinger residual over l = 0..2 and n_r = 0..2, in one basis.
  struct Worst {
    double eig = 0.0;
    double resid = 0.0;
  };

  Worst solve_trap(int nnodes, int nelem, double rmax, int n_quad, bool verbose) {
    namespace pb = helfem::polynomial_basis;
    using atomic::basis::FEMRadialBasis;

    std::shared_ptr<const pb::PolynomialBasis> poly(pb::make_basis(4, nnodes));
    const helfem::Vector bval = helfem::Vector::LinSpaced(nelem + 1, 0.0, rmax);
    pb::FiniteElementBasis fem(poly, bval, true, false, true, false);
    FEMRadialBasis radial(fem, n_quad);

    const helfem::Matrix S = radial.overlap();
    const helfem::Matrix Tkin = radial.kinetic();
    const helfem::Matrix Kl = radial.kinetic_l();
    // V_ij = integral B_i (w^2 r^2 / 2) B_j dr; the weight is a degree-2
    // polynomial, so say so and let the quadrature seed itself accordingly.
    const helfem::Matrix V = radial.matrix_element(
        FEMRadialBasis::BasisKind::B0, FEMRadialBasis::BasisKind::B0,
        [](double r) { return 0.5 * omega * omega * r * r; },
        std::vector<double>(), 2);

    // Radii at which the residual is evaluated. Deliberately dense in the
    // FIRST element, where R = u/r is 0/0 and the deflation has to work.
    const double h1 = rmax / nelem;
    std::vector<double> rg;
    for(int i = 1; i <= 20; i++) rg.push_back(h1 * i / 21.0);
    for(int i = 0; i < 40; i++) rg.push_back(h1 + 0.125 * i);

    Worst w;
    if(verbose)
      printf("  nnodes=%-3i nelem=%-3i Nbf=%-4i", nnodes, nelem,
             (int) radial.Nbf());

    for(int l = 0; l <= 2; l++) {
      helfem::Matrix H = Tkin + V;
      if(l > 0) H += (double) (l * (l + 1)) * Kl;

      Eigen::GeneralizedSelfAdjointEigenSolver<helfem::Matrix> es(H, S);
      const helfem::Vector E = es.eigenvalues();
      const helfem::Matrix C = es.eigenvectors();

      for(int nr = 0; nr < 3; nr++) {
        const double exact = (2.0 * nr + l + 1.5) * omega;
        w.eig = std::max(w.eig, std::abs(E(nr) - exact) / exact);

        const helfem::Matrix Cn = C.col(nr);
        double worst = 0.0, scale = 0.0;
        for(size_t i = 0; i < rg.size(); i++) {
          const double r = rg[i];
          const double R0 = radial.eval_dorbs(Cn, r, 0)(0);
          const double R1 = radial.eval_dorbs(Cn, r, 1)(0);
          const double R2 = radial.eval_dorbs(Cn, r, 2)(0);
          // -1/2 [ R'' + (2/r) R' - l(l+1) R/r^2 ] + V R  -  E R
          const double lhs =
              -0.5 * (R2 + 2.0 / r * R1
                      - (double) (l * (l + 1)) * R0 / (r * r))
              + 0.5 * omega * omega * r * r * R0;
          worst = std::max(worst, std::abs(lhs - E(nr) * R0));
          scale = std::max(scale, std::abs(E(nr) * R0));
        }
        w.resid = std::max(w.resid, worst / scale);
      }
    }
    if(verbose)
      printf("   worst |dE|/E %8.2e   worst residual %8.2e\n", w.eig, w.resid);
    return w;
  }

} // namespace

int main() {
  namespace pb = helfem::polynomial_basis;
  using atomic::basis::FEMRadialBasis;

  const double rmax = 10.0;
  const int n_quad = 30;

  // The harmonic ground state decays as exp(-w r^2 / 2), so a boundary at
  // r = 10 sits at exp(-50): the Dirichlet condition there is exact to far
  // beyond double precision, and nothing below is a boundary artefact.
  printf("harmonic trap, worst over l = 0..2 and n_r = 0..2\n");
  const Worst coarse = solve_trap(8, 5, rmax, n_quad, true);
  const Worst fine = solve_trap(15, 10, rmax, n_quad, true);
  printf("\n");

  // The eigenvalues first: a residual test says nothing unless the state
  // really is an eigenstate of the analytic problem.
  below("eigenvalues vs (2 n_r + l + 3/2) w", fine.eig, 1e-10);

  // The self-calibrating check. A correct derivative makes the pointwise
  // residual a basis-truncation error, which refinement drives down; a wrong
  // formula leaves an O(1) residual that refinement cannot touch. Requiring
  // merely a factor of two is deliberately weak -- the measured drop is far
  // larger -- because the point is to separate "converges" from "does not",
  // not to pin an exponent the basis, not eval_dorbs, controls.
  {
    const double ratio = coarse.resid / fine.resid;
    char detail[96];
    snprintf(detail, sizeof(detail), "-- %.2e -> %.2e, a factor of %.0f",
             coarse.resid, fine.resid, ratio);
    report("residual falls when the basis is refined", ratio > 2.0, detail);
  }

  // A loose absolute floor. Not the real test: it is a statement about this
  // basis, and it exists only to catch a residual that converges to the
  // wrong thing.
  below("residual in the fine basis (sanity floor)", fine.resid, 1e-6);

  printf("\ncontracts\n");
  {
    std::shared_ptr<const pb::PolynomialBasis> poly(pb::make_basis(4, 15));
    const helfem::Vector bval = helfem::Vector::LinSpaced(11, 0.0, rmax);
    pb::FiniteElementBasis fem(poly, bval, true, false, true, false);
    FEMRadialBasis radial(fem, n_quad);
    const Eigen::Index nbf = (Eigen::Index) radial.Nbf();
    const helfem::Matrix Cn = helfem::Matrix::Random(nbf, 3);

    // n = 0 must BE eval_orbs, bit for bit. That is not a coincidence to be
    // checked loosely: eval_orbs delegates to eval_dorbs(C, r, 0), so any
    // difference at all would mean the delegation had been broken.
    double d0 = 0.0;
    for(int i = 1; i <= 50; i++) {
      const double r = rmax * i / 51.0;   // spans every element
      const helfem::Vector a = radial.eval_orbs(Cn, r);
      const helfem::Vector b = radial.eval_dorbs(Cn, r, 0);
      d0 = std::max(d0, (a - b).cwiseAbs().maxCoeff());
    }
    report("eval_dorbs(n=0) is eval_orbs exactly", d0 == 0.0);

    for(int n = 0; n <= 2; n++) {
      const helfem::Vector far = radial.eval_dorbs(Cn, rmax + 5.0, n);
      char lbl[64];
      snprintf(lbl, sizeof(lbl), "zero beyond practical infinity, n=%i", n);
      report(lbl, far.cwiseAbs().maxCoeff() == 0.0);
    }
  }

  printf("\nthe defaulted base-class method\n");
  {
    BareRadialBasis bare;
    const helfem::Matrix C1 = helfem::Matrix::Zero(1, 2);
    bool n0_ok = false, threw = false;
    try {
      n0_ok = (bare.eval_dorbs(C1, 1.0, 0).array() == 7.0).all();
    } catch (const std::exception &) {}
    try {
      bare.eval_dorbs(C1, 1.0, 1);
    } catch (const std::logic_error &) {
      threw = true;
    }
    report("n=0 falls through to eval_orbs", n0_ok);
    report("n>0 throws rather than returning garbage", threw);
  }

  printf("\n%s\n", nfail ? "EVAL_DORBS TEST FAILED" : "EVAL_DORBS TEST PASSED");
  return nfail ? 1 : 0;
}
