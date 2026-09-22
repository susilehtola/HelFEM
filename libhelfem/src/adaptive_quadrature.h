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
#ifndef HELFEM_ADAPTIVE_QUADRATURE_H
#define HELFEM_ADAPTIVE_QUADRATURE_H

// The one adaptive-quadrature refinement loop in HelFEM.
//
// There used to be three -- FiniteElementBasis.cpp's converge_panel (every
// one-electron matrix element), RadialBasis.cpp's converge_rule (atomic 2e,
// Yukawa, cross-basis projection) and diatomic's converge_block -- copies of
// one algorithm that had drifted apart:
//
//   * their stopping tests differed: the two libhelfem copies had a
//     two-doubling stall exit the diatomic one lacked, and the diatomic one
//     had a machine-floor exit the libhelfem ones lacked;
//   * the libhelfem copies reported NO numbers when they hit the order cap --
//     not the magnitude, not the change -- so a near miss and a wildly
//     unconverged block printed the same line, and converge_panel's label was
//     a hardcoded string naming no element, panel or weight;
//   * each wrote its warn-once flag as a plain bool from inside an OpenMP
//     parallel region, formally a data race.
//
// This header is private to the build (it is not installed). It is reached
// from libhelfem/src directly and from src/ through the libhelfem/src entry
// on helfem-fem's build-interface include path.

#include <helfem/Matrix.h>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>

namespace helfem {
  namespace adaptive {

    /// What a refinement concluded.
    ///
    /// `rel` is the relative change the convergence test last saw -- diff /
    /// scale over the judged block -- with two sentinels:
    ///   -1  the cap was reached on the very first probe, so no comparison was
    ///       ever made (distinct from "the change was small");
    ///   -2  seed_fallback returned the seed block.
    /// `printed` is true for the ONE call per warn-flag that emitted the
    /// warning, which is what lets a test check "exactly once" directly.
    struct CapReport {
      bool capped = false;
      int n = 0;
      double scale = 0.0;
      double rel = 0.0;
      bool printed = false;
      /// Which exit ended the refinement: 1-4 as numbered at refine(), 0 for
      /// the order cap. Worth having because the exits are not equally
      /// strict: (4) also fires on a difference that merely HALVES per
      /// doubling -- algebraic convergence -- and would stop such a block
      /// at ~sqrt(eps) rather than at eps.
      int exit = 0;
    };

    /// Tuning of one refinement. Defaults are the stopping rule; the rest are
    /// per-call-site behaviour.
    struct Options {
      /// Starting order.
      int nstart = 2;
      /// Order cap: refinement doubles n, clamped here.
      int nmax = 512;
      /// On hitting the cap, return the SEED block quietly instead of warning.
      /// For integrands known to be genuinely non-convergent where the block
      /// is never used (diatomic's innermost-element Q_L for |M| >= 2).
      bool seed_fallback = false;
      /// Exclude the first `nskip` rows and columns from the convergence test;
      /// the block is still returned whole. For entries that diverge but are
      /// structurally discarded (diatomic radial_integral(-1,0)).
      Eigen::Index nskip = 0;
    };

    /// Max |entry| of the part of `M` that is judged.
    template <typename T>
    T judged_max(const helfem::Mat<T> & M, Eigen::Index nskip) {
      const Eigen::Index nk = (nskip < M.rows() && nskip < M.cols()) ? nskip : 0;
      return M.bottomRightCorner(M.rows() - nk, M.cols() - nk).cwiseAbs().maxCoeff();
    }

    /// Refine `probe(n)` -- which builds its block from its own n-point rule
    /// -- by doubling n from opt.nstart until the block is stable, and return
    /// it. The block's shape must be independent of n.
    ///
    /// Stopping rule: the UNION of the tests the three former copies used.
    /// Each returns the current block, so adding an exit can only make a loop
    /// stop earlier, never later.
    ///
    ///   (1) eps convergence: the block stops changing to 8*eps(T). The common
    ///       exit, and the only one a polynomial-exact block ever needs.
    ///   (2) machine floor: deep in the asymptotic regime (<= sqrt(eps)) AND
    ///       already at a small multiple of eps -- all of T's precision has
    ///       been extracted even if the difference is still halving.
    ///   (3) one-doubling stall: deep in the asymptotic regime and a doubling
    ///       no longer at least halves the difference -- it is roundoff noise.
    ///   (4) two-doubling stall: noise at the floor can happen to keep halving
    ///       and so dodge (3); genuine convergence gains far more than 8x over
    ///       two doublings (geometric convergence squares the error per
    ///       doubling), while noise stays flat.
    ///
    /// On hitting the cap without converging, the current block is returned
    /// as the best estimate and ONE warning is printed per `warned` flag,
    /// naming the integral (`label()`, only evaluated then) and reporting the
    /// block magnitude and the last relative change. `warned` is claimed with
    /// an atomic exchange, so exactly one caller prints even when many hit the
    /// cap concurrently from an OpenMP region.
    template <typename T, typename Fn, typename LabelFn>
    helfem::Mat<T> refine(const Fn & probe, const Options & opt,
                          const LabelFn & label, std::atomic<bool> & warned,
                          CapReport * report = nullptr, int * nconv = nullptr) {
      const T eps = std::numeric_limits<T>::epsilon();
      const T tol = T(8) * eps;
      const T sqrteps = std::sqrt(eps);
      // A small multiple of eps above 8*eps: the assembly roundoff floor of a
      // block that is not polynomial-exact, far below any tolerance that
      // matters downstream.
      const T floor_rel = T(256) * eps;

      helfem::Mat<T> prev, cur, seed;
      bool have = false;
      T prevdiff = T(-1), prevprevdiff = T(-1);
      int n = std::max(opt.nstart, 2);
      for (;;) {
        cur = probe(n);
        if (!have)
          seed = cur;
        if (have) {
          const T diff = judged_max<T>(cur - prev, opt.nskip);
          const T scale = judged_max<T>(cur, opt.nskip);
          const T s = scale + tol;
          const bool asymptotic = diff <= sqrteps * s;
          int exit = 0;
          if (diff <= tol * s)
            exit = 1;
          else if (asymptotic && diff <= floor_rel * s)
            exit = 2;
          else if (asymptotic && prevdiff >= T(0) && diff > T(0.5) * prevdiff)
            exit = 3;
          else if (asymptotic && prevprevdiff >= T(0) && diff > T(0.125) * prevprevdiff)
            exit = 4;
          if (exit) {
            if (nconv) *nconv = n;
            if (report) {
              report->capped = false;
              report->n = n;
              report->scale = (double) scale;
              report->rel = (double) (scale > T(0) ? diff / scale : T(0));
              report->printed = false;
              report->exit = exit;
            }
            return cur;
          }
          prevprevdiff = prevdiff;
          prevdiff = diff;
        }
        prev = cur;
        have = true;

        if (n >= opt.nmax) {
          if (opt.seed_fallback) {
            if (nconv) *nconv = std::max(opt.nstart, 2);
            if (report) {
              report->capped = true;
              report->n = n;
              report->scale = (double) seed.cwiseAbs().maxCoeff();
              report->rel = -2.0;
              report->printed = false;
              report->exit = 0;
            }
            return seed;
          }

          const T scale = judged_max<T>(cur, opt.nskip);
          // prevdiff is the last difference the convergence test itself saw.
          // It is -1 only when the cap was reached on the first probe.
          const double rel = (prevdiff >= T(0) && scale > T(0))
              ? (double) (prevdiff / scale) : -1.0;
          // exchange() rather than a plain test-and-set: many threads can hit
          // the cap at once from an OpenMP region, and exactly one may print.
          const bool first = !warned.exchange(true);
          if (first) {
            const std::string what = label();
            if (rel >= 0.0)
              printf("Warning: %s hit the quadrature order cap (n=%d) without"
                     " converging to eps; using the best estimate.\n"
                     "  block magnitude %.3e, relative change still %.3e\n",
                     what.c_str(), opt.nmax, (double) scale, rel);
            else
              printf("Warning: %s was seeded at or above the quadrature order"
                     " cap (n=%d), so its convergence was never tested.\n"
                     "  block magnitude %.3e\n",
                     what.c_str(), opt.nmax, (double) scale);
            if ((double) scale > 1e12)
              printf("  ** The integrand spans too many orders of magnitude for\n"
                     "     this precision; results built on it are not\n"
                     "     trustworthy. Reduce the angular momentum or the\n"
                     "     extent of the grid.\n");
            fflush(stdout);
          }
          if (nconv) *nconv = n;
          if (report) {
            report->capped = true;
            report->n = n;
            report->scale = (double) scale;
            report->rel = rel;
            report->printed = first;
            report->exit = 0;
          }
          return cur;
        }
        n = std::min(2 * n, opt.nmax);
      }
    }

  } // namespace adaptive
} // namespace helfem

#endif // HELFEM_ADAPTIVE_QUADRATURE_H
