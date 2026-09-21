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
#ifndef HELFEM_DIATOMIC_CONVERGE_BLOCK_H
#define HELFEM_DIATOMIC_CONVERGE_BLOCK_H

// Split out of basis.cpp so it can be driven by a test. It is a template over
// the probe, so a header is its natural home anyway; before, it sat in an
// anonymous namespace where nothing outside basis.cpp could reach it, and the
// numbers its cap warning reports could not be checked at all.

#include <helfem/Matrix.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

namespace helfem {
  namespace diatomic {
    namespace basis {
      namespace detail {

  // --------------------------------------------------------------------
  // Auto-converging quadrature for the in-element two-electron kernel.
  //
  // Mirrors the atomic Stage-1 helper (libhelfem/src/RadialBasis.cpp,
  // converge_rule): recompute a probe block at a rising quadrature order
  // until it stops changing, so the accuracy of the two-electron radial
  // integrals is set by the floating-point type (here double's eps)
  // instead of a user-supplied --nquad. The diatomic path is double-only,
  // so this is a plain double specialization rather than a template.
  //
  // The probe builds its own rule, and the FAMILY is chosen per
  // integrand class (see libhelfem/src/RadialBasis.cpp for the full
  // analysis of why this matters):
  //   * Gauss-Lobatto for the analytic blocks -- radial_integral's
  //     sinh^m cosh^n weights, kinetic's sinh weight, and the
  //     cross-basis overlap projection. These integrands are entire (or
  //     polynomial x entire), so a polynomial-exact rule converges
  //     geometrically and exit (1) fires after a doubling or two. The
  //     modified Gauss-Chebyshev rule, being the trapezoid rule under
  //     the sin^4-Jacobian Perez-Jorda transformation, has a FIXED
  //     Euler-Maclaurin order O(n^-10) no matter how smooth the
  //     integrand -- ample for double, but needlessly slow to refine.
  //   * Gauss-Chebyshev for the P_L/Q_L Green's-function blocks
  //     (Plm/Qlm integrals, the in-element two-electron kernel). Over
  //     the element touching mu=0 the Q_L(cosh mu) weight is endpoint-
  //     singular (Q_L(1) = +inf), which a Lobatto endpoint node would
  //     evaluate directly, and for endpoint-singular integrands the
  //     Chebyshev rule's sin^4 node clustering is the right treatment
  //     anyway.
  //
  // TWO stopping conditions, both meaning "all of double's precision has
  // been extracted":
  //   (1) True eps convergence: the block stops changing to 8*eps, the
  //       same criterion FiniteElementBasis's 1e matrix_element uses.
  //   (2) Roundoff-floor stall: the 2e Coulomb P_L(cosh mu_<)Q_L(cosh mu_>)
  //       Green's function is NOT polynomial-exact, so (unlike the 1e
  //       Gauss-Lobatto block) the block difference converges only down to
  //       the assembly roundoff floor and then wobbles rather than
  //       collapsing to zero. Once it is deep in the asymptotic regime
  //       (diff <= sqrt(eps)*(scale+tol)) and a doubling no longer at least
  //       halves it (diff > 0.5*prevdiff), it has converged: return
  //       quietly. Without this every production element would grind to the
  //       cap and print a spurious warning.
  //
  // nmax cap = 512 with a one-shot warning is a genuine backstop, not the
  // common path; the start order is seeded from --nquad so the common case
  // converges in 1-2 steps.
  // --------------------------------------------------------------------

        /// Order cap for the refinement loop.
        inline constexpr int twoe_nmax = 512;
        /// Warn at most once per process if the refinement hits the order cap.
        /// `inline` so there is exactly one flag per program, as there was
        /// when this lived in a single translation unit.
        inline bool twoe_cap_warned = false;

        /// What converge_block concluded. `rel` is the relative change the
        /// convergence test last saw -- diff / scale over the block, excluding
        /// the first `nskip` rows and columns -- with two sentinels:
        ///   -1  the cap was reached on the very first probe, so no comparison
        ///       was ever made (formerly reported, indistinguishably, as 0);
        ///   -2  seed_fallback returned the seed block after comparisons.
        struct CapReport {
          bool capped = false;   ///< hit the order cap without converging
          int n = 0;             ///< final quadrature order
          double scale = 0.0;    ///< max |block| entry judged
          double rel = 0.0;      ///< see above
        };

        /// Refine `probe(n)` -- which must rebuild its block using its own
        /// n-point Gauss-Chebyshev rule -- by doubling n from nstart until the
        /// block is stable, and return the converged block. If `nconv` is
        /// non-null it receives the (finer) converged order, so a caller can
        /// rebuild a heavier object (e.g. a TwoElectronElement) once at that
        /// order. The block's shape is independent of n, so the
        /// block-difference comparison is well defined.
        ///
        /// `seed_fallback` selects what happens if the order cap is reached
        /// without convergence. The disjoint Q_L integral over the element that
        /// touches mu=0 is genuinely NON-convergent for |M| >= 2 -- there
        /// Q_{L,|M|}(cosh mu) ~ mu^{-|M|}, so the bare integral (without the
        /// companion P_{L,|M|} ~ mu^{|M|} that regularizes it inside the
        /// in-element kernel) diverges. That block is never actually used (the
        /// innermost element is never the OUTER element of a disjoint pair), so
        /// forcing it is pointless. With seed_fallback the routine then returns
        /// the block at the seed order -- exactly the fixed --nquad value the
        /// pre-auto-convergence code produced -- quietly. Without it (the
        /// in-element kernel, which IS convergent) a cap is a real anomaly and
        /// gets the one-shot warning + best estimate.
        /// `nskip` excludes the first `nskip` rows and columns from the
        /// convergence test (the block itself is returned whole). It exists
        /// for radial_integral(-1,0): that integrand is 1/sinh(mu), which
        /// diverges logarithmically at mu=0, so the entries involving the one
        /// basis function that does not vanish there never converge -- the
        /// block magnitude GROWS with the quadrature order. Those entries are
        /// structurally discarded (remove_boundaries drops that function from
        /// every m != 0 shell, which is the only place the integral is used),
        /// so judging the block on them reports a failure that cannot happen
        /// and hides one that could.
        ///
        /// `report`, if non-null, receives the outcome -- in particular what
        /// the cap warning prints, filled on EVERY call rather than only the
        /// first. The warning itself is one-shot per process, so without this
        /// the numbers it reports were unobservable after the first cap.
        template <typename Fn>
        helfem::Matrix converge_block(const Fn & probe, int nstart,
                                      const char * what, int * nconv = nullptr,
                                      bool seed_fallback = false,
                                      Eigen::Index nskip = 0,
                                      CapReport * report = nullptr) {
          const double eps     = std::numeric_limits<double>::epsilon();
          const double tol     = 8.0 * eps;
          const double sqrteps = std::sqrt(eps);
          // Machine-precision floor. These blocks are not polynomial-exact and
          // the LIP assembly carries a roundoff floor of a small multiple of
          // eps (empirically ~1e-15 relative), so the block difference
          // plateaus there rather than collapsing to 8*eps. 256*eps ~ 5.7e-14
          // is comfortably above that floor yet far below both the cd_thresh
          // (1e-12) the kernel is later factorized to and the 1e-10 total-
          // energy tolerance: once the difference reaches it, all of double's
          // precision has been extracted.
          const double floor_rel = 256.0 * eps;

          helfem::Matrix prev, cur, seed;
          bool have = false;
          double prevdiff = -1.0;
          int n = std::max(nstart, 2);
          for(;;) {
            cur = probe(n);
            if(!have)
              seed = cur;   // the fixed --nquad value, for seed_fallback
            if(have) {
              const Eigen::Index nk =
                  (nskip < cur.rows() && nskip < cur.cols()) ? nskip : 0;
              const Eigen::Index nr = cur.rows() - nk, nc = cur.cols() - nk;
              const double diff =
                  (cur - prev).bottomRightCorner(nr, nc).cwiseAbs().maxCoeff();
              const double scale =
                  cur.bottomRightCorner(nr, nc).cwiseAbs().maxCoeff();
              // (1) true eps convergence (well-conditioned, polynomial-exact
              // blocks reach this).
              if(diff <= tol * (scale + tol)) {
                if(nconv) *nconv = n;
                if(report) *report = CapReport{false, n, scale, diff / scale};
                return cur;
              }
              // (2) roundoff-floor stall: the Coulomb P_L(cosh mu_<)Q_L(cosh
              // mu_>) Green's function is not polynomial-exact, and over the
              // element that touches mu=0 the Q_L weight is only C^0 (a
              // mu*ln(mu) endpoint kink, Q_L(1)=+inf). So the block difference
              // converges only down to the assembly roundoff floor and then
              // wobbles. Once it is deep in the asymptotic regime
              // (diff <= sqrt(eps)*scale) AND it has either reached the
              // machine-precision floor (diff <= floor_rel*scale) or a doubling
              // no longer at least halves it, all of double's precision has
              // been extracted: stop quietly. The floor test makes this robust
              // when only a couple of doublings separate the seed from the cap.
              if(diff <= sqrteps * (scale + tol) &&
                 (diff <= floor_rel * (scale + tol) ||
                  (prevdiff >= 0.0 && diff > 0.5 * prevdiff))) {
                if(nconv) *nconv = n;
                if(report) *report = CapReport{false, n, scale, diff / scale};
                return cur;
              }
              prevdiff = diff;
            }
            prev = cur;
            have = true;
            if(n >= twoe_nmax) {
              if(seed_fallback) {
                // Non-convergent integrand (the divergent innermost-element Q_L
                // for |M| >= 2). Fall back to the requested --nquad order
                // quietly -- the same value the fixed-order code produced.
                if(nconv) *nconv = std::max(nstart, 2);
                if(report) *report = CapReport{true, n, seed.cwiseAbs().maxCoeff(),
                                               prevdiff >= 0.0 ? -2.0 : -1.0};
                return seed;
              }
              // Report the MAGNITUDE, not just the fact. The failure mode
              // here is P_L(cosh mu) ~ (cosh mu)^L reaching 1e60 and more
              // at high L, which no relative convergence test can resolve;
              // the resulting Fock matrix can carry elements many orders
              // of magnitude too large, and the SCF then stops early
              // because the noise floor it infers from that spectrum
              // swamps the convergence threshold. Printing the scale makes
              // that visible instead of leaving it to be inferred.
              // Deliberately no failure counter: converge_block runs inside
              // the OpenMP exchange loop, and a shared counter would be a
              // data race for a cosmetic number. The magnitude below is
              // what identifies the problem anyway.
              const Eigen::Index nk =
                  (nskip < cur.rows() && nskip < cur.cols()) ? nskip : 0;
              const double scale = cur.bottomRightCorner(cur.rows() - nk,
                                                         cur.cols() - nk)
                                       .cwiseAbs().maxCoeff();
                // prevdiff, NOT (cur - prev): prev was overwritten with cur a
                // few lines above, so that difference is identically zero and
                // the warning used to report "relative change still 0.000e+00"
                // however badly the block was actually converging. prevdiff is
                // the last difference the convergence test itself saw, and is
                // -1 only if the cap was already reached at the seed order, in
                // which case no difference exists to report.
              const double rel = (prevdiff >= 0.0 && scale > 0.0)
                  ? prevdiff / scale : prevdiff;
              if(report) *report = CapReport{true, n, scale, rel};
              if(!twoe_cap_warned) {
                twoe_cap_warned = true;
                if(rel >= 0.0)
                  printf("Warning: diatomic %s hit the quadrature order cap"
                         " (n=%d) without converging to eps(double).\n"
                         "  block magnitude %.3e, relative change still %.3e\n",
                         what, twoe_nmax, scale, rel);
                else
                  printf("Warning: diatomic %s was seeded at or above the"
                         " quadrature order cap (n=%d), so its convergence was"
                         " never tested.\n"
                         "  block magnitude %.3e\n",
                         what, twoe_nmax, scale);
                if(scale > 1e12)
                  printf("  ** The integrand spans too many orders of magnitude for\n"
                         "     double precision. Results from this run are NOT\n"
                         "     trustworthy: the Fock matrix built from these\n"
                         "     integrals can be wrong by orders of magnitude, and\n"
                         "     the SCF may stop early because the noise floor it\n"
                         "     infers from that spectrum exceeds the convergence\n"
                         "     threshold. Reduce lmax, or reduce Rmax/Rbond.\n");
                fflush(stdout);
              }
              if(nconv) *nconv = n;
              return cur;
            }
            n = std::min(2 * n, twoe_nmax);
          }
        }

      } // namespace detail
    } // namespace basis
  } // namespace diatomic
} // namespace helfem

#endif // HELFEM_DIATOMIC_CONVERGE_BLOCK_H
