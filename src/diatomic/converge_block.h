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
#include "adaptive_quadrature.h"
#include <string>
#include <atomic>

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
  // Shares its loop with the atomic 2e primitives (libhelfem/src/
  // RadialBasis.cpp, converge_rule): recompute a probe block at a rising
  // quadrature order
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
  // The stopping rule and the cap warning are NOT defined here: this is a
  // thin wrapper over helfem::adaptive::refine (libhelfem/src/
  // adaptive_quadrature.h), the one refinement loop shared with the atomic
  // 2e primitives and every one-electron matrix element. See there for the
  // four exits. What stays here is the choice of quadrature FAMILY above,
  // which is a property of each probe, and the two diatomic-specific
  // behaviours below (seed_fallback and nskip).
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
        inline std::atomic<bool> twoe_cap_warned{false};

        /// The shared report (see adaptive_quadrature.h): capped, n, scale,
        /// rel (-1 = no comparison made, -2 = seed_fallback) and printed.
        using CapReport = helfem::adaptive::CapReport;

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
          helfem::adaptive::Options opt;
          opt.nstart = nstart;
          opt.nmax = twoe_nmax;
          opt.seed_fallback = seed_fallback;
          opt.nskip = nskip;
          // The stopping rule is adaptive::refine's -- the union of the three
          // former copies. This one used to lack the two-doubling stall exit
          // the libhelfem copies had, making it the least able of the three
          // to recognise a roundoff floor.
          return helfem::adaptive::refine<double>(
              probe, opt,
              [what]() { return std::string("diatomic ") + what; },
              twoe_cap_warned, report, nconv);
        }

      } // namespace detail
    } // namespace basis
  } // namespace diatomic
} // namespace helfem

#endif // HELFEM_DIATOMIC_CONVERGE_BLOCK_H
