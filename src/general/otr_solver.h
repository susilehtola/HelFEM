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
#ifndef HELFEM_OTR_SOLVER_H
#define HELFEM_OTR_SOLVER_H

#include <functional>
#include <string>

namespace helfem {
  /// Thin C++ wrapper around OpenTrustRegion's C interface.
  ///
  /// OpenTrustRegion minimizes an objective that is parametrized by a step
  /// vector taken from a *reference* point which the caller owns. The
  /// division of labour is
  ///
  ///   update(kappa, ...)  apply the step, MAKE IT THE NEW REFERENCE, and
  ///                       report the objective, gradient and Hessian
  ///                       diagonal there;
  ///   objective(kappa)    apply the step to the current reference WITHOUT
  ///                       adopting it, and report only the objective;
  ///   hessian_vector(x)   Hessian-vector product at the current reference.
  ///
  /// Every kappa the library hands out is relative to whatever `update` last
  /// installed, never to the starting point, so the caller never has to
  /// accumulate steps itself.
  ///
  /// The C interface passes bare function pointers with no user-data slot,
  /// so the callbacks are bridged through file-scope state. Only one solve
  /// can therefore be in flight at a time; solve() throws rather than
  /// corrupting a running one.
  namespace otr {
    /// The three callbacks the solver drives. Each returns false to signal
    /// failure, which aborts the solve with an exception.
    struct Callbacks {
      /// Apply the step, adopt it, and evaluate. grad and h_diag are
      /// n_param long and are written in place.
      std::function<bool(const double *kappa, double &func, double *grad,
                         double *h_diag)>
          update;
      /// Apply the step without adopting it, and evaluate the objective.
      std::function<bool(const double *kappa, double &func)> objective;
      /// Hessian-vector product at the current reference: hx = H x.
      std::function<bool(const double *x, double *hx)> hessian_vector;
      /// Optional extra convergence criterion, checked once per
      /// macroiteration: returning true stops the solve. Used to bail out
      /// when the parametrization itself has gone stale, which no
      /// gradient threshold can detect.
      std::function<bool(bool &stop)> converged;
      /// Optional preconditioner: out = (M - mu)^-1 r at level shift mu.
      /// Left empty, the library falls back to its own
      /// (level-shifted) absolute-diagonal preconditioner built from the
      /// h_diag `update` reported.
      std::function<bool(const double *r, double mu, double *out)> precondition;
    };

    /// Solver settings. The defaults here are OpenTrustRegion's own, except
    /// where noted; anything left at the library default is not touched.
    struct Settings {
      /// Convergence threshold on the RMS gradient
      double conv_tol = 1e-5;
      /// Initial trust radius
      double start_trust_radius = 0.4;
      /// Maximum number of macroiterations
      int n_macro = 150;
      /// Maximum number of microiterations
      int n_micro = 50;
      /// Number of random trial vectors seeding the microiterations
      int n_random_trial_vectors = 1;
      /// How far the microiterations must reduce the residual before the
      /// trust-region step counts as solved, far from and near a solution.
      ///
      /// These matter more than their obscurity suggests. A step whose
      /// subproblem did not meet this target is REJECTED outright, however
      /// good it is -- measured, a rejected step here lowered the energy by
      /// 5.3e-4 and landed on the minimum. On an ill-conditioned Hessian
      /// the target can be unreachable within n_micro, and the solve then
      /// throws away good steps until it stalls.
      double global_red_factor = 3e-1;
      double local_red_factor = 3e-2;
      /// Microiteration solver: "davidson", "jacobi-davidson" or "tcg"
      std::string subsystem_solver = "davidson";
      /// Line search after every macroiteration
      bool line_search = false;
      /// Stability check on convergence
      bool stability = false;
      /// Output detail, 0 silent
      int verbose = 3;
      /// Seed for the random trial vectors
      int seed = 42;
    };

    /// Was HelFEM built against OpenTrustRegion?
    bool available();

    /// Run the trust-region solver over n_param parameters. Throws if the
    /// library is absent, if a callback failed, or if the solver reported
    /// an error (including non-convergence).
    void solve(int n_param, const Callbacks &cb, const Settings &settings);
  } // namespace otr
} // namespace helfem

#endif
