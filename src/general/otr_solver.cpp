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
#include "otr_solver.h"

#include <cstring>
#include <sstream>
#include <stdexcept>

#ifdef HELFEM_HAVE_OPENTRUSTREGION
#include <opentrustregion.h>

extern "C" {
/// The library records the integer width it was compiled with in this
/// global. The C header picks c_int from USE_ILP64 on OUR side, so a
/// mismatch silently changes the layout of the settings struct; the
/// symbol lets us refuse instead.
extern bool ilp64;
}
#endif

namespace helfem {
  namespace otr {

#ifdef HELFEM_HAVE_OPENTRUSTREGION
    namespace {
      /// The C interface passes bare function pointers with no user-data
      /// slot, so the only way to reach the caller's closures is
      /// file-scope state. `active` guards against a second, nested solve
      /// silently stealing it.
      const Callbacks *callbacks = nullptr;
      int nparam = 0;
      bool active = false;
      /// Set when a callback throws or reports failure. The C boundary is
      /// not exception-safe, so the failure is recorded, the library is
      /// told to stop, and the exception is re-raised in solve().
      std::string failure;

      /// Report a callback failure to the library. The error codes < 100
      /// are the range the library reserves for callback errors.
      int fail(const char *what) {
        if (failure.empty())
          failure = what;
        return 1;
      }

      c_int hess_x_bridge(const c_real *x, c_real *hx) {
        try {
          if (!callbacks->hessian_vector(x, hx))
            return fail("the Hessian-vector product failed");
        } catch (std::exception &e) {
          return fail(e.what());
        }
        return 0;
      }

      c_int update_orbs_bridge(const c_real *kappa, c_real *func, c_real *grad,
                               c_real *h_diag, hess_x_fp *hess_x_ptr) {
        *hess_x_ptr = hess_x_bridge;
        try {
          double f = 0.0;
          if (!callbacks->update(kappa, f, grad, h_diag))
            return fail("the orbital update failed");
          *func = f;
        } catch (std::exception &e) {
          return fail(e.what());
        }
        return 0;
      }

      c_int precond_bridge(const c_real *residual, const c_real *mu,
                           c_real *out) {
        try {
          if (!callbacks->precondition(residual, *mu, out))
            return fail("the preconditioner failed");
        } catch (std::exception &e) {
          return fail(e.what());
        }
        return 0;
      }

      c_int conv_check_bridge(c_bool *converged) {
        try {
          bool stop = false;
          if (!callbacks->converged(stop))
            return fail("the convergence check failed");
          *converged = stop;
        } catch (std::exception &e) {
          return fail(e.what());
        }
        return 0;
      }

      c_int obj_func_bridge(const c_real *kappa, c_real *func) {
        try {
          double f = 0.0;
          if (!callbacks->objective(kappa, f))
            return fail("the objective function evaluation failed");
          *func = f;
        } catch (std::exception &e) {
          return fail(e.what());
        }
        return 0;
      }

      /// Decode the library's OOEE error code: OO names the component that
      /// reported the failure, EE the failure itself.
      std::string describe_error(int error) {
        static const struct {
          int origin;
          const char *name;
        } origins[] = {{1, "solver"},      {2, "stability check"},
                       {11, "obj_func"},   {12, "update_orbs"},
                       {13, "hess_x"},     {14, "precond"},
                       {15, "conv_check"}, {16, "project"}};

        std::ostringstream oss;
        oss << "OpenTrustRegion failed with error code " << error;
        for (const auto &o : origins)
          if (error / 100 == o.origin) {
            oss << " (reported by " << o.name << ")";
            break;
          }
        return oss.str();
      }

      /// Scoped reset of the file-scope bridge state, so an exception
      /// thrown out of solve() cannot leave the slot claimed.
      struct Guard {
        ~Guard() {
          callbacks = nullptr;
          nparam = 0;
          active = false;
        }
      };
    } // namespace

    bool available() { return true; }

    void solve(int n_param, const Callbacks &cb, const Settings &settings) {
      if (ilp64)
        throw std::logic_error(
            "OpenTrustRegion was built with 64-bit integers, but HelFEM "
            "compiles its header in the 32-bit (LP64) layout. Rebuild the "
            "library with -DINTEGER_SIZE=4.\n");
      if (n_param <= 0)
        throw std::logic_error("OpenTrustRegion: no parameters to optimize.\n");
      if (active)
        throw std::logic_error(
            "OpenTrustRegion: a solve is already in progress. The C "
            "interface has no user-data slot, so solves cannot nest.\n");
      if (!cb.update || !cb.objective || !cb.hessian_vector)
        throw std::logic_error("OpenTrustRegion: incomplete callback set.\n");
      if (settings.subsystem_solver.size() > OTR_KW_LEN)
        throw std::logic_error("OpenTrustRegion: subsystem solver name too "
                               "long.\n");

      Guard guard;
      callbacks = &cb;
      nparam = n_param;
      active = true;
      failure.clear();

      solver_settings_type s = solver_settings_init();
      s.conv_tol = settings.conv_tol;
      s.start_trust_radius = settings.start_trust_radius;
      s.n_macro = settings.n_macro;
      s.n_micro = settings.n_micro;
      s.n_random_trial_vectors = settings.n_random_trial_vectors;
      s.global_red_factor = settings.global_red_factor;
      s.local_red_factor = settings.local_red_factor;
      s.line_search = settings.line_search;
      s.stability = settings.stability;
      s.verbose = settings.verbose;
      s.seed = settings.seed;
      if (cb.precondition)
        s.precond = precond_bridge;
      if (cb.converged)
        s.conv_check = conv_check_bridge;
      std::memset(s.subsystem_solver, 0, sizeof(s.subsystem_solver));
      std::strncpy(s.subsystem_solver, settings.subsystem_solver.c_str(),
                   OTR_KW_LEN);

      const c_int error = ::solver(update_orbs_bridge, obj_func_bridge,
                                   (c_int)n_param, s);

      // A callback failure is reported through the library, which unwinds
      // and returns an error code; the real message is the one we recorded.
      if (!failure.empty())
        throw std::runtime_error("OpenTrustRegion aborted: " + failure + "\n");
      if (error != 0)
        throw std::runtime_error(describe_error((int)error) + "\n");
    }

#else

    bool available() { return false; }

    void solve(int, const Callbacks &, const Settings &) {
      throw std::logic_error(
          "This HelFEM was built without OpenTrustRegion, so the "
          "second-order optimizer is unavailable. Configure with "
          "-DHELFEM_OPENTRUSTREGION=ON (needs a Fortran compiler and an "
          "LP64 BLAS/LAPACK).\n");
    }

#endif

  } // namespace otr
} // namespace helfem
