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
#ifndef HELFEM_DFTGRID_COMMON_H
#define HELFEM_DFTGRID_COMMON_H

// Shared DFT-grid worker plumbing: libxc dispatch, xc buffer
// allocation, gradient/tau/lapl need detection, energy accumulation.
// Extracted from three near-identical copies in src/atomic/dftgrid.cpp,
// src/sadatom/dftgrid.cpp, src/diatomic/dftgrid.cpp. The
// geometry-specific bits (compute_bf, update_density, eval_Fxc,
// compute_Nel, etc.) stay in the derived classes.

#include <armadillo>

namespace helfem {
  namespace dftgrid_common {
    /// Base class holding the shared XC state and libxc-facing
    /// plumbing for the three DFTGridWorker variants. Geometry-specific
    /// derived classes inherit and add their own basis-function buffers,
    /// density update, and Fxc reassembly.
    class DFTGridWorkerBase {
    protected:
      /// Total quadrature weights on the current element's grid
      arma::rowvec wtot;

      /// Is gradient needed?
      bool do_grad;
      /// Is kinetic energy density needed?
      bool do_tau;
      /// Is laplacian needed?
      bool do_lapl;
      /// Spin-polarized calculation?
      bool polarized;

      /// GGA functional used? (Set in compute_xc, only affects eval_Fxc)
      bool do_gga;
      /// Meta-GGA tau used? (Set in compute_xc, only affects eval_Fxc)
      bool do_mgga_t;
      /// Meta-GGA lapl used? (Set in compute_xc, only affects eval_Fxc)
      bool do_mgga_l;

      // LDA
      /// Density, Nrho x Npts
      arma::mat rho;
      /// Energy density, Npts
      arma::rowvec exc;
      /// Functional derivative wrt density
      arma::mat vxc;

      // GGA
      /// Dot products of density gradient
      arma::mat sigma;
      /// Functional derivative wrt density gradient
      arma::mat vsigma;

      // Meta-GGA
      /// Laplacian of density
      arma::mat lapl;
      /// Kinetic energy density
      arma::mat tau;
      /// Functional derivative wrt laplacian
      arma::mat vlapl;
      /// Functional derivative wrt kinetic energy density
      arma::mat vtau;

    public:
      DFTGridWorkerBase();
      virtual ~DFTGridWorkerBase();

      /// Check necessity of computing gradient / tau / laplacian for the
      /// given exchange + correlation functional ids.
      void check_grad_tau_lapl(int x_func, int c_func);
      /// Query the do_grad / do_tau / do_lapl flags
      /// Explicit override of the do_grad / do_tau / do_lapl flags
      void set_grad_tau_lapl(bool grad, bool tau, bool lapl);

      /// Zero the exchange-correlation energy density buffer
      void zero_Exc();

      /// Initialise vxc / vsigma / vtau / vlapl buffers with the
      /// correct shape and zero exc.
      void init_xc();

      /// Evaluate int wtot * exc * rho_total
      double eval_Exc() const;

      /// Compute libxc functional contribution and add to exc / vxc /
      /// vsigma / vtau / vlapl. pot=true also computes potentials.
      void compute_xc(int func_id, const arma::vec & params, double thr, bool pot = true);
    };
  }
}

#endif
