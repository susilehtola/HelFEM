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
#include <Matrix.h>
#include <sstream>
#include <stdexcept>

namespace helfem {
  namespace dftgrid_common {
    /// Base class holding the shared XC state and libxc-facing
    /// plumbing for the three DFTGridWorker variants. Geometry-specific
    /// derived classes inherit and add their own basis-function buffers,
    /// density update, and Fxc reassembly.
    class DFTGridWorkerBase {
    protected:
      /// Total quadrature weights on the current element's grid (Npts)
      helfem::Vector wtot;

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
      helfem::Matrix rho;
      /// Energy density, Npts
      helfem::Vector exc;
      /// Functional derivative wrt density
      helfem::Matrix vxc;

      // GGA
      /// Dot products of density gradient
      helfem::Matrix sigma;
      /// Functional derivative wrt density gradient
      helfem::Matrix vsigma;

      // Meta-GGA
      /// Laplacian of density
      helfem::Matrix lapl;
      /// Kinetic energy density
      helfem::Matrix tau;
      /// Functional derivative wrt laplacian
      helfem::Matrix vlapl;
      /// Functional derivative wrt kinetic energy density
      helfem::Matrix vtau;

    public:
      DFTGridWorkerBase();
      virtual ~DFTGridWorkerBase();

      /// Check necessity of computing gradient / tau / laplacian for the
      /// given exchange + correlation functional ids.
      void check_grad_tau_lapl(int x_func, int c_func);
      /// Explicit override of the do_grad / do_tau / do_lapl flags
      void set_grad_tau_lapl(bool grad, bool tau, bool lapl);

      /// Initialise vxc / vsigma / vtau / vlapl buffers with the
      /// correct shape and zero exc.
      void init_xc();

      /// Evaluate int wtot * exc * rho_total
      double eval_Exc() const;

      /// Integrate the (total) electron density over the current
      /// element grid: int wtot * rho. Geometry-independent -- reads
      /// only the shared wtot / rho / polarized state.
      double compute_Nel() const;

      /// Compute libxc functional contribution and add to exc / vxc /
      /// vsigma / vtau / vlapl. pot=true also computes potentials.
      void compute_xc(int func_id, const helfem::Vector & params, double thr, bool pot = true);
    };

    /// BLAS routine for LDA-type quadrature: accumulate
    /// H += Re( (f .* vxc) * f^T ), i.e. the weighted outer product of
    /// the basis-function values f (Nbf x Npts) against themselves with
    /// per-point potential weights vxc (1 x Npts). Geometry-independent
    /// -- shared verbatim by all three DFTGridWorker variants (real f
    /// for sadatom, complex f for atomic/diatomic).
    template<typename T> void increment_lda(helfem::Matrix & H, const helfem::Vector & vxc,
                                             const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> & f) {
      if(f.cols() != vxc.size()) {
        std::ostringstream oss;
        oss << "Number of functions " << f.cols() << " and potential values " << vxc.size() << " do not match!\n";
        throw std::runtime_error(oss.str());
      }
      if(H.rows() != f.rows() || H.cols() != f.rows()) {
        std::ostringstream oss;
        oss << "Size of basis function (" << f.rows() << "," << f.cols() << ") and Fock matrix (" << H.rows() << "," << H.cols() << ") doesn't match!\n";
        throw std::runtime_error(oss.str());
      }

      // Weighted helper: fhlp(:,j) = f(:,j) * vxc(j). Then
      // H += Re( fhlp * f^H ). arma::trans on a complex matrix is the
      // conjugate transpose, so this is .adjoint() (== .transpose() for
      // the real, sadatom, instantiation).
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> fhlp = f;
      for(Eigen::Index j=0;j<fhlp.cols();j++)
        fhlp.col(j) *= vxc(j);
      H += (fhlp * f.adjoint()).real();
    }

    /// BLAS routine for the Laplacian part of meta-GGA quadrature:
    ///   H += f diag(w vlapl) l^dagger + l diag(w vlapl) f^dagger
    template<typename T> void increment_mgga_lapl(helfem::Matrix & H, const helfem::Vector & vlapl, const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> & f, const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> & l) {
      if(f.cols() != vlapl.size()) {
        std::ostringstream oss;
        oss << "Number of functions " << f.cols() << " and potential values " << vlapl.size() << " do not match!\n";
        throw std::runtime_error(oss.str());
      }
      if(H.rows() != f.rows() || H.cols() != f.rows()) {
        std::ostringstream oss;
        oss << "Size of basis function (" << f.rows() << "," << f.cols() << ") and Fock matrix (" << H.rows() << "," << H.cols() << ") doesn't match!\n";
        throw std::runtime_error(oss.str());
      }
      if(l.rows() != f.rows() || l.cols() != f.cols()) {
        std::ostringstream oss;
        oss << "Size of basis function (" << f.rows() << "," << f.cols() << ") and Laplacian matrix (" << l.rows() << "," << l.cols() << ") doesn't match!\n";
        throw std::runtime_error(oss.str());
      }

      // Form helper matrix
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> fhlp = f;
      for(Eigen::Index j=0;j<fhlp.cols();j++)
        fhlp.col(j) *= vlapl(j);
      H += (fhlp*l.adjoint() + l*fhlp.adjoint()).real();
    }
  }
}

#endif
