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

#ifndef ATOMIC_DFTGRID_H
#define ATOMIC_DFTGRID_H

#include "basis.h"
#include "../general/dftgrid_common.h"

namespace helfem {
  namespace atomic {
    namespace dftgrid {

      /// Worker class. Shares XC plumbing (init_xc, compute_xc,
      /// check_grad_tau_lapl, eval_Exc, zero_Exc, grad/tau/lapl flag
      /// storage, and the LDA/GGA/mGGA buffers rho/exc/vxc/sigma/etc.)
      /// with the sadatom and diatomic variants via
      /// helfem::dftgrid_common::DFTGridWorkerBase.
      class DFTGridWorker : public helfem::dftgrid_common::DFTGridWorkerBase {
      protected:
        /// Basis set
        const helfem::atomic::basis::TwoDBasis *basp;

        /// Angular grid
        helfem::Vector cth, phi, wang;

        /// Scale factors
        helfem::Vector scale_r, scale_theta, scale_phi;
        /// Pre-computed 1 / scale^2 used by the kinetic / mGGA terms.
        /// Filled together with scale_*; cuts the square + division out
        /// of every Fxc evaluation.
        helfem::Vector inv_scale_r2, inv_scale_theta2, inv_scale_phi2;

        /// List of basis functions in element
        std::vector<Eigen::Index> bf_ind;
        /// Values of important functions in grid points, Nbf * Ngrid
        Eigen::MatrixXcd bf;
        /// Radial gradient
        Eigen::MatrixXcd bf_rho;
        /// Theta gradient
        Eigen::MatrixXcd bf_theta;
        /// Phi gradient
        Eigen::MatrixXcd bf_phi;
        /// Values of laplacians in grid points, (3*Nbf) * Ngrid
        Eigen::MatrixXcd bf_lapl;

        /// Real and imaginary parts of the basis values above, split once
        /// per element in compute_bf.
        ///
        /// The density matrix is REAL while the basis is complex, so the
        /// natural-looking P.cast<complex>() * bf.conjugate() runs a full
        /// complex GEMM against an operand whose imaginary half is
        /// identically zero -- four real products where two suffice. It is
        /// where most of this driver's time goes: 72% of an oxygen run sat
        /// in zgemm.
        ///
        /// Writing bf = a + i b, and noting P is symmetric so
        /// Pv = P conj(bf) = P a - i P b, every contraction the density and
        /// its gradients need has the form
        ///     Re( sum_u Pv(u,ip) Y(u,ip) ) = (P a).Re(Y) + (P b).Im(Y),
        /// i.e. purely real throughout. Splitting here rather than at the
        /// point of use matters: extracting .real()/.imag() inside the
        /// density loop costs about what the halved flops save.
        helfem::Matrix bf_re, bf_im;
        helfem::Matrix bf_rho_re, bf_rho_im;
        helfem::Matrix bf_theta_re, bf_theta_im;
        helfem::Matrix bf_phi_re, bf_phi_im;
        helfem::Matrix bf_lapl_re, bf_lapl_im;

        /// P*Re(X) and P*Im(X) for X = bf and its gradients -- the real
        /// stand-ins for the complex Pv/Pv_rho/... helpers.
        helfem::Matrix PvA, PvB, PvA_rho, PvB_rho, PvA_theta, PvB_theta, PvA_phi, PvB_phi;
        /// Same for spin-polarized
        helfem::Matrix PavA, PavB, PavA_rho, PavB_rho, PavA_theta, PavB_theta, PavA_phi, PavB_phi;
        helfem::Matrix PbvA, PbvB, PbvA_rho, PbvB_rho, PbvA_theta, PbvB_theta, PbvA_phi, PbvB_phi;

        /// Gradient of electron density, (3 x Nrho) x Npts (atomic-only:
        /// diatomic keeps its own decomposition; sadatom uses cube layout)
        helfem::Matrix grho;

        // The following members are provided by
        // helfem::dftgrid_common::DFTGridWorkerBase and used by the
        // shared XC plumbing:
        //   wtot, exc, rho, sigma, vxc, vsigma, lapl, tau, vlapl, vtau
        //   polarized, do_grad, do_tau, do_lapl,
        //   do_gga, do_mgga_t, do_mgga_l

      public:
        /// Dummy constructor
        DFTGridWorker();
        /// Constructor
        DFTGridWorker(const helfem::atomic::basis::TwoDBasis * basp, int lang, int mang);
        /// Destructor
        ~DFTGridWorker();

        // check_grad_tau_lapl / grad_tau_lapl / set_grad_tau_lapl
        // are inherited from DFTGridWorkerBase.

        /// Compute basis functions on grid points
        void compute_bf(size_t iel);

        /// Update values of density, restricted calculation
        void update_density(const helfem::Matrix & P);
        /// Update values of density, unrestricted calculation
        void update_density(const helfem::Matrix & Pa, const helfem::Matrix & Pb);

        /// Density of a SECOND density matrix on the current element's
        /// grid, without disturbing the reference the worker is holding.
        /// This is the same bilinear form update_density uses, applied to
        /// the perturbation, and it is all an LDA response kernel needs.
        /// 1 x Npts restricted, 2 x Npts polarized.
        helfem::Matrix eval_density(const helfem::Matrix & dP) const;
        /// Same for a spin-polarized perturbation
        helfem::Matrix eval_density(const helfem::Matrix & dPa,
                                     const helfem::Matrix & dPb) const;

        // compute_Nel() is inherited from DFTGridWorkerBase.
        /// Compute integral over density laplacian
        double compute_laplsum() const;
        /// Compute kinetic energy
        double compute_Ekin() const;

        // init_xc / compute_xc / eval_Exc / zero_Exc are inherited
        // from DFTGridWorkerBase.

        /// Numerical clean up of xc

        /// Assemble the gradient coefficient of the assembly from
        /// vsigma and the stored density gradient. Call after
        /// compute_xc and before eval_Fxc; the response path fills
        /// vgrad itself and must not call this.
        void build_vgrad() { DFTGridWorkerBase::build_vgrad(grho); }

        /// Evaluate Fock matrix, restricted calculation
        void eval_Fxc(helfem::Matrix & H) const;
        /// Evaluate Fock matrix, unrestricted calculation
        void eval_Fxc(helfem::Matrix & Ha, helfem::Matrix & Hb, bool beta=true) const;
      };

      /// Wrapper routine
      class DFTGrid {
      private:
        /// Pointer to basis set
        const helfem::atomic::basis::TwoDBasis * basp;
        /// Angular rule
        int lang, mang;

      public:
        /// Dummy constructor
        DFTGrid();
        /// Constructor
        DFTGrid(const helfem::atomic::basis::TwoDBasis * basp, int lang, int mang);
        /// Destructor
        ~DFTGrid();

        /// Compute Fock matrix, exchange-correlation energy and integrated
        /// electron density, restricted case. Eigen-typed public boundary
        /// (functional parameters, density, and Fock matrix); the quadrature
        /// interior stays arma-native with a single bridge at entry/exit.
        void eval_Fxc(int x_func, const helfem::Vector & x_pars, int c_func, const helfem::Vector & c_pars, const helfem::Matrix & P, helfem::Matrix & H, double & Exc, double & Nel, double & Ekin, double thr);
        /// Compute Fock matrix, exchange-correlation energy and integrated
        /// electron density, unrestricted case. Eigen-typed public boundary.
        void eval_Fxc(int x_func, const helfem::Vector & x_pars, int c_func, const helfem::Vector & c_pars, const helfem::Matrix & Pa, const helfem::Matrix & Pb, helfem::Matrix & Ha, helfem::Matrix & Hb, double & Exc, double & Nel, double & Ekin, bool beta, double thr);

        /// Linear response of the XC matrix to a BATCH of density
        /// perturbations dP, at the reference density P.
        ///
        /// Batched because the expensive parts -- the basis values on the
        /// grid, the reference density and the libxc kernel -- are shared
        /// across the perturbations, so a d-dimensional Hessian subspace
        /// costs far less than d separate response builds.
        ///
        /// LDA-shaped: only the density-density block of the kernel is
        /// used, which is exact for an LDA and an approximation beyond
        /// it. That is sound here because this builds the model Hessian
        /// of a trust-region method, never the energy or the gradient --
        /// an approximate kernel costs iterations, not correctness, since
        /// the step is validated against the true energy by the ratio
        /// test. The seam for the gradient and tau channels is
        /// DFTGridWorkerBase::set_response_potential, which already takes
        /// them; passing empty matrices selects the LDA-shaped path.
        void eval_Fxc_response(int x_func, const helfem::Vector & x_pars,
                                int c_func, const helfem::Vector & c_pars,
                                const helfem::Matrix & P,
                                const std::vector<helfem::Matrix> & dP,
                                std::vector<helfem::Matrix> & dH, double thr);
        /// Spin-polarized counterpart
        void eval_Fxc_response(int x_func, const helfem::Vector & x_pars,
                                int c_func, const helfem::Vector & c_pars,
                                const helfem::Matrix & Pa, const helfem::Matrix & Pb,
                                const std::vector<helfem::Matrix> & dPa,
                                const std::vector<helfem::Matrix> & dPb,
                                std::vector<helfem::Matrix> & dHa,
                                std::vector<helfem::Matrix> & dHb, double thr);

      };

      /// LDA quadrature accumulation is shared across geometries.
      using helfem::dftgrid_common::increment_lda;
      /// GGA accumulation is shared across geometries too.
      using helfem::dftgrid_common::increment_gga_split;


    }
  }
}

#endif
