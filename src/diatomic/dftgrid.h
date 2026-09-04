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

#ifndef DFTGRID
#define DFTGRID

#include "basis.h"
#include "../general/dftgrid_common.h"
#include <complex>
#include <vector>

namespace helfem {
  namespace diatomic {
    namespace dftgrid {

      /// Worker class. Shares XC plumbing with the atomic and sadatom
      /// variants via helfem::dftgrid_common::DFTGridWorkerBase.
      class DFTGridWorker : public helfem::dftgrid_common::DFTGridWorkerBase {
      protected:
        /// Basis set
        const helfem::diatomic::basis::TwoDBasis *basp;

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

        /// Density helper matrices: P_{uv} chi_v, and P_{uv} nabla(chi_v)
        Eigen::MatrixXcd Pv, Pv_rho, Pv_theta, Pv_phi;

        /// Real and imaginary parts of the basis values above, split once
        /// per grid point in compute_bf. See the atomic worker for the
        /// algebra: the density matrix is real, so every density and Fock
        /// contraction reduces to real products and the complex forms are
        /// pure overhead.
        helfem::Matrix bf_re, bf_im;
        helfem::Matrix bf_rho_re, bf_rho_im;
        helfem::Matrix bf_theta_re, bf_theta_im;
        helfem::Matrix bf_phi_re, bf_phi_im;
        helfem::Matrix bf_lapl_re, bf_lapl_im;
        /// P*Re(X) and P*Im(X), the real stand-ins for Pv and friends.
        helfem::Matrix PvA, PvB, PvA_rho, PvB_rho, PvA_theta, PvB_theta, PvA_phi, PvB_phi;
        helfem::Matrix PavA, PavB, PavA_rho, PavB_rho, PavA_theta, PavB_theta, PavA_phi, PavB_phi;
        helfem::Matrix PbvA, PbvB, PbvA_rho, PbvB_rho, PbvA_theta, PbvB_theta, PbvA_phi, PbvB_phi;
        /// Same for spin-polarized
        Eigen::MatrixXcd Pav, Pav_rho, Pav_theta, Pav_phi;
        Eigen::MatrixXcd Pbv, Pbv_rho, Pbv_theta, Pbv_phi;

        /// Gradient of electron density
        helfem::Matrix grho;

        // Members provided by helfem::dftgrid_common::DFTGridWorkerBase:
        //   wtot, exc, rho, sigma, vxc, vsigma, lapl, tau, vlapl, vtau
        //   polarized, do_grad, do_tau, do_lapl,
        //   do_gga, do_mgga_t, do_mgga_l

      public:
        /// Dummy constructor
        DFTGridWorker();
        /// Constructor
        DFTGridWorker(const helfem::diatomic::basis::TwoDBasis * basp, int lang, int mang);
        /// Destructor
        ~DFTGridWorker();

        // check_grad_tau_lapl / grad_tau_lapl / set_grad_tau_lapl
        // are inherited from DFTGridWorkerBase.

        /// Compute basis functions on grid points
        void compute_bf(size_t iel, size_t irad);

        /// Update values of density, restricted calculation
        void update_density(const helfem::Matrix & Pexp);
        /// Update values of density, unrestricted calculation
        void update_density(const helfem::Matrix & Paexp, const helfem::Matrix & Pbexp);

        /// Density of a SECOND density matrix on the current grid point,
        /// without disturbing the reference the worker is holding. The
        /// same bilinear form update_density uses, applied to the
        /// perturbation, which is all an LDA response kernel needs.
        /// dP is expanded to the dummy basis, as update_density's is.
        helfem::Matrix eval_density(const helfem::Matrix & dPexp) const;
        /// Same for a spin-polarized perturbation
        helfem::Matrix eval_density(const helfem::Matrix & dPaexp,
                                     const helfem::Matrix & dPbexp) const;

        /// The reference density gradient, which the response kernel's
        /// gradient channel needs alongside the perturbed one.
        /// (3 x Nrho) x Npts, the layout update_density fills.
        const helfem::Matrix & get_grho() const { return grho; }

        /// Perturbed density, density gradient and kinetic energy density
        /// of one perturbation: the SAME bilinear forms in the density
        /// matrix that update_density evaluates for the reference, with
        /// the perturbed matrix substituted. Keeping the two in step is
        /// what makes the response kernel exact, so this mirrors that
        /// code deliberately -- the metric factors in particular.
        /// drho is 1 x Npts, dgrho 3 x Npts, dtau 1 x Npts; empty
        /// matrices come back for the channels the functional does not
        /// use.
        void eval_response_fields(const helfem::Matrix & dPexp,
                                  helfem::Matrix & drho,
                                  helfem::Matrix & dgrho,
                                  helfem::Matrix & dtau) const;
        /// Spin-resolved perturbed fields: 2, 6 and 2 rows.
        void eval_response_fields(const helfem::Matrix & dPaexp,
                                  const helfem::Matrix & dPbexp,
                                  helfem::Matrix & drho,
                                  helfem::Matrix & dgrho,
                                  helfem::Matrix & dtau) const;
        /// Debug: check the perturbed fields against central differences
        /// of the reference fields. Destroys the cached density.
        void check_response_fields(const helfem::Matrix & Pexp,
                                   const helfem::Matrix & dPexp,
                                   const helfem::Matrix & drho,
                                   const helfem::Matrix & dgrho,
                                   const helfem::Matrix & dtau);
        /// Debug: spin-resolved perturbed fields vs central differences.
        void check_response_fields(const helfem::Matrix & Paexp,
                                   const helfem::Matrix & Pbexp,
                                   const helfem::Matrix & dPaexp,
                                   const helfem::Matrix & dPbexp,
                                   const helfem::Matrix & drho,
                                   const helfem::Matrix & dgrho,
                                   const helfem::Matrix & dtau);

        // compute_Nel() is inherited from DFTGridWorkerBase.
        /// Compute kinetic energy
        double compute_Ekin() const;

        // init_xc / compute_xc / eval_Exc / zero_Exc are inherited
        // from DFTGridWorkerBase.

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
        const helfem::diatomic::basis::TwoDBasis * basp;
        /// Angular rule
        int lang, mang;

      public:
        /// Dummy constructor
        DFTGrid();
        /// Constructor
        DFTGrid(const helfem::diatomic::basis::TwoDBasis * basp, int lang, int mang);
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
        /// perturbations, at the reference density P. Batched because the
        /// basis values, the reference density and the libxc kernel are
        /// shared across the perturbations.
        ///
        /// The kernel is the functional's own: the density-density
        /// block for every rung, the gradient channel for a GGA and the
        /// tau channel for a tau-meta-GGA. Only the LAPLACIAN channel is
        /// still missing, so a laplacian-dependent meta-GGA still gets an
        /// approximate Hessian -- sound in a trust-region method, where
        /// the step is validated against the true energy by the ratio
        /// test, so an approximate model costs iterations, not
        /// correctness.
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
