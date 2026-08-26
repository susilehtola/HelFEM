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

#ifndef SADATOM_DFTGRID_H
#define SADATOM_DFTGRID_H

#include "basis.h"
#include "../general/dftgrid_common.h"
#include <vector>

namespace helfem {
  namespace sadatom {
    namespace dftgrid {

      /// Worker class. Shares XC plumbing with the atomic and diatomic
      /// variants via helfem::dftgrid_common::DFTGridWorkerBase.
      class DFTGridWorker : public helfem::dftgrid_common::DFTGridWorkerBase {
      protected:
        /// Basis set
        const helfem::sadatom::basis::TwoDBasis *basp;

        /// Distance from nucleus
        helfem::Vector r;
        /// Radial quadrature weight
        helfem::Vector wrad;

        /// List of basis functions in element
        std::vector<Eigen::Index> bf_ind;
        /// Values of important functions in grid points, Nbf * Ngrid
        helfem::Matrix bf;
        /// Radial gradient
        helfem::Matrix bf_rho;
        /// Radial laplacian
        helfem::Matrix bf_rho2;

        /// Density helper matrices: P_{uv} chi_v, and P_{uv} nabla(chi_v)
        helfem::Matrix Pv, Pv_rho;
        /// Same for spin-polarized
        helfem::Matrix Pav, Pav_rho;
        helfem::Matrix Pbv, Pbv_rho;

        /// Gradient of electron density
        helfem::Matrix grho;
        /// Where the centrifugal term of tau survived the clamp in
        /// update_density, per spin channel and grid point (1.0 / 0.0).
        /// A response must differentiate max(term2, 0), whose derivative
        /// follows the REFERENCE sign, so the mask has to be recorded
        /// when tau is built rather than inferred from the perturbation.
        helfem::Matrix tau_centrifugal;

        // Members provided by helfem::dftgrid_common::DFTGridWorkerBase:
        //   wtot, exc, rho, sigma, vxc, vsigma, lapl, tau, vlapl, vtau
        //   polarized, do_grad, do_tau, do_lapl,
        //   do_gga, do_mgga_t, do_mgga_l

      public:
        /// Dummy constructor
        DFTGridWorker();
        /// Constructor
        DFTGridWorker(const helfem::sadatom::basis::TwoDBasis * basp);
        /// Destructor
        ~DFTGridWorker();

        // check_grad_tau_lapl / grad_tau_lapl / set_grad_tau_lapl
        // are inherited from DFTGridWorkerBase.

        /// Compute basis functions on grid points
        void compute_bf(size_t iel);

        /// Update values of density, restricted calculation. The per-l
        /// density cube is passed as one helfem::Matrix per l-slice.
        void update_density(const std::vector<helfem::Matrix> & P);
        /// Update values of density, unrestricted calculation
        void update_density(const std::vector<helfem::Matrix> & Pa, const std::vector<helfem::Matrix> & Pb);

        /// Evaluate the density of a *second* density matrix on the
        /// current element grid and return it in the rho layout, leaving
        /// the reference density -- and with it the kernel computed at
        /// that density -- untouched. This is how a response evaluation
        /// gets its perturbation density without a second pass over the
        /// element. Only the density itself is formed: the LDA kernel
        /// needs nothing else.
        helfem::Matrix eval_density(const std::vector<helfem::Matrix> & P) const;
        /// Same for a spin-polarized perturbation
        helfem::Matrix eval_density(const std::vector<helfem::Matrix> & Pa, const std::vector<helfem::Matrix> & Pb) const;
        /// Perturbed density, radial density gradient and kinetic energy
        /// density of one perturbation, as the same bilinear forms in the
        /// perturbed density matrix that update_density uses for the
        /// reference. Empty matrices come back for the channels the
        /// functional does not use.
        /// The reference density gradient, which the response kernel's
        /// gradient channel needs alongside the perturbed one.
        const helfem::Matrix & get_grho() const { return grho; }
        /// Debug: check the perturbed fields against central differences
        /// of the reference fields. Destroys the cached density.
        void check_response_fields(const helfem::Cube & P,
                                   const helfem::Cube & dP,
                                   const helfem::Matrix & drho,
                                   const helfem::Matrix & dgrho,
                                   const helfem::Matrix & dtau);
        /// Debug: spin-resolved perturbed fields vs central differences.
        void check_response_fields_spin(const helfem::Cube & Pa,
                                        const helfem::Cube & Pb,
                                        const helfem::Cube & dPa,
                                        const helfem::Cube & dPb,
                                        const helfem::Matrix & drho,
                                        const helfem::Matrix & dgrho,
                                        const helfem::Matrix & dtau);
        /// Spin-resolved perturbed fields, two rows per quantity.
        void eval_response_fields(const std::vector<helfem::Matrix> & Pa,
                                  const std::vector<helfem::Matrix> & Pb,
                                  helfem::Matrix & drho,
                                  helfem::Matrix & dgrho,
                                  helfem::Matrix & dtau) const;
        void eval_response_fields(const std::vector<helfem::Matrix> & P,
                                  helfem::Matrix & drho,
                                  helfem::Matrix & dgrho,
                                  helfem::Matrix & dtau) const;

        // compute_Nel() is inherited from DFTGridWorkerBase.
        // init_xc / compute_xc / eval_Exc / zero_Exc are inherited
        // from DFTGridWorkerBase.

        /// Assemble the gradient coefficient of the assembly from
        /// vsigma and the stored density gradient. Call after
        /// compute_xc and before eval_Fxc; the response path fills
        /// vgrad itself and must not call this.
        void build_vgrad() { DFTGridWorkerBase::build_vgrad(grho); }

        /// Evaluate Fock matrix, restricted calculation. One
        /// helfem::Matrix per l-slice.
        void eval_Fxc(std::vector<helfem::Matrix> & H) const;
        /// Evaluate Fock matrix, unrestricted calculation
        void eval_Fxc(std::vector<helfem::Matrix> & Ha, std::vector<helfem::Matrix> & Hb, bool beta=true) const;
      };

      /// Wrapper routine
      class DFTGrid {
      private:
        /// Pointer to basis set
        const helfem::sadatom::basis::TwoDBasis * basp;

      public:
        /// Fill the basis set's quadrature-value cache for the orders the
        /// given functionals need, so the grid loops become table lookups
        /// rather than a polynomial evaluation per element per Fock
        /// build. eval_Fxc calls this itself; it is public so that a
        /// caller about to evaluate several Fock matrices concurrently
        /// can fill the cache once, serially, first -- filling it from
        /// several threads at once would be a data race, reading it from
        /// several threads is fine.
        void prime_quadrature_cache(int x_func, int c_func) const;

        /// Dummy constructor
        DFTGrid();
        /// Constructor
        DFTGrid(const helfem::sadatom::basis::TwoDBasis * basp);
        /// Destructor
        ~DFTGrid();

        /// Compute Fock matrix, exchange-correlation energy and integrated
        /// electron density, restricted case. Functional parameters are
        /// Eigen-typed (helfem::Vector), matching the atomic/diatomic grids;
        /// the per-l density/Fock are helfem::Cube (a stack of radial
        /// matrices indexed by angular momentum l).
        void eval_Fxc(int x_func, const helfem::Vector & x_pars, int c_func, const helfem::Vector & c_pars, const helfem::Cube & P, helfem::Cube & H, double & Exc, double & Nel, double thr);
        /// Compute Fock matrix, exchange-correlation energy and integrated
        /// electron density, unrestricted case.
        void eval_Fxc(int x_func, const helfem::Vector & x_pars, int c_func, const helfem::Vector & c_pars, const helfem::Cube & Pa, const helfem::Cube & Pb, helfem::Cube & Ha, helfem::Cube & Hb, double & Exc, double & Nel, bool beta, double thr);

        /// Linear response of the XC matrix to each of the density
        /// perturbations dP, evaluated at the reference density P:
        /// dH[i] = f_xc . dP[i]. Exact for an LDA; for a GGA or meta-GGA
        /// it is the density-density part of the kernel only, which is
        /// what a model Hessian needs (see
        /// DFTGridWorkerBase::compute_fxc).
        ///
        /// The perturbations are passed as a batch because everything
        /// except the last contraction -- the basis values, the reference
        /// density, the libxc kernel evaluation -- is shared between
        /// them. Building an exact Hessian in a d-dimensional subspace
        /// therefore costs far less than d separate response builds.
        void eval_Fxc_response(int x_func, const helfem::Vector & x_pars, int c_func, const helfem::Vector & c_pars, const helfem::Cube & P, const std::vector<helfem::Cube> & dP, std::vector<helfem::Cube> & dH, double thr);
        /// Unrestricted counterpart; alpha and beta perturbations are
        /// batched together because the kernel mixes the spin channels.
        void eval_Fxc_response(int x_func, const helfem::Vector & x_pars, int c_func, const helfem::Vector & c_pars, const helfem::Cube & Pa, const helfem::Cube & Pb, const std::vector<helfem::Cube> & dPa, const std::vector<helfem::Cube> & dPb, std::vector<helfem::Cube> & dHa, std::vector<helfem::Cube> & dHb, double thr);

      };

      /// LDA quadrature accumulation is shared across geometries.
      using helfem::dftgrid_common::increment_lda;


      /// BLAS routine for mGGA-type quadrature
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
}

#endif
