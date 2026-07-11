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

        /// Density helper matrices: P_{uv} chi_v, and P_{uv} nabla(chi_v)
        Eigen::MatrixXcd Pv, Pv_rho, Pv_theta, Pv_phi;
        /// Same for spin-polarized
        Eigen::MatrixXcd Pav, Pav_rho, Pav_theta, Pav_phi;
        Eigen::MatrixXcd Pbv, Pbv_rho, Pbv_theta, Pbv_phi;

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

        // check_grad_tau_lapl / get_grad_tau_lapl / set_grad_tau_lapl
        // are inherited from DFTGridWorkerBase.

        /// Compute basis functions on grid points
        void compute_bf(size_t iel);

        /// Update values of density, restricted calculation
        void update_density(const helfem::Matrix & P);
        /// Update values of density, unrestricted calculation
        void update_density(const helfem::Matrix & Pa, const helfem::Matrix & Pb);

        // compute_Nel() is inherited from DFTGridWorkerBase.
        /// Compute integral over density laplacian
        double compute_laplsum() const;
        /// Compute kinetic energy
        double compute_Ekin() const;

        // init_xc / compute_xc / eval_Exc / zero_Exc are inherited
        // from DFTGridWorkerBase.

        /// Numerical clean up of xc

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

      };

      /// LDA quadrature accumulation is shared across geometries.
      using helfem::dftgrid_common::increment_lda;

      /// BLAS routine for GGA-type quadrature
      template<typename T> void increment_gga(helfem::Matrix & H, const helfem::Matrix & gn, const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> & f, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> f_x, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> f_y, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> f_z) {
        if(gn.cols()!=3) {
          throw std::runtime_error("Grad rho must have three columns!\n");
        }
        if(f.rows() != f_x.rows() || f.cols() != f_x.cols() || f.rows() != f_y.rows() || f.cols() != f_y.cols() || f.rows() != f_z.rows() || f.cols() != f_z.cols()) {
          throw std::runtime_error("Sizes of basis function and derivative matrices doesn't match!\n");
        }
        if(H.rows() != f.rows() || H.cols() != f.rows()) {
          throw std::runtime_error("Sizes of basis function and Fock matrices doesn't match!\n");
        }

        // Compute helper: gamma_{ip} = \sum_c \chi_{ip;c} gr_{p;c}
        //                 (N, Np)    =        (N Np; c)    (Np, 3)
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> gamma =
          Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(f.rows(), f.cols());

        // x gradient: scale column j of f_x by gn(j,0)
        for(Eigen::Index j=0;j<f_x.cols();j++)
          f_x.col(j) *= gn(j,0);
        gamma += f_x;

        // y gradient
        for(Eigen::Index j=0;j<f_y.cols();j++)
          f_y.col(j) *= gn(j,1);
        gamma += f_y;

        // z gradient
        for(Eigen::Index j=0;j<f_z.cols();j++)
          f_z.col(j) *= gn(j,2);
        gamma += f_z;

        // Form Fock matrix (arma::trans on complex is the conjugate
        // transpose -> .adjoint()).
        H += (gamma*f.adjoint() + f*gamma.adjoint()).real();
      }

      /// BLAS routine for meta-GGA-type quadrature
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
