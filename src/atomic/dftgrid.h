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
        arma::vec cth, phi, wang;

        /// Scale factors
        arma::rowvec scale_r, scale_theta, scale_phi;
        /// Pre-computed 1 / scale^2 used by the kinetic / mGGA terms.
        /// Filled together with scale_*; cuts arma::square + division out
        /// of every Fxc evaluation.
        arma::rowvec inv_scale_r2, inv_scale_theta2, inv_scale_phi2;

        /// List of basis functions in element
        arma::uvec bf_ind;
        /// Values of important functions in grid points, Nbf * Ngrid
        arma::cx_mat bf;
        /// Radial gradient
        arma::cx_mat bf_rho;
        /// Theta gradient
        arma::cx_mat bf_theta;
        /// Phi gradient
        arma::cx_mat bf_phi;
        /// Values of laplacians in grid points, (3*Nbf) * Ngrid
        arma::cx_mat bf_lapl;

        /// Density helper matrices: P_{uv} chi_v, and P_{uv} nabla(chi_v)
        arma::cx_mat Pv, Pv_rho, Pv_theta, Pv_phi;
        /// Same for spin-polarized
        arma::cx_mat Pav, Pav_rho, Pav_theta, Pav_phi;
        arma::cx_mat Pbv, Pbv_rho, Pbv_theta, Pbv_phi;

        /// Gradient of electron density, (3 x Nrho) x Npts (atomic-only:
        /// diatomic keeps its own decomposition; sadatom uses cube layout)
        arma::mat grho;

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
        void update_density(const arma::mat & P);
        /// Update values of density, unrestricted calculation
        void update_density(const arma::mat & Pa, const arma::mat & Pb);

        /// Compute number of electrons
        double compute_Nel() const;
        /// Compute integral over density laplacian
        double compute_laplsum() const;
        /// Compute kinetic energy
        double compute_Ekin() const;

        // init_xc / compute_xc / eval_Exc / zero_Exc are inherited
        // from DFTGridWorkerBase.

        /// Numerical clean up of xc

        /// Evaluate Fock matrix, restricted calculation
        void eval_Fxc(arma::mat & H) const;
        /// Evaluate Fock matrix, unrestricted calculation
        void eval_Fxc(arma::mat & Ha, arma::mat & Hb, bool beta=true) const;
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

      /// BLAS routine for LDA-type quadrature
      template<typename T> void increment_lda(arma::mat & H, const arma::rowvec & vxc, const arma::Mat<T> & f) {
        if(f.n_cols != vxc.n_elem) {
          std::ostringstream oss;
          oss << "Number of functions " << f.n_cols << " and potential values " << vxc.n_elem << " do not match!\n";
          throw std::runtime_error(oss.str());
        }
        if(H.n_rows != f.n_rows || H.n_cols != f.n_rows) {
          std::ostringstream oss;
          oss << "Size of basis function (" << f.n_rows << "," << f.n_cols << ") and Fock matrix (" << H.n_rows << "," << H.n_cols << ") doesn't match!\n";
          throw std::runtime_error(oss.str());
        }

        // Form helper matrix
        arma::Mat<T> fhlp(f);
        for(size_t i=0;i<fhlp.n_rows;i++)
          for(size_t j=0;j<fhlp.n_cols;j++)
            fhlp(i,j)*=vxc(j);
        H+=arma::real(fhlp*arma::trans(f));
      }

      /// BLAS routine for GGA-type quadrature
      template<typename T> void increment_gga(arma::mat & H, const arma::mat & gn, const arma::Mat<T> & f, arma::Mat<T> f_x, arma::Mat<T> f_y, arma::Mat<T> f_z) {
        if(gn.n_cols!=3) {
          throw std::runtime_error("Grad rho must have three columns!\n");
        }
        if(f.n_rows != f_x.n_rows || f.n_cols != f_x.n_cols || f.n_rows != f_y.n_rows || f.n_cols != f_y.n_cols || f.n_rows != f_z.n_rows || f.n_cols != f_z.n_cols) {
          throw std::runtime_error("Sizes of basis function and derivative matrices doesn't match!\n");
        }
        if(H.n_rows != f.n_rows || H.n_cols != f.n_rows) {
          throw std::runtime_error("Sizes of basis function and Fock matrices doesn't match!\n");
        }

        // Compute helper: gamma_{ip} = \sum_c \chi_{ip;c} gr_{p;c}
        //                 (N, Np)    =        (N Np; c)    (Np, 3)
        arma::Mat<T> gamma(f.n_rows,f.n_cols);
        gamma.zeros();
        {
          // Helper
          arma::rowvec gc;

          // x gradient
          gc=arma::strans(gn.col(0));
          for(size_t j=0;j<f_x.n_cols;j++)
            for(size_t i=0;i<f_x.n_rows;i++)
              f_x(i,j)*=gc(j);
          gamma+=f_x;

          // x gradient
          gc=arma::strans(gn.col(1));
          for(size_t j=0;j<f_y.n_cols;j++)
            for(size_t i=0;i<f_y.n_rows;i++)
              f_y(i,j)*=gc(j);
          gamma+=f_y;

          // z gradient
          gc=arma::strans(gn.col(2));
          for(size_t j=0;j<f_z.n_cols;j++)
            for(size_t i=0;i<f_z.n_rows;i++)
              f_z(i,j)*=gc(j);
          gamma+=f_z;
        }

        // Form Fock matrix
        H+=arma::real(gamma*arma::trans(f) + f*arma::trans(gamma));
      }

      /// BLAS routine for meta-GGA-type quadrature
      template<typename T> void increment_mgga_lapl(arma::mat & H, const arma::rowvec & vlapl, const arma::Mat<T> & f, const arma::Mat<T> & l) {
        if(f.n_cols != vlapl.n_elem) {
          std::ostringstream oss;
          oss << "Number of functions " << f.n_cols << " and potential values " << vlapl.n_elem << " do not match!\n";
          throw std::runtime_error(oss.str());
        }
        if(H.n_rows != f.n_rows || H.n_cols != f.n_rows) {
          std::ostringstream oss;
          oss << "Size of basis function (" << f.n_rows << "," << f.n_cols << ") and Fock matrix (" << H.n_rows << "," << H.n_cols << ") doesn't match!\n";
          throw std::runtime_error(oss.str());
        }
        if(l.n_rows != f.n_rows || l.n_cols != f.n_cols) {
          std::ostringstream oss;
          oss << "Size of basis function (" << f.n_rows << "," << f.n_cols << ") and Laplacian matrix (" << l.n_rows << "," << l.n_cols << ") doesn't match!\n";
          throw std::runtime_error(oss.str());
        }

        // Form helper matrix
        arma::Mat<T> fhlp(f);
        for(size_t i=0;i<fhlp.n_rows;i++)
          for(size_t j=0;j<fhlp.n_cols;j++)
            fhlp(i,j)*=vlapl(j);
        H+=arma::real(fhlp*arma::trans(l)+l*arma::trans(fhlp));
      }
    }
  }
}

#endif
