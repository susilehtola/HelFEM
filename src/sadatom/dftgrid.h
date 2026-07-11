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

        // check_grad_tau_lapl / get_grad_tau_lapl / set_grad_tau_lapl
        // are inherited from DFTGridWorkerBase.

        /// Compute basis functions on grid points
        void compute_bf(size_t iel);

        /// Update values of density, restricted calculation. The per-l
        /// density cube is passed as one helfem::Matrix per l-slice.
        void update_density(const std::vector<helfem::Matrix> & P);
        /// Update values of density, unrestricted calculation
        void update_density(const std::vector<helfem::Matrix> & Pa, const std::vector<helfem::Matrix> & Pb);

        // compute_Nel() is inherited from DFTGridWorkerBase.

        // init_xc / compute_xc / eval_Exc / zero_Exc are inherited
        // from DFTGridWorkerBase.


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
        /// Dummy constructor
        DFTGrid();
        /// Constructor
        DFTGrid(const helfem::sadatom::basis::TwoDBasis * basp);
        /// Destructor
        ~DFTGrid();

        /// Compute Fock matrix, exchange-correlation energy and integrated
        /// electron density, restricted case. Functional parameters are
        /// Eigen-typed (helfem::Vector), matching the atomic/diatomic grids;
        /// the per-l density/Fock stay arma::cube (no helfem cube type).
        void eval_Fxc(int x_func, const helfem::Vector & x_pars, int c_func, const helfem::Vector & c_pars, const arma::cube & P, arma::cube & H, double & Exc, double & Nel, double thr);
        /// Compute Fock matrix, exchange-correlation energy and integrated
        /// electron density, unrestricted case.
        void eval_Fxc(int x_func, const helfem::Vector & x_pars, int c_func, const helfem::Vector & c_pars, const arma::cube & Pa, const arma::cube & Pb, arma::cube & Ha, arma::cube & Hb, double & Exc, double & Nel, bool beta, double thr);

      };

      /// LDA quadrature accumulation is shared across geometries.
      using helfem::dftgrid_common::increment_lda;

      /// BLAS routine for GGA-type quadrature
      template<typename T> void increment_gga(helfem::Matrix & H, const helfem::Matrix & gn, const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> & f, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> f_x) {
        if(gn.cols()!=1) {
          throw std::runtime_error("Grad rho must have three columns!\n");
        }
        if(f.rows() != f_x.rows() || f.cols() != f_x.cols()) {
          throw std::runtime_error("Sizes of basis function and derivative matrices doesn't match!\n");
        }
        if(H.rows() != f.rows() || H.cols() != f.rows()) {
          throw std::runtime_error("Sizes of basis function and Fock matrices doesn't match!\n");
        }

        // Compute helper: gamma_{ip} = \sum_c \chi_{ip;c} gr_{p;c}
        //                 (N, Np)    =        (N Np; c)    (Np, 3)
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> gamma =
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(f.rows(), f.cols());
        {
          // gn.col(0) holds the (single) radial gradient weight per point;
          // scale column j of f_x by gn(j,0).
          for(Eigen::Index j=0;j<f_x.cols();j++)
            f_x.col(j) *= gn(j,0);
          gamma += f_x;
        }

        // Form Fock matrix
        H += (gamma*f.adjoint() + f*gamma.adjoint()).real();
      }

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
