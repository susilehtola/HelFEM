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
        arma::rowvec r;
        /// Radial quadrature weight
        arma::rowvec wrad;

        /// List of basis functions in element
        arma::uvec bf_ind;
        /// Values of important functions in grid points, Nbf * Ngrid
        arma::mat bf;
        /// Radial gradient
        arma::mat bf_rho;
        /// Radial laplacian
        arma::mat bf_rho2;

        /// Density helper matrices: P_{uv} chi_v, and P_{uv} nabla(chi_v)
        arma::mat Pv, Pv_rho;
        /// Same for spin-polarized
        arma::mat Pav, Pav_rho;
        arma::mat Pbv, Pbv_rho;

        /// Gradient of electron density
        arma::mat grho;

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
        /// Free memory
        void free();

        /// Update values of density, restricted calculation
        void update_density(const arma::cube & P);
        /// Update values of density, unrestricted calculation
        void update_density(const arma::cube & Pa, const arma::cube & Pb);

        /// Compute number of electrons
        double compute_Nel() const;
        /// Compute kinetic energy density
        double compute_tau() const;
        /// Compute laplacian
        double compute_lapl() const;

        // init_xc / compute_xc / eval_Exc / zero_Exc are inherited
        // from DFTGridWorkerBase.

        /// Evaluate overlap matrix
        void eval_overlap(arma::mat & S) const;

        /// Evaluate Fock matrix, restricted calculation
        void eval_Fxc(arma::cube & H) const;
        /// Evaluate Fock matrix, unrestricted calculation
        void eval_Fxc(arma::cube & Ha, arma::cube & Hb, bool beta=true) const;

        /// Get the potential
        void get_pot(arma::mat & pot) const;
        /// Get the ingredients
        void get_ingredients(arma::mat & ing) const;
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

        /// Compute Fock matrix, exchange-correlation energy and integrated electron density, restricted case
        void eval_Fxc(int x_func, const arma::vec & x_pars, int c_func, const arma::vec & c_pars, const arma::cube & P, arma::cube & H, double & Exc, double & Nel, double thr);
        /// Compute Fock matrix, exchange-correlation energy and integrated electron density, unrestricted case
        void eval_Fxc(int x_func, const arma::vec & x_pars, int c_func, const arma::vec & c_pars, const arma::cube & Pa, const arma::cube & Pb, arma::cube & Ha, arma::cube & Hb, double & Exc, double & Nel, bool beta, double thr);

        /// Evaluate the potential
        void eval_pot(int x_func, const arma::vec & x_pars, int c_func, const arma::vec & c_pars, const arma::cube & P, arma::mat & pot, double thr);
        /// Compute the potential
        void eval_pot(int x_func, const arma::vec & x_pars, int c_func, const arma::vec & c_pars, const arma::cube & Pa, const arma::cube & Pb, arma::mat & pot, double thr);

        /// Evaluate the ingredients
        void eval_ing(int x_func, const arma::vec & x_pars, int c_func, const arma::vec & c_pars, const arma::cube & P, arma::mat & ing, double thr);
        /// Compute the ingredients
        void eval_ing(int x_func, const arma::vec & x_pars, int c_func, const arma::vec & c_pars, const arma::cube & Pa, const arma::cube & Pb, arma::mat & ing, double thr);

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
      template<typename T> void increment_gga(arma::mat & H, const arma::mat & gn, const arma::Mat<T> & f, arma::Mat<T> f_x) {
        if(gn.n_cols!=1) {
          throw std::runtime_error("Grad rho must have three columns!\n");
        }
        if(f.n_rows != f_x.n_rows || f.n_cols != f_x.n_cols) {
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

          gc=arma::strans(gn.col(0));
          for(size_t j=0;j<f_x.n_cols;j++)
            for(size_t i=0;i<f_x.n_rows;i++)
              f_x(i,j)*=gc(j);
          gamma+=f_x;
        }

        // Form Fock matrix
        H+=arma::real(gamma*arma::trans(f) + f*arma::trans(gamma));
      }

      /// BLAS routine for mGGA-type quadrature
      template<typename T> void increment_mgga_lapl(arma::mat & H, const arma::mat & vlapl, const arma::Mat<T> & f, const arma::Mat<T> & l) {
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
