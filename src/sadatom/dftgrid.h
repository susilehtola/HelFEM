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
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#ifndef SADATOM_DFTGRID_H
#define SADATOM_DFTGRID_H

#include "basis.h"

namespace helfem {
  namespace sadatom {
    namespace dftgrid {

      /// Worker class
      class DFTGridWorker {
      protected:
        /// Basis set
        const helfem::sadatom::basis::TwoDBasis *basp;

        /// Distance from nucleus
        arma::vec r;
        /// Radial quadrature weight
        arma::rowvec wrad;
        /// Total quadrature weight
        arma::rowvec wtot;

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

        // LDA stuff:

        /// Density, Nrho x Npts
        arma::mat rho;
        /// Energy density, Npts
        arma::rowvec exc;
        /// Functional derivative of energy wrt electron density, Nrho x Npts
        arma::mat vxc;

        // GGA stuff

        /// Gradient of electron density, (3 x Nrho) x Npts
        arma::mat grho;
        /// Dot products of gradient of electron density, N x Npts; N=1 for closed-shell and 3 for open-shell
        arma::mat sigma;
        /// Functional derivative of energy wrt gradient of electron density
        arma::mat vsigma;

        // Meta-GGA stuff

        /// Laplacian of electron density
        arma::mat lapl;
        /// Kinetic energy density
        arma::mat tau;

        /// Functional derivative of energy wrt laplacian of electron density
        arma::mat vlapl;
        /// Functional derivative of energy wrt kinetic energy density
        arma::mat vtau;

      public:
        /// Dummy constructor
        DFTGridWorker();
        /// Constructor
        DFTGridWorker(const helfem::sadatom::basis::TwoDBasis * basp);
        /// Destructor
        ~DFTGridWorker();

        /// Check necessity of computing gradient and laplacians, necessary for compute_bf!
        void check_grad_tau_lapl(int x_func, int c_func);
        /// Get necessity of computing gradient and laplacians
        void get_grad_tau_lapl(bool & grad, bool & tau, bool & lapl) const;
        /// Set necessity of computing gradient and laplacians, necessary for compute_bf!
        void set_grad_tau_lapl(bool grad, bool tau, bool lapl);

        /// Compute basis functions on grid points
        void compute_bf(size_t iel);
        /// Free memory
        void free();

        /// Update values of density, restricted calculation
        void update_density(const arma::cube & P);
        /// Update values of density, unrestricted calculation
        void update_density(const arma::cube & Pa, const arma::cube & Pb);
        /// Screen out small densities
        void screen_density(double thr);

        /// Compute number of electrons
        double compute_Nel() const;
        /// Compute kinetic energy density
        double compute_tau() const;
        /// Compute laplacian
        double compute_lapl() const;

        /// Initialize XC arrays
        void init_xc();
        /// Compute XC functional from density and add to total XC
        /// array. Pot toggles evaluation of potential
        void compute_xc(int func_id, const arma::vec & params, bool pot=true);
        /// Evaluate exchange/correlation energy
        double eval_Exc() const;
        /// Zero out energy
        void zero_Exc();

        /// Evaluate overlap matrix
        void eval_overlap(arma::mat & S) const;

        /// Evaluate Fock matrix, restricted calculation
        void eval_Fxc(arma::cube & H) const;
        /// Evaluate Fock matrix, unrestricted calculation
        void eval_Fxc(arma::cube & Ha, arma::cube & Hb, bool beta=true) const;
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

        /// Evaluate overlap
        arma::mat eval_overlap();
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
