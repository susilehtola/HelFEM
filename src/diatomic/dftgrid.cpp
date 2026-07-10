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

#include <cfloat>
#include <cmath>
#include <cstdio>
// LibXC
#include <xc.h>

#include "dftgrid.h"
#include <ArmaEigen.h>
#include "../general/dftfuncs.h"
// Angular quadrature
#include "../general/angular.h"

// OpenMP parallellization for XC calculations
#ifdef _OPENMP
#include <omp.h>
#endif

namespace helfem {
  namespace diatomic {
    namespace dftgrid {
      DFTGridWorker::DFTGridWorker() {
      }

      DFTGridWorker::DFTGridWorker(const helfem::diatomic::basis::TwoDBasis * basp_, int lang, int mang) : basp(basp_) {
        do_grad=false;
        do_tau=false;
        do_lapl=false;

        // Get angular grid
        helfem::angular::angular_chebyshev(lang,mang,cth,phi,wang);
      }

      DFTGridWorker::~DFTGridWorker() {
      }

      void DFTGridWorker::update_density(const arma::mat & P0) {
        // Update values of density
        if(!P0.n_elem) {
          throw std::runtime_error("Error - density matrix is empty!\n");
        }
        arma::mat P(basp->expand_boundaries(P0)(bf_ind,bf_ind));

        // Non-polarized calculation.
        polarized=false;

        // Update density vector
        Pv=P*arma::conj(bf);

        // Calculate density
        rho.zeros(1,wtot.n_elem);
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(size_t ip=0;ip<wtot.n_elem;ip++)
          rho(0,ip)=std::real(arma::dot(Pv.col(ip),bf.col(ip)));

        // Calculate gradient
        if(do_grad) {
          grho.zeros(3,wtot.n_elem);
          sigma.zeros(1,wtot.n_elem);
#ifdef _OPENMP
#pragma omp parallel for
#endif
          for(size_t ip=0;ip<wtot.n_elem;ip++) {
            // Calculate values
            double g_rad=grho(0,ip)=2.0*std::real(arma::dot(Pv.col(ip),bf_rho.col(ip)))/scale_r(ip);
            double g_th=grho(1,ip)=2.0*std::real(arma::dot(Pv.col(ip),bf_theta.col(ip)))/scale_theta(ip);
            double g_phi=grho(2,ip)=2.0*std::real(arma::dot(Pv.col(ip),bf_phi.col(ip)))/scale_phi(ip);
            // Compute sigma as well
            sigma(0,ip)=g_rad*g_rad + g_th*g_th + g_phi*g_phi;
          }
        }

        // Calculate laplacian and kinetic energy density
        if(do_tau) {
          // Adjust size of grid
          tau.zeros(1,wtot.n_elem);

          // Update helpers
          Pv_rho=P*arma::conj(bf_rho);
          Pv_theta=P*arma::conj(bf_theta);
          Pv_phi=P*arma::conj(bf_phi);

          // Calculate values
#ifdef _OPENMP
#pragma omp parallel for
#endif
          for(size_t ip=0;ip<wtot.n_elem;ip++) {
            // Gradient term
            double kinrho(std::real(arma::dot(Pv_rho.col(ip),bf_rho.col(ip)))/std::pow(scale_r(ip),2));
            double kintheta(std::real(arma::dot(Pv_theta.col(ip),bf_theta.col(ip)))/std::pow(scale_theta(ip),2));
            double kinphi(std::real(arma::dot(Pv_phi.col(ip),bf_phi.col(ip)))/std::pow(scale_phi(ip),2));
            double kin(kinrho + kintheta + kinphi);

            // Store values
            tau(0,ip)=0.5*kin;
          }
        }

        if(do_lapl)
          throw std::logic_error("Laplacian not implemented!\n");
      }

      void DFTGridWorker::update_density(const arma::mat & Pa0, const arma::mat & Pb0) {
        if(!Pa0.n_elem || !Pb0.n_elem) {
          throw std::runtime_error("Error - density matrix is empty!\n");
        }

        // Polarized calculation.
        polarized=true;

        // Update density vector
        arma::mat Pa(basp->expand_boundaries(Pa0)(bf_ind,bf_ind));
        arma::mat Pb(basp->expand_boundaries(Pb0)(bf_ind,bf_ind));

        Pav=Pa*arma::conj(bf);
        Pbv=Pb*arma::conj(bf);

        // Calculate density
        rho.zeros(2,wtot.n_elem);
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(size_t ip=0;ip<wtot.n_elem;ip++) {
          rho(0,ip)=std::real(arma::dot(Pav.col(ip),bf.col(ip)));
          rho(1,ip)=std::real(arma::dot(Pbv.col(ip),bf.col(ip)));

          /*
            double na=compute_density(Pa0,*basp,grid[ip].r);
            double nb=compute_density(Pb0,*basp,grid[ip].r);
            if(fabs(da-na)>1e-6 || fabs(db-nb)>1e-6)
            printf("Density at point % .3f % .3f % .3f: %e vs %e, %e vs %e\n",grid[ip].r.x,grid[ip].r.y,grid[ip].r.z,da,na,db,nb);
          */
        }

        // Calculate gradient

        if(do_grad) {
          grho.zeros(6,wtot.n_elem);
          sigma.zeros(3,wtot.n_elem);
#ifdef _OPENMP
#pragma omp parallel for
#endif
          for(size_t ip=0;ip<wtot.n_elem;ip++) {
            double ga_rad=grho(0,ip)=2.0*std::real(arma::dot(Pav.col(ip),bf_rho.col(ip)))/scale_r(ip);
            double ga_th=grho(1,ip)=2.0*std::real(arma::dot(Pav.col(ip),bf_theta.col(ip)))/scale_theta(ip);
            double ga_phi=grho(2,ip)=2.0*std::real(arma::dot(Pav.col(ip),bf_phi.col(ip)))/scale_phi(ip);

            double gb_rad=grho(3,ip)=2.0*std::real(arma::dot(Pbv.col(ip),bf_rho.col(ip)))/scale_r(ip);
            double gb_th=grho(4,ip)=2.0*std::real(arma::dot(Pbv.col(ip),bf_theta.col(ip)))/scale_theta(ip);
            double gb_phi=grho(5,ip)=2.0*std::real(arma::dot(Pbv.col(ip),bf_phi.col(ip)))/scale_phi(ip);

            // Compute sigma as well
            sigma(0,ip)=ga_rad*ga_rad + ga_th*ga_th + ga_phi*ga_phi;
            sigma(1,ip)=ga_rad*gb_rad + ga_th*gb_th + ga_phi*gb_phi;
            sigma(2,ip)=gb_rad*gb_rad + gb_th*gb_th + gb_phi*gb_phi;
          }
        }

        // Calculate kinetic energy density
        if(do_tau) {
          // Adjust size of grid
          tau.resize(2,wtot.n_elem);

          // Update helpers
          Pav_rho=Pa*arma::conj(bf_rho);
          Pav_theta=Pa*arma::conj(bf_theta);
          Pav_phi=Pa*arma::conj(bf_phi);

          Pbv_rho=Pb*arma::conj(bf_rho);
          Pbv_theta=Pb*arma::conj(bf_theta);
          Pbv_phi=Pb*arma::conj(bf_phi);

          // Calculate values
#ifdef _OPENMP
#pragma omp parallel for
#endif
          for(size_t ip=0;ip<wtot.n_elem;ip++) {
            // Gradient term
            double kinar=std::real(arma::dot(Pav_rho.col(ip),bf_rho.col(ip)))/std::pow(scale_r(ip),2);
            double kinath=std::real(arma::dot(Pav_theta.col(ip),bf_theta.col(ip)))/std::pow(scale_theta(ip),2);
            double kinaphi=std::real(arma::dot(Pav_phi.col(ip),bf_phi.col(ip)))/std::pow(scale_phi(ip),2);
            double kina(kinar + kinath + kinaphi);

            double kinbr=std::real(arma::dot(Pbv_rho.col(ip),bf_rho.col(ip)))/std::pow(scale_r(ip),2);
            double kinbth=std::real(arma::dot(Pbv_theta.col(ip),bf_theta.col(ip)))/std::pow(scale_theta(ip),2);
            double kinbphi=std::real(arma::dot(Pbv_phi.col(ip),bf_phi.col(ip)))/std::pow(scale_phi(ip),2);
            double kinb(kinbr + kinbth + kinbphi);

            // Store values
            tau(0,ip)=0.5*kina;
            tau(1,ip)=0.5*kinb;
          }
          if(do_lapl)
            throw std::logic_error("Laplacian not implemented!\n");
        }
      }


      double DFTGridWorker::compute_Ekin() const {
        double ekin=0.0;

        if(do_tau) {
          if(!polarized) {
            for(size_t ip=0;ip<wtot.n_elem;ip++)
              ekin+=wtot(ip)*tau(0,ip);
          } else {
            for(size_t ip=0;ip<wtot.n_elem;ip++)
              ekin+=wtot(ip)*(tau(0,ip)+tau(1,ip));
          }
        }
        return ekin;
      }

      // init_xc, zero_Exc: inherited from
      // helfem::dftgrid_common::DFTGridWorkerBase.

      void check_array(const std::vector<double> & x, size_t n, std::vector<size_t> & idx) {
        if(x.size()%n!=0) {
          std::ostringstream oss;
          oss << "Size of array " << x.size() << " is not divisible by " << n << "!\n";
          throw std::runtime_error(oss.str());
        }

        for(size_t i=0;i<x.size()/n;i++) {
          // Check for failed entry
          bool fail=false;
          for(size_t j=0;j<n;j++)
            if(!std::isfinite(x[i*n+j]))
              fail=true;

          // If failed i is not in the list, add it
          if(fail) {
            if (!std::binary_search (idx.begin(), idx.end(), i)) {
              idx.push_back(i);
              std::sort(idx.begin(),idx.end());
            }
          }
        }
      }

      // compute_xc: inherited from DFTGridWorkerBase.

      // eval_Exc: inherited from DFTGridWorkerBase.

      void DFTGridWorker::eval_Fxc(arma::mat & Ho) const {
        if(polarized) {
          throw std::runtime_error("Refusing to compute restricted Fock matrix with unrestricted density.\n");
        }

        // Work matrix
        arma::mat H(bf_ind.n_elem,bf_ind.n_elem);
        H.zeros();

        {
          // LDA potential
          arma::rowvec vrho(vxc.row(0));
          // Multiply weights into potential
          vrho%=wtot;
          // Increment matrix
          increment_lda< std::complex<double> >(H,vrho,bf);
        }

        if(do_gga) {
          // Get vsigma
          arma::rowvec vs(vsigma.row(0));
          // Get grad rho
          arma::uvec idx(arma::linspace<arma::uvec>(0,2,3));
          arma::mat gr(arma::trans(grho.rows(idx)));
          // Multiply grad rho by vsigma and the weights
          for(size_t i=0;i<gr.n_rows;i++) {
            gr(i,0)*=2.0*wtot(i)*vs(i)/scale_r(i);
            gr(i,1)*=2.0*wtot(i)*vs(i)/scale_theta(i);
            gr(i,2)*=2.0*wtot(i)*vs(i)/scale_phi(i);
          }
          // Increment matrix
          increment_gga< std::complex<double> >(H,gr,bf,bf_rho,bf_theta,bf_phi);
        }

        if(do_mgga_t) {
          arma::rowvec vt(vtau.row(0));
          vt%=0.5*wtot;

          increment_lda< std::complex<double> >(H,vt % inv_scale_r2,bf_rho);
          increment_lda< std::complex<double> >(H,vt % inv_scale_theta2,bf_theta);
          increment_lda< std::complex<double> >(H,vt % inv_scale_phi2,bf_phi);
        }
        if(do_mgga_l)
          throw std::logic_error("Laplacian not implemented!\n");

        Ho(bf_ind,bf_ind)+=H;
      }

      void DFTGridWorker::eval_Fxc(arma::mat & Hao, arma::mat & Hbo, bool beta) const {
        if(!polarized) {
          throw std::runtime_error("Refusing to compute unrestricted Fock matrix with restricted density.\n");
        }

        arma::mat Ha, Hb;
        Ha.zeros(bf_ind.n_elem,bf_ind.n_elem);
        if(beta)
          Hb.zeros(bf_ind.n_elem,bf_ind.n_elem);

        {
          // LDA potential
          arma::rowvec vrhoa(vxc.row(0));
          // Multiply weights into potential
          vrhoa%=wtot;
          // Increment matrix
          increment_lda< std::complex<double> >(Ha,vrhoa,bf);

          if(beta) {
            arma::rowvec vrhob(vxc.row(1));
            vrhob%=wtot;
            increment_lda< std::complex<double> >(Hb,vrhob,bf);
          }
        }
        if(Ha.has_nan() || (beta && Hb.has_nan()))
          //throw std::logic_error("NaN encountered!\n");
          fprintf(stderr,"NaN in Hamiltonian!\n");

        if(do_gga) {
          // Get vsigma
          arma::rowvec vs_aa(vsigma.row(0));
          arma::rowvec vs_ab(vsigma.row(1));

          // Get grad rho
          arma::uvec idxa(arma::linspace<arma::uvec>(0,2,3));
          arma::uvec idxb(arma::linspace<arma::uvec>(3,5,3));
          arma::mat gr_a0(arma::trans(grho.rows(idxa)));
          arma::mat gr_b0(arma::trans(grho.rows(idxb)));

          // Multiply grad rho by vsigma and the weights
          arma::mat gr_a(gr_a0);
          for(size_t i=0;i<gr_a.n_rows;i++) {
            gr_a(i,0)=wtot(i)*(2.0*vs_aa(i)*gr_a0(i,0) + vs_ab(i)*gr_b0(i,0))/scale_r(i);
            gr_a(i,1)=wtot(i)*(2.0*vs_aa(i)*gr_a0(i,1) + vs_ab(i)*gr_b0(i,1))/scale_theta(i);
            gr_a(i,2)=wtot(i)*(2.0*vs_aa(i)*gr_a0(i,2) + vs_ab(i)*gr_b0(i,2))/scale_phi(i);
          }
          // Increment matrix
          increment_gga< std::complex<double> >(Ha,gr_a,bf,bf_rho,bf_theta,bf_phi);

          if(beta) {
            arma::rowvec vs_bb(vsigma.row(2));
            arma::mat gr_b(gr_b0);
            for(size_t i=0;i<gr_b.n_rows;i++) {
              gr_b(i,0)=wtot(i)*(2.0*vs_bb(i)*gr_b0(i,0) + vs_ab(i)*gr_a0(i,0))/scale_r(i);
              gr_b(i,1)=wtot(i)*(2.0*vs_bb(i)*gr_b0(i,1) + vs_ab(i)*gr_a0(i,1))/scale_theta(i);
              gr_b(i,2)=wtot(i)*(2.0*vs_bb(i)*gr_b0(i,2) + vs_ab(i)*gr_a0(i,2))/scale_phi(i);
            }
            increment_gga< std::complex<double> >(Hb,gr_b,bf,bf_rho,bf_theta,bf_phi);
          }
        }


        if(do_mgga_t) {
          arma::rowvec vt_a(vtau.row(0));
          vt_a%=0.5*wtot;

          increment_lda< std::complex<double> >(Ha,vt_a % inv_scale_r2,bf_rho);
          increment_lda< std::complex<double> >(Ha,vt_a % inv_scale_theta2,bf_theta);
          increment_lda< std::complex<double> >(Ha,vt_a % inv_scale_phi2,bf_phi);
          if(beta) {
            arma::rowvec vt_b(vtau.row(1));
            vt_b%=0.5*wtot;

            increment_lda< std::complex<double> >(Hb,vt_b % inv_scale_r2,bf_rho);
            increment_lda< std::complex<double> >(Hb,vt_b % inv_scale_theta2,bf_theta);
            increment_lda< std::complex<double> >(Hb,vt_b % inv_scale_phi2,bf_phi);
          }
        }
        if(do_mgga_l) {
          throw std::logic_error("Laplacian not implemented!\n");
        }

        Hao(bf_ind,bf_ind)+=Ha;
        if(beta)
          Hbo(bf_ind,bf_ind)+=Hb;
      }

      // check_grad_tau_lapl, get_grad_tau_lapl, set_grad_tau_lapl:
      // inherited from DFTGridWorkerBase.

      void DFTGridWorker::compute_bf(size_t iel, size_t irad) {
        // Update function list
        bf_ind=basp->bf_list_dummy(iel);

        // Get radial weights. Only do one radial quadrature point at a
        // time, since this is an easy way to save a lot of memory.
        arma::vec wrad(1), r(1);
        wrad(0)=basp->get_wrad(iel)(irad);
        r(0)=basp->get_r(iel)(irad);

        double Rhalf(basp->get_Rhalf());

        // Calculate helpers
        arma::vec shmu(arma::sinh(r));

        arma::vec sth(cth.n_elem);
        for(size_t ia=0;ia<cth.n_elem;ia++)
          sth(ia)=sqrt(1.0 - cth(ia)*cth(ia));

        // Radial is
        scale_r.resize(wrad.n_elem*wang.n_elem);
        for(size_t ia=0;ia<wang.n_elem;ia++)
          for(size_t ir=0;ir<wrad.n_elem;ir++)
            // h_mu = R_{h}\sqrt{\sinh^{2}\mu+\sin^{2}\nu}
            scale_r(ia*wrad.n_elem+ir)=Rhalf*sqrt(std::pow(shmu(ir),2) + std::pow(sth(ia),2));
        // Theta is same as radial
        scale_theta=scale_r;
        // phi is simple
        scale_phi.resize(wrad.n_elem*wang.n_elem);
        for(size_t ia=0;ia<wang.n_elem;ia++)
          for(size_t ir=0;ir<wrad.n_elem;ir++)
            scale_phi(ia*wrad.n_elem+ir)=Rhalf*shmu(ir)*sth(ia);
        // Pre-compute 1/scale^2 for the kinetic / mGGA terms.
        inv_scale_r2 = 1.0 / arma::square(scale_r);
        inv_scale_theta2 = 1.0 / arma::square(scale_theta);
        inv_scale_phi2 = 1.0 / arma::square(scale_phi);
        // Update total weights
        wtot.zeros(wrad.n_elem*wang.n_elem);
        for(size_t ia=0;ia<wang.n_elem;ia++)
          for(size_t ir=0;ir<wrad.n_elem;ir++) {
            size_t idx=ia*wrad.n_elem+ir;
            // sin(th) is already contained within wang, but we don't want to divide by it since it may be zero.
            wtot(idx)=wang(ia)*wrad(ir)*std::pow(Rhalf,3)*shmu(ir)*(std::pow(shmu(ir),2)+std::pow(sth(ia),2));
          }

        // Compute basis function values
        bf.zeros(bf_ind.n_elem,wtot.n_elem);
        // Loop over angular grid
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(size_t ia=0;ia<cth.n_elem;ia++) {
          // Evaluate basis functions at angular point
          arma::cx_mat abf(basp->eval_bf(iel, irad, cth(ia), phi(ia)));
          if(abf.n_cols != bf_ind.n_elem) {
            std::ostringstream oss;
            oss << "Mismatch! Have " << bf_ind.n_elem << " basis function indices but " << abf.n_cols << " basis functions!\n";
            throw std::logic_error(oss.str());
          }
          // Store functions
          bf.cols(ia*wrad.n_elem,(ia+1)*wrad.n_elem-1)=arma::trans(abf);
        }

        if(do_grad) {
          bf_rho.zeros(bf_ind.n_elem,wtot.n_elem);
          bf_theta.zeros(bf_ind.n_elem,wtot.n_elem);
          bf_phi.zeros(bf_ind.n_elem,wtot.n_elem);

#ifdef _OPENMP
#pragma omp parallel for
#endif
          for(size_t ia=0;ia<cth.n_elem;ia++) {
            // Evaluate basis functions at angular point
            arma::cx_mat dr, dth, dphi;
            basp->eval_df(iel, irad, cth(ia), phi(ia), dr, dth, dphi);
            if(dr.n_cols != bf_ind.n_elem) {
              std::ostringstream oss;
              oss << "Mismatch! Have " << bf_ind.n_elem << " basis function indices but " << dr.n_cols << " basis functions!\n";
              throw std::logic_error(oss.str());
            }
            // Store functions
            bf_rho.cols(ia*wrad.n_elem,(ia+1)*wrad.n_elem-1)=arma::trans(dr);
            bf_theta.cols(ia*wrad.n_elem,(ia+1)*wrad.n_elem-1)=arma::trans(dth);
            bf_phi.cols(ia*wrad.n_elem,(ia+1)*wrad.n_elem-1)=arma::trans(dphi);
          }
        }

        if(do_lapl) {
          throw std::logic_error("Laplacian not implemented.\n");
        }
      }

      DFTGrid::DFTGrid() {
      }

      DFTGrid::DFTGrid(const helfem::diatomic::basis::TwoDBasis * basp_, int lang_, int mang_) : basp(basp_), lang(lang_), mang(mang_) {
        arma::vec cth, phi, wang;
        helfem::angular::angular_chebyshev(lang,mang,cth,phi,wang);
        printf("DFT angular grid of order l=%i m=%i has %i points\n",lang,mang,(int) wang.n_elem);
      }

      DFTGrid::~DFTGrid() {
      }

      void DFTGrid::eval_Fxc(int x_func, const helfem::Vector & x_pars, int c_func, const helfem::Vector & c_pars, const helfem::Matrix & P_e, helfem::Matrix & H_e, double & Exc, double & Nel, double & Ekin, double thr) {
        // Eigen public boundary; bridge the density to the arma interior once
        // (functional parameters flow straight through to compute_xc).
        const arma::mat P(helfem::to_arma(P_e));
        arma::mat H;
        H.zeros(basp->Ndummy(),basp->Ndummy());

        double exc=0.0;
        double ekin=0.0;
        double nel=0.0;
        {
          DFTGridWorker grid(basp,lang,mang);
          grid.check_grad_tau_lapl(x_func,c_func);

          for(size_t iel=0;iel<basp->get_rad_Nel();iel++) {
            for(size_t irad=0;irad<basp->get_r(iel).n_elem;irad++) {
              grid.compute_bf(iel,irad);
              grid.update_density(P);
              nel+=grid.compute_Nel();
              ekin+=grid.compute_Ekin();

              grid.init_xc();
              if(x_func>0)
                grid.compute_xc(x_func, x_pars, thr);
              if(c_func>0)
                grid.compute_xc(c_func, c_pars, thr);

              exc+=grid.eval_Exc();
              grid.eval_Fxc(H);
            }
          }
        }

        // Save outputs
        Exc=exc;
        Ekin=ekin;
        Nel=nel;

        H_e=helfem::to_eigen(basp->remove_boundaries(H));
      }

      void DFTGrid::eval_Fxc(int x_func, const helfem::Vector & x_pars, int c_func, const helfem::Vector & c_pars, const helfem::Matrix & Pa_e, const helfem::Matrix & Pb_e, helfem::Matrix & Ha_e, helfem::Matrix & Hb_e, double & Exc, double & Nel, double & Ekin, bool beta, double thr) {
        // Eigen public boundary; bridge the density to the arma interior once
        // (functional parameters flow straight through to compute_xc).
        const arma::mat Pa(helfem::to_arma(Pa_e));
        const arma::mat Pb(helfem::to_arma(Pb_e));
        arma::mat Ha, Hb;
        Ha.zeros(basp->Ndummy(),basp->Ndummy());
        Hb.zeros(basp->Ndummy(),basp->Ndummy());

        double exc=0.0;
        double nel=0.0;
        double ekin=0.0;
        {
          DFTGridWorker grid(basp,lang,mang);
          grid.check_grad_tau_lapl(x_func,c_func);

          for(size_t iel=0;iel<basp->get_rad_Nel();iel++) {
            for(size_t irad=0;irad<basp->get_r(iel).n_elem;irad++) {
              grid.compute_bf(iel,irad);
              grid.update_density(Pa,Pb);
              nel+=grid.compute_Nel();
              ekin+=grid.compute_Ekin();

              grid.init_xc();
              if(x_func>0)
                grid.compute_xc(x_func, x_pars, thr);
              if(c_func>0)
                grid.compute_xc(c_func, c_pars, thr);

              exc+=grid.eval_Exc();
              grid.eval_Fxc(Ha,Hb,beta);
            }
          }
        }

        // Save outputs
        Exc=exc;
        Ekin=ekin;
        Nel=nel;

        // Clean up matrices
        Ha_e=helfem::to_eigen(basp->remove_boundaries(Ha));
        Hb_e=helfem::to_eigen(basp->remove_boundaries(Hb));
      }

    }
  }
}
