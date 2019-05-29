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

#include <cfloat>
#include <cmath>
#include <cstdio>
// LibXC
#include <xc.h>

#include "dftgrid.h"
#include "../general/dftfuncs.h"

// OpenMP parallellization for XC calculations
#ifdef _OPENMP
#include <omp.h>
#endif

namespace helfem {
  namespace sadatom {
    namespace dftgrid {
      DFTGridWorker::DFTGridWorker() {
      }

      DFTGridWorker::DFTGridWorker(const helfem::sadatom::basis::TwoDBasis * basp_) : basp(basp_) {
        do_grad=false;
        do_tau=false;
        do_lapl=false;
      }

      DFTGridWorker::~DFTGridWorker() {
      }

      void DFTGridWorker::update_density(const arma::mat & P0) {
        // Update values of density
        arma::mat P(P0(bf_ind,bf_ind));

        // Non-polarized calculation.
        polarized=false;

        // Update density vector
        Pv=P*arma::conj(bf);

        // Calculate density
        rho.zeros(1,wtot.n_elem);
        for(size_t ip=0;ip<wtot.n_elem;ip++)
          rho(0,ip)=std::real(arma::dot(Pv.col(ip),bf.col(ip)));

        // Calculate gradient
        if(do_grad) {
          grho.zeros(3,wtot.n_elem);
          sigma.zeros(1,wtot.n_elem);
          for(size_t ip=0;ip<wtot.n_elem;ip++) {
            // Calculate values
            double g_rad=grho(0,ip)=2.0*std::real(arma::dot(Pv.col(ip),bf_rho.col(ip)));
            // Compute sigma as well
            sigma(0,ip)=g_rad*g_rad;
          }
        }

        // Calculate laplacian and kinetic energy density
        if(do_tau || do_lapl)
          throw std::logic_error("Meta-GGAs not implemented!\n");
      }

      void DFTGridWorker::update_density(const arma::mat & Pa0, const arma::mat & Pb0) {
        if(!Pa0.n_elem || !Pb0.n_elem) {
          throw std::runtime_error("Error - density matrix is empty!\n");
        }
        arma::mat Pa(Pa0(bf_ind,bf_ind));
        arma::mat Pb(Pb0(bf_ind,bf_ind));

        // Polarized calculation.
        polarized=true;

        // Update density vector
        Pav=Pa*arma::conj(bf);
        Pbv=Pb*arma::conj(bf);

        // Calculate density
        rho.zeros(2,wtot.n_elem);
        for(size_t ip=0;ip<wtot.n_elem;ip++) {
          rho(0,ip)=std::real(arma::dot(Pav.col(ip),bf.col(ip)));
          rho(1,ip)=std::real(arma::dot(Pbv.col(ip),bf.col(ip)));
        }

        // Calculate gradient
        if(do_grad) {
          grho.zeros(6,wtot.n_elem);
          sigma.zeros(3,wtot.n_elem);
          for(size_t ip=0;ip<wtot.n_elem;ip++) {
            double ga_rad=grho(0,ip)=2.0*std::real(arma::dot(Pav.col(ip),bf_rho.col(ip)));
            double gb_rad=grho(3,ip)=2.0*std::real(arma::dot(Pbv.col(ip),bf_rho.col(ip)));

            // Compute sigma as well
            sigma(0,ip)=ga_rad*ga_rad;
            sigma(1,ip)=ga_rad*gb_rad;
            sigma(2,ip)=gb_rad*gb_rad;
          }
        }

        // Calculate kinetic energy density
        if(do_tau || do_lapl)
          throw std::logic_error("Meta-GGA not implemented!\n");
      }

      void DFTGridWorker::screen_density(double thr) {
        if(polarized) {
          for(size_t ip=0;ip<wtot.n_elem;ip++) {
            if(rho(0,ip)+rho(1,ip) <= thr) {
              rho(0,ip)=0.0;
              rho(1,ip)=0.0;

              if(do_grad) {
                sigma(0,ip)=0.0;
                sigma(1,ip)=0.0;
                sigma(2,ip)=0.0;
              }

              if(do_tau) {
                tau(0,ip)=0.0;
                tau(1,ip)=0.0;
              }
            }
          }
        } else {
          for(size_t ip=0;ip<wtot.n_elem;ip++) {
            if(rho(0,ip) <= thr) {
              rho(0,ip)=0.0;
              if(do_grad) {
                sigma(0,ip)=0.0;
              }
              if(do_tau) {
                tau(0,ip)=0.0;
              }
            }
          }
        }
      }

      double DFTGridWorker::compute_Nel() const {
        double nel=0.0;
        if(!polarized) {
          for(size_t ip=0;ip<wtot.n_elem;ip++)
            nel+=wtot(ip)*rho(0,ip);
        } else {
          for(size_t ip=0;ip<wtot.n_elem;ip++)
            nel+=wtot(ip)*(rho(0,ip)+rho(1,ip));
        }

        return nel;
      }

      void DFTGridWorker::init_xc() {
        // Size of grid.
        const size_t N=wtot.n_elem;

        // Zero energy
        zero_Exc();

        if(!polarized) {
          // Restricted case
          vxc.zeros(1,N);
          if(do_grad)
            vsigma.zeros(1,N);
          if(do_tau)
            vtau.zeros(1,N);
          if(do_lapl)
            vlapl.zeros(1,N);
        } else {
          // Unrestricted case
          vxc.zeros(2,N);
          if(do_grad)
            vsigma.zeros(3,N);
          if(do_tau)
            vtau.zeros(2,N);
          if(do_lapl) {
            vlapl.zeros(2,N);
          }
        }

        // Initial values
        do_gga=false;
        do_mgga_l=false;
        do_mgga_t=false;
      }

      void DFTGridWorker::zero_Exc() {
        exc.zeros(wtot.n_elem);
      }

      void DFTGridWorker::compute_xc(int func_id, bool pot) {
        // Compute exchange-correlation functional

        // Which functional is in question?
        bool gga, mgga_t, mgga_l;
        is_gga_mgga(func_id,gga,mgga_t,mgga_l);

        // Update controlling flags for eval_Fxc (exchange and correlation
        // parts might be of different type)
        do_gga=do_gga || gga || mgga_t || mgga_l;
        do_mgga_t=do_mgga_t || mgga_t;
        do_mgga_l=do_mgga_l || mgga_l;

        // Amount of grid points
        const size_t N=wtot.n_elem;

        // Work arrays - exchange and correlation are computed separately
        arma::rowvec exc_wrk;
        arma::mat vxc_wrk;
        arma::mat vsigma_wrk;
        arma::mat vlapl_wrk;
        arma::mat vtau_wrk;

        if(has_exc(func_id))
          exc_wrk.zeros(exc.n_elem);
        if(pot) {
          vxc_wrk.zeros(vxc.n_rows,vxc.n_cols);
          if(gga || mgga_t || mgga_l)
            vsigma_wrk.zeros(vsigma.n_rows,vsigma.n_cols);
          if(mgga_t)
            vtau_wrk.zeros(vtau.n_rows,vtau.n_cols);
          if(mgga_l)
            vlapl_wrk.zeros(vlapl.n_rows,vlapl.n_cols);
        }

        // Spin variable for libxc
        int nspin;
        if(!polarized)
          nspin=XC_UNPOLARIZED;
        else
          nspin=XC_POLARIZED;

        // Initialize libxc worker
        xc_func_type func;
        if(xc_func_init(&func, func_id, nspin) != 0) {
          std::ostringstream oss;
          oss << "Functional "<<func_id<<" not found!";
          throw std::runtime_error(oss.str());
        }

        // Evaluate functionals.
        if(has_exc(func_id)) {
          if(pot) {
            if(mgga_t || mgga_l) {// meta-GGA
              double * laplp = mgga_t ? lapl.memptr() : NULL;
              double * taup = mgga_t ? tau.memptr() : NULL;
              double * vlaplp = mgga_t ? vlapl_wrk.memptr() : NULL;
              double * vtaup = mgga_t ? vtau_wrk.memptr() : NULL;
              xc_mgga_exc_vxc(&func, N, rho.memptr(), sigma.memptr(), laplp, taup, exc_wrk.memptr(), vxc_wrk.memptr(), vsigma_wrk.memptr(), vlaplp, vtaup);
            } else if(gga) // GGA
              xc_gga_exc_vxc(&func, N, rho.memptr(), sigma.memptr(), exc_wrk.memptr(), vxc_wrk.memptr(), vsigma_wrk.memptr());
            else // LDA
              xc_lda_exc_vxc(&func, N, rho.memptr(), exc_wrk.memptr(), vxc_wrk.memptr());
          } else {
            if(mgga_t || mgga_l) { // meta-GGA
              double * laplp = mgga_t ? lapl.memptr() : NULL;
              double * taup = mgga_t ? tau.memptr() : NULL;
              xc_mgga_exc(&func, N, rho.memptr(), sigma.memptr(), laplp, taup, exc_wrk.memptr());
            } else if(gga) // GGA
              xc_gga_exc(&func, N, rho.memptr(), sigma.memptr(), exc_wrk.memptr());
            else // LDA
              xc_lda_exc(&func, N, rho.memptr(), exc_wrk.memptr());
          }

        } else {
          if(pot) {
            if(mgga_t || mgga_l) { // meta-GGA
              double * laplp = mgga_t ? lapl.memptr() : NULL;
              double * taup = mgga_t ? tau.memptr() : NULL;
              double * vlaplp = mgga_t ? vlapl_wrk.memptr() : NULL;
              double * vtaup = mgga_t ? vtau_wrk.memptr() : NULL;
              xc_mgga_vxc(&func, N, rho.memptr(), sigma.memptr(), laplp, taup, vxc_wrk.memptr(), vsigma_wrk.memptr(), vlaplp, vtaup);
            } else if(gga) // GGA
              xc_gga_vxc(&func, N, rho.memptr(), sigma.memptr(), vxc_wrk.memptr(), vsigma_wrk.memptr());
            else // LDA
              xc_lda_vxc(&func, N, rho.memptr(), vxc_wrk.memptr());
          }
        }

        // Sum to total arrays containing both exchange and correlation
        if(has_exc(func_id))
          exc+=exc_wrk;
        if(pot) {
          if(mgga_l)
            vlapl+=vlapl_wrk;
          if(mgga_t)
            vtau+=vtau_wrk;
          if(mgga_t || mgga_l || gga)
            vsigma+=vsigma_wrk;
          vxc+=vxc_wrk;
        }

        // Free functional
        xc_func_end(&func);
      }

      double DFTGridWorker::eval_Exc() const {
        arma::rowvec dens(rho.row(0));
        if(polarized)
          dens+=rho.row(1);

        return arma::sum(wtot%exc%dens);
      }

      void DFTGridWorker::eval_overlap(arma::mat & So) const {
        // Calculate in subspace
        arma::mat S(bf_ind.n_elem,bf_ind.n_elem);
        S.zeros();
        increment_lda<double>(S,wtot,bf);
        // Increment
        So.submat(bf_ind,bf_ind)+=S;
      }

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
          increment_lda<double>(H,vrho,bf);
        }

        if(do_gga) {
          // Get vsigma
          arma::rowvec vs(vsigma.row(0));
          // Get grad rho
          arma::uvec idx(arma::linspace<arma::uvec>(0,2,3));
          arma::mat gr(arma::trans(grho.rows(idx)));
          // Multiply grad rho by vsigma and the weights
          for(size_t i=0;i<gr.n_rows;i++) {
            gr(i,0)*=2.0*wtot(i)*vs(i);
          }
          // Increment matrix
          increment_gga<double>(H,gr,bf,bf_rho);
        }

        if(do_mgga_t) {
          arma::rowvec vt(vtau.row(0));
          vt%=0.5*wtot;

          increment_lda<double>(H,vt,bf_rho);
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
          increment_lda<double>(Ha,vrhoa,bf);

          if(beta) {
            arma::rowvec vrhob(vxc.row(1));
            vrhob%=wtot;
            increment_lda<double>(Hb,vrhob,bf);
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
            gr_a(i,0)=wtot(i)*(2.0*vs_aa(i)*gr_a0(i,0) + vs_ab(i)*gr_b0(i,0));
          }
          // Increment matrix
          increment_gga<double>(Ha,gr_a,bf,bf_rho);

          if(beta) {
            arma::rowvec vs_bb(vsigma.row(2));
            arma::mat gr_b(gr_b0);
            for(size_t i=0;i<gr_b.n_rows;i++) {
              gr_b(i,0)=wtot(i)*(2.0*vs_bb(i)*gr_b0(i,0) + vs_ab(i)*gr_a0(i,0));
            }
            increment_gga<double>(Hb,gr_b,bf,bf_rho);
          }
        }

        if(do_mgga_t || do_mgga_l) {
          throw std::logic_error("Meta-GGA not implemented!\n");
        }

        Hao(bf_ind,bf_ind)+=Ha;
        if(beta)
          Hbo(bf_ind,bf_ind)+=Hb;
      }

      void DFTGridWorker::check_grad_tau_lapl(int x_func, int c_func) {
        // Do we need gradients?
        do_grad=false;
        if(x_func>0)
          do_grad=do_grad || gradient_needed(x_func);
        if(c_func>0)
          do_grad=do_grad || gradient_needed(c_func);

        // Do we need laplacians?
        do_tau=false;
        if(x_func>0)
          do_tau=do_tau || tau_needed(x_func);
        if(c_func>0)
          do_tau=do_tau || tau_needed(c_func);

        // Do we need laplacians?
        do_lapl=false;
        if(x_func>0)
          do_lapl=do_lapl || laplacian_needed(x_func);
        if(c_func>0)
          do_lapl=do_lapl || laplacian_needed(c_func);
      }

      void DFTGridWorker::get_grad_tau_lapl(bool & grad_, bool & tau_, bool & lap_) const {
        grad_=do_grad;
        tau_=do_tau;
        lap_=do_lapl;
      }

      void DFTGridWorker::set_grad_tau_lapl(bool grad_, bool tau_, bool lap_) {
        do_grad=grad_;
        do_tau=tau_;
        do_lapl=lap_;
      }

      void DFTGridWorker::compute_bf(size_t iel) {
        // Update function list
        bf_ind=basp->bf_list(iel);

        // Get radii and radial weights
        arma::vec r(basp->get_r(iel));
        arma::vec wrad(basp->get_wrad(iel));

        // Update total weights
        wtot.zeros(wrad.n_elem);
        for(size_t ir=0;ir<wrad.n_elem;ir++) {
          wtot(ir)=4.0*M_PI*wrad(ir)*std::pow(r(ir),2);
        }

        // Compute basis function values
        bf=arma::trans(basp->eval_bf(iel));

        if(do_grad) {
          bf_rho=arma::trans(basp->eval_df(iel));
        }

        if(do_lapl) {
          throw std::logic_error("Laplacian not implemented.\n");
        }
      }

      DFTGrid::DFTGrid() {
      }

      DFTGrid::DFTGrid(const helfem::sadatom::basis::TwoDBasis * basp_) : basp(basp_) {
      }

      DFTGrid::~DFTGrid() {
      }

      void DFTGrid::eval_Fxc(int x_func, int c_func, const arma::mat & P, arma::mat & H, double & Exc, double & Nel, double thr) {
        H.zeros(P.n_rows,P.n_rows);

        double exc=0.0;
        double nel=0.0;
#ifdef _OPENMP
#pragma omp parallel reduction(+:exc,nel)
#endif
        {
          DFTGridWorker grid(basp);
          grid.check_grad_tau_lapl(x_func,c_func);

#ifdef _OPENMP
#pragma omp for
#endif
          for(size_t iel=0;iel<basp->get_rad_Nel();iel+=2) {
            grid.compute_bf(iel);
            grid.update_density(P);
            nel+=grid.compute_Nel();

            grid.init_xc();
            if(thr>0.0)
              grid.screen_density(thr);
            if(x_func>0)
              grid.compute_xc(x_func);
            if(c_func>0)
              grid.compute_xc(c_func);

            exc+=grid.eval_Exc();
            grid.eval_Fxc(H);
          }
#ifdef _OPENMP
#pragma omp for
#endif
          for(size_t iel=1;iel<basp->get_rad_Nel();iel+=2) {
            grid.compute_bf(iel);
            grid.update_density(P);
            nel+=grid.compute_Nel();

            grid.init_xc();
            if(thr>0.0)
              grid.screen_density(thr);
            if(x_func>0)
              grid.compute_xc(x_func);
            if(c_func>0)
              grid.compute_xc(c_func);

            exc+=grid.eval_Exc();
            grid.eval_Fxc(H);
          }
        }

        // Save outputs
        Exc=exc;
        Nel=nel;
      }

      void DFTGrid::eval_Fxc(int x_func, int c_func, const arma::mat & Pa, const arma::mat & Pb, arma::mat & Ha, arma::mat & Hb, double & Exc, double & Nel, bool beta, double thr) {
        Ha.zeros(Pa.n_rows,Pa.n_rows);
        Hb.zeros(Pb.n_rows,Pb.n_rows);

        double exc=0.0;
        double nel=0.0;
#ifdef _OPENMP
#pragma omp parallel reduction(+:exc,nel)
#endif
        {
          DFTGridWorker grid(basp);
          grid.check_grad_tau_lapl(x_func,c_func);

#ifdef _OPENMP
#pragma omp for
#endif
          for(size_t iel=0;iel<basp->get_rad_Nel();iel+=2) {
            grid.compute_bf(iel);
            grid.update_density(Pa,Pb);
            nel+=grid.compute_Nel();

            grid.init_xc();
            if(thr>0.0)
              grid.screen_density(thr);
            if(x_func>0)
              grid.compute_xc(x_func);
            if(c_func>0)
              grid.compute_xc(c_func);

            exc+=grid.eval_Exc();
            grid.eval_Fxc(Ha,Hb,beta);
          }
#ifdef _OPENMP
#pragma omp for
#endif
          for(size_t iel=1;iel<basp->get_rad_Nel();iel+=2) {
            grid.compute_bf(iel);
            grid.update_density(Pa,Pb);
            nel+=grid.compute_Nel();

            grid.init_xc();
            if(thr>0.0)
              grid.screen_density(thr);
            if(x_func>0)
              grid.compute_xc(x_func);
            if(c_func>0)
              grid.compute_xc(c_func);

            exc+=grid.eval_Exc();
            grid.eval_Fxc(Ha,Hb,beta);
          }
        }

        // Save outputs
        Exc=exc;
        Nel=nel;
      }

      arma::mat DFTGrid::eval_overlap() {
        arma::mat S(basp->Nbf(),basp->Nbf());
        S.zeros();

#ifdef _OPENMP
#pragma omp parallel
#endif
        {
          DFTGridWorker grid(basp);
          grid.set_grad_tau_lapl(false,false,false);

#ifdef _OPENMP
#pragma omp for
#endif
          for(size_t iel=0;iel<basp->get_rad_Nel();iel+=2) {
            grid.compute_bf(iel);
            grid.eval_overlap(S);
          }
#ifdef _OPENMP
#pragma omp for
#endif
          for(size_t iel=1;iel<basp->get_rad_Nel();iel+=2) {
            grid.compute_bf(iel);
            grid.eval_overlap(S);
          }
        }

        return S;
      }
    }
  }
}
