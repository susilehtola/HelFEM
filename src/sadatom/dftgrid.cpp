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

      void DFTGridWorker::update_density(const arma::cube & Pc0) {
        // In-element cube
        arma::cube Pc(bf_ind.n_elem, bf_ind.n_elem, Pc0.n_slices);
        for(size_t islice=0; islice<Pc0.n_slices;islice++) {
          Pc.slice(islice) = Pc0.slice(islice)(bf_ind, bf_ind);
        }
        // Total density matrix
        arma::mat P(bf_ind.n_elem, bf_ind.n_elem, arma::fill::zeros);
        for(size_t islice=0; islice<Pc.n_slices;islice++) {
          P += Pc.slice(islice);
        }
        // and the one multiplied by l(l+1)
        arma::mat Pl(bf_ind.n_elem, bf_ind.n_elem, arma::fill::zeros);
        for(size_t islice=0; islice<Pc.n_slices;islice++) {
          Pl += islice*(islice+1)*Pc.slice(islice);
        }

        // Non-polarized calculation.
        polarized=false;

        // Update density vector
        Pv=P*arma::conj(bf);

        // Calculate density
        rho.zeros(1,wtot.n_elem);
        for(size_t ip=0;ip<wtot.n_elem;ip++)
          rho(0,ip)=arma::dot(Pv.col(ip),bf.col(ip));

        // Calculate gradient
        if(do_grad) {
          grho.zeros(1,wtot.n_elem);
          sigma.zeros(1,wtot.n_elem);
          for(size_t ip=0;ip<wtot.n_elem;ip++) {
            // Calculate values
            double g_rad=grho(0,ip)=2.0*arma::dot(Pv.col(ip),bf_rho.col(ip));
            // Compute sigma as well
            sigma(0,ip)=g_rad*g_rad;
          }
        }

        // Calculate kinetic energy density
        if(do_tau || do_lapl) {
          arma::mat Pvp(P*arma::conj(bf_rho));

          if(do_tau) {
            arma::mat Plv(Pl*arma::conj(bf));
            tau.zeros(1,wtot.n_elem);
            for(size_t ip=0;ip<wtot.n_elem;ip++) {
              // First term: P(u,v) * \chi_u' \chi_v'
              double term1 = arma::dot(Pvp.col(ip), bf_rho.col(ip));
              // Second term: l(l+1) Pl(u,v) \chi_u \chi_v / r^2
              double term2 = arma::dot(Plv.col(ip), bf.col(ip))/(r(ip)*r(ip));
              // The second term is ill-behaved near the nucleus since
              // only s orbitals contribute to density but that gets
              // killed by the l(l+1) factor
              tau(0,ip) = 0.5*(term1 + std::max(term2, 0.0));
            }
          }

          if(do_lapl) {
            lapl.zeros(1,wtot.n_elem);
            for(size_t ip=0;ip<wtot.n_elem;ip++) {
              // First term: P(u,v) * \chi_u' \chi_v'
              double term1 = 2.0*arma::dot(Pvp.col(ip), bf_rho.col(ip));
              // Second term: P(u,v) \chi_u \chi_v''
              double term2 = 2.0*arma::dot(Pv.col(ip), bf_rho2.col(ip));
              // Third term: P(u,v) * \chi_u \chi_v' / r
              double term3 = 4.0*arma::dot(Pv.col(ip), bf_rho.col(ip))/r(ip);

              // Store values
              lapl(0,ip)=term1+term2+term3;
            }
          }
        }
      }

      void DFTGridWorker::update_density(const arma::cube & Pac0, const arma::cube & Pbc0) {
        if(!Pac0.n_elem || !Pbc0.n_elem) {
          throw std::runtime_error("Error - density matrix is empty!\n");
        }

        // In-element cube
        arma::cube Pac(bf_ind.n_elem, bf_ind.n_elem, Pac0.n_slices);
        for(size_t islice=0; islice<Pac0.n_slices;islice++) {
          Pac.slice(islice) = Pac0.slice(islice)(bf_ind, bf_ind);
        }
        arma::cube Pbc(bf_ind.n_elem, bf_ind.n_elem, Pbc0.n_slices);
        for(size_t islice=0; islice<Pbc0.n_slices;islice++) {
          Pbc.slice(islice) = Pbc0.slice(islice)(bf_ind, bf_ind);
        }
        // Total density matrix
        arma::mat Pa(bf_ind.n_elem, bf_ind.n_elem, arma::fill::zeros);
        for(size_t islice=0; islice<Pac.n_slices;islice++) {
          Pa += Pac.slice(islice);
        }
        arma::mat Pb(bf_ind.n_elem, bf_ind.n_elem, arma::fill::zeros);
        for(size_t islice=0; islice<Pbc.n_slices;islice++) {
          Pb += Pbc.slice(islice);
        }
        // and the one multiplied by l(l+1)
        arma::mat Pal(bf_ind.n_elem, bf_ind.n_elem, arma::fill::zeros);
        for(size_t islice=0; islice<Pac.n_slices;islice++) {
          Pal += islice*(islice+1)*Pac.slice(islice);
        }
        arma::mat Pbl(bf_ind.n_elem, bf_ind.n_elem, arma::fill::zeros);
        for(size_t islice=0; islice<Pbc.n_slices;islice++) {
          Pbl += islice*(islice+1)*Pbc.slice(islice);
        }

        // Polarized calculation.
        polarized=true;

        // Update density vector
        Pav=Pa*arma::conj(bf);
        Pbv=Pb*arma::conj(bf);

        // Calculate density
        rho.zeros(2,wtot.n_elem);
        for(size_t ip=0;ip<wtot.n_elem;ip++) {
          rho(0,ip)=arma::dot(Pav.col(ip),bf.col(ip));
          rho(1,ip)=arma::dot(Pbv.col(ip),bf.col(ip));
        }

        // Calculate gradient
        if(do_grad) {
          grho.zeros(2,wtot.n_elem);
          sigma.zeros(3,wtot.n_elem);
          for(size_t ip=0;ip<wtot.n_elem;ip++) {
            double ga_rad=grho(0,ip)=2.0*arma::dot(Pav.col(ip),bf_rho.col(ip));
            double gb_rad=grho(1,ip)=2.0*arma::dot(Pbv.col(ip),bf_rho.col(ip));

            // Compute sigma as well
            sigma(0,ip)=ga_rad*ga_rad;
            sigma(1,ip)=ga_rad*gb_rad;
            sigma(2,ip)=gb_rad*gb_rad;
          }
        }

        // Calculate kinetic energy density
        if(do_tau || do_lapl) {
          arma::mat Pavp(Pa*arma::conj(bf_rho));
          arma::mat Pbvp(Pb*arma::conj(bf_rho));

          if(do_tau) {
            arma::mat Palv(Pal*arma::conj(bf));
            arma::mat Pblv(Pbl*arma::conj(bf));
            tau.zeros(2,wtot.n_elem);
            for(size_t ip=0;ip<wtot.n_elem;ip++) {
              // First term: P(u,v) * \chi_u' \chi_v'
              double term1a = arma::dot(Pavp.col(ip), bf_rho.col(ip));
              double term1b = arma::dot(Pbvp.col(ip), bf_rho.col(ip));
              // Second term: l(l+1) Pl(u,v) \chi_u \chi_v / r^2
              double term2a = arma::dot(Palv.col(ip), bf.col(ip))/(r(ip)*r(ip));
              double term2b = arma::dot(Pblv.col(ip), bf.col(ip))/(r(ip)*r(ip));
              // The second term is ill-behaved near the nucleus since
              // only s orbitals contribute to density but that gets
              // killed by the l(l+1) factor
              tau(0,ip) = 0.5*(term1a + std::max(term2a, 0.0));
              tau(1,ip) = 0.5*(term1b + std::max(term2b, 0.0));
            }
          }

          if(do_lapl) {
            lapl.zeros(2,wtot.n_elem);
            for(size_t ip=0;ip<wtot.n_elem;ip++) {
              // First term: P(u,v) * \chi_u' \chi_v'
              double term1a = 2.0*arma::dot(Pavp.col(ip), bf_rho.col(ip));
              double term1b = 2.0*arma::dot(Pbvp.col(ip), bf_rho.col(ip));
              // Second term: P(u,v) \chi_u \chi_v''
              double term2a = 2.0*arma::dot(Pav.col(ip), bf_rho2.col(ip));
              double term2b = 2.0*arma::dot(Pbv.col(ip), bf_rho2.col(ip));
              // Third term: P(u,v) * \chi_u \chi_v' / r
              double term3a = 4.0*arma::dot(Pav.col(ip), bf_rho.col(ip))/r(ip);
              double term3b = 4.0*arma::dot(Pbv.col(ip), bf_rho.col(ip))/r(ip);

              // Store values
              lapl(0,ip)=term1a+term2a+term3a;
              lapl(1,ip)=term1b+term2b+term3b;
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

      double DFTGridWorker::compute_tau() const {
        double t=0.0;
        if(!polarized) {
          for(size_t ip=0;ip<tau.n_cols;ip++)
            t+=wtot(ip)*tau(0,ip);
        } else {
          for(size_t ip=0;ip<tau.n_cols;ip++)
            t+=wtot(ip)*(tau(0,ip)+tau(1,ip));
        }

        return t;
      }

      double DFTGridWorker::compute_lapl() const {
        double l=0.0;
        if(!polarized) {
          for(size_t ip=0;ip<lapl.n_cols;ip++)
            l+=wtot(ip)*lapl(0,ip);
        } else {
          for(size_t ip=0;ip<lapl.n_cols;ip++)
            l+=wtot(ip)*(lapl(0,ip)+lapl(1,ip));
        }

        return l;
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

      void DFTGridWorker::compute_xc(int func_id, const arma::vec & p, double thr, bool pot) {
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
        // Set density threshold
        xc_func_set_dens_threshold(&func, thr);

        // Set parameters
        if(p.n_elem) {
          // Check sanity
          if(p.n_elem != (arma::uword) xc_func_info_get_n_ext_params(func.info))
            throw std::logic_error("Incompatible number of parameters!\n");
          arma::vec phlp(p);
          xc_func_set_ext_params(&func, phlp.memptr());
        }

        // Evaluate functionals.
        if(has_exc(func_id)) {
          if(pot) {
            if(mgga_t || mgga_l) {// meta-GGA
              double * laplp = mgga_l ? lapl.memptr() : NULL;
              double * taup = mgga_t ? tau.memptr() : NULL;
              double * vlaplp = mgga_l ? vlapl_wrk.memptr() : NULL;
              double * vtaup = mgga_t ? vtau_wrk.memptr() : NULL;
              xc_mgga_exc_vxc(&func, N, rho.memptr(), sigma.memptr(), laplp, taup, exc_wrk.memptr(), vxc_wrk.memptr(), vsigma_wrk.memptr(), vlaplp, vtaup);
            } else if(gga) // GGA
              xc_gga_exc_vxc(&func, N, rho.memptr(), sigma.memptr(), exc_wrk.memptr(), vxc_wrk.memptr(), vsigma_wrk.memptr());
            else // LDA
              xc_lda_exc_vxc(&func, N, rho.memptr(), exc_wrk.memptr(), vxc_wrk.memptr());
          } else {
            if(mgga_t || mgga_l) { // meta-GGA
              double * laplp = mgga_l ? lapl.memptr() : NULL;
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
              double * laplp = mgga_l ? lapl.memptr() : NULL;
              double * taup = mgga_t ? tau.memptr() : NULL;
              double * vlaplp = mgga_l ? vlapl_wrk.memptr() : NULL;
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

      void DFTGridWorker::eval_Fxc(arma::cube & Ho) const {
        if(polarized) {
          throw std::runtime_error("Refusing to compute restricted Fock matrix with unrestricted density.\n");
        }

        // Work matrix
        arma::mat H(bf_ind.n_elem,bf_ind.n_elem);
        H.zeros();

        // l-dependent term
        arma::mat Hl(bf_ind.n_elem,bf_ind.n_elem);
        Hl.zeros();

        {
          // LDA potential
          arma::rowvec vrho(vxc.row(0));
          // Multiply weights into potential
          vrho%=wtot;
          // Increment matrix
          increment_lda<double>(H,vrho,bf);
        }
        if(H.has_nan())
          fprintf(stderr,"NaN in Hamiltonian after LDA!\n");

        if(do_gga) {
          // Get vsigma
          arma::rowvec vs(vsigma.row(0));
          // Get grad rho
          arma::uvec idx(arma::linspace<arma::uvec>(0,0,1));
          arma::mat gr(arma::trans(grho.rows(idx)));
          // Multiply grad rho by vsigma and the weights
          gr.col(0)%=2.0*(wtot%vs).t();
          // If we also have laplacian dependence, we get an extra term
          if(do_mgga_l) {
            gr.col(0)+=(2.0*vlapl.row(0)%r%(wrad*4.0*M_PI)).t();
          }
          // Increment matrix
          increment_gga<double>(H,gr,bf,bf_rho);
          if(H.has_nan())
            fprintf(stderr,"NaN in Hamiltonian after GGA!\n");
        }

        if(do_mgga_t || do_mgga_l) {
          arma::rowvec vtl(wtot.n_elem, arma::fill::zeros);
          if(do_mgga_t)
            vtl+=0.5*vtau.row(0);
          if(do_mgga_l)
            vtl+=2.0*vlapl.row(0);
          vtl%=wtot;
          // Base term
          increment_lda<double>(H,vtl,bf_rho);

          if(do_mgga_t) {
            // l(l+1) term: r^-2 cancels out the factor in the total weight
            vtl=vtau.row(0)%(0.5*wrad*4.0*M_PI);
            increment_lda<double>(Hl,vtl,bf);
          }
          if(do_mgga_l) {
            // Laplacian term
            vtl=vlapl.row(0)%wtot;
            increment_mgga_lapl<double>(H,vtl,bf,bf_rho2);
          }
          if(H.has_nan())
            fprintf(stderr,"NaN in Hamiltonian after mGGA!\n");
        }

        // Collect results
        for(size_t islice=0;islice<Ho.n_slices;islice++) {
          Ho.slice(islice)(bf_ind,bf_ind) += H + islice*(islice+1)*Hl;
        }
      }

      void DFTGridWorker::eval_Fxc(arma::cube & Hao, arma::cube & Hbo, bool beta) const {
        if(!polarized) {
          throw std::runtime_error("Refusing to compute unrestricted Fock matrix with restricted density.\n");
        }

        arma::mat Ha, Hb;
        Ha.zeros(bf_ind.n_elem,bf_ind.n_elem);
        if(beta)
          Hb.zeros(bf_ind.n_elem,bf_ind.n_elem);

        arma::mat Hal, Hbl;
        Hal.zeros(bf_ind.n_elem,bf_ind.n_elem);
        if(beta)
          Hbl.zeros(bf_ind.n_elem,bf_ind.n_elem);

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
          fprintf(stderr,"NaN in Hamiltonian after LDA!\n");

        if(do_gga) {
          // Get vsigma
          arma::rowvec vs_aa(vsigma.row(0));
          arma::rowvec vs_ab(vsigma.row(1));

          // Get grad rho
          arma::uvec idxa(arma::linspace<arma::uvec>(0,0,1));
          arma::uvec idxb(arma::linspace<arma::uvec>(1,1,1));
          arma::mat gr_a0(arma::trans(grho.rows(idxa)));
          arma::mat gr_b0(arma::trans(grho.rows(idxb)));

          // Multiply grad rho by vsigma and the weights
          arma::mat gr_a(gr_a0);
          gr_a.col(0)=(wtot%(2.0*vs_aa%gr_a0.col(0).t() + vs_ab%gr_b0.col(0).t())).t();
          // If we also have laplacian dependence, we get an extra term
          if(do_mgga_l) {
            gr_a.col(0)+=2.0*(vlapl.row(0)%r%(wrad*4.0*M_PI)).t();
          }
          // Increment matrix
          increment_gga<double>(Ha,gr_a,bf,bf_rho);

          if(beta) {
            arma::rowvec vs_bb(vsigma.row(2));
            arma::mat gr_b(gr_b0);
            gr_b.col(0)=(wtot%(2.0*vs_bb%gr_b0.col(0).t() + vs_ab%gr_a0.col(0).t())).t();
            if(do_mgga_l) {
              gr_b.col(0)+=2.0*(vlapl.row(1)%r%(wrad*4.0*M_PI)).t();
            }
            increment_gga<double>(Hb,gr_b,bf,bf_rho);
          }
          if(Ha.has_nan() || (beta && Hb.has_nan()))
            //throw std::logic_error("NaN encountered!\n");
            fprintf(stderr,"NaN in Hamiltonian after GGA!\n");
        }

        if(do_mgga_t || do_mgga_l) {
          arma::rowvec vtl_a(wtot.n_elem, arma::fill::zeros);
          if(do_mgga_t)
            vtl_a += 0.5*vtau.row(0);
          if(do_mgga_l)
            vtl_a += 2.0*vlapl.row(0);
          vtl_a %= wtot;

          // Base term
          increment_lda<double>(Ha,vtl_a,bf_rho);

          if(do_mgga_t) {
            // l(l+1) term: r^-2 cancels out the factor in the total weight
            vtl_a=vtau.row(0)%(0.5*wrad*4.0*M_PI);
            increment_lda<double>(Hal,vtl_a,bf);
          }
          if(do_mgga_l) {
            vtl_a=vlapl.row(0)%wtot;
            increment_mgga_lapl<double>(Ha,vtl_a,bf,bf_rho2);
          }
          if(beta) {
            arma::rowvec vtl_b(wtot.n_elem, arma::fill::zeros);
            if(do_mgga_t)
              vtl_b += 0.5*vtau.row(1);
            if(do_mgga_l)
              vtl_b += 2.0*vlapl.row(1);
            vtl_b %= wtot;

            // Base term
            increment_lda<double>(Hb,vtl_b,bf_rho);

            if(do_mgga_t) {
              // l(l+1) term: r^-2 cancels out the factor in the total weight
              vtl_b=vtau.row(1)%(0.5*wrad*4.0*M_PI);
              increment_lda<double>(Hbl,vtl_b,bf);
            }
            if(do_mgga_l) {
              vtl_b=vlapl.row(1)%wtot;
              increment_mgga_lapl<double>(Hb,vtl_b,bf,bf_rho2);
            }
          }
          if(Ha.has_nan() || (beta && Hb.has_nan()))
            //throw std::logic_error("NaN encountered!\n");
            fprintf(stderr,"NaN in Hamiltonian after mGGA!\n");
        }

        // Collect results
        for(size_t islice=0;islice<Hao.n_slices;islice++) {
          Hao.slice(islice)(bf_ind,bf_ind) += Ha + islice*(islice+1)*Hal;
        }
        if(beta) {
          for(size_t islice=0;islice<Hbo.n_slices;islice++) {
            Hbo.slice(islice)(bf_ind,bf_ind) += Hb + islice*(islice+1)*Hbl;
          }
        }
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
        // Get radii
        r=basp->get_r(iel).t();
        // Get radial weights
        wrad=basp->get_wrad(iel).t();

        // Update total weights
        wtot = 4.0*M_PI*wrad%arma::square(r);

        // Compute basis function values
        bf=arma::trans(basp->eval_bf(iel));

        if(do_grad) {
          bf_rho=arma::trans(basp->eval_df(iel));
        }

        if(do_lapl) {
          bf_rho2=arma::trans(basp->eval_lf(iel));
        }
      }

      DFTGrid::DFTGrid() {
      }

      DFTGrid::DFTGrid(const helfem::sadatom::basis::TwoDBasis * basp_) : basp(basp_) {
      }

      DFTGrid::~DFTGrid() {
      }

      void DFTGrid::eval_Fxc(int x_func, const arma::vec & x_pars, int c_func, const arma::vec & c_pars, const arma::cube & P, arma::cube & H, double & Exc, double & Nel, double thr) {
        H.zeros(P.n_rows,P.n_rows,P.n_slices);

        double exc=0.0;
        double nel=0.0;
        double tau=0.0;
        double lapl=0.0;

#ifdef _OPENMP
#pragma omp parallel reduction(+:exc,nel,tau,lapl)
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
            tau+=grid.compute_tau();
            lapl+=grid.compute_lapl();

            grid.init_xc();
            if(x_func>0)
              grid.compute_xc(x_func,x_pars,thr);
            if(c_func>0)
              grid.compute_xc(c_func,c_pars,thr);

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
            tau+=grid.compute_tau();
            lapl+=grid.compute_lapl();

            grid.init_xc();
            if(x_func>0)
              grid.compute_xc(x_func,x_pars,thr);
            if(c_func>0)
              grid.compute_xc(c_func,c_pars,thr);

            exc+=grid.eval_Exc();
            grid.eval_Fxc(H);
          }
        }

        // Save outputs
        Exc=exc;
        Nel=nel;

#if 0
        if(tau!=0.0)
          printf("Tau integral %.10e\n",tau);
        if(lapl!=0.0)
          printf("Laplacian integral %.10e\n",lapl);
#endif
      }

      void DFTGrid::eval_Fxc(int x_func, const arma::vec & x_pars, int c_func, const arma::vec & c_pars, const arma::cube & Pa, const arma::cube & Pb, arma::cube & Ha, arma::cube & Hb, double & Exc, double & Nel, bool beta, double thr) {
        Ha.zeros(Pa.n_rows,Pa.n_rows,Pa.n_slices);
        Hb.zeros(Pb.n_rows,Pb.n_rows,Pb.n_slices);

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
            if(x_func>0)
              grid.compute_xc(x_func,x_pars,thr);
            if(c_func>0)
              grid.compute_xc(c_func,c_pars,thr);

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
            if(x_func>0)
              grid.compute_xc(x_func,x_pars,thr);
            if(c_func>0)
              grid.compute_xc(c_func,c_pars,thr);

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
