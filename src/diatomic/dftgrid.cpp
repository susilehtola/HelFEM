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
#include <helfem.h>
#include <cmath>
#include <cstdio>
// LibXC
#include <xc.h>

#include "dftgrid.h"
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

        // Get angular grid (angular_chebyshev is Eigen-typed).
        helfem::angular::angular_chebyshev(lang,mang,cth,phi,wang);
      }

      DFTGridWorker::~DFTGridWorker() {
      }

      void DFTGridWorker::update_density(const helfem::Matrix & Pexp) {
        // Update values of density. Pexp is the density already expanded
        // to the dummy (Ndummy) basis: the expansion is the same at every
        // grid point, so eval_Fxc does it once outside the loop rather
        // than allocating and zeroing an Ndummy x Ndummy matrix here for
        // every (element, radial point) pair.
        if(!Pexp.size()) {
          throw std::runtime_error("Error - density matrix is empty!\n");
        }
        helfem::Matrix P(bf_ind.size(), bf_ind.size());
        for(size_t i=0;i<bf_ind.size();i++)
          for(size_t j=0;j<bf_ind.size();j++)
            P(i,j)=Pexp(bf_ind[i],bf_ind[j]);

        // Non-polarized calculation.
        polarized=false;

        // Update density vector
        PvA.noalias()=P*bf_re;  PvB.noalias()=P*bf_im;

        // Calculate density
        rho = helfem::Matrix::Zero(1,wtot.size());
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(size_t ip=0;ip<(size_t) wtot.size();ip++)
          rho(0,ip)=(PvA.col(ip).dot(bf_re.col(ip))+PvB.col(ip).dot(bf_im.col(ip)));

        // Calculate gradient
        if(do_grad) {
          grho = helfem::Matrix::Zero(3,wtot.size());
          sigma = helfem::Matrix::Zero(1,wtot.size());
#ifdef _OPENMP
#pragma omp parallel for
#endif
          for(size_t ip=0;ip<(size_t) wtot.size();ip++) {
            // Calculate values
            double g_rad=grho(0,ip)=2.0*(PvA.col(ip).dot(bf_rho_re.col(ip))+PvB.col(ip).dot(bf_rho_im.col(ip)))/scale_r(ip);
            double g_th=grho(1,ip)=2.0*(PvA.col(ip).dot(bf_theta_re.col(ip))+PvB.col(ip).dot(bf_theta_im.col(ip)))/scale_theta(ip);
            double g_phi=grho(2,ip)=2.0*(PvA.col(ip).dot(bf_phi_re.col(ip))+PvB.col(ip).dot(bf_phi_im.col(ip)))/scale_phi(ip);
            // Compute sigma as well
            sigma(0,ip)=g_rad*g_rad + g_th*g_th + g_phi*g_phi;
          }
        }

        // Calculate laplacian and kinetic energy density
        if(do_tau) {
          // Adjust size of grid
          tau = helfem::Matrix::Zero(1,wtot.size());

          // Update helpers
          PvA_rho.noalias()=P*bf_rho_re;  PvB_rho.noalias()=P*bf_rho_im;
          PvA_theta.noalias()=P*bf_theta_re;  PvB_theta.noalias()=P*bf_theta_im;
          PvA_phi.noalias()=P*bf_phi_re;  PvB_phi.noalias()=P*bf_phi_im;

          // Calculate values
#ifdef _OPENMP
#pragma omp parallel for
#endif
          for(size_t ip=0;ip<(size_t) wtot.size();ip++) {
            // Gradient term
            double kinrho((PvA_rho.col(ip).dot(bf_rho_re.col(ip))+PvB_rho.col(ip).dot(bf_rho_im.col(ip)))/std::pow(scale_r(ip),2));
            double kintheta((PvA_theta.col(ip).dot(bf_theta_re.col(ip))+PvB_theta.col(ip).dot(bf_theta_im.col(ip)))/std::pow(scale_theta(ip),2));
            double kinphi((PvA_phi.col(ip).dot(bf_phi_re.col(ip))+PvB_phi.col(ip).dot(bf_phi_im.col(ip)))/std::pow(scale_phi(ip),2));
            double kin(kinrho + kintheta + kinphi);

            // Store values
            tau(0,ip)=0.5*kin;
          }
        }

        if(do_lapl)
          throw std::logic_error("Laplacian not implemented!\n");
      }

      void DFTGridWorker::update_density(const helfem::Matrix & Paexp, const helfem::Matrix & Pbexp) {
        // Both densities arrive already expanded to the dummy basis; see
        // the restricted overload above for why.
        if(!Paexp.size() || !Pbexp.size()) {
          throw std::runtime_error("Error - density matrix is empty!\n");
        }

        // Polarized calculation.
        polarized=true;

        // Update density vector.
        helfem::Matrix Pa(bf_ind.size(), bf_ind.size());
        helfem::Matrix Pb(bf_ind.size(), bf_ind.size());
        for(size_t i=0;i<bf_ind.size();i++)
          for(size_t j=0;j<bf_ind.size();j++) {
            Pa(i,j)=Paexp(bf_ind[i],bf_ind[j]);
            Pb(i,j)=Pbexp(bf_ind[i],bf_ind[j]);
          }

        PavA.noalias()=Pa*bf_re;  PavB.noalias()=Pa*bf_im;
        PbvA.noalias()=Pb*bf_re;  PbvB.noalias()=Pb*bf_im;

        // Calculate density
        rho = helfem::Matrix::Zero(2,wtot.size());
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(size_t ip=0;ip<(size_t) wtot.size();ip++) {
          rho(0,ip)=(PavA.col(ip).dot(bf_re.col(ip))+PavB.col(ip).dot(bf_im.col(ip)));
          rho(1,ip)=(PbvA.col(ip).dot(bf_re.col(ip))+PbvB.col(ip).dot(bf_im.col(ip)));

          /*
            double na=compute_density(Pa0,*basp,grid[ip].r);
            double nb=compute_density(Pb0,*basp,grid[ip].r);
            if(fabs(da-na)>1e-6 || fabs(db-nb)>1e-6)
            printf("Density at point % .3f % .3f % .3f: %e vs %e, %e vs %e\n",grid[ip].r.x,grid[ip].r.y,grid[ip].r.z,da,na,db,nb);
          */
        }

        // Calculate gradient

        if(do_grad) {
          grho = helfem::Matrix::Zero(6,wtot.size());
          sigma = helfem::Matrix::Zero(3,wtot.size());
#ifdef _OPENMP
#pragma omp parallel for
#endif
          for(size_t ip=0;ip<(size_t) wtot.size();ip++) {
            double ga_rad=grho(0,ip)=2.0*(PavA.col(ip).dot(bf_rho_re.col(ip))+PavB.col(ip).dot(bf_rho_im.col(ip)))/scale_r(ip);
            double ga_th=grho(1,ip)=2.0*(PavA.col(ip).dot(bf_theta_re.col(ip))+PavB.col(ip).dot(bf_theta_im.col(ip)))/scale_theta(ip);
            double ga_phi=grho(2,ip)=2.0*(PavA.col(ip).dot(bf_phi_re.col(ip))+PavB.col(ip).dot(bf_phi_im.col(ip)))/scale_phi(ip);

            double gb_rad=grho(3,ip)=2.0*(PbvA.col(ip).dot(bf_rho_re.col(ip))+PbvB.col(ip).dot(bf_rho_im.col(ip)))/scale_r(ip);
            double gb_th=grho(4,ip)=2.0*(PbvA.col(ip).dot(bf_theta_re.col(ip))+PbvB.col(ip).dot(bf_theta_im.col(ip)))/scale_theta(ip);
            double gb_phi=grho(5,ip)=2.0*(PbvA.col(ip).dot(bf_phi_re.col(ip))+PbvB.col(ip).dot(bf_phi_im.col(ip)))/scale_phi(ip);

            // Compute sigma as well
            sigma(0,ip)=ga_rad*ga_rad + ga_th*ga_th + ga_phi*ga_phi;
            sigma(1,ip)=ga_rad*gb_rad + ga_th*gb_th + ga_phi*gb_phi;
            sigma(2,ip)=gb_rad*gb_rad + gb_th*gb_th + gb_phi*gb_phi;
          }
        }

        // Calculate kinetic energy density
        if(do_tau) {
          // Adjust size of grid
          tau.resize(2,wtot.size());

          // Update helpers
          PavA_rho.noalias()=Pa*bf_rho_re;  PavB_rho.noalias()=Pa*bf_rho_im;
          PavA_theta.noalias()=Pa*bf_theta_re;  PavB_theta.noalias()=Pa*bf_theta_im;
          PavA_phi.noalias()=Pa*bf_phi_re;  PavB_phi.noalias()=Pa*bf_phi_im;

          PbvA_rho.noalias()=Pb*bf_rho_re;  PbvB_rho.noalias()=Pb*bf_rho_im;
          PbvA_theta.noalias()=Pb*bf_theta_re;  PbvB_theta.noalias()=Pb*bf_theta_im;
          PbvA_phi.noalias()=Pb*bf_phi_re;  PbvB_phi.noalias()=Pb*bf_phi_im;

          // Calculate values
#ifdef _OPENMP
#pragma omp parallel for
#endif
          for(size_t ip=0;ip<(size_t) wtot.size();ip++) {
            // Gradient term
            double kinar=(PavA_rho.col(ip).dot(bf_rho_re.col(ip))+PavB_rho.col(ip).dot(bf_rho_im.col(ip)))/std::pow(scale_r(ip),2);
            double kinath=(PavA_theta.col(ip).dot(bf_theta_re.col(ip))+PavB_theta.col(ip).dot(bf_theta_im.col(ip)))/std::pow(scale_theta(ip),2);
            double kinaphi=(PavA_phi.col(ip).dot(bf_phi_re.col(ip))+PavB_phi.col(ip).dot(bf_phi_im.col(ip)))/std::pow(scale_phi(ip),2);
            double kina(kinar + kinath + kinaphi);

            double kinbr=(PbvA_rho.col(ip).dot(bf_rho_re.col(ip))+PbvB_rho.col(ip).dot(bf_rho_im.col(ip)))/std::pow(scale_r(ip),2);
            double kinbth=(PbvA_theta.col(ip).dot(bf_theta_re.col(ip))+PbvB_theta.col(ip).dot(bf_theta_im.col(ip)))/std::pow(scale_theta(ip),2);
            double kinbphi=(PbvA_phi.col(ip).dot(bf_phi_re.col(ip))+PbvB_phi.col(ip).dot(bf_phi_im.col(ip)))/std::pow(scale_phi(ip),2);
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
            for(size_t ip=0;ip<(size_t) wtot.size();ip++)
              ekin+=wtot(ip)*tau(0,ip);
          } else {
            for(size_t ip=0;ip<(size_t) wtot.size();ip++)
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

      void DFTGridWorker::eval_Fxc(helfem::Matrix & Ho) const {
        if(polarized) {
          throw std::runtime_error("Refusing to compute restricted Fock matrix with unrestricted density.\n");
        }

        // Work matrix
        helfem::Matrix H = helfem::Matrix::Zero(bf_ind.size(),bf_ind.size());

        {
          // LDA potential
          helfem::Vector vrho = vxc.row(0).transpose();
          // Multiply weights into potential
          vrho = vrho.array() * wtot.array();
          // Increment matrix
          helfem::dftgrid_common::increment_lda_split(H,vrho,bf_re,bf_im);
        }

        if(do_gga) {
          // vgrad is the vector coefficient of the basis-function
          // gradient pair, assembled either from vsigma and the density
          // gradient (build_vgrad, ground state) or from the kernel
          // chain rule (set_response_potential). Reading it here rather
          // than rebuilding 2 vsigma grad(rho) inline is what lets the
          // response reuse this assembly: its coefficient is a general
          // vector field, not a multiple of grad(rho).
          helfem::Matrix gr = vgrad.topRows(3).transpose();
          for(Eigen::Index i=0;i<gr.rows();i++) {
            gr(i,0)*=wtot(i)/scale_r(i);
            gr(i,1)*=wtot(i)/scale_theta(i);
            gr(i,2)*=wtot(i)/scale_phi(i);
          }
          // Increment matrix
          increment_gga_split(H,gr,bf_re,bf_im,{&bf_rho_re,&bf_theta_re,&bf_phi_re},{&bf_rho_im,&bf_theta_im,&bf_phi_im});
        }

        if(do_mgga_t) {
          helfem::Vector vt = vtau.row(0).transpose();
          vt = vt.array() * wtot.array() * 0.5;

          helfem::dftgrid_common::increment_lda_split(H,helfem::Vector(vt.array()*inv_scale_r2.array()),bf_rho_re,bf_rho_im);
          helfem::dftgrid_common::increment_lda_split(H,helfem::Vector(vt.array()*inv_scale_theta2.array()),bf_theta_re,bf_theta_im);
          helfem::dftgrid_common::increment_lda_split(H,helfem::Vector(vt.array()*inv_scale_phi2.array()),bf_phi_re,bf_phi_im);
        }
        if(do_mgga_l)
          throw std::logic_error("Laplacian not implemented!\n");

        for(size_t i=0;i<bf_ind.size();i++)
          for(size_t j=0;j<bf_ind.size();j++)
            Ho(bf_ind[i],bf_ind[j])+=H(i,j);
      }

      void DFTGridWorker::eval_Fxc(helfem::Matrix & Hao, helfem::Matrix & Hbo, bool beta) const {
        if(!polarized) {
          throw std::runtime_error("Refusing to compute unrestricted Fock matrix with restricted density.\n");
        }

        helfem::Matrix Ha = helfem::Matrix::Zero(bf_ind.size(),bf_ind.size());
        helfem::Matrix Hb;
        if(beta)
          Hb = helfem::Matrix::Zero(bf_ind.size(),bf_ind.size());

        {
          // LDA potential
          helfem::Vector vrhoa = vxc.row(0).transpose();
          // Multiply weights into potential
          vrhoa = vrhoa.array() * wtot.array();
          // Increment matrix
          helfem::dftgrid_common::increment_lda_split(Ha,vrhoa,bf_re,bf_im);

          if(beta) {
            helfem::Vector vrhob = vxc.row(1).transpose();
            vrhob = vrhob.array() * wtot.array();
            helfem::dftgrid_common::increment_lda_split(Hb,vrhob,bf_re,bf_im);
          }
        }
        if(!Ha.allFinite() || (beta && !Hb.allFinite()))
          //throw std::logic_error("NaN encountered!\n");
          fprintf(stderr,"NaN in Hamiltonian!\n");

        if(do_gga) {
          // See the restricted overload: vgrad already carries the whole
          // gradient coefficient, spin coupling (the vsigma_ab term)
          // included, whether it came from the ground state or from the
          // response kernel.
          helfem::Matrix gr_a = vgrad.topRows(3).transpose();
          for(Eigen::Index i=0;i<gr_a.rows();i++) {
            gr_a(i,0)*=wtot(i)/scale_r(i);
            gr_a(i,1)*=wtot(i)/scale_theta(i);
            gr_a(i,2)*=wtot(i)/scale_phi(i);
          }
          // Increment matrix
          increment_gga_split(Ha,gr_a,bf_re,bf_im,{&bf_rho_re,&bf_theta_re,&bf_phi_re},{&bf_rho_im,&bf_theta_im,&bf_phi_im});

          if(beta) {
            helfem::Matrix gr_b = vgrad.bottomRows(3).transpose();
            for(Eigen::Index i=0;i<gr_b.rows();i++) {
              gr_b(i,0)*=wtot(i)/scale_r(i);
              gr_b(i,1)*=wtot(i)/scale_theta(i);
              gr_b(i,2)*=wtot(i)/scale_phi(i);
            }
            increment_gga_split(Hb,gr_b,bf_re,bf_im,{&bf_rho_re,&bf_theta_re,&bf_phi_re},{&bf_rho_im,&bf_theta_im,&bf_phi_im});
          }
        }


        if(do_mgga_t) {
          helfem::Vector vt_a = vtau.row(0).transpose();
          vt_a = vt_a.array() * wtot.array() * 0.5;

          helfem::dftgrid_common::increment_lda_split(Ha,helfem::Vector(vt_a.array()*inv_scale_r2.array()),bf_rho_re,bf_rho_im);
          helfem::dftgrid_common::increment_lda_split(Ha,helfem::Vector(vt_a.array()*inv_scale_theta2.array()),bf_theta_re,bf_theta_im);
          helfem::dftgrid_common::increment_lda_split(Ha,helfem::Vector(vt_a.array()*inv_scale_phi2.array()),bf_phi_re,bf_phi_im);
          if(beta) {
            helfem::Vector vt_b = vtau.row(1).transpose();
            vt_b = vt_b.array() * wtot.array() * 0.5;

            helfem::dftgrid_common::increment_lda_split(Hb,helfem::Vector(vt_b.array()*inv_scale_r2.array()),bf_rho_re,bf_rho_im);
            helfem::dftgrid_common::increment_lda_split(Hb,helfem::Vector(vt_b.array()*inv_scale_theta2.array()),bf_theta_re,bf_theta_im);
            helfem::dftgrid_common::increment_lda_split(Hb,helfem::Vector(vt_b.array()*inv_scale_phi2.array()),bf_phi_re,bf_phi_im);
          }
        }
        if(do_mgga_l) {
          throw std::logic_error("Laplacian not implemented!\n");
        }

        for(size_t i=0;i<bf_ind.size();i++)
          for(size_t j=0;j<bf_ind.size();j++) {
            Hao(bf_ind[i],bf_ind[j])+=Ha(i,j);
            if(beta)
              Hbo(bf_ind[i],bf_ind[j])+=Hb(i,j);
          }
      }

      // check_grad_tau_lapl, grad_tau_lapl, set_grad_tau_lapl:
      // inherited from DFTGridWorkerBase.

      void DFTGridWorker::compute_bf(size_t iel, size_t irad) {
        // Update function list
        bf_ind=basp->bf_list_dummy(iel);

        // Get radial weights. Only do one radial quadrature point at a
        // time, since this is an easy way to save a lot of memory.
        helfem::Vector wrad(1), r(1);
        wrad(0)=basp->wrad(iel)(irad);
        r(0)=basp->r(iel)(irad);

        double Rhalf(basp->Rhalf());

        // Calculate helpers
        helfem::Vector shmu = r.array().sinh();

        helfem::Vector sth(cth.size());
        for(Eigen::Index ia=0;ia<cth.size();ia++)
          sth(ia)=sqrt(1.0 - cth(ia)*cth(ia));

        const Eigen::Index nwrad=wrad.size();
        const Eigen::Index nwang=wang.size();

        // Radial is
        scale_r.resize(nwrad*nwang);
        for(Eigen::Index ia=0;ia<nwang;ia++)
          for(Eigen::Index ir=0;ir<nwrad;ir++)
            // h_mu = R_{h}\sqrt{\sinh^{2}\mu+\sin^{2}\nu}
            scale_r(ia*nwrad+ir)=Rhalf*sqrt(std::pow(shmu(ir),2) + std::pow(sth(ia),2));
        // Theta is same as radial
        scale_theta=scale_r;
        // phi is simple
        scale_phi.resize(nwrad*nwang);
        for(Eigen::Index ia=0;ia<nwang;ia++)
          for(Eigen::Index ir=0;ir<nwrad;ir++)
            scale_phi(ia*nwrad+ir)=Rhalf*shmu(ir)*sth(ia);
        // Pre-compute 1/scale^2 for the kinetic / mGGA terms.
        inv_scale_r2 = scale_r.array().square().inverse();
        inv_scale_theta2 = scale_theta.array().square().inverse();
        inv_scale_phi2 = scale_phi.array().square().inverse();
        // Update total weights
        wtot = helfem::Vector::Zero(nwrad*nwang);
        for(Eigen::Index ia=0;ia<nwang;ia++)
          for(Eigen::Index ir=0;ir<nwrad;ir++) {
            Eigen::Index idx=ia*nwrad+ir;
            // sin(th) is already contained within wang, but we don't want to divide by it since it may be zero.
            wtot(idx)=wang(ia)*wrad(ir)*std::pow(Rhalf,3)*shmu(ir)*(std::pow(shmu(ir),2)+std::pow(sth(ia),2));
          }

        // Compute basis function values
        bf = Eigen::MatrixXcd::Zero(bf_ind.size(),wtot.size());
        // Loop over angular grid
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(Eigen::Index ia=0;ia<cth.size();ia++) {
          // Evaluate basis functions at angular point (Eigen-native).
          const Eigen::MatrixXcd abf(basp->eval_bf(iel, irad, cth(ia), phi(ia)));
          if((size_t) abf.cols() != bf_ind.size()) {
            std::ostringstream oss;
            oss << "Mismatch! Have " << bf_ind.size() << " basis function indices but " << abf.cols() << " basis functions!\n";
            throw std::logic_error(oss.str());
          }
          // Store functions (arma::trans was the conjugate transpose -> adjoint).
          bf.middleCols(ia*nwrad,nwrad)=abf.adjoint();
        }

        if(do_grad) {
          bf_rho = Eigen::MatrixXcd::Zero(bf_ind.size(),wtot.size());
          bf_theta = Eigen::MatrixXcd::Zero(bf_ind.size(),wtot.size());
          bf_phi = Eigen::MatrixXcd::Zero(bf_ind.size(),wtot.size());

#ifdef _OPENMP
#pragma omp parallel for
#endif
          for(Eigen::Index ia=0;ia<cth.size();ia++) {
            // Evaluate basis functions at angular point (Eigen-native).
            Eigen::MatrixXcd dr, dth, dphi;
            basp->eval_df(iel, irad, cth(ia), phi(ia), dr, dth, dphi);
            if((size_t) dr.cols() != bf_ind.size()) {
              std::ostringstream oss;
              oss << "Mismatch! Have " << bf_ind.size() << " basis function indices but " << dr.cols() << " basis functions!\n";
              throw std::logic_error(oss.str());
            }
            // Store functions (arma::trans was the conjugate transpose -> adjoint).
            bf_rho.middleCols(ia*nwrad,nwrad)=dr.adjoint();
            bf_theta.middleCols(ia*nwrad,nwrad)=dth.adjoint();
            bf_phi.middleCols(ia*nwrad,nwrad)=dphi.adjoint();
          }
        }

        if(do_lapl) {
          throw std::logic_error("Laplacian not implemented.\n");
        }
        // Split once per grid point: the density and Fock contractions
        // below are real. See the atomic worker for the algebra.
        bf_re = bf.real();  bf_im = bf.imag();
        if(do_grad) {
          bf_rho_re   = bf_rho.real();   bf_rho_im   = bf_rho.imag();
          bf_theta_re = bf_theta.real(); bf_theta_im = bf_theta.imag();
          bf_phi_re   = bf_phi.real();   bf_phi_im   = bf_phi.imag();
        }
        if(do_lapl) {
          bf_lapl_re = bf_lapl.real(); bf_lapl_im = bf_lapl.imag();
        }
      }

      DFTGrid::DFTGrid() {
      }

      DFTGrid::DFTGrid(const helfem::diatomic::basis::TwoDBasis * basp_, int lang_, int mang_) : basp(basp_), lang(lang_), mang(mang_) {
        helfem::Vector cth, phi, wang;
        helfem::angular::angular_chebyshev(lang,mang,cth,phi,wang);
        if(helfem::verbose)
          printf("DFT angular grid of order l=%i m=%i has %i points\n",lang,mang,(int) wang.size());
      }

      DFTGrid::~DFTGrid() {
      }

      static inline int helfem_omp_max_threads() {
#ifdef _OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
      }
      static inline int helfem_omp_thread_num() {
#ifdef _OPENMP
        return omp_get_thread_num();
#else
        return 0;
#endif
      }

      /// The (element, radial point) pairs the grid loops over, flattened.
      /// A molecule has only a handful of radial elements, so parallelising
      /// the element loop alone would leave most of a machine idle; one task
      /// per radial point is dozens.
      static std::vector<std::pair<size_t, size_t>>
      grid_tasks(const helfem::diatomic::basis::TwoDBasis *basp) {
        std::vector<std::pair<size_t, size_t>> tasks;
        for (size_t iel = 0; iel < basp->rad_Nel(); iel++)
          for (size_t irad = 0; irad < (size_t) basp->r(iel).size(); irad++)
            tasks.push_back({iel, irad});
        return tasks;
      }

      helfem::Matrix DFTGridWorker::eval_density(const helfem::Matrix & dPexp) const {
        // The same contraction update_density performs, on a different
        // density matrix and without touching any member state.
        helfem::Matrix dP(bf_ind.size(), bf_ind.size());
        for(size_t i=0;i<bf_ind.size();i++)
          for(size_t j=0;j<bf_ind.size();j++)
            dP(i,j)=dPexp(bf_ind[i],bf_ind[j]);

        const helfem::Matrix dPvA(dP*bf_re), dPvB(dP*bf_im);
        helfem::Matrix drho=helfem::Matrix::Zero(1,wtot.size());
        for(Eigen::Index ip=0;ip<wtot.size();ip++)
          drho(0,ip)=dPvA.col(ip).dot(bf_re.col(ip))+dPvB.col(ip).dot(bf_im.col(ip));
        return drho;
      }

      helfem::Matrix DFTGridWorker::eval_density(const helfem::Matrix & dPaexp, const helfem::Matrix & dPbexp) const {
        helfem::Matrix dPa(bf_ind.size(), bf_ind.size()), dPb(bf_ind.size(), bf_ind.size());
        for(size_t i=0;i<bf_ind.size();i++)
          for(size_t j=0;j<bf_ind.size();j++) {
            dPa(i,j)=dPaexp(bf_ind[i],bf_ind[j]);
            dPb(i,j)=dPbexp(bf_ind[i],bf_ind[j]);
          }
        const helfem::Matrix dPavA(dPa*bf_re), dPavB(dPa*bf_im);
        const helfem::Matrix dPbvA(dPb*bf_re), dPbvB(dPb*bf_im);
        helfem::Matrix drho=helfem::Matrix::Zero(2,wtot.size());
        for(Eigen::Index ip=0;ip<wtot.size();ip++) {
          drho(0,ip)=dPavA.col(ip).dot(bf_re.col(ip))+dPavB.col(ip).dot(bf_im.col(ip));
          drho(1,ip)=dPbvA.col(ip).dot(bf_re.col(ip))+dPbvB.col(ip).dot(bf_im.col(ip));
        }
        return drho;
      }

      void DFTGridWorker::eval_response_fields(const helfem::Matrix & dPexp,
                                               helfem::Matrix & drho,
                                               helfem::Matrix & dgrho,
                                               helfem::Matrix & dtau) const {
        // The perturbed fields are the SAME bilinear forms in the density
        // matrix as the reference ones in update_density; only the matrix
        // differs. Keeping the two in step is what makes the response
        // kernel exact, so this mirrors that code deliberately -- the
        // 1/scale_* of every gradient component and the 1/scale_*^2 of
        // every kinetic term included. dPexp arrives expanded to the
        // dummy basis, exactly as update_density's reference does.
        helfem::Matrix dP(bf_ind.size(), bf_ind.size());
        for(size_t i=0;i<bf_ind.size();i++)
          for(size_t j=0;j<bf_ind.size();j++)
            dP(i,j)=dPexp(bf_ind[i],bf_ind[j]);

        const helfem::Matrix dPvA(dP*bf_re), dPvB(dP*bf_im);

        drho=helfem::Matrix::Zero(1,wtot.size());
        for(Eigen::Index ip=0;ip<wtot.size();ip++)
          drho(0,ip)=dPvA.col(ip).dot(bf_re.col(ip))+dPvB.col(ip).dot(bf_im.col(ip));

        if(do_grad) {
          dgrho=helfem::Matrix::Zero(3,wtot.size());
          for(Eigen::Index ip=0;ip<wtot.size();ip++) {
            dgrho(0,ip)=2.0*(dPvA.col(ip).dot(bf_rho_re.col(ip))+dPvB.col(ip).dot(bf_rho_im.col(ip)))/scale_r(ip);
            dgrho(1,ip)=2.0*(dPvA.col(ip).dot(bf_theta_re.col(ip))+dPvB.col(ip).dot(bf_theta_im.col(ip)))/scale_theta(ip);
            dgrho(2,ip)=2.0*(dPvA.col(ip).dot(bf_phi_re.col(ip))+dPvB.col(ip).dot(bf_phi_im.col(ip)))/scale_phi(ip);
          }
        } else {
          dgrho=helfem::Matrix();
        }

        if(do_tau) {
          const helfem::Matrix dPvA_rho(dP*bf_rho_re), dPvB_rho(dP*bf_rho_im);
          const helfem::Matrix dPvA_theta(dP*bf_theta_re), dPvB_theta(dP*bf_theta_im);
          const helfem::Matrix dPvA_phi(dP*bf_phi_re), dPvB_phi(dP*bf_phi_im);
          dtau=helfem::Matrix::Zero(1,wtot.size());
          for(Eigen::Index ip=0;ip<wtot.size();ip++) {
            double kinrho((dPvA_rho.col(ip).dot(bf_rho_re.col(ip))+dPvB_rho.col(ip).dot(bf_rho_im.col(ip)))/std::pow(scale_r(ip),2));
            double kintheta((dPvA_theta.col(ip).dot(bf_theta_re.col(ip))+dPvB_theta.col(ip).dot(bf_theta_im.col(ip)))/std::pow(scale_theta(ip),2));
            double kinphi((dPvA_phi.col(ip).dot(bf_phi_re.col(ip))+dPvB_phi.col(ip).dot(bf_phi_im.col(ip)))/std::pow(scale_phi(ip),2));
            dtau(0,ip)=0.5*(kinrho + kintheta + kinphi);
          }
        } else {
          dtau=helfem::Matrix();
        }
      }

      void DFTGridWorker::eval_response_fields(const helfem::Matrix & dPaexp,
                                               const helfem::Matrix & dPbexp,
                                               helfem::Matrix & drho,
                                               helfem::Matrix & dgrho,
                                               helfem::Matrix & dtau) const {
        helfem::Matrix dPa(bf_ind.size(), bf_ind.size()), dPb(bf_ind.size(), bf_ind.size());
        for(size_t i=0;i<bf_ind.size();i++)
          for(size_t j=0;j<bf_ind.size();j++) {
            dPa(i,j)=dPaexp(bf_ind[i],bf_ind[j]);
            dPb(i,j)=dPbexp(bf_ind[i],bf_ind[j]);
          }

        const helfem::Matrix dPavA(dPa*bf_re), dPavB(dPa*bf_im);
        const helfem::Matrix dPbvA(dPb*bf_re), dPbvB(dPb*bf_im);

        drho=helfem::Matrix::Zero(2,wtot.size());
        for(Eigen::Index ip=0;ip<wtot.size();ip++) {
          drho(0,ip)=dPavA.col(ip).dot(bf_re.col(ip))+dPavB.col(ip).dot(bf_im.col(ip));
          drho(1,ip)=dPbvA.col(ip).dot(bf_re.col(ip))+dPbvB.col(ip).dot(bf_im.col(ip));
        }

        if(do_grad) {
          dgrho=helfem::Matrix::Zero(6,wtot.size());
          for(Eigen::Index ip=0;ip<wtot.size();ip++) {
            dgrho(0,ip)=2.0*(dPavA.col(ip).dot(bf_rho_re.col(ip))+dPavB.col(ip).dot(bf_rho_im.col(ip)))/scale_r(ip);
            dgrho(1,ip)=2.0*(dPavA.col(ip).dot(bf_theta_re.col(ip))+dPavB.col(ip).dot(bf_theta_im.col(ip)))/scale_theta(ip);
            dgrho(2,ip)=2.0*(dPavA.col(ip).dot(bf_phi_re.col(ip))+dPavB.col(ip).dot(bf_phi_im.col(ip)))/scale_phi(ip);

            dgrho(3,ip)=2.0*(dPbvA.col(ip).dot(bf_rho_re.col(ip))+dPbvB.col(ip).dot(bf_rho_im.col(ip)))/scale_r(ip);
            dgrho(4,ip)=2.0*(dPbvA.col(ip).dot(bf_theta_re.col(ip))+dPbvB.col(ip).dot(bf_theta_im.col(ip)))/scale_theta(ip);
            dgrho(5,ip)=2.0*(dPbvA.col(ip).dot(bf_phi_re.col(ip))+dPbvB.col(ip).dot(bf_phi_im.col(ip)))/scale_phi(ip);
          }
        } else {
          dgrho=helfem::Matrix();
        }

        if(do_tau) {
          const helfem::Matrix dPavA_rho(dPa*bf_rho_re), dPavB_rho(dPa*bf_rho_im);
          const helfem::Matrix dPavA_theta(dPa*bf_theta_re), dPavB_theta(dPa*bf_theta_im);
          const helfem::Matrix dPavA_phi(dPa*bf_phi_re), dPavB_phi(dPa*bf_phi_im);
          const helfem::Matrix dPbvA_rho(dPb*bf_rho_re), dPbvB_rho(dPb*bf_rho_im);
          const helfem::Matrix dPbvA_theta(dPb*bf_theta_re), dPbvB_theta(dPb*bf_theta_im);
          const helfem::Matrix dPbvA_phi(dPb*bf_phi_re), dPbvB_phi(dPb*bf_phi_im);
          dtau=helfem::Matrix::Zero(2,wtot.size());
          for(Eigen::Index ip=0;ip<wtot.size();ip++) {
            double kinar=(dPavA_rho.col(ip).dot(bf_rho_re.col(ip))+dPavB_rho.col(ip).dot(bf_rho_im.col(ip)))/std::pow(scale_r(ip),2);
            double kinath=(dPavA_theta.col(ip).dot(bf_theta_re.col(ip))+dPavB_theta.col(ip).dot(bf_theta_im.col(ip)))/std::pow(scale_theta(ip),2);
            double kinaphi=(dPavA_phi.col(ip).dot(bf_phi_re.col(ip))+dPavB_phi.col(ip).dot(bf_phi_im.col(ip)))/std::pow(scale_phi(ip),2);

            double kinbr=(dPbvA_rho.col(ip).dot(bf_rho_re.col(ip))+dPbvB_rho.col(ip).dot(bf_rho_im.col(ip)))/std::pow(scale_r(ip),2);
            double kinbth=(dPbvA_theta.col(ip).dot(bf_theta_re.col(ip))+dPbvB_theta.col(ip).dot(bf_theta_im.col(ip)))/std::pow(scale_theta(ip),2);
            double kinbphi=(dPbvA_phi.col(ip).dot(bf_phi_re.col(ip))+dPbvB_phi.col(ip).dot(bf_phi_im.col(ip)))/std::pow(scale_phi(ip),2);

            dtau(0,ip)=0.5*(kinar + kinath + kinaphi);
            dtau(1,ip)=0.5*(kinbr + kinbth + kinbphi);
          }
        } else {
          dtau=helfem::Matrix();
        }
      }

      void DFTGridWorker::check_response_fields(const helfem::Matrix & P,
                                                const helfem::Matrix & dP,
                                                const helfem::Matrix & drho,
                                                const helfem::Matrix & dgrho,
                                                const helfem::Matrix & dtau) {
        // The perturbed fields are linear in the perturbation, so they
        // must equal a central difference of the reference ones. Going
        // through update_density rather than a second copy of the
        // formulas above is the point: a dropped metric factor made in
        // both places would cancel out of any self-consistent check.
        const double h = 1e-5;
        update_density(helfem::Matrix(P + h*dP));
        const helfem::Matrix rp(rho), gp(grho), tp(tau);
        update_density(helfem::Matrix(P - h*dP));
        const helfem::Matrix rm(rho), gm(grho), tm(tau);
        helfem::dftgrid_common::report_dfields("drho", drho, helfem::Matrix((rp-rm)/(2*h)));
        if(do_grad)
          helfem::dftgrid_common::report_dfields("dgrho", dgrho, helfem::Matrix((gp-gm)/(2*h)));
        if(do_tau)
          helfem::dftgrid_common::report_dfields("dtau", dtau, helfem::Matrix((tp-tm)/(2*h)));
      }

      void DFTGridWorker::check_response_fields(const helfem::Matrix & Pa,
                                                const helfem::Matrix & Pb,
                                                const helfem::Matrix & dPa,
                                                const helfem::Matrix & dPb,
                                                const helfem::Matrix & drho,
                                                const helfem::Matrix & dgrho,
                                                const helfem::Matrix & dtau) {
        const double h = 1e-5;
        update_density(helfem::Matrix(Pa + h*dPa), helfem::Matrix(Pb + h*dPb));
        const helfem::Matrix rp(rho), gp(grho), tp(tau);
        update_density(helfem::Matrix(Pa - h*dPa), helfem::Matrix(Pb - h*dPb));
        const helfem::Matrix rm(rho), gm(grho), tm(tau);
        helfem::dftgrid_common::report_dfields("drho", drho, helfem::Matrix((rp-rm)/(2*h)));
        if(do_grad)
          helfem::dftgrid_common::report_dfields("dgrho", dgrho, helfem::Matrix((gp-gm)/(2*h)));
        if(do_tau)
          helfem::dftgrid_common::report_dfields("dtau", dtau, helfem::Matrix((tp-tm)/(2*h)));
      }

      void DFTGrid::eval_Fxc(int x_func, const helfem::Vector & x_pars, int c_func, const helfem::Vector & c_pars, const helfem::Matrix & P_e, helfem::Matrix & H_e, double & Exc, double & Nel, double & Ekin, double thr) {
        // Eigen flows straight through the worker and remove_boundaries.
        helfem::Matrix H = helfem::Matrix::Zero(basp->Ndummy(),basp->Ndummy());

        double exc=0.0;
        double ekin=0.0;
        double nel=0.0;
        {
          // Loop-invariant: expand once, not once per grid point.
          const helfem::Matrix P_exp(basp->expand_boundaries(P_e));
          const std::vector<std::pair<size_t, size_t>> tasks = grid_tasks(basp);
          // Each thread accumulates its own Fock matrix and its own energy
          // sums; the partials are summed afterwards in thread order. The
          // scatter writes basis
          // functions that neighbouring radial points share, so no partition
          // of the tasks makes the writes disjoint -- and summing under a
          // critical section would make the result depend on which thread
          // arrived first, since floating-point addition is not associative.
          // The schedule is left static for the same reason: a dynamic one
          // hands a different set of points to each thread on every run,
          // which perturbs the partial sums even at a fixed thread count.
          const int nthread = helfem_omp_max_threads();
          std::vector<helfem::Matrix> Hpart(nthread);
          std::vector<double> excp(nthread, 0.0), ekinp(nthread, 0.0), nelp(nthread, 0.0);
#ifdef _OPENMP
#pragma omp parallel
#endif
          {
            DFTGridWorker grid(basp,lang,mang);
            grid.check_grad_tau_lapl(x_func,c_func);
            const int tid = helfem_omp_thread_num();
            helfem::Matrix & Hloc = Hpart[tid];
            Hloc = helfem::Matrix::Zero(basp->Ndummy(), basp->Ndummy());

#ifdef _OPENMP
#pragma omp for
#endif
            for (long it = 0; it < (long) tasks.size(); it++) {
              const size_t iel = tasks[it].first, irad = tasks[it].second;
              grid.compute_bf(iel,irad);
              grid.update_density(P_exp);
              nelp[tid]+=grid.compute_Nel();
              ekinp[tid]+=grid.compute_Ekin();

              grid.init_xc();
              if(x_func>0)
                grid.compute_xc(x_func, x_pars, thr);
              if(c_func>0)
                grid.compute_xc(c_func, c_pars, thr);

              // the assembly contracts a general vector field; build it once
              grid.build_vgrad();

              excp[tid]+=grid.eval_Exc();
              grid.eval_Fxc(Hloc);
            }
          }
          for (int t = 0; t < nthread; t++) {
            exc += excp[t]; ekin += ekinp[t]; nel += nelp[t];
          }
          for (int t = 0; t < nthread; t++)
            if (Hpart[t].size()) {
              H += Hpart[t];
            }
        }

        // Save outputs
        Exc=exc;
        Ekin=ekin;
        Nel=nel;

        H_e=basp->remove_boundaries(H);
      }

      void DFTGrid::eval_Fxc(int x_func, const helfem::Vector & x_pars, int c_func, const helfem::Vector & c_pars, const helfem::Matrix & Pa_e, const helfem::Matrix & Pb_e, helfem::Matrix & Ha_e, helfem::Matrix & Hb_e, double & Exc, double & Nel, double & Ekin, bool beta, double thr) {
        // Eigen flows straight through the worker and remove_boundaries.
        helfem::Matrix Ha = helfem::Matrix::Zero(basp->Ndummy(),basp->Ndummy());
        helfem::Matrix Hb = helfem::Matrix::Zero(basp->Ndummy(),basp->Ndummy());

        double exc=0.0;
        double nel=0.0;
        double ekin=0.0;
        {
          // Loop-invariant: expand once, not once per grid point.
          const helfem::Matrix Pa_exp(basp->expand_boundaries(Pa_e));
          const helfem::Matrix Pb_exp(basp->expand_boundaries(Pb_e));

          const std::vector<std::pair<size_t, size_t>> tasks = grid_tasks(basp);
          // See the restricted driver above for why every partial -- Fock
          // matrices and energy sums alike -- is summed in thread order
          // rather than folded in under a critical section.
          const int nthread = helfem_omp_max_threads();
          std::vector<helfem::Matrix> Hapart(nthread), Hbpart(nthread);
          std::vector<double> excp(nthread, 0.0), ekinp(nthread, 0.0), nelp(nthread, 0.0);
#ifdef _OPENMP
#pragma omp parallel
#endif
          {
            DFTGridWorker grid(basp,lang,mang);
            grid.check_grad_tau_lapl(x_func,c_func);
            const int tid = helfem_omp_thread_num();
            helfem::Matrix & Haloc = Hapart[tid];
            helfem::Matrix & Hbloc = Hbpart[tid];
            Haloc = helfem::Matrix::Zero(basp->Ndummy(),basp->Ndummy());
            Hbloc = helfem::Matrix::Zero(basp->Ndummy(),basp->Ndummy());

#ifdef _OPENMP
#pragma omp for
#endif
            for (long it = 0; it < (long) tasks.size(); it++) {
              const size_t iel = tasks[it].first, irad = tasks[it].second;
              grid.compute_bf(iel,irad);
              grid.update_density(Pa_exp,Pb_exp);
              nelp[tid]+=grid.compute_Nel();
              ekinp[tid]+=grid.compute_Ekin();

              grid.init_xc();
              if(x_func>0)
                grid.compute_xc(x_func, x_pars, thr);
              if(c_func>0)
                grid.compute_xc(c_func, c_pars, thr);

              // the assembly contracts a general vector field; build it once
              grid.build_vgrad();

              excp[tid]+=grid.eval_Exc();
              grid.eval_Fxc(Haloc,Hbloc,beta);
            }
          }
          for (int t = 0; t < nthread; t++) {
            exc += excp[t]; ekin += ekinp[t]; nel += nelp[t];
          }
          for (int t = 0; t < nthread; t++)
            if (Hapart[t].size()) { Ha += Hapart[t]; Hb += Hbpart[t]; }
        }

        // Save outputs
        Exc=exc;
        Ekin=ekin;
        Nel=nel;

        // Clean up matrices
        Ha_e=basp->remove_boundaries(Ha);
        Hb_e=basp->remove_boundaries(Hb);
      }

      void DFTGrid::eval_Fxc_response(int x_func, const helfem::Vector & x_pars, int c_func, const helfem::Vector & c_pars, const helfem::Matrix & P_e, const std::vector<helfem::Matrix> & dP_e, std::vector<helfem::Matrix> & dH_e, double thr) {
        dH_e.assign(dP_e.size(), helfem::Matrix());
        if(dP_e.empty())
          return;

        std::vector<helfem::Matrix> dH(dP_e.size(), helfem::Matrix::Zero(basp->Ndummy(),basp->Ndummy()));
        {
          DFTGridWorker grid(basp,lang,mang);
          grid.check_grad_tau_lapl(x_func,c_func);

          // Loop-invariant: expand once, not once per grid point.
          const helfem::Matrix P_exp(basp->expand_boundaries(P_e));
          std::vector<helfem::Matrix> dP_exp(dP_e.size());
          for(size_t ip=0;ip<dP_e.size();ip++)
            dP_exp[ip]=basp->expand_boundaries(dP_e[ip]);

          for(size_t iel=0;iel<basp->rad_Nel();iel++) {
            for(size_t irad=0;irad<(size_t) basp->r(iel).size();irad++) {
              grid.compute_bf(iel,irad);
              grid.update_density(P_exp);
              // init_xc allocates the potential buffers the response is
              // written into, and resets the do_gga / do_mgga flags the
              // assembly reads.
              grid.init_xc();
              // The gradient channel of the response kernel contains the
              // GROUND-STATE vsigma (the 2 vsigma grad(drho) term), so
              // the first derivatives have to be evaluated here as well;
              // only the second ones were needed while the response was
              // LDA-shaped.
              if(x_func>0)
                grid.compute_xc(x_func, x_pars, thr);
              if(c_func>0)
                grid.compute_xc(c_func, c_pars, thr);
              grid.init_fxc();
              if(x_func>0)
                grid.compute_fxc(x_func, x_pars, thr);
              if(c_func>0)
                grid.compute_fxc(c_func, c_pars, thr);
              for(size_t ip=0;ip<dP_exp.size();ip++) {
                helfem::Matrix drho, dgrho, dtau;
                grid.eval_response_fields(dP_exp[ip], drho, dgrho, dtau);
                if(getenv("HELFEM_CHECK_DFIELDS")) {
                  // The perturbed fields are linear in the perturbation,
                  // so they must equal a central difference of the
                  // reference ones. This isolates eval_response_fields
                  // from the kernel and the assembly.
                  grid.check_response_fields(P_exp, dP_exp[ip], drho, dgrho, dtau);
                  grid.update_density(P_exp);
                }
                grid.set_response_potential(drho, grid.get_grho(), dgrho, dtau);
                grid.eval_Fxc(dH[ip]);
              }
            }
          }
        }
        for(size_t ip=0;ip<dH.size();ip++)
          dH_e[ip]=basp->remove_boundaries(dH[ip]);
        if(getenv("HELFEM_CHECK_DFOCK")) {
          // The response Fock matrix must equal a central difference of
          // the ground-state one: dF = [F(P + h dP) - F(P - h dP)]/2h.
          // This tests the whole response path -- kernel, channels and
          // assembly -- against machinery already known to be correct,
          // which the field check alone cannot do.
          double worst = 0.0, scale = 0.0;
          for(size_t ip=0;ip<dP_e.size();ip++) {
            // Scale the step to the perturbation; see the atomic grid for
            // why a fixed one measures its own truncation instead.
            const double h = 1e-5/std::max(1.0, dP_e[ip].cwiseAbs().maxCoeff());
            helfem::Matrix Hp, Hm;
            double e, n, k;
            eval_Fxc(x_func, x_pars, c_func, c_pars, helfem::Matrix(P_e + h*dP_e[ip]), Hp, e, n, k, thr);
            eval_Fxc(x_func, x_pars, c_func, c_pars, helfem::Matrix(P_e - h*dP_e[ip]), Hm, e, n, k, thr);
            const helfem::Matrix fd = (Hp-Hm)/(2*h);
            worst = std::max(worst, (dH_e[ip]-fd).cwiseAbs().maxCoeff());
            scale = std::max(scale, fd.cwiseAbs().maxCoeff());
          }
          fprintf(stderr, "DFOCK worst |analytic-FD| = %.3e  (scale %.3e, "
                  "rel %.3e)\n", worst, scale, worst/std::max(1e-30, scale));
        }
      }

      void DFTGrid::eval_Fxc_response(int x_func, const helfem::Vector & x_pars, int c_func, const helfem::Vector & c_pars, const helfem::Matrix & Pa_e, const helfem::Matrix & Pb_e, const std::vector<helfem::Matrix> & dPa_e, const std::vector<helfem::Matrix> & dPb_e, std::vector<helfem::Matrix> & dHa_e, std::vector<helfem::Matrix> & dHb_e, double thr) {
        dHa_e.assign(dPa_e.size(), helfem::Matrix());
        dHb_e.assign(dPa_e.size(), helfem::Matrix());
        if(dPa_e.empty())
          return;
        if(dPb_e.size()!=dPa_e.size())
          throw std::logic_error("Got a different number of alpha and beta perturbations.\n");

        std::vector<helfem::Matrix> dHa(dPa_e.size(), helfem::Matrix::Zero(basp->Ndummy(),basp->Ndummy()));
        std::vector<helfem::Matrix> dHb(dPa_e.size(), helfem::Matrix::Zero(basp->Ndummy(),basp->Ndummy()));
        {
          DFTGridWorker grid(basp,lang,mang);
          grid.check_grad_tau_lapl(x_func,c_func);

          const helfem::Matrix Pa_exp(basp->expand_boundaries(Pa_e));
          const helfem::Matrix Pb_exp(basp->expand_boundaries(Pb_e));
          std::vector<helfem::Matrix> dPa_exp(dPa_e.size()), dPb_exp(dPa_e.size());
          for(size_t ip=0;ip<dPa_e.size();ip++) {
            dPa_exp[ip]=basp->expand_boundaries(dPa_e[ip]);
            dPb_exp[ip]=basp->expand_boundaries(dPb_e[ip]);
          }

          for(size_t iel=0;iel<basp->rad_Nel();iel++) {
            for(size_t irad=0;irad<(size_t) basp->r(iel).size();irad++) {
              grid.compute_bf(iel,irad);
              grid.update_density(Pa_exp,Pb_exp);
              grid.init_xc();
              // The response kernel's gradient channel carries the
              // ground-state vsigma, so the first derivatives are needed
              // here too.
              if(x_func>0)
                grid.compute_xc(x_func, x_pars, thr);
              if(c_func>0)
                grid.compute_xc(c_func, c_pars, thr);
              grid.init_fxc();
              if(x_func>0)
                grid.compute_fxc(x_func, x_pars, thr);
              if(c_func>0)
                grid.compute_fxc(c_func, c_pars, thr);
              for(size_t ip=0;ip<dPa_exp.size();ip++) {
                helfem::Matrix drho, dgrho, dtau;
                grid.eval_response_fields(dPa_exp[ip], dPb_exp[ip], drho, dgrho, dtau);
                if(getenv("HELFEM_CHECK_DFIELDS")) {
                  grid.check_response_fields(Pa_exp, Pb_exp, dPa_exp[ip], dPb_exp[ip],
                                              drho, dgrho, dtau);
                  grid.update_density(Pa_exp,Pb_exp);
                }
                grid.set_response_potential(drho, grid.get_grho(), dgrho, dtau);
                grid.eval_Fxc(dHa[ip],dHb[ip],true);
              }
            }
          }
        }
        for(size_t ip=0;ip<dHa.size();ip++) {
          dHa_e[ip]=basp->remove_boundaries(dHa[ip]);
          dHb_e[ip]=basp->remove_boundaries(dHb[ip]);
        }
        if(getenv("HELFEM_CHECK_DFOCK")) {
          double worst = 0.0, scale = 0.0;
          for(size_t ip=0;ip<dPa_e.size();ip++) {
            const double h = 1e-5/std::max({1.0, dPa_e[ip].cwiseAbs().maxCoeff(),
                                             dPb_e[ip].cwiseAbs().maxCoeff()});
            helfem::Matrix Hap, Ham, Hbp, Hbm;
            double e, n, k;
            eval_Fxc(x_func, x_pars, c_func, c_pars,
                     helfem::Matrix(Pa_e + h*dPa_e[ip]), helfem::Matrix(Pb_e + h*dPb_e[ip]),
                     Hap, Hbp, e, n, k, true, thr);
            eval_Fxc(x_func, x_pars, c_func, c_pars,
                     helfem::Matrix(Pa_e - h*dPa_e[ip]), helfem::Matrix(Pb_e - h*dPb_e[ip]),
                     Ham, Hbm, e, n, k, true, thr);
            const helfem::Matrix fda = (Hap-Ham)/(2*h), fdb = (Hbp-Hbm)/(2*h);
            worst = std::max(worst, (dHa_e[ip]-fda).cwiseAbs().maxCoeff());
            worst = std::max(worst, (dHb_e[ip]-fdb).cwiseAbs().maxCoeff());
            scale = std::max(scale, fda.cwiseAbs().maxCoeff());
            scale = std::max(scale, fdb.cwiseAbs().maxCoeff());
          }
          fprintf(stderr, "DFOCK worst |analytic-FD| = %.3e  (scale %.3e, "
                  "rel %.3e)\n", worst, scale, worst/std::max(1e-30, scale));
        }
      }

    }
  }
}
