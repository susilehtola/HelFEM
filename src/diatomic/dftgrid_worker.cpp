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
#include <helfem/helfem.h>
#include <cmath>
#include <cstdio>

#include "dftgrid.h"
// Angular quadrature
#include "../general/angular.h"

// OpenMP parallellization for XC calculations
#ifdef _OPENMP
#include <omp.h>
#endif

// The diatomic DFT-grid worker: basis functions on the grid, densities, and
// Fock assembly from potentials already on the grid.  Nothing here calls
// libxc, so it lives in helfem-fem; the DFTGrid driver that asks libxc for the
// potentials stays in dftgrid.cpp in helfem-common.  See
// general/dftgrid_core.cpp for why a split by translation unit is enough.

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

    }
  }
}
