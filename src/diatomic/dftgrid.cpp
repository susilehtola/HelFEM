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
          // Get vsigma
          helfem::Vector vs = vsigma.row(0).transpose();
          // Get grad rho (rows 0..2 transposed to Npts x 3)
          helfem::Matrix gr = grho.topRows(3).transpose();
          // Multiply grad rho by vsigma and the weights
          for(Eigen::Index i=0;i<gr.rows();i++) {
            gr(i,0)*=2.0*wtot(i)*vs(i)/scale_r(i);
            gr(i,1)*=2.0*wtot(i)*vs(i)/scale_theta(i);
            gr(i,2)*=2.0*wtot(i)*vs(i)/scale_phi(i);
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
          // Get vsigma
          helfem::Vector vs_aa = vsigma.row(0).transpose();
          helfem::Vector vs_ab = vsigma.row(1).transpose();

          // Get grad rho (rows 0..2 alpha, 3..5 beta, transposed to Npts x 3)
          helfem::Matrix gr_a0 = grho.topRows(3).transpose();
          helfem::Matrix gr_b0 = grho.bottomRows(3).transpose();

          // Multiply grad rho by vsigma and the weights
          helfem::Matrix gr_a(gr_a0);
          for(Eigen::Index i=0;i<gr_a.rows();i++) {
            gr_a(i,0)=wtot(i)*(2.0*vs_aa(i)*gr_a0(i,0) + vs_ab(i)*gr_b0(i,0))/scale_r(i);
            gr_a(i,1)=wtot(i)*(2.0*vs_aa(i)*gr_a0(i,1) + vs_ab(i)*gr_b0(i,1))/scale_theta(i);
            gr_a(i,2)=wtot(i)*(2.0*vs_aa(i)*gr_a0(i,2) + vs_ab(i)*gr_b0(i,2))/scale_phi(i);
          }
          // Increment matrix
          increment_gga_split(Ha,gr_a,bf_re,bf_im,{&bf_rho_re,&bf_theta_re,&bf_phi_re},{&bf_rho_im,&bf_theta_im,&bf_phi_im});

          if(beta) {
            helfem::Vector vs_bb = vsigma.row(2).transpose();
            helfem::Matrix gr_b(gr_b0);
            for(Eigen::Index i=0;i<gr_b.rows();i++) {
              gr_b(i,0)=wtot(i)*(2.0*vs_bb(i)*gr_b0(i,0) + vs_ab(i)*gr_a0(i,0))/scale_r(i);
              gr_b(i,1)=wtot(i)*(2.0*vs_bb(i)*gr_b0(i,1) + vs_ab(i)*gr_a0(i,1))/scale_theta(i);
              gr_b(i,2)=wtot(i)*(2.0*vs_bb(i)*gr_b0(i,2) + vs_ab(i)*gr_a0(i,2))/scale_phi(i);
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

      void DFTGrid::eval_Fxc(int x_func, const helfem::Vector & x_pars, int c_func, const helfem::Vector & c_pars, const helfem::Matrix & P_e, helfem::Matrix & H_e, double & Exc, double & Nel, double & Ekin, double thr) {
        // Eigen flows straight through the worker and remove_boundaries.
        helfem::Matrix H = helfem::Matrix::Zero(basp->Ndummy(),basp->Ndummy());

        double exc=0.0;
        double ekin=0.0;
        double nel=0.0;
        {
          DFTGridWorker grid(basp,lang,mang);
          grid.check_grad_tau_lapl(x_func,c_func);

          // Loop-invariant: expand once, not once per grid point.
          const helfem::Matrix P_exp(basp->expand_boundaries(P_e));

          for(size_t iel=0;iel<basp->rad_Nel();iel++) {
            for(size_t irad=0;irad<(size_t) basp->r(iel).size();irad++) {
              grid.compute_bf(iel,irad);
              grid.update_density(P_exp);
              nel+=grid.compute_Nel();
              ekin+=grid.compute_Ekin();

              grid.init_xc();
              if(x_func>0)
                grid.compute_xc(x_func, x_pars, thr);
              if(c_func>0)
                grid.compute_xc(c_func, c_pars, thr);

              // the assembly contracts a general vector field; build it once
              grid.build_vgrad();

              exc+=grid.eval_Exc();
              grid.eval_Fxc(H);
            }
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
          DFTGridWorker grid(basp,lang,mang);
          grid.check_grad_tau_lapl(x_func,c_func);

          // Loop-invariant: expand once, not once per grid point.
          const helfem::Matrix Pa_exp(basp->expand_boundaries(Pa_e));
          const helfem::Matrix Pb_exp(basp->expand_boundaries(Pb_e));

          for(size_t iel=0;iel<basp->rad_Nel();iel++) {
            for(size_t irad=0;irad<(size_t) basp->r(iel).size();irad++) {
              grid.compute_bf(iel,irad);
              grid.update_density(Pa_exp,Pb_exp);
              nel+=grid.compute_Nel();
              ekin+=grid.compute_Ekin();

              grid.init_xc();
              if(x_func>0)
                grid.compute_xc(x_func, x_pars, thr);
              if(c_func>0)
                grid.compute_xc(c_func, c_pars, thr);

              // the assembly contracts a general vector field; build it once
              grid.build_vgrad();

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
        Ha_e=basp->remove_boundaries(Ha);
        Hb_e=basp->remove_boundaries(Hb);
      }

    }
  }
}
