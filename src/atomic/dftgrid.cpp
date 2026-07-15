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
  namespace atomic {
    namespace dftgrid {
      DFTGridWorker::DFTGridWorker() {
      }

      DFTGridWorker::DFTGridWorker(const helfem::atomic::basis::TwoDBasis * basp_, int lang, int mang) : basp(basp_) {
        do_grad=false;
        do_tau=false;
        do_lapl=false;

        // Get angular grid (angular_chebyshev returns arma -- bridge at
        // the fill site).
        arma::vec cth_a, phi_a, wang_a;
        helfem::angular::angular_chebyshev(lang,mang,cth_a,phi_a,wang_a);
        cth=helfem::to_eigen(cth_a);
        phi=helfem::to_eigen(phi_a);
        wang=helfem::to_eigen(wang_a);
      }

      DFTGridWorker::~DFTGridWorker() {
      }

      void DFTGridWorker::update_density(const helfem::Matrix & P0) {
        // Update values of density
        if(!P0.size()) {
          throw std::runtime_error("Error - density matrix is empty!\n");
        }
        // The atomic basis has no boundary reduction, so P0 already spans
        // the full (Ndummy) basis; slice out this element's functions.
        helfem::Matrix P(bf_ind.size(), bf_ind.size());
        for(size_t i=0;i<bf_ind.size();i++)
          for(size_t j=0;j<bf_ind.size();j++)
            P(i,j)=P0(bf_ind[i],bf_ind[j]);

        // Non-polarized calculation.
        polarized=false;

        // Update density vector (real P * complex conj(bf)).
        Pv=P.cast<std::complex<double> >()*bf.conjugate();

        // Calculate density (arma::dot does not conjugate).
        rho=helfem::Matrix::Zero(1,wtot.size());
        for(Eigen::Index ip=0;ip<wtot.size();ip++)
          rho(0,ip)=std::real((Pv.col(ip).array()*bf.col(ip).array()).sum());

        // Calculate gradient
        if(do_grad) {
          grho=helfem::Matrix::Zero(3,wtot.size());
          sigma=helfem::Matrix::Zero(1,wtot.size());
          for(Eigen::Index ip=0;ip<wtot.size();ip++) {
            // Calculate values
            double g_rad=grho(0,ip)=2.0*std::real((Pv.col(ip).array()*bf_rho.col(ip).array()).sum())/scale_r(ip);
            double g_th=grho(1,ip)=2.0*std::real((Pv.col(ip).array()*bf_theta.col(ip).array()).sum())/scale_theta(ip);
            double g_phi=grho(2,ip)=2.0*std::real((Pv.col(ip).array()*bf_phi.col(ip).array()).sum())/scale_phi(ip);
            // Compute sigma as well
            sigma(0,ip)=g_rad*g_rad + g_th*g_th + g_phi*g_phi;
          }
        }

        // Calculate laplacian and kinetic energy density
        if(do_tau || do_lapl) {
          // Adjust size of grid
          tau=helfem::Matrix::Zero(1,wtot.size());

          // Update helpers
          Pv_rho=P.cast<std::complex<double> >()*bf_rho.conjugate();
          Pv_theta=P.cast<std::complex<double> >()*bf_theta.conjugate();
          Pv_phi=P.cast<std::complex<double> >()*bf_phi.conjugate();

          // Calculate values
          for(Eigen::Index ip=0;ip<wtot.size();ip++) {
            // Gradient term
            double kinrho(std::real((Pv_rho.col(ip).array()*bf_rho.col(ip).array()).sum())/std::pow(scale_r(ip),2));
            double kintheta(std::real((Pv_theta.col(ip).array()*bf_theta.col(ip).array()).sum())/std::pow(scale_theta(ip),2));
            double kinphi(std::real((Pv_phi.col(ip).array()*bf_phi.col(ip).array()).sum())/std::pow(scale_phi(ip),2));
            double kin(kinrho + kintheta + kinphi);

            // Store values
            tau(0,ip)=0.5*kin;
          }

          if(do_lapl) {
            // Adjust size of grid
            lapl=helfem::Matrix::Zero(1,wtot.size());
            // Calculate values
            for(Eigen::Index ip=0;ip<wtot.size();ip++) {
              // Gradient term
              double kinrho(std::real((Pv_rho.col(ip).array()*bf_rho.col(ip).array()).sum())/std::pow(scale_r(ip),2));
              double kintheta(std::real((Pv_theta.col(ip).array()*bf_theta.col(ip).array()).sum())/std::pow(scale_theta(ip),2));
              double kinphi(std::real((Pv_phi.col(ip).array()*bf_phi.col(ip).array()).sum())/std::pow(scale_phi(ip),2));
              double kin(kinrho + kintheta + kinphi);
              // Laplacian term
              double lap(std::real((Pv.col(ip).array()*bf_lapl.col(ip).array()).sum()));

              // Store values
              lapl(0,ip)=2.0*(kin + lap);
            }
          }
        }
      }

      void DFTGridWorker::update_density(const helfem::Matrix & Pa0, const helfem::Matrix & Pb0) {
        if(!Pa0.size() || !Pb0.size()) {
          throw std::runtime_error("Error - density matrix is empty!\n");
        }

        // Polarized calculation.
        polarized=true;

        // Update density vector (atomic basis has no boundary reduction).
        helfem::Matrix Pa(bf_ind.size(), bf_ind.size());
        helfem::Matrix Pb(bf_ind.size(), bf_ind.size());
        for(size_t i=0;i<bf_ind.size();i++)
          for(size_t j=0;j<bf_ind.size();j++) {
            Pa(i,j)=Pa0(bf_ind[i],bf_ind[j]);
            Pb(i,j)=Pb0(bf_ind[i],bf_ind[j]);
          }

        Pav=Pa.cast<std::complex<double> >()*bf.conjugate();
        Pbv=Pb.cast<std::complex<double> >()*bf.conjugate();

        // Calculate density (arma::dot does not conjugate).
        rho=helfem::Matrix::Zero(2,wtot.size());
        for(Eigen::Index ip=0;ip<wtot.size();ip++) {
          rho(0,ip)=std::real((Pav.col(ip).array()*bf.col(ip).array()).sum());
          rho(1,ip)=std::real((Pbv.col(ip).array()*bf.col(ip).array()).sum());

          /*
            double na=compute_density(Pa0,*basp,grid[ip].r);
            double nb=compute_density(Pb0,*basp,grid[ip].r);
            if(fabs(da-na)>1e-6 || fabs(db-nb)>1e-6)
            printf("Density at point % .3f % .3f % .3f: %e vs %e, %e vs %e\n",grid[ip].r.x,grid[ip].r.y,grid[ip].r.z,da,na,db,nb);
          */
        }

        // Calculate gradient

        if(do_grad) {
          grho=helfem::Matrix::Zero(6,wtot.size());
          sigma=helfem::Matrix::Zero(3,wtot.size());
          for(Eigen::Index ip=0;ip<wtot.size();ip++) {
            double ga_rad=grho(0,ip)=2.0*std::real((Pav.col(ip).array()*bf_rho.col(ip).array()).sum())/scale_r(ip);
            double ga_th=grho(1,ip)=2.0*std::real((Pav.col(ip).array()*bf_theta.col(ip).array()).sum())/scale_theta(ip);
            double ga_phi=grho(2,ip)=2.0*std::real((Pav.col(ip).array()*bf_phi.col(ip).array()).sum())/scale_phi(ip);

            double gb_rad=grho(3,ip)=2.0*std::real((Pbv.col(ip).array()*bf_rho.col(ip).array()).sum())/scale_r(ip);
            double gb_th=grho(4,ip)=2.0*std::real((Pbv.col(ip).array()*bf_theta.col(ip).array()).sum())/scale_theta(ip);
            double gb_phi=grho(5,ip)=2.0*std::real((Pbv.col(ip).array()*bf_phi.col(ip).array()).sum())/scale_phi(ip);

            // Compute sigma as well
            sigma(0,ip)=ga_rad*ga_rad + ga_th*ga_th + ga_phi*ga_phi;
            sigma(1,ip)=ga_rad*gb_rad + ga_th*gb_th + ga_phi*gb_phi;
            sigma(2,ip)=gb_rad*gb_rad + gb_th*gb_th + gb_phi*gb_phi;
          }
        }

        // Calculate kinetic energy density
        if(do_tau || do_lapl) {
          // Adjust size of grid
          tau.resize(2,wtot.size());

          // Update helpers
          Pav_rho=Pa.cast<std::complex<double> >()*bf_rho.conjugate();
          Pav_theta=Pa.cast<std::complex<double> >()*bf_theta.conjugate();
          Pav_phi=Pa.cast<std::complex<double> >()*bf_phi.conjugate();

          Pbv_rho=Pb.cast<std::complex<double> >()*bf_rho.conjugate();
          Pbv_theta=Pb.cast<std::complex<double> >()*bf_theta.conjugate();
          Pbv_phi=Pb.cast<std::complex<double> >()*bf_phi.conjugate();

          // Calculate values
          for(Eigen::Index ip=0;ip<wtot.size();ip++) {
            // Gradient term
            double kinar=std::real((Pav_rho.col(ip).array()*bf_rho.col(ip).array()).sum())/std::pow(scale_r(ip),2);
            double kinath=std::real((Pav_theta.col(ip).array()*bf_theta.col(ip).array()).sum())/std::pow(scale_theta(ip),2);
            double kinaphi=std::real((Pav_phi.col(ip).array()*bf_phi.col(ip).array()).sum())/std::pow(scale_phi(ip),2);
            double kina(kinar + kinath + kinaphi);

            double kinbr=std::real((Pbv_rho.col(ip).array()*bf_rho.col(ip).array()).sum())/std::pow(scale_r(ip),2);
            double kinbth=std::real((Pbv_theta.col(ip).array()*bf_theta.col(ip).array()).sum())/std::pow(scale_theta(ip),2);
            double kinbphi=std::real((Pbv_phi.col(ip).array()*bf_phi.col(ip).array()).sum())/std::pow(scale_phi(ip),2);
            double kinb(kinbr + kinbth + kinbphi);

            // Store values
            tau(0,ip)=0.5*kina;
            tau(1,ip)=0.5*kinb;
          }
          if(do_lapl) {
            // Adjust size of grid
            lapl=helfem::Matrix::Zero(2,wtot.size());
            // Calculate values
            for(Eigen::Index ip=0;ip<wtot.size();ip++) {
              // Gradient term
              double kinar=std::real((Pav_rho.col(ip).array()*bf_rho.col(ip).array()).sum())/std::pow(scale_r(ip),2);
              double kinath=std::real((Pav_theta.col(ip).array()*bf_theta.col(ip).array()).sum())/std::pow(scale_theta(ip),2);
              double kinaphi=std::real((Pav_phi.col(ip).array()*bf_phi.col(ip).array()).sum())/std::pow(scale_phi(ip),2);
              double kina(kinar + kinath + kinaphi);

              double kinbr=std::real((Pbv_rho.col(ip).array()*bf_rho.col(ip).array()).sum())/std::pow(scale_r(ip),2);
              double kinbth=std::real((Pbv_theta.col(ip).array()*bf_theta.col(ip).array()).sum())/std::pow(scale_theta(ip),2);
              double kinbphi=std::real((Pbv_phi.col(ip).array()*bf_phi.col(ip).array()).sum())/std::pow(scale_phi(ip),2);
              double kinb(kinbr + kinbth + kinbphi);

              // Laplacian term
              double lapa(std::real((Pav.col(ip).array()*bf_lapl.col(ip).array()).sum()));
              double lapb(std::real((Pbv.col(ip).array()*bf_lapl.col(ip).array()).sum()));

              // Store values
              lapl(0,ip)=2.0*(kina + lapa);
              lapl(1,ip)=2.0*(kinb + lapb);
            }
          }
        }
      }

      double DFTGridWorker::compute_laplsum() const {
        double sum=0.0;
        if(lapl.cols() == wtot.size()) {
          if(!polarized) {
            for(Eigen::Index ip=0;ip<wtot.size();ip++)
              sum+=wtot(ip)*lapl(0,ip);
          } else {
            for(Eigen::Index ip=0;ip<wtot.size();ip++)
              sum+=wtot(ip)*(lapl(0,ip)+lapl(1,ip));
          }
        }

        return sum;
      }

      double DFTGridWorker::compute_Ekin() const {
        double ekin=0.0;

        if(do_tau) {
          if(!polarized) {
            for(Eigen::Index ip=0;ip<wtot.size();ip++)
              ekin+=wtot(ip)*tau(0,ip);
          } else {
            for(Eigen::Index ip=0;ip<wtot.size();ip++)
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

      // compute_xc, eval_Exc: inherited from
      // helfem::dftgrid_common::DFTGridWorkerBase.

      void DFTGridWorker::eval_Fxc(helfem::Matrix & Ho) const {
        if(polarized) {
          throw std::runtime_error("Refusing to compute restricted Fock matrix with unrestricted density.\n");
        }

        // Work matrix
        helfem::Matrix H=helfem::Matrix::Zero(bf_ind.size(),bf_ind.size());

        {
          // LDA potential
          helfem::Vector vrho=vxc.row(0).transpose();
          // Multiply weights into potential
          vrho=vrho.array()*wtot.array();
          // Increment matrix
          increment_lda< std::complex<double> >(H,vrho,bf);
        }

        if(do_gga) {
          // Get vsigma
          helfem::Vector vs=vsigma.row(0).transpose();
          // Get grad rho
          helfem::Matrix gr=grho.topRows(3).transpose();
          // Multiply grad rho by vsigma and the weights
          for(Eigen::Index i=0;i<gr.rows();i++) {
            gr(i,0)*=2.0*wtot(i)*vs(i)/scale_r(i);
            gr(i,1)*=2.0*wtot(i)*vs(i)/scale_theta(i);
            gr(i,2)*=2.0*wtot(i)*vs(i)/scale_phi(i);
          }
          // Increment matrix
          increment_gga< std::complex<double> >(H,gr,bf,bf_rho,bf_theta,bf_phi);
        }

        if(do_mgga_t || do_mgga_l) {
          helfem::Vector vtl=helfem::Vector::Zero(wtot.size());
          if(do_mgga_t)
            vtl += 0.5*vtau.row(0).transpose();
          if(do_mgga_l)
            vtl += 2.0*vlapl.row(0).transpose();
          vtl=vtl.array()*wtot.array();

          helfem::Vector vtl_r=vtl.array()*inv_scale_r2.array();
          helfem::Vector vtl_th=vtl.array()*inv_scale_theta2.array();
          helfem::Vector vtl_phi=vtl.array()*inv_scale_phi2.array();
          increment_lda< std::complex<double> >(H,vtl_r,bf_rho);
          increment_lda< std::complex<double> >(H,vtl_th,bf_theta);
          increment_lda< std::complex<double> >(H,vtl_phi,bf_phi);
        }
        if(do_mgga_l) {
          helfem::Vector vl=vlapl.row(0).transpose().array()*wtot.array();
          helfem::dftgrid_common::increment_mgga_lapl< std::complex<double> >(H,vl,bf,bf_lapl);
        }

        for(size_t i=0;i<bf_ind.size();i++)
          for(size_t j=0;j<bf_ind.size();j++)
            Ho(bf_ind[i],bf_ind[j])+=H(i,j);
      }

      void DFTGridWorker::eval_Fxc(helfem::Matrix & Hao, helfem::Matrix & Hbo, bool beta) const {
        if(!polarized) {
          throw std::runtime_error("Refusing to compute unrestricted Fock matrix with restricted density.\n");
        }

        helfem::Matrix Ha=helfem::Matrix::Zero(bf_ind.size(),bf_ind.size());
        helfem::Matrix Hb;
        if(beta)
          Hb=helfem::Matrix::Zero(bf_ind.size(),bf_ind.size());

        {
          // LDA potential
          helfem::Vector vrhoa=vxc.row(0).transpose();
          // Multiply weights into potential
          vrhoa=vrhoa.array()*wtot.array();
          // Increment matrix
          increment_lda< std::complex<double> >(Ha,vrhoa,bf);

          if(beta) {
            helfem::Vector vrhob=vxc.row(1).transpose();
            vrhob=vrhob.array()*wtot.array();
            increment_lda< std::complex<double> >(Hb,vrhob,bf);
          }
        }
        if(!Ha.allFinite() || (beta && !Hb.allFinite()))
          //throw std::logic_error("NaN encountered!\n");
          fprintf(stderr,"NaN in Hamiltonian!\n");

        if(do_gga) {
          // Get vsigma
          helfem::Vector vs_aa=vsigma.row(0).transpose();
          helfem::Vector vs_ab=vsigma.row(1).transpose();

          // Get grad rho
          helfem::Matrix gr_a0=grho.topRows(3).transpose();
          helfem::Matrix gr_b0=grho.bottomRows(3).transpose();

          // Multiply grad rho by vsigma and the weights
          helfem::Matrix gr_a(gr_a0);
          for(Eigen::Index i=0;i<gr_a.rows();i++) {
            gr_a(i,0)=wtot(i)*(2.0*vs_aa(i)*gr_a0(i,0) + vs_ab(i)*gr_b0(i,0))/scale_r(i);
            gr_a(i,1)=wtot(i)*(2.0*vs_aa(i)*gr_a0(i,1) + vs_ab(i)*gr_b0(i,1))/scale_theta(i);
            gr_a(i,2)=wtot(i)*(2.0*vs_aa(i)*gr_a0(i,2) + vs_ab(i)*gr_b0(i,2))/scale_phi(i);
          }
          // Increment matrix
          increment_gga< std::complex<double> >(Ha,gr_a,bf,bf_rho,bf_theta,bf_phi);

          if(beta) {
            helfem::Vector vs_bb=vsigma.row(2).transpose();
            helfem::Matrix gr_b(gr_b0);
            for(Eigen::Index i=0;i<gr_b.rows();i++) {
              gr_b(i,0)=wtot(i)*(2.0*vs_bb(i)*gr_b0(i,0) + vs_ab(i)*gr_a0(i,0))/scale_r(i);
              gr_b(i,1)=wtot(i)*(2.0*vs_bb(i)*gr_b0(i,1) + vs_ab(i)*gr_a0(i,1))/scale_theta(i);
              gr_b(i,2)=wtot(i)*(2.0*vs_bb(i)*gr_b0(i,2) + vs_ab(i)*gr_a0(i,2))/scale_phi(i);
            }
            increment_gga< std::complex<double> >(Hb,gr_b,bf,bf_rho,bf_theta,bf_phi);
          }
        }


        if(do_mgga_t) {
          helfem::Vector vtl_a=helfem::Vector::Zero(wtot.size());
          if(do_mgga_t)
            vtl_a += 0.5*vtau.row(0).transpose();
          if(do_mgga_l)
            vtl_a += 2.0*vlapl.row(0).transpose();
          vtl_a=vtl_a.array()*wtot.array();

          helfem::Vector vtl_a_r=vtl_a.array()*inv_scale_r2.array();
          helfem::Vector vtl_a_th=vtl_a.array()*inv_scale_theta2.array();
          helfem::Vector vtl_a_phi=vtl_a.array()*inv_scale_phi2.array();
          increment_lda< std::complex<double> >(Ha,vtl_a_r,bf_rho);
          increment_lda< std::complex<double> >(Ha,vtl_a_th,bf_theta);
          increment_lda< std::complex<double> >(Ha,vtl_a_phi,bf_phi);
          if(beta) {
            helfem::Vector vtl_b=helfem::Vector::Zero(wtot.size());
            if(do_mgga_t)
              vtl_b += 0.5*vtau.row(1).transpose();
            if(do_mgga_l)
              vtl_b += 2.0*vlapl.row(1).transpose();
            vtl_b=vtl_b.array()*wtot.array();

            helfem::Vector vtl_b_r=vtl_b.array()*inv_scale_r2.array();
            helfem::Vector vtl_b_th=vtl_b.array()*inv_scale_theta2.array();
            helfem::Vector vtl_b_phi=vtl_b.array()*inv_scale_phi2.array();
            increment_lda< std::complex<double> >(Hb,vtl_b_r,bf_rho);
            increment_lda< std::complex<double> >(Hb,vtl_b_th,bf_theta);
            increment_lda< std::complex<double> >(Hb,vtl_b_phi,bf_phi);
          }
        }
        if(do_mgga_l) {
          helfem::Vector vl_a=vlapl.row(0).transpose().array()*wtot.array();
          helfem::Vector vl_b=vlapl.row(1).transpose().array()*wtot.array();
          helfem::dftgrid_common::increment_mgga_lapl< std::complex<double> >(Ha,vl_a,bf,bf_lapl);
          helfem::dftgrid_common::increment_mgga_lapl< std::complex<double> >(Hb,vl_b,bf,bf_lapl);
        }

        for(size_t i=0;i<bf_ind.size();i++)
          for(size_t j=0;j<bf_ind.size();j++)
            Hao(bf_ind[i],bf_ind[j])+=Ha(i,j);
        if(beta)
          for(size_t i=0;i<bf_ind.size();i++)
            for(size_t j=0;j<bf_ind.size();j++)
              Hbo(bf_ind[i],bf_ind[j])+=Hb(i,j);
      }

      // check_grad_tau_lapl, get_grad_tau_lapl, set_grad_tau_lapl:
      // inherited from helfem::dftgrid_common::DFTGridWorkerBase.

      void DFTGridWorker::compute_bf(size_t iel) {
        // Update function list
        bf_ind=basp->bf_list(iel);
        const Eigen::Index nbf=(Eigen::Index) bf_ind.size();

        // Get radii and radial weights
        helfem::Vector r(basp->get_r(iel));
        helfem::Vector wrad(basp->get_wrad(iel));
        const Eigen::Index nrad=wrad.size();
        const Eigen::Index nang=wang.size();

        // Calculate scale factors
        helfem::Vector sth(cth.size());
        for(Eigen::Index ia=0;ia<cth.size();ia++)
          sth(ia)=sqrt(1.0 - cth(ia)*cth(ia));

        // Radial is simple
        scale_r=helfem::Vector::Ones(nrad*nang);
        // Theta is a bit more complicated
        scale_theta.resize(nrad*nang);
        for(Eigen::Index ia=0;ia<nang;ia++)
          for(Eigen::Index ir=0;ir<nrad;ir++)
            scale_theta(ia*nrad+ir)=r(ir);
        // and so is phi
        scale_phi.resize(nrad*nang);
        for(Eigen::Index ia=0;ia<nang;ia++)
          for(Eigen::Index ir=0;ir<nrad;ir++)
            scale_phi(ia*nrad+ir)=r(ir)*sth(ia);

        // Pre-compute 1/scale^2 once. Scale_r is identically 1 (radial), so
        // its inverse-square is also a vector of ones.
        inv_scale_r2 = helfem::Vector::Ones(scale_r.size());
        inv_scale_theta2 = scale_theta.array().square().inverse();
        inv_scale_phi2 = scale_phi.array().square().inverse();

        // Update total weights
        wtot=helfem::Vector::Zero(nrad*nang);
        for(Eigen::Index ia=0;ia<nang;ia++)
          for(Eigen::Index ir=0;ir<nrad;ir++) {
            Eigen::Index idx=ia*nrad+ir;
            // sin(th) is already contained within wang, but we don't want to divide by it since it may be zero.
            wtot(idx)=wang(ia)*wrad(ir)*std::pow(r(ir),2);
          }

        // Compute basis function values
        bf=Eigen::MatrixXcd::Zero(nbf,wtot.size());
        // Loop over angular grid
        for(Eigen::Index ia=0;ia<cth.size();ia++) {
          // Evaluate basis functions at angular point (arma::cx_mat --
          // bridge to Eigen).
          arma::cx_mat abf(basp->eval_bf(iel, cth(ia), phi(ia)));
          if((Eigen::Index) abf.n_cols != nbf) {
            std::ostringstream oss;
            oss << "Mismatch! Have " << nbf << " basis function indices but " << abf.n_cols << " basis functions!\n";
            throw std::logic_error(oss.str());
          }
          Eigen::MatrixXcd abf_e(abf.n_rows, abf.n_cols);
          for(arma::uword c=0;c<abf.n_cols;c++)
            for(arma::uword rr=0;rr<abf.n_rows;rr++)
              abf_e(rr,c)=abf(rr,c);
          // Store functions (arma::trans on complex is the conjugate
          // transpose -> .adjoint()).
          bf.block(0,ia*nrad,nbf,nrad)=abf_e.adjoint();
        }

        if(do_grad) {
          bf_rho=Eigen::MatrixXcd::Zero(nbf,wtot.size());
          bf_theta=Eigen::MatrixXcd::Zero(nbf,wtot.size());
          bf_phi=Eigen::MatrixXcd::Zero(nbf,wtot.size());
          arma::cx_mat dr, dth, dphi;

          for(Eigen::Index ia=0;ia<cth.size();ia++) {
            // Evaluate basis functions at angular point
            basp->eval_df(iel, cth(ia), phi(ia), dr, dth, dphi);
            if((Eigen::Index) dr.n_cols != nbf) {
              std::ostringstream oss;
              oss << "Mismatch! Have " << nbf << " basis function indices but " << dr.n_cols << " basis functions!\n";
              throw std::logic_error(oss.str());
            }
            Eigen::MatrixXcd dr_e(dr.n_rows, dr.n_cols);
            Eigen::MatrixXcd dth_e(dth.n_rows, dth.n_cols);
            Eigen::MatrixXcd dphi_e(dphi.n_rows, dphi.n_cols);
            for(arma::uword c=0;c<dr.n_cols;c++)
              for(arma::uword rr=0;rr<dr.n_rows;rr++) {
                dr_e(rr,c)=dr(rr,c);
                dth_e(rr,c)=dth(rr,c);
                dphi_e(rr,c)=dphi(rr,c);
              }
            // Store functions
            bf_rho.block(0,ia*nrad,nbf,nrad)=dr_e.adjoint();
            bf_theta.block(0,ia*nrad,nbf,nrad)=dth_e.adjoint();
            bf_phi.block(0,ia*nrad,nbf,nrad)=dphi_e.adjoint();
          }
        }

        if(do_lapl) {
          bf_lapl=Eigen::MatrixXcd::Zero(nbf,wtot.size());
          // Loop over angular grid
          for(Eigen::Index ia=0;ia<cth.size();ia++) {
            // Evaluate basis functions at angular point
            arma::cx_mat alf(basp->eval_lf(iel, cth(ia), phi(ia)));
            if((Eigen::Index) alf.n_cols != nbf) {
              std::ostringstream oss;
              oss << "Mismatch! Have " << nbf << " basis function indices but " << alf.n_cols << " basis functions!\n";
              throw std::logic_error(oss.str());
            }
            Eigen::MatrixXcd alf_e(alf.n_rows, alf.n_cols);
            for(arma::uword c=0;c<alf.n_cols;c++)
              for(arma::uword rr=0;rr<alf.n_rows;rr++)
                alf_e(rr,c)=alf(rr,c);
            // Store functions
            bf_lapl.block(0,ia*nrad,nbf,nrad)=alf_e.adjoint();
          }
        }
      }

      DFTGrid::DFTGrid() {
      }

      DFTGrid::DFTGrid(const helfem::atomic::basis::TwoDBasis * basp_, int lang_, int mang_) : basp(basp_), lang(lang_), mang(mang_) {
        arma::vec cth, phi, wang;
        helfem::angular::angular_chebyshev(lang,mang,cth,phi,wang);
        printf("DFT angular grid of order l=%i m=%i has %i points\n",lang,mang,(int) wang.n_elem);
      }

      DFTGrid::~DFTGrid() {
      }

      void DFTGrid::eval_Fxc(int x_func, const helfem::Vector & x_pars, int c_func, const helfem::Vector & c_pars, const helfem::Matrix & P, helfem::Matrix & H, double & Exc, double & Nel, double & Ekin, double thr) {
        // Eigen throughout; the worker now consumes/produces Eigen matrices.
        H=helfem::Matrix::Zero(P.rows(),P.rows());

        double exc=0.0;
        double ekin=0.0;
        double nel=0.0;
        double lapl=0;
#ifdef _OPENMP
#pragma omp parallel reduction(+:exc,nel,lapl)
#endif
        {
          DFTGridWorker grid(basp,lang,mang);
          grid.check_grad_tau_lapl(x_func,c_func);

#ifdef _OPENMP
#pragma omp for
#endif
          for(size_t iel=0;iel<basp->get_rad_Nel();iel+=2) {
            grid.compute_bf(iel);
            grid.update_density(P);
            nel+=grid.compute_Nel();
            ekin+=grid.compute_Ekin();
            lapl+=grid.compute_laplsum();

            grid.init_xc();
            if(x_func>0)
              grid.compute_xc(x_func, x_pars, thr);
            if(c_func>0)
              grid.compute_xc(c_func, c_pars, thr);

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
            ekin+=grid.compute_Ekin();
            lapl+=grid.compute_laplsum();

            grid.init_xc();
            if(x_func>0)
              grid.compute_xc(x_func, x_pars, thr);
            if(c_func>0)
              grid.compute_xc(c_func, c_pars, thr);

            exc+=grid.eval_Exc();
            grid.eval_Fxc(H);
          }
        }

        // Save outputs
        Exc=exc;
        Ekin=ekin;
        Nel=nel;

        printf("Integral over laplacian %e\n",lapl);
      }

      void DFTGrid::eval_Fxc(int x_func, const helfem::Vector & x_pars, int c_func, const helfem::Vector & c_pars, const helfem::Matrix & Pa, const helfem::Matrix & Pb, helfem::Matrix & Ha, helfem::Matrix & Hb, double & Exc, double & Nel, double & Ekin, bool beta, double thr) {
        // Eigen throughout; the worker now consumes/produces Eigen matrices.
        Ha=helfem::Matrix::Zero(Pa.rows(),Pa.rows());
        Hb=helfem::Matrix::Zero(Pb.rows(),Pb.rows());

        double exc=0.0;
        double nel=0.0;
        double ekin=0.0;
#ifdef _OPENMP
#pragma omp parallel reduction(+:exc,nel)
#endif
        {
          DFTGridWorker grid(basp,lang,mang);
          grid.check_grad_tau_lapl(x_func,c_func);

#ifdef _OPENMP
#pragma omp for
#endif
          for(size_t iel=0;iel<basp->get_rad_Nel();iel+=2) {
            grid.compute_bf(iel);
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
#ifdef _OPENMP
#pragma omp for
#endif
          for(size_t iel=1;iel<basp->get_rad_Nel();iel+=2) {
            grid.compute_bf(iel);
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

        // Save outputs
        Exc=exc;
        Ekin=ekin;
        Nel=nel;
      }

    }
  }
}
