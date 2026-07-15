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

      void DFTGridWorker::update_density(const std::vector<helfem::Matrix> & Pc0) {
        const size_t nslices = Pc0.size();
        const Eigen::Index nbf = (Eigen::Index) bf_ind.size();

        // In-element per-l density matrices (gathered submatrices)
        std::vector<helfem::Matrix> Pc(nslices);
        for(size_t islice=0; islice<nslices; islice++) {
          Pc[islice] = helfem::Matrix(nbf, nbf);
          for(size_t i=0;i<bf_ind.size();i++)
            for(size_t j=0;j<bf_ind.size();j++)
              Pc[islice](i,j) = Pc0[islice](bf_ind[i], bf_ind[j]);
        }
        // Total density matrix
        helfem::Matrix P = helfem::Matrix::Zero(nbf, nbf);
        for(size_t islice=0; islice<nslices; islice++) {
          P += Pc[islice];
        }
        // and the one multiplied by l(l+1)
        helfem::Matrix Pl = helfem::Matrix::Zero(nbf, nbf);
        for(size_t islice=0; islice<nslices; islice++) {
          Pl += ((double)(islice*(islice+1)))*Pc[islice];
        }

        // Non-polarized calculation.
        polarized=false;

        // Update density vector (bf is real, so conj(bf)==bf)
        Pv=P*bf;

        // Calculate density
        rho = helfem::Matrix::Zero(1, wtot.size());
        for(Eigen::Index ip=0;ip<wtot.size();ip++)
          rho(0,ip)=(Pv.col(ip).array()*bf.col(ip).array()).sum();

        // Calculate gradient
        if(do_grad) {
          grho = helfem::Matrix::Zero(1, wtot.size());
          sigma = helfem::Matrix::Zero(1, wtot.size());
          for(Eigen::Index ip=0;ip<wtot.size();ip++) {
            // Calculate values
            double g_rad=grho(0,ip)=2.0*(Pv.col(ip).array()*bf_rho.col(ip).array()).sum();
            // Compute sigma as well
            sigma(0,ip)=g_rad*g_rad;
          }
        }

        // Calculate kinetic energy density
        if(do_tau || do_lapl) {
          helfem::Matrix Pvp = P*bf_rho;

          if(do_tau) {
            helfem::Matrix Plv = Pl*bf;
            tau = helfem::Matrix::Zero(1, wtot.size());
            for(Eigen::Index ip=0;ip<wtot.size();ip++) {
              // First term: P(u,v) * \chi_u' \chi_v'
              double term1 = (Pvp.col(ip).array()*bf_rho.col(ip).array()).sum();
              // Second term: l(l+1) Pl(u,v) \chi_u \chi_v / r^2
              double term2 = (Plv.col(ip).array()*bf.col(ip).array()).sum()/(r(ip)*r(ip));
              // The second term is ill-behaved near the nucleus since
              // only s orbitals contribute to density but that gets
              // killed by the l(l+1) factor
              tau(0,ip) = 0.5*(term1 + std::max(term2, 0.0));
            }
          }

          if(do_lapl) {
            lapl = helfem::Matrix::Zero(1, wtot.size());
            for(Eigen::Index ip=0;ip<wtot.size();ip++) {
              // First term: P(u,v) * \chi_u' \chi_v'
              double term1 = 2.0*(Pvp.col(ip).array()*bf_rho.col(ip).array()).sum();
              // Second term: P(u,v) \chi_u \chi_v''
              double term2 = 2.0*(Pv.col(ip).array()*bf_rho2.col(ip).array()).sum();
              // Third term: P(u,v) * \chi_u \chi_v' / r
              double term3 = 4.0*(Pv.col(ip).array()*bf_rho.col(ip).array()).sum()/r(ip);

              // Store values
              lapl(0,ip)=term1+term2+term3;
            }
          }
        }
      }

      void DFTGridWorker::update_density(const std::vector<helfem::Matrix> & Pac0, const std::vector<helfem::Matrix> & Pbc0) {
        if(Pac0.empty() || Pbc0.empty()) {
          throw std::runtime_error("Error - density matrix is empty!\n");
        }

        const size_t naslices = Pac0.size();
        const size_t nbslices = Pbc0.size();
        const Eigen::Index nbf = (Eigen::Index) bf_ind.size();

        // In-element per-l density matrices (gathered submatrices)
        std::vector<helfem::Matrix> Pac(naslices);
        for(size_t islice=0; islice<naslices; islice++) {
          Pac[islice] = helfem::Matrix(nbf, nbf);
          for(size_t i=0;i<bf_ind.size();i++)
            for(size_t j=0;j<bf_ind.size();j++)
              Pac[islice](i,j) = Pac0[islice](bf_ind[i], bf_ind[j]);
        }
        std::vector<helfem::Matrix> Pbc(nbslices);
        for(size_t islice=0; islice<nbslices; islice++) {
          Pbc[islice] = helfem::Matrix(nbf, nbf);
          for(size_t i=0;i<bf_ind.size();i++)
            for(size_t j=0;j<bf_ind.size();j++)
              Pbc[islice](i,j) = Pbc0[islice](bf_ind[i], bf_ind[j]);
        }
        // Total density matrix
        helfem::Matrix Pa = helfem::Matrix::Zero(nbf, nbf);
        for(size_t islice=0; islice<naslices; islice++) {
          Pa += Pac[islice];
        }
        helfem::Matrix Pb = helfem::Matrix::Zero(nbf, nbf);
        for(size_t islice=0; islice<nbslices; islice++) {
          Pb += Pbc[islice];
        }
        // and the one multiplied by l(l+1)
        helfem::Matrix Pal = helfem::Matrix::Zero(nbf, nbf);
        for(size_t islice=0; islice<naslices; islice++) {
          Pal += ((double)(islice*(islice+1)))*Pac[islice];
        }
        helfem::Matrix Pbl = helfem::Matrix::Zero(nbf, nbf);
        for(size_t islice=0; islice<nbslices; islice++) {
          Pbl += ((double)(islice*(islice+1)))*Pbc[islice];
        }

        // Polarized calculation.
        polarized=true;

        // Update density vector (bf is real, so conj(bf)==bf)
        Pav=Pa*bf;
        Pbv=Pb*bf;

        // Calculate density
        rho = helfem::Matrix::Zero(2, wtot.size());
        for(Eigen::Index ip=0;ip<wtot.size();ip++) {
          rho(0,ip)=(Pav.col(ip).array()*bf.col(ip).array()).sum();
          rho(1,ip)=(Pbv.col(ip).array()*bf.col(ip).array()).sum();
        }

        // Calculate gradient
        if(do_grad) {
          grho = helfem::Matrix::Zero(2, wtot.size());
          sigma = helfem::Matrix::Zero(3, wtot.size());
          for(Eigen::Index ip=0;ip<wtot.size();ip++) {
            double ga_rad=grho(0,ip)=2.0*(Pav.col(ip).array()*bf_rho.col(ip).array()).sum();
            double gb_rad=grho(1,ip)=2.0*(Pbv.col(ip).array()*bf_rho.col(ip).array()).sum();

            // Compute sigma as well
            sigma(0,ip)=ga_rad*ga_rad;
            sigma(1,ip)=ga_rad*gb_rad;
            sigma(2,ip)=gb_rad*gb_rad;
          }
        }

        // Calculate kinetic energy density
        if(do_tau || do_lapl) {
          helfem::Matrix Pavp = Pa*bf_rho;
          helfem::Matrix Pbvp = Pb*bf_rho;

          if(do_tau) {
            helfem::Matrix Palv = Pal*bf;
            helfem::Matrix Pblv = Pbl*bf;
            tau = helfem::Matrix::Zero(2, wtot.size());
            for(Eigen::Index ip=0;ip<wtot.size();ip++) {
              // First term: P(u,v) * \chi_u' \chi_v'
              double term1a = (Pavp.col(ip).array()*bf_rho.col(ip).array()).sum();
              double term1b = (Pbvp.col(ip).array()*bf_rho.col(ip).array()).sum();
              // Second term: l(l+1) Pl(u,v) \chi_u \chi_v / r^2
              double term2a = (Palv.col(ip).array()*bf.col(ip).array()).sum()/(r(ip)*r(ip));
              double term2b = (Pblv.col(ip).array()*bf.col(ip).array()).sum()/(r(ip)*r(ip));
              // The second term is ill-behaved near the nucleus since
              // only s orbitals contribute to density but that gets
              // killed by the l(l+1) factor
              tau(0,ip) = 0.5*(term1a + std::max(term2a, 0.0));
              tau(1,ip) = 0.5*(term1b + std::max(term2b, 0.0));
            }
          }

          if(do_lapl) {
            lapl = helfem::Matrix::Zero(2, wtot.size());
            for(Eigen::Index ip=0;ip<wtot.size();ip++) {
              // First term: P(u,v) * \chi_u' \chi_v'
              double term1a = 2.0*(Pavp.col(ip).array()*bf_rho.col(ip).array()).sum();
              double term1b = 2.0*(Pbvp.col(ip).array()*bf_rho.col(ip).array()).sum();
              // Second term: P(u,v) \chi_u \chi_v''
              double term2a = 2.0*(Pav.col(ip).array()*bf_rho2.col(ip).array()).sum();
              double term2b = 2.0*(Pbv.col(ip).array()*bf_rho2.col(ip).array()).sum();
              // Third term: P(u,v) * \chi_u \chi_v' / r
              double term3a = 4.0*(Pav.col(ip).array()*bf_rho.col(ip).array()).sum()/r(ip);
              double term3b = 4.0*(Pbv.col(ip).array()*bf_rho.col(ip).array()).sum()/r(ip);

              // Store values
              lapl(0,ip)=term1a+term2a+term3a;
              lapl(1,ip)=term1b+term2b+term3b;
            }
          }
        }
      }



      // init_xc, zero_Exc: inherited from
      // helfem::dftgrid_common::DFTGridWorkerBase.

      // The compute_xc implementation is inherited; the NaN-guard
      // diagnostic that lived here is now in the base and runs for
      // atomic and diatomic too.


      // eval_Exc: inherited from DFTGridWorkerBase.

      void DFTGridWorker::eval_Fxc(std::vector<helfem::Matrix> & Ho) const {
        if(polarized) {
          throw std::runtime_error("Refusing to compute restricted Fock matrix with unrestricted density.\n");
        }

        const Eigen::Index nbf = (Eigen::Index) bf_ind.size();

        // Work matrix
        helfem::Matrix H = helfem::Matrix::Zero(nbf,nbf);

        // l-dependent term
        helfem::Matrix Hl = helfem::Matrix::Zero(nbf,nbf);

        {
          // LDA potential
          helfem::Vector vrho = vxc.row(0).transpose();
          // Multiply weights into potential
          vrho = vrho.array() * wtot.array();
          // Increment matrix
          increment_lda<double>(H,vrho,bf);
        }
        if(!H.allFinite())
          fprintf(stderr,"NaN in Hamiltonian after LDA!\n");

        if(do_gga) {
          // Get vsigma
          helfem::Vector vs = vsigma.row(0).transpose();
          // Get grad rho (single spin channel) as a column
          helfem::Matrix gr = grho.row(0).transpose();
          // Multiply grad rho by vsigma and the weights
          gr.col(0).array() *= 2.0*(wtot.array()*vs.array());
          // If we also have laplacian dependence, we get an extra term
          if(do_mgga_l) {
            gr.col(0).array() += 2.0*vlapl.row(0).transpose().array()*r.array()*(wrad.array()*4.0*M_PI);
          }
          // Increment matrix
          increment_gga<double>(H,gr,bf,bf_rho);
          if(!H.allFinite())
            fprintf(stderr,"NaN in Hamiltonian after GGA!\n");
        }

        if(do_mgga_t || do_mgga_l) {
          helfem::Vector vtl = helfem::Vector::Zero(wtot.size());
          if(do_mgga_t)
            vtl += 0.5*vtau.row(0).transpose();
          if(do_mgga_l)
            vtl += 2.0*vlapl.row(0).transpose();
          vtl = vtl.array() * wtot.array();
          // Base term
          increment_lda<double>(H,vtl,bf_rho);

          if(do_mgga_t) {
            // l(l+1) term: r^-2 cancels out the factor in the total weight
            vtl = vtau.row(0).transpose().array() * (0.5*wrad.array()*4.0*M_PI);
            increment_lda<double>(Hl,vtl,bf);
          }
          if(do_mgga_l) {
            // Laplacian term
            vtl = vlapl.row(0).transpose().array() * wtot.array();
            increment_mgga_lapl<double>(H,vtl,bf,bf_rho2);
          }
          if(!H.allFinite())
            fprintf(stderr,"NaN in Hamiltonian after mGGA!\n");
        }

        // Collect results
        for(size_t islice=0;islice<Ho.size();islice++) {
          helfem::Matrix Hs = H + ((double)(islice*(islice+1)))*Hl;
          for(size_t i=0;i<bf_ind.size();i++)
            for(size_t j=0;j<bf_ind.size();j++)
              Ho[islice](bf_ind[i],bf_ind[j]) += Hs(i,j);
        }
      }

      void DFTGridWorker::eval_Fxc(std::vector<helfem::Matrix> & Hao, std::vector<helfem::Matrix> & Hbo, bool beta) const {
        if(!polarized) {
          throw std::runtime_error("Refusing to compute unrestricted Fock matrix with restricted density.\n");
        }

        const Eigen::Index nbf = (Eigen::Index) bf_ind.size();

        helfem::Matrix Ha = helfem::Matrix::Zero(nbf,nbf);
        helfem::Matrix Hb;
        if(beta)
          Hb = helfem::Matrix::Zero(nbf,nbf);

        helfem::Matrix Hal = helfem::Matrix::Zero(nbf,nbf);
        helfem::Matrix Hbl;
        if(beta)
          Hbl = helfem::Matrix::Zero(nbf,nbf);

        {
          // LDA potential
          helfem::Vector vrhoa = vxc.row(0).transpose();
          // Multiply weights into potential
          vrhoa = vrhoa.array() * wtot.array();
          // Increment matrix
          increment_lda<double>(Ha,vrhoa,bf);

          if(beta) {
            helfem::Vector vrhob = vxc.row(1).transpose();
            vrhob = vrhob.array() * wtot.array();
            increment_lda<double>(Hb,vrhob,bf);
          }
        }
        if(!Ha.allFinite() || (beta && !Hb.allFinite()))
          //throw std::logic_error("NaN encountered!\n");
          fprintf(stderr,"NaN in Hamiltonian after LDA!\n");

        if(do_gga) {
          // Get vsigma
          helfem::Vector vs_aa = vsigma.row(0).transpose();
          helfem::Vector vs_ab = vsigma.row(1).transpose();

          // Get grad rho (columns)
          helfem::Vector gr_a0 = grho.row(0).transpose();
          helfem::Vector gr_b0 = grho.row(1).transpose();

          // Multiply grad rho by vsigma and the weights
          helfem::Matrix gr_a(wtot.size(),1);
          gr_a.col(0) = (wtot.array()*(2.0*vs_aa.array()*gr_a0.array() + vs_ab.array()*gr_b0.array())).matrix();
          // If we also have laplacian dependence, we get an extra term
          if(do_mgga_l) {
            gr_a.col(0).array() += 2.0*vlapl.row(0).transpose().array()*r.array()*(wrad.array()*4.0*M_PI);
          }
          // Increment matrix
          increment_gga<double>(Ha,gr_a,bf,bf_rho);

          if(beta) {
            helfem::Vector vs_bb = vsigma.row(2).transpose();
            helfem::Matrix gr_b(wtot.size(),1);
            gr_b.col(0) = (wtot.array()*(2.0*vs_bb.array()*gr_b0.array() + vs_ab.array()*gr_a0.array())).matrix();
            if(do_mgga_l) {
              gr_b.col(0).array() += 2.0*vlapl.row(1).transpose().array()*r.array()*(wrad.array()*4.0*M_PI);
            }
            increment_gga<double>(Hb,gr_b,bf,bf_rho);
          }
          if(!Ha.allFinite() || (beta && !Hb.allFinite()))
            //throw std::logic_error("NaN encountered!\n");
            fprintf(stderr,"NaN in Hamiltonian after GGA!\n");
        }

        if(do_mgga_t || do_mgga_l) {
          helfem::Vector vtl_a = helfem::Vector::Zero(wtot.size());
          if(do_mgga_t)
            vtl_a += 0.5*vtau.row(0).transpose();
          if(do_mgga_l)
            vtl_a += 2.0*vlapl.row(0).transpose();
          vtl_a = vtl_a.array() * wtot.array();

          // Base term
          increment_lda<double>(Ha,vtl_a,bf_rho);

          if(do_mgga_t) {
            // l(l+1) term: r^-2 cancels out the factor in the total weight
            vtl_a = vtau.row(0).transpose().array() * (0.5*wrad.array()*4.0*M_PI);
            increment_lda<double>(Hal,vtl_a,bf);
          }
          if(do_mgga_l) {
            vtl_a = vlapl.row(0).transpose().array() * wtot.array();
            increment_mgga_lapl<double>(Ha,vtl_a,bf,bf_rho2);
          }
          if(beta) {
            helfem::Vector vtl_b = helfem::Vector::Zero(wtot.size());
            if(do_mgga_t)
              vtl_b += 0.5*vtau.row(1).transpose();
            if(do_mgga_l)
              vtl_b += 2.0*vlapl.row(1).transpose();
            vtl_b = vtl_b.array() * wtot.array();

            // Base term
            increment_lda<double>(Hb,vtl_b,bf_rho);

            if(do_mgga_t) {
              // l(l+1) term: r^-2 cancels out the factor in the total weight
              vtl_b = vtau.row(1).transpose().array() * (0.5*wrad.array()*4.0*M_PI);
              increment_lda<double>(Hbl,vtl_b,bf);
            }
            if(do_mgga_l) {
              vtl_b = vlapl.row(1).transpose().array() * wtot.array();
              increment_mgga_lapl<double>(Hb,vtl_b,bf,bf_rho2);
            }
          }
          if(!Ha.allFinite() || (beta && !Hb.allFinite()))
            //throw std::logic_error("NaN encountered!\n");
            fprintf(stderr,"NaN in Hamiltonian after mGGA!\n");
        }

        // Collect results
        for(size_t islice=0;islice<Hao.size();islice++) {
          helfem::Matrix Hs = Ha + ((double)(islice*(islice+1)))*Hal;
          for(size_t i=0;i<bf_ind.size();i++)
            for(size_t j=0;j<bf_ind.size();j++)
              Hao[islice](bf_ind[i],bf_ind[j]) += Hs(i,j);
        }
        if(beta) {
          for(size_t islice=0;islice<Hbo.size();islice++) {
            helfem::Matrix Hs = Hb + ((double)(islice*(islice+1)))*Hbl;
            for(size_t i=0;i<bf_ind.size();i++)
              for(size_t j=0;j<bf_ind.size();j++)
                Hbo[islice](bf_ind[i],bf_ind[j]) += Hs(i,j);
          }
        }
      }

      // check_grad_tau_lapl, get_grad_tau_lapl, set_grad_tau_lapl:
      // inherited from DFTGridWorkerBase.

      void DFTGridWorker::compute_bf(size_t iel) {
        // Update function list (basis returns vector<Eigen::Index>)
        bf_ind = basp->bf_list(iel);

        // Get radii
        r = basp->get_r(iel);
        // Get radial weights
        wrad = basp->get_wrad(iel);

        // Update total weights
        wtot = 4.0*M_PI * wrad.array() * r.array().square();

        // Compute basis function values (transpose to Nbf x Npts)
        bf = basp->eval_bf(iel).transpose();

        if(do_grad) {
          bf_rho = basp->eval_df(iel).transpose();
        }

        if(do_lapl) {
          bf_rho2 = basp->eval_lf(iel).transpose();
        }
      }

      DFTGrid::DFTGrid() {
      }

      DFTGrid::DFTGrid(const helfem::sadatom::basis::TwoDBasis * basp_) : basp(basp_) {
      }

      DFTGrid::~DFTGrid() {
      }

      void DFTGrid::eval_Fxc(int x_func, const helfem::Vector & x_pars, int c_func, const helfem::Vector & c_pars, const helfem::Cube & P, helfem::Cube & H, double & Exc, double & Nel, double thr) {
        const Eigen::Index Nrad = P[0].rows();

        // Per-l density cube is already a vector of Eigen matrices.
        const std::vector<helfem::Matrix> & Pvec = P;

        // Shared Eigen Fock accumulator (one matrix per l-slice)
        std::vector<helfem::Matrix> Hvec(P.size());
        for(size_t is=0; is<P.size(); is++)
          Hvec[is] = helfem::Matrix::Zero(Nrad, Nrad);

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
            grid.update_density(Pvec);
            nel+=grid.compute_Nel();

            grid.init_xc();
            if(x_func>0)
              grid.compute_xc(x_func,x_pars,thr);
            if(c_func>0)
              grid.compute_xc(c_func,c_pars,thr);

            exc+=grid.eval_Exc();
            grid.eval_Fxc(Hvec);
          }
#ifdef _OPENMP
#pragma omp for
#endif
          for(size_t iel=1;iel<basp->get_rad_Nel();iel+=2) {
            grid.compute_bf(iel);
            grid.update_density(Pvec);
            nel+=grid.compute_Nel();

            grid.init_xc();
            if(x_func>0)
              grid.compute_xc(x_func,x_pars,thr);
            if(c_func>0)
              grid.compute_xc(c_func,c_pars,thr);

            exc+=grid.eval_Exc();
            grid.eval_Fxc(Hvec);
          }
        }

        // Move Fock accumulator into the output cube
        H = std::move(Hvec);

        // Save outputs
        Exc=exc;
        Nel=nel;
      }

      void DFTGrid::eval_Fxc(int x_func, const helfem::Vector & x_pars, int c_func, const helfem::Vector & c_pars, const helfem::Cube & Pa, const helfem::Cube & Pb, helfem::Cube & Ha, helfem::Cube & Hb, double & Exc, double & Nel, bool beta, double thr) {
        const Eigen::Index Nrad_a = Pa[0].rows();
        const Eigen::Index Nrad_b = Pb[0].rows();

        // Per-l density cubes are already vectors of Eigen matrices.
        const std::vector<helfem::Matrix> & Pavec = Pa;
        const std::vector<helfem::Matrix> & Pbvec = Pb;

        // Shared Eigen Fock accumulators (one matrix per l-slice)
        std::vector<helfem::Matrix> Havec(Pa.size());
        for(size_t is=0; is<Pa.size(); is++)
          Havec[is] = helfem::Matrix::Zero(Nrad_a, Nrad_a);
        std::vector<helfem::Matrix> Hbvec(Pb.size());
        for(size_t is=0; is<Pb.size(); is++)
          Hbvec[is] = helfem::Matrix::Zero(Nrad_b, Nrad_b);

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
            grid.update_density(Pavec,Pbvec);
            nel+=grid.compute_Nel();

            grid.init_xc();
            if(x_func>0)
              grid.compute_xc(x_func,x_pars,thr);
            if(c_func>0)
              grid.compute_xc(c_func,c_pars,thr);

            exc+=grid.eval_Exc();
            grid.eval_Fxc(Havec,Hbvec,beta);
          }
#ifdef _OPENMP
#pragma omp for
#endif
          for(size_t iel=1;iel<basp->get_rad_Nel();iel+=2) {
            grid.compute_bf(iel);
            grid.update_density(Pavec,Pbvec);
            nel+=grid.compute_Nel();

            grid.init_xc();
            if(x_func>0)
              grid.compute_xc(x_func,x_pars,thr);
            if(c_func>0)
              grid.compute_xc(c_func,c_pars,thr);

            exc+=grid.eval_Exc();
            grid.eval_Fxc(Havec,Hbvec,beta);
          }
        }

        // Move Fock accumulators into the output cubes
        Ha = std::move(Havec);
        if(beta) {
          Hb = std::move(Hbvec);
        } else {
          Hb.assign(Pb.size(), helfem::Matrix::Zero(Nrad_b, Nrad_b));
        }

        // Save outputs
        Exc=exc;
        Nel=nel;
      }




    }
  }
}
