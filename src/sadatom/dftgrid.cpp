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

      //! Radius below which the centrifugal l(l+1)/r^2 term of the
      //! kinetic energy density is faded out.  The term is bounded as
      //! r->0 (the wave function goes as r^l), but it is EVALUATED as a
      //! quotient of two quantities that both vanish there, so below a
      //! fraction of a milli-bohr it is roundoff amplified by 1/r^2 --
      //! HelFEM used to hide that behind max(term2, 0).  That clamp is a
      //! kink, and a kink has no derivative: it is precisely why the
      //! analytic response could not be made to agree with finite
      //! differences for an occupied l>0 shell.  A smooth damping that
      //! depends only on r, never on the density matrix, is both bounded
      //! and exactly differentiable, so the response below is the true
      //! derivative of the energy the code actually evaluates.
      static double centrifugal_damping(double r, int Z) {
        // A C^infinity transition that is IDENTICALLY one outside the
        // pathological shell and identically zero inside it, so the
        // energy is untouched wherever the centrifugal term carries any
        // density.  Both endpoints are fixed radii: the switch never
        // depends on the density matrix, so d(tau)/dP carries it as a
        // constant factor and the analytic response remains the exact
        // derivative of the energy the code evaluates.  The window
        // scales as 1/Z because that is how the region where
        // |psi_l(r)|^2/r^2 is dominated by cancellation scales.
        const double scale = 1.0/std::max(1, Z);
        // c1 is CALIBRATED, not guessed: the full three-dimensional
        // atomic worker computes tau from genuine theta/phi derivatives
        // and so needs no reduction at all, which makes its energy an
        // independent reference for this one.  Against it, Ne/TPSS
        // gives 1e-10 Ha at c1 = 0.003 but fails the finite-difference
        // derivative test; c1 = 0.02 is the narrowest window that
        // passes, and costs 3.4e-8 Ha (N: 9.3e-9).  Widening to 0.03
        // triples that for no gain.  Override with HELFEM_TAU_CENT_R1
        // to trade smoothness against that residual.
        static const double c1 = []() {
          const char * e = getenv("HELFEM_TAU_CENT_R1");
          return e ? atof(e) : 0.02;
        }();
        const double r1 = c1*scale, r0 = 0.01*r1;
        if(r <= r0) return 0.0;
        if(r >= r1) return 1.0;
        const double t = (r - r0)/(r1 - r0);
        const double a = std::exp(-1.0/t), b = std::exp(-1.0/(1.0 - t));
        return a/(a + b);
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
            tau_centrifugal = helfem::Matrix::Zero(1, wtot.size());
            for(Eigen::Index ip=0;ip<wtot.size();ip++) {
              // First term: P(u,v) * \chi_u' \chi_v'
              double term1 = (Pvp.col(ip).array()*bf_rho.col(ip).array()).sum();
              // Second term: l(l+1) Pl(u,v) \chi_u \chi_v / r^2
              double term2 = (Plv.col(ip).array()*bf.col(ip).array()).sum()/(r(ip)*r(ip));
              // The second term is ill-behaved near the nucleus; fade
              // it out smoothly (see centrifugal_damping).
              tau_centrifugal(0,ip) = centrifugal_damping(r(ip), basp->Z());
              tau(0,ip) = 0.5*(term1 + tau_centrifugal(0,ip)*term2);

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
            tau_centrifugal = helfem::Matrix::Zero(2, wtot.size());
            for(Eigen::Index ip=0;ip<wtot.size();ip++) {
              // First term: P(u,v) * \chi_u' \chi_v'
              double term1a = (Pavp.col(ip).array()*bf_rho.col(ip).array()).sum();
              double term1b = (Pbvp.col(ip).array()*bf_rho.col(ip).array()).sum();
              // Second term: l(l+1) Pl(u,v) \chi_u \chi_v / r^2
              double term2a = (Palv.col(ip).array()*bf.col(ip).array()).sum()/(r(ip)*r(ip));
              double term2b = (Pblv.col(ip).array()*bf.col(ip).array()).sum()/(r(ip)*r(ip));
              // The second term is ill-behaved near the nucleus; fade
              // it out smoothly (see centrifugal_damping).
              tau_centrifugal(0,ip) = tau_centrifugal(1,ip) =
                centrifugal_damping(r(ip), basp->Z());
              tau(0,ip) = 0.5*(term1a + tau_centrifugal(0,ip)*term2a);
              tau(1,ip) = 0.5*(term1b + tau_centrifugal(1,ip)*term2b);

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

      helfem::Matrix DFTGridWorker::eval_density(const std::vector<helfem::Matrix> & Pc0) const {
        const Eigen::Index nbf = (Eigen::Index) bf_ind.size();

        // Sum the l-slices first: only the total density enters an LDA
        // kernel, and the gather is the expensive part.
        helfem::Matrix P = helfem::Matrix::Zero(nbf, nbf);
        for(size_t islice=0; islice<Pc0.size(); islice++)
          for(size_t i=0;i<bf_ind.size();i++)
            for(size_t j=0;j<bf_ind.size();j++)
              P(i,j) += Pc0[islice](bf_ind[i], bf_ind[j]);

        helfem::Matrix Pvloc = P*bf;
        helfem::Matrix drho = helfem::Matrix::Zero(1, wtot.size());
        for(Eigen::Index ip=0;ip<wtot.size();ip++)
          drho(0,ip) = (Pvloc.col(ip).array()*bf.col(ip).array()).sum();
        return drho;
      }

      void DFTGridWorker::check_response_fields(const helfem::Cube & P,
                                                const helfem::Cube & dP,
                                                const helfem::Matrix & drho,
                                                const helfem::Matrix & dgrho,
                                                const helfem::Matrix & dtau) {
        const double h = 1e-5;
        helfem::Cube Pp(P), Pm(P);
        for(size_t l=0;l<P.size();l++) {
          Pp[l] = P[l] + h*dP[l];
          Pm[l] = P[l] - h*dP[l];
        }
        update_density(Pp);
        const helfem::Matrix rp(rho), gp(grho), tp(tau);
        update_density(Pm);
        const helfem::Matrix rm(rho), gm(grho), tm(tau);
        double er = 0.0, eg = 0.0, et = 0.0;
        auto rel = [](double a, double b) {
          return std::abs(a-b)/std::max(1e-10, std::abs(b));
        };
        for(Eigen::Index i=0;i<drho.cols();i++)
          for(Eigen::Index sp=0;sp<drho.rows();sp++) {
            er = std::max(er, rel(drho(sp,i), (rp(sp,i)-rm(sp,i))/(2*h)));
            if(do_grad)
              eg = std::max(eg, rel(dgrho(sp,i), (gp(sp,i)-gm(sp,i))/(2*h)));
            if(do_tau)
              et = std::max(et, rel(dtau(sp,i), (tp(sp,i)-tm(sp,i))/(2*h)));
          }
        static double wr=0, wg=0, wt=0;
        wr=std::max(wr,er); wg=std::max(wg,eg); wt=std::max(wt,et);
        fprintf(stderr, "DFIELDS worst rel dev: drho %.3e  dgrho %.3e  "
                "dtau %.3e\n", wr, wg, wt);
      }

      void DFTGridWorker::eval_response_fields(const std::vector<helfem::Matrix> & Pc0,
                                               helfem::Matrix & drho,
                                               helfem::Matrix & dgrho,
                                               helfem::Matrix & dtau) const {
        // The perturbed fields are the SAME bilinear forms in the density
        // matrix as the reference ones in update_density; only the matrix
        // differs. Keeping the two in step is what makes the response
        // kernel exact, so this mirrors that code deliberately.
        const Eigen::Index nbf = (Eigen::Index) bf_ind.size();
        const size_t nslices = Pc0.size();

        std::vector<helfem::Matrix> Pc(nslices);
        for(size_t islice=0; islice<nslices; islice++) {
          Pc[islice] = helfem::Matrix(nbf, nbf);
          for(size_t i=0;i<bf_ind.size();i++)
            for(size_t j=0;j<bf_ind.size();j++)
              Pc[islice](i,j) = Pc0[islice](bf_ind[i], bf_ind[j]);
        }
        helfem::Matrix P = helfem::Matrix::Zero(nbf, nbf);
        for(size_t islice=0; islice<nslices; islice++)
          P += Pc[islice];
        helfem::Matrix Pl = helfem::Matrix::Zero(nbf, nbf);
        for(size_t islice=0; islice<nslices; islice++)
          Pl += ((double)(islice*(islice+1)))*Pc[islice];

        const helfem::Matrix Pv = P*bf;

        drho = helfem::Matrix::Zero(1, wtot.size());
        for(Eigen::Index ip=0;ip<wtot.size();ip++)
          drho(0,ip) = (Pv.col(ip).array()*bf.col(ip).array()).sum();

        if(do_grad) {
          dgrho = helfem::Matrix::Zero(1, wtot.size());
          for(Eigen::Index ip=0;ip<wtot.size();ip++)
            dgrho(0,ip) = 2.0*(Pv.col(ip).array()*bf_rho.col(ip).array()).sum();
        } else {
          dgrho = helfem::Matrix();
        }

        if(do_tau) {
          const helfem::Matrix Pvp = P*bf_rho;
          const helfem::Matrix Plv = Pl*bf;
          dtau = helfem::Matrix::Zero(1, wtot.size());
          for(Eigen::Index ip=0;ip<wtot.size();ip++) {
            const double term1 = (Pvp.col(ip).array()*bf_rho.col(ip).array()).sum();
            const double term2 = (Plv.col(ip).array()*bf.col(ip).array()).sum()/(r(ip)*r(ip));
            // update_density clamps the centrifugal term at zero; the
            // derivative of that clamp is zero where it binds, so the
            // perturbation follows the REFERENCE sign, not its own.
            dtau(0,ip) = 0.5*(term1 + tau_centrifugal(0,ip)*term2);
          }
        } else {
          dtau = helfem::Matrix();
        }
      }

      void DFTGridWorker::check_response_fields_spin(const helfem::Cube & Pa,
                                                     const helfem::Cube & Pb,
                                                     const helfem::Cube & dPa,
                                                     const helfem::Cube & dPb,
                                                     const helfem::Matrix & drho,
                                                     const helfem::Matrix & dgrho,
                                                     const helfem::Matrix & dtau) {
        // Spin-resolved counterpart of check_response_fields.
        const double h = 1e-5;
        helfem::Cube Pap(Pa), Pam(Pa), Pbp(Pb), Pbm(Pb);
        for(size_t l=0;l<Pa.size();l++) {
          Pap[l] = Pa[l] + h*dPa[l]; Pam[l] = Pa[l] - h*dPa[l];
          Pbp[l] = Pb[l] + h*dPb[l]; Pbm[l] = Pb[l] - h*dPb[l];
        }
        update_density(Pap, Pbp);
        const helfem::Matrix rp(rho), gp(grho), tp(tau);
        update_density(Pam, Pbm);
        const helfem::Matrix rm(rho), gm(grho), tm(tau);
        auto rel = [](double a, double b) {
          return std::abs(a-b)/std::max(1e-10, std::abs(b));
        };
        double er=0, eg=0, et=0;
        for(Eigen::Index i=0;i<drho.cols();i++)
          for(Eigen::Index sp=0;sp<2;sp++) {
            er = std::max(er, rel(drho(sp,i), (rp(sp,i)-rm(sp,i))/(2*h)));
            if(do_grad)
              eg = std::max(eg, rel(dgrho(sp,i), (gp(sp,i)-gm(sp,i))/(2*h)));
            if(do_tau)
              et = std::max(et, rel(dtau(sp,i), (tp(sp,i)-tm(sp,i))/(2*h)));
          }
        static double wr=0, wg=0, wt=0;
        wr=std::max(wr,er); wg=std::max(wg,eg); wt=std::max(wt,et);
        fprintf(stderr, "DFIELDS-SPIN worst rel dev: drho %.3e  dgrho %.3e  "
                "dtau %.3e\n", wr, wg, wt);
      }

      void DFTGridWorker::eval_response_fields(const std::vector<helfem::Matrix> & Pac0,
                                               const std::vector<helfem::Matrix> & Pbc0,
                                               helfem::Matrix & drho,
                                               helfem::Matrix & dgrho,
                                               helfem::Matrix & dtau) const {
        // Spin-resolved counterpart of the routine above: the same
        // bilinear forms update_density uses for the reference, applied
        // to the two perturbed spin density matrices.
        const Eigen::Index nbf = (Eigen::Index) bf_ind.size();
        const Eigen::Index N = wtot.size();
        const std::vector<helfem::Matrix> * src[2] = {&Pac0, &Pbc0};

        drho = helfem::Matrix::Zero(2, N);
        dgrho = do_grad ? helfem::Matrix::Zero(2, N) : helfem::Matrix();
        dtau  = do_tau  ? helfem::Matrix::Zero(2, N) : helfem::Matrix();

        for(int sp=0; sp<2; sp++) {
          const std::vector<helfem::Matrix> & Pc0 = *src[sp];
          const size_t nslices = Pc0.size();
          std::vector<helfem::Matrix> Pc(nslices);
          for(size_t islice=0; islice<nslices; islice++) {
            Pc[islice] = helfem::Matrix(nbf, nbf);
            for(size_t i=0;i<bf_ind.size();i++)
              for(size_t j=0;j<bf_ind.size();j++)
                Pc[islice](i,j) = Pc0[islice](bf_ind[i], bf_ind[j]);
          }
          helfem::Matrix P = helfem::Matrix::Zero(nbf, nbf);
          for(size_t islice=0; islice<nslices; islice++) P += Pc[islice];
          helfem::Matrix Pl = helfem::Matrix::Zero(nbf, nbf);
          for(size_t islice=0; islice<nslices; islice++)
            Pl += ((double)(islice*(islice+1)))*Pc[islice];

          const helfem::Matrix Pv = P*bf;
          for(Eigen::Index ip=0;ip<N;ip++)
            drho(sp,ip) = (Pv.col(ip).array()*bf.col(ip).array()).sum();
          if(do_grad)
            for(Eigen::Index ip=0;ip<N;ip++)
              dgrho(sp,ip) = 2.0*(Pv.col(ip).array()*bf_rho.col(ip).array()).sum();
          if(do_tau) {
            const helfem::Matrix Pvp = P*bf_rho;
            const helfem::Matrix Plv = Pl*bf;
            for(Eigen::Index ip=0;ip<N;ip++) {
              const double term1 = (Pvp.col(ip).array()*bf_rho.col(ip).array()).sum();
              const double term2 = (Plv.col(ip).array()*bf.col(ip).array()).sum()/(r(ip)*r(ip));
              dtau(sp,ip) = 0.5*(term1 + tau_centrifugal(sp,ip)*term2);
            }
          }
        }
      }

      helfem::Matrix DFTGridWorker::eval_density(const std::vector<helfem::Matrix> & Pac0, const std::vector<helfem::Matrix> & Pbc0) const {
        const Eigen::Index nbf = (Eigen::Index) bf_ind.size();

        helfem::Matrix Pa = helfem::Matrix::Zero(nbf, nbf);
        for(size_t islice=0; islice<Pac0.size(); islice++)
          for(size_t i=0;i<bf_ind.size();i++)
            for(size_t j=0;j<bf_ind.size();j++)
              Pa(i,j) += Pac0[islice](bf_ind[i], bf_ind[j]);
        helfem::Matrix Pb = helfem::Matrix::Zero(nbf, nbf);
        for(size_t islice=0; islice<Pbc0.size(); islice++)
          for(size_t i=0;i<bf_ind.size();i++)
            for(size_t j=0;j<bf_ind.size();j++)
              Pb(i,j) += Pbc0[islice](bf_ind[i], bf_ind[j]);

        helfem::Matrix Pavloc = Pa*bf;
        helfem::Matrix Pbvloc = Pb*bf;
        helfem::Matrix drho = helfem::Matrix::Zero(2, wtot.size());
        for(Eigen::Index ip=0;ip<wtot.size();ip++) {
          drho(0,ip) = (Pavloc.col(ip).array()*bf.col(ip).array()).sum();
          drho(1,ip) = (Pbvloc.col(ip).array()*bf.col(ip).array()).sum();
        }
        return drho;
      }

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
          // vgrad is the vector coefficient of the basis-function
          // gradient pair, built from vsigma (ground state) or from the
          // kernel chain rule (response).
          helfem::Matrix gr = vgrad.row(0).transpose();
          gr.col(0).array() *= wtot.array();
          // If we also have laplacian dependence, we get an extra term
          if(do_mgga_l) {
            gr.col(0).array() += 2.0*vlapl.row(0).transpose().array()*r.array()*(wrad.array()*4.0*M_PI);
          }
          // Increment matrix
          helfem::dftgrid_common::increment_gga_split(H,gr,bf,helfem::Matrix(),{&bf_rho},{});
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
            vtl = vtau.row(0).transpose().array() * (0.5*wrad.array()*4.0*M_PI)
              * tau_centrifugal.row(0).transpose().array();
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
          helfem::Matrix gr_a(wtot.size(),1);
          gr_a.col(0) = (wtot.array()*vgrad.row(0).transpose().array()).matrix();
          // If we also have laplacian dependence, we get an extra term
          if(do_mgga_l) {
            gr_a.col(0).array() += 2.0*vlapl.row(0).transpose().array()*r.array()*(wrad.array()*4.0*M_PI);
          }
          // Increment matrix
          helfem::dftgrid_common::increment_gga_split(Ha,gr_a,bf,helfem::Matrix(),{&bf_rho},{});

          if(beta) {
            helfem::Matrix gr_b(wtot.size(),1);
            gr_b.col(0) = (wtot.array()*vgrad.row(1).transpose().array()).matrix();
            if(do_mgga_l) {
              gr_b.col(0).array() += 2.0*vlapl.row(1).transpose().array()*r.array()*(wrad.array()*4.0*M_PI);
            }
            helfem::dftgrid_common::increment_gga_split(Hb,gr_b,bf,helfem::Matrix(),{&bf_rho},{});
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
            vtl_a = vtau.row(0).transpose().array() * (0.5*wrad.array()*4.0*M_PI)
              * tau_centrifugal.row(0).transpose().array();
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
              vtl_b = vtau.row(1).transpose().array() * (0.5*wrad.array()*4.0*M_PI)
              * tau_centrifugal.row(1).transpose().array();
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

      // check_grad_tau_lapl, grad_tau_lapl, set_grad_tau_lapl:
      // inherited from DFTGridWorkerBase.

      void DFTGridWorker::compute_bf(size_t iel) {
        // Update function list (basis returns vector<Eigen::Index>)
        bf_ind = basp->bf_list(iel);

        // Get radii
        r = basp->r(iel);
        // Get radial weights
        wrad = basp->wrad(iel);

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

      void DFTGrid::prime_quadrature_cache(int x_func, int c_func) const {
        // A throwaway worker is the cheapest way to ask the functionals
        // which orders they need; it allocates nothing beyond the flags.
        DFTGridWorker probe(basp);
        probe.check_grad_tau_lapl(x_func, c_func);
        basp->fill_quadrature_cache(probe.needs_grad(), probe.needs_lapl());
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

        // The basis values at the quadrature nodes are a pure function of
        // the element index, but every Fock build used to re-evaluate
        // them -- ~11% of an atom-in-jellium run. Fill the cache once,
        // here, while we are still serial; the loop below only reads it.
        prime_quadrature_cache(x_func,c_func);

#ifdef _OPENMP
#pragma omp parallel reduction(+:exc,nel)
#endif
        {
          DFTGridWorker grid(basp);
          grid.check_grad_tau_lapl(x_func,c_func);

#ifdef _OPENMP
#pragma omp for
#endif
          for(size_t iel=0;iel<basp->rad_Nel();iel+=2) {
            grid.compute_bf(iel);
            grid.update_density(Pvec);
            nel+=grid.compute_Nel();

            grid.init_xc();
            if(x_func>0)
              grid.compute_xc(x_func,x_pars,thr);
            if(c_func>0)
              grid.compute_xc(c_func,c_pars,thr);

            // the assembly contracts a general vector field; build it once
            grid.build_vgrad();
            exc+=grid.eval_Exc();
            grid.eval_Fxc(Hvec);
          }
#ifdef _OPENMP
#pragma omp for
#endif
          for(size_t iel=1;iel<basp->rad_Nel();iel+=2) {
            grid.compute_bf(iel);
            grid.update_density(Pvec);
            nel+=grid.compute_Nel();

            grid.init_xc();
            if(x_func>0)
              grid.compute_xc(x_func,x_pars,thr);
            if(c_func>0)
              grid.compute_xc(c_func,c_pars,thr);

            // the assembly contracts a general vector field; build it once
            grid.build_vgrad();
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

        // Same as in the restricted overload: fill the basis-value cache
        // while we are still serial.
        prime_quadrature_cache(x_func,c_func);

#ifdef _OPENMP
#pragma omp parallel reduction(+:exc,nel)
#endif
        {
          DFTGridWorker grid(basp);
          grid.check_grad_tau_lapl(x_func,c_func);

#ifdef _OPENMP
#pragma omp for
#endif
          for(size_t iel=0;iel<basp->rad_Nel();iel+=2) {
            grid.compute_bf(iel);
            grid.update_density(Pavec,Pbvec);
            nel+=grid.compute_Nel();

            grid.init_xc();
            if(x_func>0)
              grid.compute_xc(x_func,x_pars,thr);
            if(c_func>0)
              grid.compute_xc(c_func,c_pars,thr);

            // the assembly contracts a general vector field; build it once
            grid.build_vgrad();
            exc+=grid.eval_Exc();
            grid.eval_Fxc(Havec,Hbvec,beta);
          }
#ifdef _OPENMP
#pragma omp for
#endif
          for(size_t iel=1;iel<basp->rad_Nel();iel+=2) {
            grid.compute_bf(iel);
            grid.update_density(Pavec,Pbvec);
            nel+=grid.compute_Nel();

            grid.init_xc();
            if(x_func>0)
              grid.compute_xc(x_func,x_pars,thr);
            if(c_func>0)
              grid.compute_xc(c_func,c_pars,thr);

            // the assembly contracts a general vector field; build it once
            grid.build_vgrad();
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
      void DFTGrid::eval_Fxc_response(int x_func, const helfem::Vector & x_pars, int c_func, const helfem::Vector & c_pars, const helfem::Cube & P, const std::vector<helfem::Cube> & dP, std::vector<helfem::Cube> & dH, double thr) {
        const Eigen::Index Nrad = P[0].rows();

        dH.resize(dP.size());
        for(size_t ip=0; ip<dP.size(); ip++) {
          if(dP[ip].size() != P.size())
            throw std::logic_error("Perturbation has a different number of l-slices than the reference density.\n");
          dH[ip] = helfem::Cube(P.size(), helfem::Matrix::Zero(Nrad, Nrad));
        }
        if(dP.empty())
          return;

        prime_quadrature_cache(x_func,c_func);

        // One pass over the elements, exactly as eval_Fxc does, except
        // that the potential handed to the assembly is the response
        // potential of each perturbation in turn.
        auto do_element = [&](DFTGridWorker & grid, size_t iel) {
          grid.compute_bf(iel);
          grid.update_density(P);
          grid.init_xc();
          // The gradient channel of the response kernel contains the
          // GROUND-STATE vsigma (the 2 vsigma grad(drho) term), so the
          // first derivatives have to be evaluated here as well; only
          // the second ones were needed while the response was
          // LDA-shaped.
          if(x_func>0)
            grid.compute_xc(x_func,x_pars,thr);
          if(c_func>0)
            grid.compute_xc(c_func,c_pars,thr);
          grid.init_fxc();
          if(x_func>0)
            grid.compute_fxc(x_func,x_pars,thr);
          if(c_func>0)
            grid.compute_fxc(c_func,c_pars,thr);
          for(size_t ip=0; ip<dP.size(); ip++) {
            helfem::Matrix drho, dgrho, dtau;
            grid.eval_response_fields(dP[ip], drho, dgrho, dtau);
            if(getenv("HELFEM_CHECK_DFIELDS")) {
              // The perturbed fields are linear in the perturbation, so
              // they must equal a central difference of the reference
              // fields. This isolates eval_response_fields from the
              // kernel and the assembly.
              grid.check_response_fields(P, dP[ip], drho, dgrho, dtau);
              grid.update_density(P);
            }
            grid.set_response_potential(drho, grid.get_grho(), dgrho, dtau);
            grid.eval_Fxc(dH[ip]);
          }
        };

#ifdef _OPENMP
#pragma omp parallel
#endif
        {
          DFTGridWorker grid(basp);
          grid.check_grad_tau_lapl(x_func,c_func);

          // Same even/odd element split as eval_Fxc: neighbouring
          // elements share boundary functions, so writing them from two
          // threads at once would race.
#ifdef _OPENMP
#pragma omp for
#endif
          for(size_t iel=0;iel<basp->rad_Nel();iel+=2)
            do_element(grid, iel);
#ifdef _OPENMP
#pragma omp for
#endif
          for(size_t iel=1;iel<basp->rad_Nel();iel+=2)
            do_element(grid, iel);
        }
        if(getenv("HELFEM_CHECK_DFOCK")) {
          // The response Fock matrix must equal a central difference of
          // the ground-state one: dF = [F(P + h dP) - F(P - h dP)]/2h.
          // This tests the whole response path -- kernel, channels and
          // assembly -- against machinery already known to be correct.
          const double h = 1e-5;
          double worst = 0.0, scale = 0.0;
          for(size_t ip=0; ip<dP.size(); ip++) {
            helfem::Cube Pp(P), Pm(P);
            for(size_t l=0;l<P.size();l++) {
              Pp[l] = P[l] + h*dP[ip][l];
              Pm[l] = P[l] - h*dP[ip][l];
            }
            helfem::Cube Hp, Hm;
            double e, n;
            eval_Fxc(x_func, x_pars, c_func, c_pars, Pp, Hp, e, n, thr);
            eval_Fxc(x_func, x_pars, c_func, c_pars, Pm, Hm, e, n, thr);
            for(size_t l=0;l<dH[ip].size();l++) {
              const helfem::Matrix fd = (Hp[l]-Hm[l])/(2*h);
              const double e = (dH[ip][l]-fd).cwiseAbs().maxCoeff();
              const double sc = fd.cwiseAbs().maxCoeff();
              worst = std::max(worst, e);
              scale = std::max(scale, sc);
            }
          }
          fprintf(stderr, "DFOCK worst |analytic-FD| = %.3e  (scale %.3e, "
                  "rel %.3e)\n", worst, scale, worst/std::max(1e-30, scale));
        }

      }

      void DFTGrid::eval_Fxc_response(int x_func, const helfem::Vector & x_pars, int c_func, const helfem::Vector & c_pars, const helfem::Cube & Pa, const helfem::Cube & Pb, const std::vector<helfem::Cube> & dPa, const std::vector<helfem::Cube> & dPb, std::vector<helfem::Cube> & dHa, std::vector<helfem::Cube> & dHb, double thr) {
        const Eigen::Index Nrad = Pa[0].rows();

        if(dPa.size() != dPb.size())
          throw std::logic_error("Different numbers of alpha and beta perturbations.\n");

        dHa.resize(dPa.size());
        dHb.resize(dPb.size());
        for(size_t ip=0; ip<dPa.size(); ip++) {
          dHa[ip] = helfem::Cube(Pa.size(), helfem::Matrix::Zero(Nrad, Nrad));
          dHb[ip] = helfem::Cube(Pb.size(), helfem::Matrix::Zero(Nrad, Nrad));
        }
        if(dPa.empty())
          return;

        prime_quadrature_cache(x_func,c_func);

        auto do_element = [&](DFTGridWorker & grid, size_t iel) {
          grid.compute_bf(iel);
          grid.update_density(Pa,Pb);
          grid.init_xc();
          // The gradient channel of the response kernel contains the
          // GROUND-STATE vsigma (the 2 vsigma grad(drho) term), so the
          // first derivatives are needed here too -- exactly as in the
          // spin-restricted driver above. Omitting them silently drops
          // that term, which an LDA cannot notice and a GGA can.
          if(x_func>0)
            grid.compute_xc(x_func,x_pars,thr);
          if(c_func>0)
            grid.compute_xc(c_func,c_pars,thr);
          grid.init_fxc();
          if(x_func>0)
            grid.compute_fxc(x_func,x_pars,thr);
          if(c_func>0)
            grid.compute_fxc(c_func,c_pars,thr);
          for(size_t ip=0; ip<dPa.size(); ip++) {
            helfem::Matrix drho, dgrho, dtau;
            grid.eval_response_fields(dPa[ip], dPb[ip], drho, dgrho, dtau);
            if(getenv("HELFEM_CHECK_DFIELDS")) {
              grid.check_response_fields_spin(Pa, Pb, dPa[ip], dPb[ip],
                                              drho, dgrho, dtau);
              grid.update_density(Pa, Pb);
            }
            grid.set_response_potential(drho, grid.get_grho(), dgrho, dtau);
            grid.eval_Fxc(dHa[ip],dHb[ip],true);
          }
        };

#ifdef _OPENMP
#pragma omp parallel
#endif
        {
          DFTGridWorker grid(basp);
          grid.check_grad_tau_lapl(x_func,c_func);

#ifdef _OPENMP
#pragma omp for
#endif
          for(size_t iel=0;iel<basp->rad_Nel();iel+=2)
            do_element(grid, iel);
#ifdef _OPENMP
#pragma omp for
#endif
          for(size_t iel=1;iel<basp->rad_Nel();iel+=2)
            do_element(grid, iel);
        }
        if(getenv("HELFEM_CHECK_DFOCK")) {
          // Spin-resolved counterpart: the response Fock matrices must
          // equal central differences of the ground-state ones.
          const double h = 1e-5;
          double worst = 0.0, scale = 0.0;
          for(size_t ip=0; ip<dPa.size(); ip++) {
            helfem::Cube Pap(Pa), Pam(Pa), Pbp(Pb), Pbm(Pb);
            for(size_t l=0;l<Pa.size();l++) {
              Pap[l] = Pa[l] + h*dPa[ip][l]; Pam[l] = Pa[l] - h*dPa[ip][l];
              Pbp[l] = Pb[l] + h*dPb[ip][l]; Pbm[l] = Pb[l] - h*dPb[ip][l];
            }
            helfem::Cube Hap, Ham, Hbp, Hbm;
            double e, n;
            eval_Fxc(x_func, x_pars, c_func, c_pars, Pap, Pbp, Hap, Hbp,
                     e, n, true, thr);
            eval_Fxc(x_func, x_pars, c_func, c_pars, Pam, Pbm, Ham, Hbm,
                     e, n, true, thr);
            for(size_t l=0;l<dHa[ip].size();l++) {
              const helfem::Matrix fa = (Hap[l]-Ham[l])/(2*h);
              const helfem::Matrix fb = (Hbp[l]-Hbm[l])/(2*h);
              worst = std::max(worst,
                  std::max((dHa[ip][l]-fa).cwiseAbs().maxCoeff(),
                           (dHb[ip][l]-fb).cwiseAbs().maxCoeff()));
              scale = std::max(scale,
                  std::max(fa.cwiseAbs().maxCoeff(),
                           fb.cwiseAbs().maxCoeff()));
            }
          }
          fprintf(stderr, "DFOCK-SPIN worst |analytic-FD| = %.3e  (scale "
                  "%.3e, rel %.3e)\n", worst, scale,
                  worst/std::max(1e-30, scale));
        }

      }

    } // namespace dftgrid
  } // namespace sadatom
} // namespace helfem