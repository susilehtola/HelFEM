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
