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

// Shared DFT-grid worker plumbing. Extracted from
// src/atomic/dftgrid.cpp, src/sadatom/dftgrid.cpp,
// src/diatomic/dftgrid.cpp -- each copy of these methods was
// byte-identical (init_xc, check_grad_tau_lapl, eval_Exc, zero_Exc)
// or near-identical (compute_xc; the sadatom version's optional NaN
// validation guard is preserved as an inline check here so the
// unified path stays safe for the sadatom callers).

#include "dftgrid_common.h"
#include "dftfuncs.h"
#include <cmath>
#include <cstdio>
#include <sstream>
#include <stdexcept>
extern "C" {
#include <xc.h>
}

namespace helfem {
  namespace dftgrid_common {

    DFTGridWorkerBase::DFTGridWorkerBase() {}
    DFTGridWorkerBase::~DFTGridWorkerBase() {}

    void DFTGridWorkerBase::check_grad_tau_lapl(int x_func, int c_func) {
      do_grad = false;
      if (x_func > 0) do_grad = do_grad || gradient_needed(x_func);
      if (c_func > 0) do_grad = do_grad || gradient_needed(c_func);

      do_tau = false;
      if (x_func > 0) do_tau = do_tau || tau_needed(x_func);
      if (c_func > 0) do_tau = do_tau || tau_needed(c_func);

      do_lapl = false;
      if (x_func > 0) do_lapl = do_lapl || laplacian_needed(x_func);
      if (c_func > 0) do_lapl = do_lapl || laplacian_needed(c_func);
    }

    void DFTGridWorkerBase::set_grad_tau_lapl(bool grad_, bool tau_, bool lap_) {
      do_grad = grad_;
      do_tau  = tau_;
      do_lapl = lap_;
    }

    void DFTGridWorkerBase::zero_Exc() {
      exc.zeros(wtot.n_elem);
    }

    void DFTGridWorkerBase::init_xc() {
      const size_t N = wtot.n_elem;
      zero_Exc();
      if (!polarized) {
        vxc.zeros(1, N);
        if (do_grad) vsigma.zeros(1, N);
        if (do_tau)  vtau.zeros(1, N);
        if (do_lapl) vlapl.zeros(1, N);
      } else {
        vxc.zeros(2, N);
        if (do_grad) vsigma.zeros(3, N);
        if (do_tau)  vtau.zeros(2, N);
        if (do_lapl) vlapl.zeros(2, N);
      }
      do_gga    = false;
      do_mgga_l = false;
      do_mgga_t = false;
    }

    double DFTGridWorkerBase::eval_Exc() const {
      arma::rowvec dens(rho.row(0));
      if (polarized) dens += rho.row(1);
      return arma::sum(wtot % exc % dens);
    }

    void DFTGridWorkerBase::compute_xc(int func_id, const arma::vec & p, double thr, bool pot) {
      bool gga, mgga_t, mgga_l;
      is_gga_mgga(func_id, gga, mgga_t, mgga_l);

      do_gga    = do_gga    || gga || mgga_t || mgga_l;
      do_mgga_t = do_mgga_t || mgga_t;
      do_mgga_l = do_mgga_l || mgga_l;

      const size_t N = wtot.n_elem;

      arma::rowvec exc_wrk;
      arma::mat vxc_wrk, vsigma_wrk, vlapl_wrk, vtau_wrk;

      if (has_exc(func_id))
        exc_wrk.zeros(exc.n_elem);
      if (pot) {
        vxc_wrk.zeros(vxc.n_rows, vxc.n_cols);
        if (gga || mgga_t || mgga_l)
          vsigma_wrk.zeros(vsigma.n_rows, vsigma.n_cols);
        if (mgga_t)
          vtau_wrk.zeros(vtau.n_rows, vtau.n_cols);
        if (mgga_l)
          vlapl_wrk.zeros(vlapl.n_rows, vlapl.n_cols);
      }

      const int nspin = polarized ? XC_POLARIZED : XC_UNPOLARIZED;

      xc_func_type func;
      if (xc_func_init(&func, func_id, nspin) != 0) {
        std::ostringstream oss;
        oss << "Functional " << func_id << " not found!";
        throw std::runtime_error(oss.str());
      }
      xc_func_set_dens_threshold(&func, thr);

      if (p.n_elem) {
        if (p.n_elem != (arma::uword) xc_func_info_get_n_ext_params((xc_func_info_type *) func.info))
          throw std::logic_error("Incompatible number of parameters!\n");
        arma::vec phlp(p);
        xc_func_set_ext_params(&func, phlp.memptr());
      }

      if (has_exc(func_id)) {
        if (pot) {
          if (mgga_t || mgga_l) {
            double * laplp  = mgga_l ? lapl.memptr()      : NULL;
            double * taup   = mgga_t ? tau.memptr()       : NULL;
            double * vlaplp = mgga_l ? vlapl_wrk.memptr() : NULL;
            double * vtaup  = mgga_t ? vtau_wrk.memptr()  : NULL;
            xc_mgga_exc_vxc(&func, N, rho.memptr(), sigma.memptr(),
                             laplp, taup,
                             exc_wrk.memptr(), vxc_wrk.memptr(),
                             vsigma_wrk.memptr(), vlaplp, vtaup);
          } else if (gga) {
            xc_gga_exc_vxc(&func, N, rho.memptr(), sigma.memptr(),
                            exc_wrk.memptr(), vxc_wrk.memptr(),
                            vsigma_wrk.memptr());
          } else {
            xc_lda_exc_vxc(&func, N, rho.memptr(),
                            exc_wrk.memptr(), vxc_wrk.memptr());
          }
        } else {
          if (mgga_t || mgga_l) {
            double * laplp = mgga_l ? lapl.memptr() : NULL;
            double * taup  = mgga_t ? tau.memptr()  : NULL;
            xc_mgga_exc(&func, N, rho.memptr(), sigma.memptr(),
                         laplp, taup, exc_wrk.memptr());
          } else if (gga) {
            xc_gga_exc(&func, N, rho.memptr(), sigma.memptr(), exc_wrk.memptr());
          } else {
            xc_lda_exc(&func, N, rho.memptr(), exc_wrk.memptr());
          }
        }
      } else {
        if (pot) {
          if (mgga_t || mgga_l) {
            double * laplp  = mgga_l ? lapl.memptr()      : NULL;
            double * taup   = mgga_t ? tau.memptr()       : NULL;
            double * vlaplp = mgga_l ? vlapl_wrk.memptr() : NULL;
            double * vtaup  = mgga_t ? vtau_wrk.memptr()  : NULL;
            xc_mgga_vxc(&func, N, rho.memptr(), sigma.memptr(),
                         laplp, taup,
                         vxc_wrk.memptr(), vsigma_wrk.memptr(),
                         vlaplp, vtaup);
          } else if (gga) {
            xc_gga_vxc(&func, N, rho.memptr(), sigma.memptr(),
                        vxc_wrk.memptr(), vsigma_wrk.memptr());
          } else {
            xc_lda_vxc(&func, N, rho.memptr(), vxc_wrk.memptr());
          }
        }
      }

      // NaN-guard diagnostic (originally sadatom-only; unified here
      // so atomic and diatomic get the same warning). Prints only;
      // never modifies state.
      for (size_t i = 0; i < N; ++i) {
        const double e = has_exc(func_id) ? exc_wrk[i] : 0.0;
        double rhoa = 0.0, rhob = 0.0;
        double sigmaaa = 0.0, sigmaab = 0.0, sigmabb = 0.0;
        double lapla = 0.0, laplb = 0.0, taua = 0.0, taub = 0.0;
        double vrhoa = 0.0, vrhob = 0.0;
        double vsigmaaa = 0.0, vsigmaab = 0.0, vsigmabb = 0.0;
        double vlapla = 0.0, vlaplb = 0.0, vtaua = 0.0, vtaub = 0.0;
        if (polarized) {
          rhoa = rho(0, i); rhob = rho(1, i);
          vrhoa = vxc_wrk(0, i); vrhob = vxc_wrk(1, i);
          if (gga || mgga_t || mgga_l) {
            sigmaaa = sigma(0, i); sigmaab = sigma(1, i); sigmabb = sigma(2, i);
            vsigmaaa = vsigma_wrk(0, i); vsigmaab = vsigma_wrk(1, i); vsigmabb = vsigma_wrk(2, i);
          }
          if (mgga_l) {
            lapla = lapl(0, i); laplb = lapl(1, i);
            vlapla = vlapl_wrk(0, i); vlaplb = vlapl_wrk(1, i);
          }
          if (mgga_t) {
            taua = tau(0, i); taub = tau(1, i);
            vtaua = vtau_wrk(0, i); vtaub = vtau_wrk(1, i);
          }
        } else {
          rhoa = 0.5 * rho(0, i); rhob = 0.5 * rho(0, i);
          vrhoa = vxc_wrk(0, i);  vrhob = vxc_wrk(0, i);
          if (gga || mgga_t || mgga_l) {
            sigmaaa = 0.25 * sigma(0, i); sigmaab = 0.25 * sigma(0, i); sigmabb = 0.25 * sigma(0, i);
            vsigmaaa = vsigma_wrk(0, i); vsigmaab = vsigma_wrk(0, i); vsigmabb = vsigma_wrk(0, i);
          }
          if (mgga_l) {
            lapla = 0.5 * lapl(0, i); laplb = 0.5 * lapl(0, i);
            vlapla = vlapl_wrk(0, i); vlaplb = vlapl_wrk(0, i);
          }
          if (mgga_t) {
            taua = 0.5 * tau(0, i); taub = 0.5 * tau(0, i);
            vtaua = vtau_wrk(0, i); vtaub = vtau_wrk(0, i);
          }
        }
        if (std::isnan(e)   || std::isnan(vrhoa)    || std::isnan(vrhob)
             || std::isnan(vsigmaaa) || std::isnan(vsigmaab) || std::isnan(vsigmabb)
             || std::isnan(vlapla)   || std::isnan(vlaplb)
             || std::isnan(vtaua)    || std::isnan(vtaub)) {
          printf("NaN encountered for functional id = %i with input\n", func_id);
          printf("input: %e %e %e % e %e %e %e % e % e\n",
                  rhoa, rhob, sigmaaa, sigmaab, sigmabb, lapla, laplb, taua, taub);
          printf("output: % e % e % e % e % e % e % e % e % e % e\n",
                  e, vrhoa, vrhob, vsigmaaa, vsigmaab, vsigmabb, vlapla, vlaplb, vtaua, vtaub);
        }
      }

      if (has_exc(func_id))
        exc += exc_wrk;
      if (pot) {
        if (mgga_l)                     vlapl  += vlapl_wrk;
        if (mgga_t)                     vtau   += vtau_wrk;
        if (mgga_t || mgga_l || gga)    vsigma += vsigma_wrk;
        vxc += vxc_wrk;
      }

      xc_func_end(&func);
    }

  } // namespace dftgrid_common
} // namespace helfem
