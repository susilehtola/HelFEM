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
#include "xckernel_fxc.h"
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

    const DFTGridWorkerBase::Functional &
    DFTGridWorkerBase::functional(int func_id, const helfem::Vector & p,
                                  double thr) {
      const int nspin = polarized ? XC_POLARIZED : XC_UNPOLARIZED;
      for (const Functional & f : funcs)
        if (f.id == func_id && f.nspin == nspin && f.thr == thr &&
            f.params.size() == p.size() &&
            (p.size() == 0 || (f.params - p).cwiseAbs().maxCoeff() == 0.0))
          return f;

      Functional f;
      f.id = func_id;
      f.nspin = nspin;
      f.thr = thr;
      f.params = p;
      is_gga_mgga(func_id, f.gga, f.mgga_t, f.mgga_l);
      f.have_exc = has_exc(func_id);

      // Initialize first, wrap second: the deleter calls xc_func_end, and
      // running that on a functional xc_func_init refused would be worse
      // than the failure it reports.
      xc_func_type * raw = new xc_func_type;
      if (xc_func_init(raw, func_id, nspin) != 0) {
        delete raw;
        std::ostringstream oss;
        oss << "Functional " << func_id << " not found!";
        throw std::runtime_error(oss.str());
      }
      f.func = std::shared_ptr<xc_func_type>(raw, [](xc_func_type * q) {
        xc_func_end(q);
        delete q;
      });

      xc_func_set_dens_threshold(raw, thr);
      if (p.size()) {
        if (p.size() != (Eigen::Index) xc_func_info_get_n_ext_params((xc_func_info_type *) raw->info))
          throw std::logic_error("Incompatible number of parameters!\n");
        helfem::Vector phlp(p);
        xc_func_set_ext_params(raw, phlp.data());
      }

      funcs.push_back(f);
      return funcs.back();
    }

    void DFTGridWorkerBase::compute_xc(int func_id, const helfem::Vector & p, double thr, bool pot) {
      const Functional & fc = functional(func_id, p, thr);
      xc_func_type & func = *fc.func;
      const bool gga = fc.gga, mgga_t = fc.mgga_t, mgga_l = fc.mgga_l;
      const bool have_exc = fc.have_exc;

      do_gga    = do_gga    || gga || mgga_t || mgga_l;
      do_mgga_t = do_mgga_t || mgga_t;
      do_mgga_l = do_mgga_l || mgga_l;

      const size_t N = (size_t) wtot.size();

      helfem::Vector exc_wrk;
      helfem::Matrix vxc_wrk, vsigma_wrk, vlapl_wrk, vtau_wrk;

      if (have_exc)
        exc_wrk = helfem::Vector::Zero(exc.size());
      if (pot) {
        vxc_wrk = helfem::Matrix::Zero(vxc.rows(), vxc.cols());
        if (gga || mgga_t || mgga_l)
          vsigma_wrk = helfem::Matrix::Zero(vsigma.rows(), vsigma.cols());
        if (mgga_t)
          vtau_wrk = helfem::Matrix::Zero(vtau.rows(), vtau.cols());
        if (mgga_l)
          vlapl_wrk = helfem::Matrix::Zero(vlapl.rows(), vlapl.cols());
      }

      if (have_exc) {
        if (pot) {
          if (mgga_t || mgga_l) {
            double * laplp  = mgga_l ? lapl.data()      : NULL;
            double * taup   = mgga_t ? tau.data()       : NULL;
            double * vlaplp = mgga_l ? vlapl_wrk.data() : NULL;
            double * vtaup  = mgga_t ? vtau_wrk.data()  : NULL;
            xc_mgga_exc_vxc(&func, N, rho.data(), sigma.data(),
                             laplp, taup,
                             exc_wrk.data(), vxc_wrk.data(),
                             vsigma_wrk.data(), vlaplp, vtaup);
          } else if (gga) {
            xc_gga_exc_vxc(&func, N, rho.data(), sigma.data(),
                            exc_wrk.data(), vxc_wrk.data(),
                            vsigma_wrk.data());
          } else {
            xc_lda_exc_vxc(&func, N, rho.data(),
                            exc_wrk.data(), vxc_wrk.data());
          }
        } else {
          if (mgga_t || mgga_l) {
            double * laplp = mgga_l ? lapl.data() : NULL;
            double * taup  = mgga_t ? tau.data()  : NULL;
            xc_mgga_exc(&func, N, rho.data(), sigma.data(),
                         laplp, taup, exc_wrk.data());
          } else if (gga) {
            xc_gga_exc(&func, N, rho.data(), sigma.data(), exc_wrk.data());
          } else {
            xc_lda_exc(&func, N, rho.data(), exc_wrk.data());
          }
        }
      } else {
        if (pot) {
          if (mgga_t || mgga_l) {
            double * laplp  = mgga_l ? lapl.data()      : NULL;
            double * taup   = mgga_t ? tau.data()       : NULL;
            double * vlaplp = mgga_l ? vlapl_wrk.data() : NULL;
            double * vtaup  = mgga_t ? vtau_wrk.data()  : NULL;
            xc_mgga_vxc(&func, N, rho.data(), sigma.data(),
                         laplp, taup,
                         vxc_wrk.data(), vsigma_wrk.data(),
                         vlaplp, vtaup);
          } else if (gga) {
            xc_gga_vxc(&func, N, rho.data(), sigma.data(),
                        vxc_wrk.data(), vsigma_wrk.data());
          } else {
            xc_lda_vxc(&func, N, rho.data(), vxc_wrk.data());
          }
        }
      }

      // NaN-guard diagnostic (originally sadatom-only; unified here
      // so atomic and diatomic get the same warning). Prints only;
      // never modifies state.
      for (size_t i = 0; i < N; ++i) {
        const double e = have_exc ? exc_wrk(i) : 0.0;
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

      if (have_exc)
        exc += exc_wrk;
      if (pot) {
        if (mgga_l)                     vlapl  += vlapl_wrk;
        if (mgga_t)                     vtau   += vtau_wrk;
        if (mgga_t || mgga_l || gga)    vsigma += vsigma_wrk;
        vxc += vxc_wrk;
      }
    }

    void DFTGridWorkerBase::compute_fxc(int func_id, const helfem::Vector & p,
                                        double thr) {
      const Functional & fc = functional(func_id, p, thr);
      xc_func_type & func = *fc.func;
      const bool gga = fc.gga, mgga_t = fc.mgga_t, mgga_l = fc.mgga_l;

      const size_t N = (size_t) wtot.size();

      // A functional without XC_FLAGS_HAVE_FXC would leave the buffer
      // untouched, i.e. contribute a silently zero kernel; the Hessian
      // would then be wrong in a way nothing downstream can detect.
      if (!(func.info->flags & XC_FLAGS_HAVE_FXC)) {
        std::ostringstream oss;
        oss << "Functional " << func_id << " does not implement its second "
            << "derivatives, so no response kernel can be formed.\n";
        throw std::logic_error(oss.str());
      }

      // Every block the functional actually populates is kept; the
      // laplacian ones are still scratch, as no laplacian response is
      // assembled yet.
      const Eigen::Index n2 = polarized ? 3 : 1;
      const Eigen::Index n4 = polarized ? 4 : 1;
      const Eigen::Index n6 = polarized ? 6 : 1;
      helfem::Matrix wrk = helfem::Matrix::Zero(n2, N);
      helfem::Matrix rs = helfem::Matrix::Zero(n6, N);
      helfem::Matrix s2 = helfem::Matrix::Zero(n6, N);
      helfem::Matrix rt = helfem::Matrix::Zero(n4, N);
      helfem::Matrix st = helfem::Matrix::Zero(n6, N);
      helfem::Matrix t2 = helfem::Matrix::Zero(n2, N);
      if (mgga_t || mgga_l) {
        helfem::Matrix rl(n4, N), sl(n6, N), l2(n2, N), lt(n4, N);
        // libxc dereferences lapl / tau only when the functional asks for
        // them, exactly as compute_xc above relies on.
        double *laplp = mgga_l ? lapl.data() : NULL;
        double *taup = mgga_t ? tau.data() : NULL;
        double *l2p = mgga_l ? l2.data() : NULL;
        double *ltp = mgga_l ? lt.data() : NULL;
        double *rlp = mgga_l ? rl.data() : NULL;
        double *slp = mgga_l ? sl.data() : NULL;
        double *t2p = mgga_t ? t2.data() : NULL;
        double *rtp = mgga_t ? rt.data() : NULL;
        double *stp = mgga_t ? st.data() : NULL;
        xc_mgga_fxc(&func, N, rho.data(), sigma.data(), laplp, taup, wrk.data(),
                    rs.data(), rlp, rtp, s2.data(), slp, stp, l2p, ltp, t2p);
      } else if (gga) {
        xc_gga_fxc(&func, N, rho.data(), sigma.data(), wrk.data(), rs.data(),
                   s2.data());
      } else {
        xc_lda_fxc(&func, N, rho.data(), wrk.data());
      }
      v2rho2 += wrk;
      if (do_grad && (gga || mgga_t || mgga_l)) {
        v2rhosigma += rs;
        v2sigma2   += s2;
      }
      if (do_tau && mgga_t) {
        v2rhotau += rt;
        v2tau2   += t2;
        if (do_grad) v2sigmatau += st;
      }
    }

  } // namespace dftgrid_common
} // namespace helfem
