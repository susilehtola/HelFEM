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

// The libxc-free half of the DFT-grid worker base: construction, the
// potential buffers, the gradient channel and the response bookkeeping.
// Everything here works on potentials that are ALREADY on the grid, however
// they got there, so it lives in helfem-fem with the basis machinery.  The
// half that asks libxc for them -- functional(), compute_xc(), compute_fxc()
// and check_grad_tau_lapl() -- stays in dftgrid_common.cpp in helfem-common.
//
// The split is by translation unit, not by class: the destructor is the only
// virtual function, so the vtable and typeinfo are emitted HERE, and a
// consumer linking only helfem-fem can build, fill and contract a worker as
// long as it never calls the libxc methods.  That is what lets a potential
// from somewhere other than libxc -- a fitted electron-proton correlation
// functional, say -- use the same assembly as every XC Fock matrix.

#include "dftgrid_common.h"
#include "xckernel_fxc.h"
#include <cmath>
#include <cstdio>
#include <sstream>
#include <stdexcept>

namespace helfem {
  namespace dftgrid_common {


    DFTGridWorkerBase::DFTGridWorkerBase() {}
    DFTGridWorkerBase::~DFTGridWorkerBase() {}

    void DFTGridWorkerBase::set_grad_tau_lapl(bool grad_, bool tau_, bool lap_) {
      do_grad = grad_;
      do_tau  = tau_;
      do_lapl = lap_;
    }

    void DFTGridWorkerBase::init_xc() {
      const Eigen::Index N = wtot.size();
      exc = helfem::Vector::Zero(N);
      if (!polarized) {
        vxc = helfem::Matrix::Zero(1, N);
        if (do_grad) vsigma = helfem::Matrix::Zero(1, N);
        if (do_tau)  vtau   = helfem::Matrix::Zero(1, N);
        if (do_lapl) vlapl  = helfem::Matrix::Zero(1, N);
      } else {
        vxc = helfem::Matrix::Zero(2, N);
        if (do_grad) vsigma = helfem::Matrix::Zero(3, N);
        if (do_tau)  vtau   = helfem::Matrix::Zero(2, N);
        if (do_lapl) vlapl  = helfem::Matrix::Zero(2, N);
      }
      do_gga    = false;
      do_mgga_l = false;
      do_mgga_t = false;
    }

    double DFTGridWorkerBase::eval_Exc() const {
      helfem::Vector dens = rho.row(0).transpose();
      if (polarized) dens += rho.row(1).transpose();
      return (wtot.array() * exc.array() * dens.array()).sum();
    }

    double DFTGridWorkerBase::compute_Nel() const {
      double nel=0.0;
      if(!polarized) {
        for(Eigen::Index ip=0;ip<wtot.size();ip++)
          nel+=wtot(ip)*rho(0,ip);
      } else {
        for(Eigen::Index ip=0;ip<wtot.size();ip++)
          nel+=wtot(ip)*(rho(0,ip)+rho(1,ip));
      }

      return nel;
    }

    void DFTGridWorkerBase::init_fxc() {
      const Eigen::Index N = wtot.size();
      const Eigen::Index n2 = polarized ? 3 : 1;
      const Eigen::Index n4 = polarized ? 4 : 1;
      const Eigen::Index n6 = polarized ? 6 : 1;
      v2rho2 = helfem::Matrix::Zero(n2, N);
      if (do_grad) {
        v2rhosigma = helfem::Matrix::Zero(n6, N);
        v2sigma2   = helfem::Matrix::Zero(n6, N);
      }
      if (do_tau) {
        v2rhotau   = helfem::Matrix::Zero(n4, N);
        v2tau2     = helfem::Matrix::Zero(n2, N);
        if (do_grad) v2sigmatau = helfem::Matrix::Zero(n6, N);
      }
    }

    void DFTGridWorkerBase::build_vgrad(const helfem::Matrix & grho) {
      if (!do_grad) return;
      // The number of gradient components is the caller's business: the
      // radial atomic worker carries one, the three-dimensional workers
      // three. Everything below is written per component.
      const Eigen::Index nsp = polarized ? 2 : 1;
      if (grho.rows() % nsp) {
        std::ostringstream oss;
        oss << "Density gradient has " << grho.rows() << " rows, which is "
            << "not divisible by the " << nsp << " spin channels.\n";
        throw std::logic_error(oss.str());
      }
      const Eigen::Index nc = grho.rows() / nsp;
      const Eigen::Index N = grho.cols();
      vgrad = helfem::Matrix::Zero(grho.rows(), N);
      // The chain rule is GENERATED, not written here: these are
      // libxckernel's ground-state potential channels (emitters/
      // helfemwriter.py, from engine/fock.vxc_channels), so the
      // assembly cannot drift from the kernel it is differentiated
      // into. The generated kernel is single-component because the
      // gradient channel of a semilocal functional depends only on its
      // OWN component -- sigma is a sum of squares -- which is why one
      // call per component serves the radial worker's single component,
      // the pure-m worker's two and the three-dimensional workers'
      // three alike.
      for (Eigen::Index c = 0; c < nc; c++)
        for (Eigen::Index i = 0; i < N; i++) {
          if (!polarized) {
            helfem::xckernel::xck_helfem_vxc_grad(
                grho(c, i), vsigma(0, i), vgrad(c, i));
          } else {
            helfem::xckernel::xck_helfem_vxc_grad_spin(
                grho(c, i), grho(nc + c, i), vsigma(0, i), vsigma(1, i),
                vsigma(2, i), vgrad(c, i), vgrad(nc + c, i));
          }
        }
    }

    void DFTGridWorkerBase::set_response_potential(const helfem::Matrix & drho,
                                                   const helfem::Matrix & grho,
                                                   const helfem::Matrix & dgrad_rho,
                                                   const helfem::Matrix & dtau) {
      if (drho.rows() != rho.rows() || drho.cols() != rho.cols()) {
        std::ostringstream oss;
        oss << "Perturbation density is " << drho.rows() << "x" << drho.cols()
            << " but the reference density is " << rho.rows() << "x"
            << rho.cols() << ".\n";
        throw std::logic_error(oss.str());
      }
      // A caller that does not supply the perturbed gradient / tau gets
      // the density-density block alone, i.e. exactly the LDA-shaped
      // response this routine produced before the other channels
      // existed. That keeps every existing caller working unchanged
      // while a caller that does supply them gets the exact kernel.
      const bool have_grad = do_grad && dgrad_rho.size() > 0 && grho.size() > 0;
      const bool have_tau  = do_tau && dtau.size() > 0;
      if (have_grad && (dgrad_rho.rows() != grho.rows() ||
                        dgrad_rho.cols() != grho.cols()))
        throw std::logic_error("Perturbed and reference density gradients "
                               "have different shapes.\n");
      if (!have_grad || (do_tau && !have_tau)) {
        if (!polarized) {
          vxc.row(0) = v2rho2.row(0).array() * drho.row(0).array();
        } else {
          vxc.row(0) = v2rho2.row(0).array() * drho.row(0).array() +
                       v2rho2.row(1).array() * drho.row(1).array();
          vxc.row(1) = v2rho2.row(1).array() * drho.row(0).array() +
                       v2rho2.row(2).array() * drho.row(1).array();
        }
        do_gga    = false;
        do_mgga_t = false;
        do_mgga_l = false;
        return;
      }

      const Eigen::Index N = wtot.size();
      const Eigen::Index nsp = polarized ? 2 : 1;
      // How many gradient components the caller carries is its own
      // business, and it differs per geometry: one for the spherically
      // averaged radial worker, two for the diatomic pure-m one (the
      // azimuthal component of a phi-independent density vanishes
      // identically), three for the atomic and diatomic 3D workers.
      // Unlike the ground-state potential, the response cannot be
      // applied one component at a time: sigma_ab = grad rho_a . grad
      // rho_b couples them, so the kernel is emitted per component
      // count and selected here.
      if (grho.rows() % nsp) {
        std::ostringstream oss;
        oss << "Density gradient has " << grho.rows() << " rows, which is "
            << "not divisible by the " << nsp << " spin channels.\n";
        throw std::logic_error(oss.str());
      }
      const Eigen::Index nc = grho.rows() / nsp;
      if (nc < 1 || nc > 3) {
        std::ostringstream oss;
        oss << "No response kernel is emitted for " << nc << " gradient "
            << "components.\n";
        throw std::logic_error(oss.str());
      }
      vgrad = helfem::Matrix::Zero(grho.rows(), N);

      if (!polarized) {
        // The chain rule itself is GENERATED, not written here: the
        // per-point channels come from libxckernel's fxc_channels via
        // emitters/helfemwriter.py, so the expressions cannot drift from
        // the ones the generator validates.
        for (Eigen::Index i = 0; i < N; i++) {
          double u = 0.0, w_tau = 0.0, vg[3] = {0.0, 0.0, 0.0};
          if (have_tau) {
            switch (nc) {
            case 1:
              helfem::xckernel::xck_helfem_fxc_mgga_tau(dgrad_rho(0, i),
                  grho(0, i), drho(0, i), dtau(0, i), v2rho2(0, i),
                  v2rhosigma(0, i), v2rhotau(0, i), v2sigma2(0, i),
                  v2sigmatau(0, i), v2tau2(0, i), vsigma(0, i), u, vg[0],
                  w_tau);
              break;
            case 2:
              helfem::xckernel::xck_helfem_fxc_mgga_tau_2d(dgrad_rho(0, i),
                  dgrad_rho(1, i), grho(0, i), grho(1, i), drho(0, i),
                  dtau(0, i), v2rho2(0, i), v2rhosigma(0, i),
                  v2rhotau(0, i), v2sigma2(0, i), v2sigmatau(0, i),
                  v2tau2(0, i), vsigma(0, i), u, vg[0], vg[1], w_tau);
              break;
            case 3:
              helfem::xckernel::xck_helfem_fxc_mgga_tau_3d(dgrad_rho(0, i),
                  dgrad_rho(1, i), dgrad_rho(2, i), grho(0, i), grho(1, i),
                  grho(2, i), drho(0, i), dtau(0, i), v2rho2(0, i),
                  v2rhosigma(0, i), v2rhotau(0, i), v2sigma2(0, i),
                  v2sigmatau(0, i), v2tau2(0, i), vsigma(0, i), u, vg[0],
                  vg[1], vg[2], w_tau);
              break;
            }
          } else {
            switch (nc) {
            case 1:
              helfem::xckernel::xck_helfem_fxc_gga(dgrad_rho(0, i),
                  grho(0, i), drho(0, i), v2rho2(0, i), v2rhosigma(0, i),
                  v2sigma2(0, i), vsigma(0, i), u, vg[0]);
              break;
            case 2:
              helfem::xckernel::xck_helfem_fxc_gga_2d(dgrad_rho(0, i),
                  dgrad_rho(1, i), grho(0, i), grho(1, i), drho(0, i),
                  v2rho2(0, i), v2rhosigma(0, i), v2sigma2(0, i),
                  vsigma(0, i), u, vg[0], vg[1]);
              break;
            case 3:
              helfem::xckernel::xck_helfem_fxc_gga_3d(dgrad_rho(0, i),
                  dgrad_rho(1, i), dgrad_rho(2, i), grho(0, i), grho(1, i),
                  grho(2, i), drho(0, i), v2rho2(0, i), v2rhosigma(0, i),
                  v2sigma2(0, i), vsigma(0, i), u, vg[0], vg[1], vg[2]);
              break;
            }
          }
          vxc(0, i) = u;
          for (Eigen::Index c = 0; c < nc; c++) vgrad(c, i) = vg[c];
          if (have_tau) vtau(0, i) = w_tau;
        }
      } else {
        // Spin-resolved channels, likewise generated: the polarized Libxc
        // arrays keep their flat packing, and the call sites below were
        // emitted from the generated signatures rather than ordered by
        // hand.
        for (Eigen::Index i = 0; i < N; i++) {
          double u[2] = {0.0, 0.0}, w[2] = {0.0, 0.0};
          double vg[2][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
          if (have_tau) {
            switch (nc) {
            case 1:
              helfem::xckernel::xck_helfem_fxc_mgga_tau_spin(
                  dgrad_rho(0*nc + 0, i), grho(0*nc + 0, i),
                  dgrad_rho(1*nc + 0, i), grho(1*nc + 0, i), drho(0, i),
                  drho(1, i), dtau(0, i), dtau(1, i), v2rho2(0, i),
                  v2rho2(1, i), v2rho2(2, i), v2rhosigma(0, i),
                  v2rhosigma(1, i), v2rhosigma(2, i), v2rhosigma(3, i),
                  v2rhosigma(4, i), v2rhosigma(5, i), v2rhotau(0, i),
                  v2rhotau(1, i), v2rhotau(2, i), v2rhotau(3, i),
                  v2sigma2(0, i), v2sigma2(1, i), v2sigma2(2, i),
                  v2sigma2(3, i), v2sigma2(4, i), v2sigma2(5, i),
                  v2sigmatau(0, i), v2sigmatau(1, i), v2sigmatau(2, i),
                  v2sigmatau(3, i), v2sigmatau(4, i), v2sigmatau(5, i),
                  v2tau2(0, i), v2tau2(1, i), v2tau2(2, i), vsigma(0, i),
                  vsigma(1, i), vsigma(2, i), u[0], vg[0][0], w[0], u[1],
                  vg[1][0], w[1]);
              break;
            case 2:
              helfem::xckernel::xck_helfem_fxc_mgga_tau_2d_spin(
                  dgrad_rho(0*nc + 0, i), dgrad_rho(0*nc + 1, i),
                  grho(0*nc + 0, i), grho(0*nc + 1, i),
                  dgrad_rho(1*nc + 0, i), dgrad_rho(1*nc + 1, i),
                  grho(1*nc + 0, i), grho(1*nc + 1, i), drho(0, i),
                  drho(1, i), dtau(0, i), dtau(1, i), v2rho2(0, i),
                  v2rho2(1, i), v2rho2(2, i), v2rhosigma(0, i),
                  v2rhosigma(1, i), v2rhosigma(2, i), v2rhosigma(3, i),
                  v2rhosigma(4, i), v2rhosigma(5, i), v2rhotau(0, i),
                  v2rhotau(1, i), v2rhotau(2, i), v2rhotau(3, i),
                  v2sigma2(0, i), v2sigma2(1, i), v2sigma2(2, i),
                  v2sigma2(3, i), v2sigma2(4, i), v2sigma2(5, i),
                  v2sigmatau(0, i), v2sigmatau(1, i), v2sigmatau(2, i),
                  v2sigmatau(3, i), v2sigmatau(4, i), v2sigmatau(5, i),
                  v2tau2(0, i), v2tau2(1, i), v2tau2(2, i), vsigma(0, i),
                  vsigma(1, i), vsigma(2, i), u[0], vg[0][0], vg[0][1],
                  w[0], u[1], vg[1][0], vg[1][1], w[1]);
              break;
            case 3:
              helfem::xckernel::xck_helfem_fxc_mgga_tau_3d_spin(
                  dgrad_rho(0*nc + 0, i), dgrad_rho(0*nc + 1, i),
                  dgrad_rho(0*nc + 2, i), grho(0*nc + 0, i),
                  grho(0*nc + 1, i), grho(0*nc + 2, i),
                  dgrad_rho(1*nc + 0, i), dgrad_rho(1*nc + 1, i),
                  dgrad_rho(1*nc + 2, i), grho(1*nc + 0, i),
                  grho(1*nc + 1, i), grho(1*nc + 2, i), drho(0, i),
                  drho(1, i), dtau(0, i), dtau(1, i), v2rho2(0, i),
                  v2rho2(1, i), v2rho2(2, i), v2rhosigma(0, i),
                  v2rhosigma(1, i), v2rhosigma(2, i), v2rhosigma(3, i),
                  v2rhosigma(4, i), v2rhosigma(5, i), v2rhotau(0, i),
                  v2rhotau(1, i), v2rhotau(2, i), v2rhotau(3, i),
                  v2sigma2(0, i), v2sigma2(1, i), v2sigma2(2, i),
                  v2sigma2(3, i), v2sigma2(4, i), v2sigma2(5, i),
                  v2sigmatau(0, i), v2sigmatau(1, i), v2sigmatau(2, i),
                  v2sigmatau(3, i), v2sigmatau(4, i), v2sigmatau(5, i),
                  v2tau2(0, i), v2tau2(1, i), v2tau2(2, i), vsigma(0, i),
                  vsigma(1, i), vsigma(2, i), u[0], vg[0][0], vg[0][1],
                  vg[0][2], w[0], u[1], vg[1][0], vg[1][1], vg[1][2], w[1]);
              break;
            }
          } else {
            switch (nc) {
            case 1:
              helfem::xckernel::xck_helfem_fxc_gga_spin(
                  dgrad_rho(0*nc + 0, i), grho(0*nc + 0, i),
                  dgrad_rho(1*nc + 0, i), grho(1*nc + 0, i), drho(0, i),
                  drho(1, i), v2rho2(0, i), v2rho2(1, i), v2rho2(2, i),
                  v2rhosigma(0, i), v2rhosigma(1, i), v2rhosigma(2, i),
                  v2rhosigma(3, i), v2rhosigma(4, i), v2rhosigma(5, i),
                  v2sigma2(0, i), v2sigma2(1, i), v2sigma2(2, i),
                  v2sigma2(3, i), v2sigma2(4, i), v2sigma2(5, i),
                  vsigma(0, i), vsigma(1, i), vsigma(2, i), u[0], vg[0][0],
                  u[1], vg[1][0]);
              break;
            case 2:
              helfem::xckernel::xck_helfem_fxc_gga_2d_spin(
                  dgrad_rho(0*nc + 0, i), dgrad_rho(0*nc + 1, i),
                  grho(0*nc + 0, i), grho(0*nc + 1, i),
                  dgrad_rho(1*nc + 0, i), dgrad_rho(1*nc + 1, i),
                  grho(1*nc + 0, i), grho(1*nc + 1, i), drho(0, i),
                  drho(1, i), v2rho2(0, i), v2rho2(1, i), v2rho2(2, i),
                  v2rhosigma(0, i), v2rhosigma(1, i), v2rhosigma(2, i),
                  v2rhosigma(3, i), v2rhosigma(4, i), v2rhosigma(5, i),
                  v2sigma2(0, i), v2sigma2(1, i), v2sigma2(2, i),
                  v2sigma2(3, i), v2sigma2(4, i), v2sigma2(5, i),
                  vsigma(0, i), vsigma(1, i), vsigma(2, i), u[0], vg[0][0],
                  vg[0][1], u[1], vg[1][0], vg[1][1]);
              break;
            case 3:
              helfem::xckernel::xck_helfem_fxc_gga_3d_spin(
                  dgrad_rho(0*nc + 0, i), dgrad_rho(0*nc + 1, i),
                  dgrad_rho(0*nc + 2, i), grho(0*nc + 0, i),
                  grho(0*nc + 1, i), grho(0*nc + 2, i),
                  dgrad_rho(1*nc + 0, i), dgrad_rho(1*nc + 1, i),
                  dgrad_rho(1*nc + 2, i), grho(1*nc + 0, i),
                  grho(1*nc + 1, i), grho(1*nc + 2, i), drho(0, i),
                  drho(1, i), v2rho2(0, i), v2rho2(1, i), v2rho2(2, i),
                  v2rhosigma(0, i), v2rhosigma(1, i), v2rhosigma(2, i),
                  v2rhosigma(3, i), v2rhosigma(4, i), v2rhosigma(5, i),
                  v2sigma2(0, i), v2sigma2(1, i), v2sigma2(2, i),
                  v2sigma2(3, i), v2sigma2(4, i), v2sigma2(5, i),
                  vsigma(0, i), vsigma(1, i), vsigma(2, i), u[0], vg[0][0],
                  vg[0][1], vg[0][2], u[1], vg[1][0], vg[1][1], vg[1][2]);
              break;
            }
          }
          vxc(0, i) = u[0];
          vxc(1, i) = u[1];
          for (Eigen::Index c = 0; c < nc; c++) {
            vgrad(c, i) = vg[0][c];
            vgrad(nc + c, i) = vg[1][c];
          }
          if (have_tau) {
            vtau(0, i) = w[0];
            vtau(1, i) = w[1];
          }
        }
      }

      // Unlike the previous LDA-shaped response, the assembly now sees the
      // functional's own rungs: the channels above are exact.
      do_gga    = have_grad;
      do_mgga_t = have_tau;
      do_mgga_l = false;
    }

  }
}
