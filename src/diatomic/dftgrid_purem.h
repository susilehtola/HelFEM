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

#ifndef DIATOMIC_DFTGRID_PUREM_H
#define DIATOMIC_DFTGRID_PUREM_H

// Pure-m DFT quadrature: the azimuthal (phi) integration is performed
// ANALYTICALLY rather than numerically.
//
// For pure-m orbitals psi = R(mu) Y_l^m(nu) e^{i m phi} the density is
// phi-independent, because |e^{i m phi}| = 1. Consequently
//
//   * rho, sigma and tau depend only on (mu, nu);
//   * the phi integral of a matrix element contributes 2 pi delta(m_i, m_j),
//     so V_xc is block diagonal in m and each block carries a factor 2 pi;
//   * d rho / d phi == 0, so the density gradient has no phi component;
//   * the only surviving phi contribution is in tau, where
//     d psi / d phi = i m psi gives the analytic term m^2 |psi|^2 / h_phi^2.
//
// The quadrature therefore runs over the (mu, nu) plane only -- the same
// grid and measure src/diatomic/twodquadrature.cpp already uses (its weight
// carries the explicit 2*pi from the phi integral) -- and all arithmetic is
// REAL: no complex basis-function arrays, no bf_phi, no phi loop.
//
// Valid whenever the orbitals cannot mix m, i.e. --symmetry >= 1 and no
// in-plane (x,y) field. The general (m-mixing) case keeps the 3D complex
// grid in dftgrid.{h,cpp}.

#include "basis.h"
#include "../general/dftgrid_common.h"
#include <vector>

namespace helfem {
  namespace diatomic {
    namespace dftgrid_purem {

      /// Worker for the pure-m (analytic-phi) DFT quadrature. Shares the XC
      /// plumbing (rho/sigma/tau/vxc/... buffers, libxc dispatch, energy
      /// accumulation) with the other DFTGridWorker variants via
      /// helfem::dftgrid_common::DFTGridWorkerBase.
      class PureMDFTGridWorker : public helfem::dftgrid_common::DFTGridWorkerBase {
      protected:
        /// Basis set
        const helfem::diatomic::basis::TwoDBasis *basp;

        /// Angular (nu) grid -- NO phi
        helfem::Vector cth, wang;

        /// Scale factors at the current (mu, nu) points.
        /// h_mu == h_nu == Rhalf sqrt(sinh^2 mu + sin^2 nu)
        helfem::Vector scale_r;
        /// 1 / h_mu^2 (== 1 / h_nu^2)
        helfem::Vector inv_scale_r2;
        /// 1 / h_phi^2, h_phi = Rhalf sinh(mu) sin(nu). Only needed for the
        /// analytic m^2 term in tau.
        helfem::Vector inv_scale_phi2;

        /// The m values present in the basis
        std::vector<int> mlist;
        /// l values of the angular shells in each m block, in the same order
        /// as bf_list_dummy lists them
        std::vector<std::vector<int>> lval_m;
        /// Angular factors, precomputed ONCE: they depend only on (l, m, nu),
        /// never on the radial point, so they must not be rebuilt per element.
        /// sph_m[im][ish](ia) = Y_l^m(nu_ia, 0);
        /// dsph_m[im][ish](ia) = d Y_l^m / d nu at nu_ia.
        std::vector<std::vector<helfem::Vector>> sph_m, dsph_m;

        /// One-element cache of the radial FEM functions and their first two
        /// mu derivatives at the element's quadrature points (rows = radial
        /// points, cols = element primitives). The FEM polynomials depend only
        /// on the element, so evaluating them per angular point -- which is
        /// what going through TwoDBasis::eval_bf(iel,irad,cth,m) does -- redoes
        /// the same work nrad * nang * n_m times and throws all but one row
        /// away. Profiling put >50% of the run time there.
        size_t cached_iel;
        bool cache_valid;
        helfem::Matrix rad_f, rad_df, rad_d2f;
        /// Dummy-basis indices of the functions in each m block
        std::vector<std::vector<Eigen::Index>> bf_ind_m;
        /// Per-m REAL basis function values and (mu, nu) derivatives,
        /// each (nbf_m x npts)
        std::vector<helfem::Matrix> bf_m, dr_m, dth_m;
        /// Per-m REAL Laplacian of the basis functions (nbf_m x npts). Only
        /// built for Laplacian-dependent meta-GGAs; see TwoDBasis::eval_lf --
        /// the associated Legendre equation makes it purely radial.
        std::vector<helfem::Matrix> bf_lapl_m;
        /// Per-m density helper P^m * bf_m (and the derivative analogues)
        std::vector<helfem::Matrix> Pv_m, Pv_dr_m, Pv_dth_m;
        std::vector<helfem::Matrix> Pav_m, Pav_dr_m, Pav_dth_m;
        std::vector<helfem::Matrix> Pbv_m, Pbv_dr_m, Pbv_dth_m;
        /// Per-m density contribution rho_m (needed for tau's analytic m^2
        /// term), 1 x npts. Spin-resolved in the polarized case.
        std::vector<helfem::Vector> rho_m, rhoa_m, rhob_m;

        /// Density gradient, (2 or 4) x npts: (d rho/d mu, d rho/d nu) per
        /// spin channel, already divided by the scale factors.
        helfem::Matrix grho;

      public:
        PureMDFTGridWorker();
        PureMDFTGridWorker(const helfem::diatomic::basis::TwoDBasis * basp, int lang);
        ~PureMDFTGridWorker();

        /// Build the per-m real basis functions (and derivatives, if needed)
        /// on the (nu) grid at radial point irad of element iel.
        void compute_bf(size_t iel, size_t irad);

        /// Update rho / sigma / tau, restricted
        void update_density(const helfem::Matrix & P);
        /// Update rho / sigma / tau, unrestricted
        void update_density(const helfem::Matrix & Pa, const helfem::Matrix & Pb);

        /// Kinetic energy density integral
        double compute_Ekin() const;

        /// Accumulate the XC Fock contribution, restricted. Only the
        /// m-diagonal blocks are built -- the off-diagonal blocks vanish by
        /// the analytic phi integration.
        void eval_Fxc(helfem::Matrix & H) const;
        /// Accumulate the XC Fock contribution, unrestricted
        void eval_Fxc(helfem::Matrix & Ha, helfem::Matrix & Hb, bool beta=true) const;
      };

      /// Wrapper with the same public interface as
      /// helfem::diatomic::dftgrid::DFTGrid, so the driver can select it
      /// whenever the orbitals are pure-m.
      class PureMDFTGrid {
      private:
        const helfem::diatomic::basis::TwoDBasis * basp;
        /// Order of the nu (angular) rule. There is no phi rule.
        int lang;

      public:
        PureMDFTGrid();
        PureMDFTGrid(const helfem::diatomic::basis::TwoDBasis * basp, int lang);
        ~PureMDFTGrid();

        /// Restricted
        void eval_Fxc(int x_func, const helfem::Vector & x_pars, int c_func, const helfem::Vector & c_pars,
                       const helfem::Matrix & P, helfem::Matrix & H,
                       double & Exc, double & Nel, double & Ekin, double thr);
        /// Unrestricted
        void eval_Fxc(int x_func, const helfem::Vector & x_pars, int c_func, const helfem::Vector & c_pars,
                       const helfem::Matrix & Pa, const helfem::Matrix & Pb,
                       helfem::Matrix & Ha, helfem::Matrix & Hb,
                       double & Exc, double & Nel, double & Ekin, bool beta, double thr);
      };

    }
  }
}

#endif
