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

// Validation of the current-corrected tau-meta-GGA,
//   tau~ = tau - |j_p|^2 / (2 rho)   per spin,
// in the two diatomic workers that carry it.
//
// The two are independent implementations of the current: the 3D worker
// forms j_p = sum_uv P_uv Im(chi_u^* grad chi_v) from the complex basis
// values and assembles the y channel against the same bilinear form,
// while the pure-m worker never sees a phi derivative and uses
// j_phi = sum_m m rho_m / h_phi with the y channel folded into an
// m-weighted LDA-type increment. Agreeing on the energy and the whole XC
// matrix for a density that carries a current is therefore a real test
// of both, and of the sign conventions between j and y in each.
//
// Checks, each with a failure path:
//   1. j = 0 (+m and -m occupied equally): --current changes nothing.
//   2. The single-orbital limit: one occupied orbital has tau~ = tau_W,
//      where TPSS correlation vanishes identically. Without the current
//      it does not for m != 0 -- the M_L dependence the correction removes.
//   3. pure-m and 3D agree on Exc and the XC matrix, current on.
//   4. The pure-m XC matrix is the derivative of its energy.
//   5. The response of both grids against a central difference of their
//      own XC matrices. Dropping the perturbed current from the kernel
//      fails this by five orders of magnitude.
// 4 and 5 use a density in which no spin channel is a single orbital;
// see the comment on `gen` below for why that matters.

#include <helfem/helfem.h>
#include "../general/dftfuncs.h"
#include "../general/elements.h"

#include "utils.h"
#include "basis.h"
#include "dftgrid.h"
#include "dftgrid_purem.h"
#include "../atomic/basis.h"
#include <Eigen/Eigenvalues>
#include <cstdio>
#include <random>

using namespace helfem;

namespace {
  int nfail = 0;

  void check(bool ok, const char * what, double value, double tol) {
    printf("  %-58s %10.3e  (tol %.0e)  %s\n", what, value, tol, ok ? "ok" : "FAIL");
    if(!ok) nfail++;
  }

  /// Maximum of |A| over all elements
  double maxabs(const helfem::Matrix & A) { return A.cwiseAbs().maxCoeff(); }

  struct Density {
    helfem::Matrix Pa, Pb;
  };

  /// One Fock build on either grid, polarized.
  template<typename Grid>
  double fxc(Grid & grid, int x_func, int c_func, const Density & D,
             helfem::Matrix & Ha, helfem::Matrix & Hb) {
    const helfem::Vector nopars;
    double exc = 0.0, nel = 0.0, ekin = 0.0;
    grid.eval_Fxc(x_func, nopars, c_func, nopars, D.Pa, D.Pb, Ha, Hb, exc, nel, ekin, true, 1e-12);
    return exc;
  }
}

int main(int, char **) {
  // H2. The densities below are built by hand from core-Hamiltonian
  // orbitals of each m block, so the molecule only has to supply a basis
  // with m = -1, 0, +1; the pi orbitals are occupied however the test
  // needs, physical or not.
  const int Z1 = 1, Z2 = 1;
  const double Rhalf = 0.5 * 2.0;
  const int lmax = 3, mmax = 1;
  const int Nelem = 2, Nnodes = 8, primbas = 5;
  const double Rmax = 20.0;

  auto poly = std::shared_ptr<const polynomial_basis::PolynomialBasis>(
      polynomial_basis::make_basis(primbas, Nnodes));
  const int Nquad = 5 * poly->nbf();
  const Eigen::VectorXi lmmax = Eigen::VectorXi::Constant(mmax + 1, lmax);
  Eigen::VectorXi lval, mval;
  diatomic::basis::lm_to_l_m(lmmax, lval, mval);
  const double mumax = utils::arcosh(Rmax / Rhalf);
  const helfem::Vector bval = atomic::basis::normal_grid(Nelem, mumax, 4, 1.0);
  diatomic::basis::TwoDBasis basis(Z1, Z2, Rhalf, poly, Nquad, bval, lval, mval);
  const Eigen::Index Nbf = basis.Nbf();
  printf("Basis: %i functions, m = -%i .. %i\n", (int) Nbf, mmax, mmax);

  // Core-Hamiltonian orbitals block by block, so every orbital has a
  // definite m and the density is exactly m-block diagonal -- the pure-m
  // worker presumes that, and it is what --symmetry>=1 produces.
  const helfem::Matrix S = basis.overlap();
  const helfem::Matrix H0 = basis.kinetic() + basis.nuclear();
  auto orbital = [&](int m, int k) {
    const std::vector<Eigen::Index> idx(basis.m_indices(m));
    const Eigen::Index n = (Eigen::Index) idx.size();
    helfem::Matrix Hb(n, n), Sb(n, n);
    for(Eigen::Index i=0;i<n;i++)
      for(Eigen::Index j=0;j<n;j++) {
        Hb(i,j) = H0(idx[i], idx[j]);
        Sb(i,j) = S(idx[i], idx[j]);
      }
    Eigen::GeneralizedSelfAdjointEigenSolver<helfem::Matrix> eig(Hb, Sb);
    helfem::Vector c = helfem::Vector::Zero(Nbf);
    for(Eigen::Index i=0;i<n;i++)
      c(idx[i]) = eig.eigenvectors()(i, k);
    return c;
  };
  const helfem::Vector s0 = orbital(0, 0), s1 = orbital(0, 1), pp = orbital(1, 0), pm = orbital(-1, 0);
  auto proj = [](const helfem::Vector & c, double w) -> helfem::Matrix { return w * c * c.transpose(); };
  const helfem::Matrix Z = helfem::Matrix::Zero(Nbf, Nbf);

  const int ldft = 4 * lmax + 12, mdft = 4 * mmax + 5;
  auto grid   = diatomic::dftgrid::DFTGrid(&basis, ldft, mdft);
  auto pmgrid = diatomic::dftgrid_purem::PureMDFTGrid(&basis, ldft);

  int tpss_x, tpss_c, dummy;
  ::parse_xc_func(tpss_x, dummy, "mgga_x_tpss");
  ::parse_xc_func(tpss_c, dummy, "mgga_c_tpss");

  // A current-carrying open shell: sigma^2 pi_+1^1 in alpha, sigma in beta.
  const Density cur{ proj(s0, 1.0) + proj(pp, 1.0), proj(s0, 1.0) };
  // The same pi electron shared equally between +1 and -1: no current.
  const Density nocur{ proj(s0, 1.0) + proj(pp, 0.5) + proj(pm, 0.5), proj(s0, 1.0) };
  // For the derivative checks, a density with no spin channel made of a
  // single orbital. One orbital has tau = tau_W, i.e. z = 1, which is
  // exactly where libxc clamps sigma <= 8 rho tau: the energy has a kink
  // there and a central difference straddles it, so the plain TPSS Fock
  // matrix "fails" a derivative check on such a density by several per
  // cent with or without the current. Fractional occupations of several
  // orbitals keep z < 1; +1 and -1 occupied unequally keep the current.
  const Density gen{ proj(s0, 1.0) + proj(s1, 0.6) + proj(pp, 0.8) + proj(pm, 0.2),
                     proj(s0, 0.9) + proj(s1, 0.4) + proj(pp, 0.3) + proj(pm, 0.5) };

  printf("\n1. no current: --current changes nothing\n");
  {
    helfem::Matrix Ha0, Hb0, Ha1, Hb1;
    grid.set_current(false);
    const double e0 = fxc(grid, tpss_x, tpss_c, nocur, Ha0, Hb0);
    grid.set_current(true);
    const double e1 = fxc(grid, tpss_x, tpss_c, nocur, Ha1, Hb1);
    check(std::abs(e1-e0) < 1e-12, "3D:     |dExc|", std::abs(e1-e0), 1e-12);
    check(std::max(maxabs(Ha1-Ha0), maxabs(Hb1-Hb0)) < 1e-10, "3D:     max|dH|",
          std::max(maxabs(Ha1-Ha0), maxabs(Hb1-Hb0)), 1e-10);

    pmgrid.set_current(false);
    const double p0 = fxc(pmgrid, tpss_x, tpss_c, nocur, Ha0, Hb0);
    pmgrid.set_current(true);
    const double p1 = fxc(pmgrid, tpss_x, tpss_c, nocur, Ha1, Hb1);
    check(std::abs(p1-p0) < 1e-12, "pure-m: |dExc|", std::abs(p1-p0), 1e-12);
    check(std::max(maxabs(Ha1-Ha0), maxabs(Hb1-Hb0)) < 1e-10, "pure-m: max|dH|",
          std::max(maxabs(Ha1-Ha0), maxabs(Hb1-Hb0)), 1e-10);
  }

  printf("\n2. single-orbital limit: TPSS correlation of one pi_+1 electron\n");
  {
    const Density one{ proj(pp, 1.0), Z };
    helfem::Matrix Ha, Hb;
    grid.set_current(false);
    const double off3 = fxc(grid, 0, tpss_c, one, Ha, Hb);
    grid.set_current(true);
    const double on3 = fxc(grid, 0, tpss_c, one, Ha, Hb);
    pmgrid.set_current(false);
    const double offp = fxc(pmgrid, 0, tpss_c, one, Ha, Hb);
    pmgrid.set_current(true);
    const double onp = fxc(pmgrid, 0, tpss_c, one, Ha, Hb);
    printf("  Ec without current: 3D % .10e  pure-m % .10e\n", off3, offp);
    printf("  Ec with current:    3D % .10e  pure-m % .10e\n", on3, onp);
    check(std::abs(off3) > 1e-4, "without the current, Ec is not zero", std::abs(off3), 1e-4);
    check(std::abs(on3) < 1e-9*std::abs(off3), "3D:     |Ec| / |Ec(no current)|", std::abs(on3/off3), 1e-9);
    check(std::abs(onp) < 1e-9*std::abs(offp), "pure-m: |Ec| / |Ec(no current)|", std::abs(onp/offp), 1e-9);
  }

  grid.set_current(true);
  pmgrid.set_current(true);

  printf("\n3. pure-m vs 3D, current-carrying density\n");
  for(int fn : {tpss_x, tpss_c}) {
    helfem::Matrix Ha3, Hb3, HaP, HbP;
    const int x = (fn == tpss_x) ? fn : 0, c = (fn == tpss_c) ? fn : 0;
    const double e3 = fxc(grid, x, c, cur, Ha3, Hb3);
    const double eP = fxc(pmgrid, x, c, cur, HaP, HbP);
    const double scale = std::max(maxabs(Ha3), maxabs(Hb3));
    const double dH = std::max(maxabs(Ha3-HaP), maxabs(Hb3-HbP));
    printf("  %s: Exc 3D % .12f  pure-m % .12f\n", fn == tpss_x ? "x" : "c", e3, eP);
    check(std::abs(e3-eP) < 1e-9, "|Exc(3D) - Exc(pure-m)|", std::abs(e3-eP), 1e-9);
    check(dH < 1e-7*scale, "max|H(3D) - H(pure-m)| / max|H|", dH/scale, 1e-7);
  }

  // A random symmetric perturbation that keeps the density m-block
  // diagonal and moves the current.
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> uni(-1.0, 1.0);
  auto perturbation = [&]() {
    helfem::Matrix D = helfem::Matrix::Zero(Nbf, Nbf);
    for(int m = -mmax; m <= mmax; m++) {
      const std::vector<Eigen::Index> idx(basis.m_indices(m));
      for(size_t i=0;i<idx.size();i++)
        for(size_t j=0;j<=i;j++)
          D(idx[i], idx[j]) = D(idx[j], idx[i]) = uni(rng);
    }
    // Keep it in the occupied neighbourhood, so that the perturbed
    // densities stay positive and the functional smooth: a random dP
    // spread over the whole basis puts most of its weight where the
    // density is negligible.
    const helfem::Matrix Pocc = gen.Pa + gen.Pb;
    return helfem::Matrix(Pocc * S * D * S * Pocc + (Pocc * S * D * S * Pocc).transpose());
  };

  printf("\n4. pure-m: the XC matrix is the derivative of the energy\n");
  for(int t = 0; t < 4; t++) {
    pmgrid.set_current(t >= 2);
    if(t == 0 || t == 2) printf("  current %s\n", t >= 2 ? "on" : "off");
    const helfem::Matrix dPa = perturbation(), dPb = perturbation();
    helfem::Matrix Ha, Hb, Hp, Hq;
    fxc(pmgrid, tpss_x, tpss_c, gen, Ha, Hb);
    const double analytic = Ha.cwiseProduct(dPa).sum() + Hb.cwiseProduct(dPb).sum();
    double fd = 0.0;
    for(double h : {1e-5}) {
      const Density p{ gen.Pa + h*dPa, gen.Pb + h*dPb }, q{ gen.Pa - h*dPa, gen.Pb - h*dPb };
      fd = (fxc(pmgrid, tpss_x, tpss_c, p, Hp, Hq) - fxc(pmgrid, tpss_x, tpss_c, q, Hp, Hq)) / (2*h);
      printf("  direction %i  h %.0e  Tr(H dP) % .12e  FD % .12e\n", t, h, analytic, fd);
    }
    check(std::abs(fd-analytic) < 1e-7*std::abs(analytic), "|FD - Tr(H dP)| / |Tr(H dP)|",
          std::abs(fd-analytic)/std::abs(analytic), 1e-7);
  }

  printf("\n5. response against a central difference of the XC matrix\n");
  auto response_error = [&](auto & g, bool current) {
    g.set_current(current);
    std::vector<helfem::Matrix> dPa{ perturbation() }, dPb{ perturbation() }, dHa, dHb;
    const helfem::Vector nopars;
    g.eval_Fxc_response(tpss_x, nopars, tpss_c, nopars, gen.Pa, gen.Pb, dPa, dPb, dHa, dHb, 1e-12);
    const double h = 1e-5;
    helfem::Matrix Hap, Hbp, Ham, Hbm;
    fxc(g, tpss_x, tpss_c, Density{gen.Pa + h*dPa[0], gen.Pb + h*dPb[0]}, Hap, Hbp);
    fxc(g, tpss_x, tpss_c, Density{gen.Pa - h*dPa[0], gen.Pb - h*dPb[0]}, Ham, Hbm);
    const helfem::Matrix fda = (Hap-Ham)/(2*h), fdb = (Hbp-Hbm)/(2*h);
    const double scale = std::max(maxabs(fda), maxabs(fdb));
    return std::max(maxabs(dHa[0]-fda), maxabs(dHb[0]-fdb)) / scale;
  };
  rng.seed(11);
  const double p_off = response_error(pmgrid, false);
  rng.seed(11);
  const double p_on  = response_error(pmgrid, true);
  rng.seed(11);
  const double g_off = response_error(grid, false);
  rng.seed(11);
  const double g_on  = response_error(grid, true);
  printf("  relative max|dH - FD|:  pure-m  off %.3e  on %.3e    3D  off %.3e  on %.3e\n",
         p_off, p_on, g_off, g_on);
  // Dropping the perturbed current from the kernel puts both at 0.17.
  check(p_on < 1e-6, "pure-m, current on", p_on, 1e-6);
  check(g_on < std::max(1e-6, 10*g_off), "3D, current on (floor: 10x current off)", g_on, std::max(1e-6, 10*g_off));

  printf("\n%s: %i failure(s)\n", nfail ? "FAILED" : "PASSED", nfail);
  return nfail ? 1 : 0;
}
