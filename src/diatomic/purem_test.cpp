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

// Validation of the pure-m (analytic-phi) DFT grid against the general
// 3D complex grid.
//
// Both grids are handed the SAME density matrix and asked for a SINGLE
// Fock build, so this compares the quadratures themselves rather than the
// endpoint of an SCF -- no convergence behaviour is folded in, and the
// whole XC matrix is compared, not just a scalar energy.
//
// Additionally finite-difference-checks TwoDBasis::eval_lf, whose closed
// form relies on the associated Legendre equation, against a numerical
// Laplacian built from the arbitrary-point evaluator eval_bf(mu,cth,phi).

#include "../general/cmdline.h"
#include "../general/constants.h"
#include "../general/dftfuncs.h"
#include "../general/elements.h"
#include "../general/scf_helpers.h"
#include "../general/spherical_harmonics.h"
#include "../general/timer.h"

#include "utils.h"
#include "basis.h"
#include "dftgrid.h"
#include "dftgrid_purem.h"
#include "../atomic/basis.h"
#include <ArmaEigen.h>
#include <Eigen/Eigenvalues>
#include <set>
#include <map>

using namespace helfem;

// Numerical Laplacian of basis function `idx` at (mu, nu), built from
// eval_bf(mu, cth, phi) via
//   grad^2 f = (1/h^2)[ f_mumu + coth(mu) f_mu + f_nunu + cot(nu) f_nu ]
//              - (m^2/h_phi^2) f
// evaluated at phi = 0, where the pure-m functions are real.
static double fd_laplacian(const diatomic::basis::TwoDBasis & basis, size_t idx,
                           int m, double mu, double nu, double Rhalf, double d) {
  auto f = [&](double u, double v) {
    return std::real(basis.eval_bf(u, std::cos(v), 0.0)(idx));
  };

  const double f0   = f(mu, nu);
  const double fmp  = f(mu + d, nu), fmm = f(mu - d, nu);
  const double fnp  = f(mu, nu + d), fnm = f(mu, nu - d);

  const double f_mu   = (fmp - fmm) / (2.0 * d);
  const double f_mumu = (fmp - 2.0 * f0 + fmm) / (d * d);
  const double f_nu   = (fnp - fnm) / (2.0 * d);
  const double f_nunu = (fnp - 2.0 * f0 + fnm) / (d * d);

  const double shmu = std::sinh(mu), chmu = std::cosh(mu);
  const double snu = std::sin(nu), cnu = std::cos(nu);
  const double h2   = Rhalf * Rhalf * (shmu * shmu + snu * snu);
  const double hphi2 = Rhalf * Rhalf * shmu * shmu * snu * snu;

  return (f_mumu + (chmu / shmu) * f_mu + f_nunu + (cnu / snu) * f_nu) / h2
          - (m * m) * f0 / hphi2;
}

int main(int argc, char **argv) {
  cmdline::parser parser;
  parser.add<std::string>("Z1", 0, "first nuclear charge", false, "1");
  parser.add<std::string>("Z2", 0, "second nuclear charge", false, "1");
  parser.add<double>("Rbond", 0, "internuclear distance", false, 1.4);
  parser.add<int>("nela", 0, "number of alpha electrons", false, 1);
  parser.add<int>("nelb", 0, "number of beta electrons", false, 1);
  parser.add<int>("lmax", 0, "maximum l", false, 2);
  parser.add<int>("mmax", 0, "maximum m", false, 2);
  parser.add<double>("Rmax", 0, "practical infinity", false, 20.0);
  parser.add<int>("nelem", 0, "number of elements", false, 3);
  parser.add<int>("nnodes", 0, "nodes per element", false, 15);
  parser.add<int>("primbas", 0, "primitive radial basis", false, 4);
  parser.add<int>("ldft", 0, "theta rule (0 auto)", false, 0);
  parser.add<int>("mdft", 0, "phi rule (0 auto)", false, 0);
  parser.add<double>("dftthr", 0, "density screening threshold", false, 1e-12);
  parser.add<bool>("dumplf", 0, "dump eval_lf vs FD values", false, false);
  parser.add<bool>("mproject", 0, "project the density onto exact m-block-diagonal form", false, false);
  parser.add<bool>("fd", 0, "run the eval_lf finite-difference check", false, true);
  parser.add<std::string>("func", 0, "test only this functional (empty = all)", false, "");
  parser.add<double>("perturb", 0, "relative perturbation of P: report how much H moves (functional conditioning)", false, 0.0);
  parser.parse_check(argc, argv);

  const int Z1 = get_Z(parser.get<std::string>("Z1"));
  const int Z2 = get_Z(parser.get<std::string>("Z2"));
  const double Rbond = parser.get<double>("Rbond");
  const int nela = parser.get<int>("nela");
  const int nelb = parser.get<int>("nelb");
  const int lmax = parser.get<int>("lmax");
  const int mmax = parser.get<int>("mmax");
  const double Rmax = parser.get<double>("Rmax");
  const int Nelem = parser.get<int>("nelem");
  const int Nnodes = parser.get<int>("nnodes");
  const int primbas = parser.get<int>("primbas");

  auto poly = std::shared_ptr<const polynomial_basis::PolynomialBasis>(
      polynomial_basis::get_basis(primbas, Nnodes));
  const int Nquad = 5 * poly->get_nbf();

  const Eigen::VectorXi lmmax = Eigen::VectorXi::Constant(mmax + 1, lmax);
  Eigen::VectorXi lval, mval;
  diatomic::basis::lm_to_l_m(lmmax, lval, mval);

  const double Rhalf = 0.5 * Rbond;
  const double mumax = utils::arcosh(Rmax / Rhalf);
  const helfem::Vector bval = atomic::basis::normal_grid(Nelem, mumax, 4, 1.0);

  diatomic::basis::TwoDBasis basis(Z1, Z2, Rhalf, poly, Nquad, bval, lval, mval);
  printf("Basis: %i angular x %i radial = %i functions\n",
          (int) basis.Nang(), (int) basis.Nrad(), (int) basis.Nbf());

  const int lang = parser.get<int>("ldft") > 0 ? parser.get<int>("ldft")
                                                : 4 * lval.maxCoeff() + 12;
  const int mang = parser.get<int>("mdft") > 0 ? parser.get<int>("mdft")
                                                : 4 * mval.maxCoeff() + 5;

  // ------------------------------------------------------------------
  // 0) The angular substitution behind eval_lf, checked on the special
  //    function itself -- no basis, no indexing:
  //      Y_nunu + cot(nu) Y_nu == [ m^2/sin^2(nu) - l(l+1) ] Y
  //    This is what removes the second angular derivative AND makes the
  //    1/sin^2(nu) cancel against -m^2/h_phi^2.
  // ------------------------------------------------------------------
  printf("=== associated Legendre ODE (the angular step in eval_lf) ===\n");
  {
    double worst = 0.0;
    const double d = 1e-5;
    auto Y = [](int l, int m, double nu) {
      return std::real(::spherical_harmonics(l, m, std::cos(nu), 0.0));
    };
    for (int l = 0; l <= 4; l++)
      for (int m = 0; m <= l; m++)
        for (double nu : {0.35, 0.8, 1.2, 1.9, 2.5, 2.9}) {
          const double y0 = Y(l,m,nu), yp = Y(l,m,nu+d), ym = Y(l,m,nu-d);
          const double y_nu   = (yp - ym) / (2.0*d);
          const double y_nunu = (yp - 2.0*y0 + ym) / (d*d);
          const double lhs = y_nunu + (std::cos(nu)/std::sin(nu)) * y_nu;
          const double rhs = (m*m/(std::sin(nu)*std::sin(nu)) - l*(l+1)) * y0;
          worst = std::max(worst, std::abs(lhs-rhs)/std::max(1.0,std::abs(rhs)));
        }
    printf("worst deviation over l<=4, 0<=m<=l, 6 nu: %.3e   -> %s\n\n",
           worst, worst < 1e-4 ? "ODE CONFIRMED (FD-limited)" : "ODE MISMATCH");
  }

  // ------------------------------------------------------------------
  // 1) Finite-difference check of eval_lf
  // ------------------------------------------------------------------
  printf("\n=== eval_lf vs finite differences ===\n");
  if (parser.get<bool>("fd")) {
    double worst = 0.0;
    size_t nchecked = 0;
    // eval_bf(mu,cth,phi) returns the vector in the REAL basis (it ends with
    // `return bf(pure_indices())`), while eval_lf's columns and bf_list_dummy
    // live in the DUMMY basis. The two coincide only up to the first m != 0
    // shell, whose psi(mu=0,nu) function the boundary condition drops. So map
    // dummy -> real explicitly, and skip the dropped functions, which have no
    // real-basis counterpart to difference.
    const std::vector<Eigen::Index> pure(basis.pure_indices());
    std::map<Eigen::Index, size_t> dummy2real;
    for (size_t r = 0; r < pure.size(); r++)
      dummy2real[pure[r]] = r;

    for (int m = 0; m <= std::min(mmax, 2); m++) {
      // A radial point comfortably inside the second element, and a few nu
      const size_t iel = std::min<size_t>(1, basis.get_rad_Nel() - 1);
      const helfem::Vector rr(basis.get_r(iel));
      // The FEM basis is only C0 across element boundaries -- the second
      // derivative jumps -- so a finite-difference stencil must not step out
      // of the element. Gauss-Lobatto points cluster at the edges, so skip
      // any radial point within `edge` of a boundary.
      const helfem::Vector bv(basis.get_bval());
      const double edge = 100.0 * 1e-4;
      for (size_t irad = 0; irad < (size_t) rr.size(); irad += 7) {
        const double mu = rr(irad);
        if (mu - bv(iel) < edge || bv(iel + 1) - mu < edge) continue;
        nchecked++;
        for (double nu : {0.7, 1.3, 2.2}) {
          helfem::Matrix lf;
          basis.eval_lf(iel, irad, std::cos(nu), m, lf);

          // Map the m-block columns back to dummy indices so the same
          // function can be evaluated by eval_bf(mu, cth, phi).
          const std::vector<Eigen::Index> dummy(basis.bf_list_dummy(iel, m));
          const double d = 1e-4;
          for (size_t k = 0; k < dummy.size(); k++) {
            auto it = dummy2real.find(dummy[k]);
            if (it == dummy2real.end()) continue;   // dropped by the m != 0 BC
            const double ana = lf(0, k);
            const double num = fd_laplacian(basis, it->second, m, mu, nu, Rhalf, d);
            const double scale = std::max(1.0, std::abs(num));
            const double err = std::abs(ana - num) / scale;
            if (parser.get<bool>("dumplf") && k < 3)
              printf("  m=%d mu=%.4f nu=%.2f k=%zu: ana=%14.6e  num=%14.6e  ratio=%8.4f\n",
                     m, mu, nu, k, ana, num, (num != 0.0) ? ana/num : 0.0);
            worst = std::max(worst, err);
          }
        }
      }
    }
    printf("checked %zu (mu, m) points, %d nu each, all functions in the block\n", nchecked, 3);
    printf("worst relative deviation: %.3e\n", worst);
    // Tolerance is set by the finite-difference reference (central differences
    // on a degree-14 LIP), not by eval_lf.
    printf("%s\n", worst < 1e-3 ? "  -> eval_lf CONFIRMED (FD-limited)" : "  -> eval_lf MISMATCH");
  }

  // ------------------------------------------------------------------
  // 2) Same density, one Fock build, both grids
  // ------------------------------------------------------------------
  const helfem::Matrix S = basis.overlap();
  const helfem::Matrix T = basis.kinetic();
  const helfem::Matrix V = basis.nuclear();
  const helfem::Matrix H0 = T + V;
  const helfem::Matrix Sinvh = basis.Sinvh(false, 0);

  // Core-Hamiltonian density
  const helfem::Matrix Fao = Sinvh.transpose() * H0 * Sinvh;
  Eigen::SelfAdjointEigenSolver<helfem::Matrix> eig(Fao);
  const helfem::Matrix C = Sinvh * eig.eigenvectors();

  auto make_P = [&](int nocc, double occ) {
    helfem::Matrix P = helfem::Matrix::Zero(basis.Nbf(), basis.Nbf());
    for (int i = 0; i < nocc; i++)
      P += occ * C.col(i) * C.col(i).transpose();
    return P;
  };
  helfem::Matrix Pa = make_P(nela, 1.0);
  helfem::Matrix Pb = make_P(nelb, 1.0);

  // The pure-m grid PRESUMES pure-m orbitals, i.e. a density that is exactly
  // m-block diagonal -- which is what a --symmetry>=1 SCF builds. Here P comes
  // from diagonalising H0 in the full basis, so degenerate +m/-m pairs can come
  // back with ~1e-16 of m mixing. That gives rho a ~1e-16 phi ripple, which an
  // ill-conditioned potential (e.g. TPSS correlation in the low-density tail,
  // where dv/drho is huge) amplifies enormously. Projecting removes it.
  if (parser.get<bool>("mproject")) {
    helfem::Matrix Ma = helfem::Matrix::Zero(basis.Nbf(), basis.Nbf());
    helfem::Matrix Mb = helfem::Matrix::Zero(basis.Nbf(), basis.Nbf());
    for (int m = -mmax; m <= mmax; m++) {
      const std::vector<Eigen::Index> idx(basis.m_indices(m));
      for (size_t i = 0; i < idx.size(); i++)
        for (size_t j = 0; j < idx.size(); j++) {
          Ma(idx[i], idx[j]) = Pa(idx[i], idx[j]);
          Mb(idx[i], idx[j]) = Pb(idx[i], idx[j]);
        }
    }
    printf("m-projection changed the density by %.2e (alpha)\n", (Ma - Pa).cwiseAbs().maxCoeff());
    Pa = Ma; Pb = Mb;
  }
  const helfem::Matrix P  = Pa + Pb;

  auto grid   = diatomic::dftgrid::DFTGrid(&basis, lang, mang);
  auto pmgrid = diatomic::dftgrid_purem::PureMDFTGrid(&basis, lang);

  std::vector<std::string> funcs =
    {"lda_x", "lda_c_vwn", "gga_x_pbe", "gga_c_pbe", "mgga_x_tpss", "mgga_c_tpss", "mgga_x_br89"};
  if (parser.get<std::string>("func").size())
    funcs = {parser.get<std::string>("func")};

  helfem::Vector nopars;
  printf("\n=== single Fock build, identical density: pure-m vs 3D grid ===\n");
  printf("%-14s %18s %18s %12s %12s\n", "functional", "Exc (3D)", "Exc (pure-m)", "|dExc|", "max|dH|");
  for (const std::string & fn : funcs) {
    int x_func, c_func;
    try {
      ::parse_xc_func(x_func, c_func, fn);
    } catch (...) {
      printf("%-14s  (unavailable)\n", fn.c_str());
      continue;
    }

    double Exc3 = 0.0, Nel3 = 0.0, Ekin3 = 0.0;
    double ExcP = 0.0, NelP = 0.0, EkinP = 0.0;
    helfem::Matrix H3 = helfem::Matrix::Zero(basis.Nbf(), basis.Nbf());
    helfem::Matrix HP = helfem::Matrix::Zero(basis.Nbf(), basis.Nbf());

    bool have3 = true;
    try {
      grid.eval_Fxc(x_func, nopars, c_func, nopars, P, H3, Exc3, Nel3, Ekin3, parser.get<double>("dftthr"));
    } catch (const std::exception & e) {
      have3 = false;  // e.g. "Laplacian not implemented" in the 3D grid
    }

    try {
      pmgrid.eval_Fxc(x_func, nopars, c_func, nopars, P, HP, ExcP, NelP, EkinP, parser.get<double>("dftthr"));
    } catch (const std::exception & e) {
      printf("%-14s  pure-m threw: %s", fn.c_str(), e.what());
      continue;
    }

    const double pert = parser.get<double>("perturb");
    if (pert != 0.0) {
      // Conditioning probe: feed the SAME grid a density perturbed at the
      // level of floating-point noise, and see how far the Fock matrix moves.
      double e2 = 0.0, n2 = 0.0, k2 = 0.0;
      helfem::Matrix HP2 = helfem::Matrix::Zero(basis.Nbf(), basis.Nbf());
      const helfem::Matrix P2 = P * (1.0 + pert);
      pmgrid.eval_Fxc(x_func, nopars, c_func, nopars, P2, HP2, e2, n2, k2, parser.get<double>("dftthr"));
      const double dHp = (HP2 - HP).cwiseAbs().maxCoeff();
      const double hmax = HP.cwiseAbs().maxCoeff();
      printf("%-14s  PERTURB rel=%.0e -> max|dH|=%.3e (rel %.2e), amplification %.1e\n",
             fn.c_str(), pert, dHp, dHp/hmax, (dHp/hmax)/pert);
    }

    if (!have3) {
      printf("%-14s %18s %18.10f %12s %12s   (3D grid cannot do this)\n",
             fn.c_str(), "n/a", ExcP, "-", "-");
      continue;
    }

    const double dE = std::abs(Exc3 - ExcP);
    Eigen::Index bi = 0, bj = 0;
    const double dH = (H3 - HP).cwiseAbs().maxCoeff(&bi, &bj);
    bool gga, mgga_t, mgga_l;
    ::is_gga_mgga(x_func ? x_func : c_func, gga, mgga_t, mgga_l);
    const bool lapl = (x_func ? ::laplacian_needed(x_func) : false)
                       || (c_func ? ::laplacian_needed(c_func) : false);
    printf("%-14s %18.10f %18.10f %12.2e %12.2e   [gga=%d tau=%d mggal=%d lapl=%d] "
           "dNel=%.1e dEkin=%.1e  worst H(%d,%d)\n",
           fn.c_str(), Exc3, ExcP, dE, dH, (int)gga, (int)mgga_t, (int)mgga_l, (int)lapl,
           std::abs(Nel3-NelP), std::abs(Ekin3-EkinP), (int)bi, (int)bj);
    printf("%-14s   max|H|=%.3e  -> relative max|dH| = %.2e\n", "", H3.cwiseAbs().maxCoeff(), dH/std::max(1e-300,H3.cwiseAbs().maxCoeff()));
    if (dH > 1e-8) {
      int m_bi = -99, m_bj = -99;
      for (int m = -mmax; m <= mmax; m++) {
        const std::vector<Eigen::Index> idx(basis.m_indices(m));
        for (size_t q = 0; q < idx.size(); q++) {
          if (idx[q] == bi) m_bi = m;
          if (idx[q] == bj) m_bj = m;
        }
      }
      printf("%-14s   H3(%d,%d)=%.10e  HP=%.10e   [m(%d)=%d, m(%d)=%d]\n", "",
             (int)bi,(int)bj, H3(bi,bj), HP(bi,bj), (int)bi, m_bi, (int)bj, m_bj);
    }
  }

  return 0;
}
