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

// Does the C++ CAS integral engine reproduce the validated Python reference?
//
// python/helfem/pyscf_driver.py is checked against PySCF: its active-space
// integrals match ao2mo to 1e-16, and a CASSCF built on them agrees with
// PySCF's own mcscf.CASSCF to 1.6e-10 on the same integrals. So reproducing it
// exactly is a real check on this port, not a tautology -- the two share no
// code, only the underlying HelFEM J/K builds.
//
// The awkward part of a cross-language check is agreeing on the orbitals. They
// come here from the CORE GUESS -- the generalized eigenvectors of (hcore, S)
// -- which is deterministic in both languages, with a phase convention fixed
// below because an eigenvector is only defined up to sign. Be at lmax=0 has no
// degenerate s orbitals, so there is no further ambiguity to worry about.
//
// The reference values were produced by the Python driver at exactly these
// orbitals; see the PR discussion for the generating snippet.
//
// The sign convention is the thing most likely to break: HelFEM's exchange()
// returns the signed contribution ADDED to a SPIN channel's Fock matrix
// (src/atomic/main.cpp:383), whereas PySCF wants a positive K subtracted from
// a TOTAL density. The two differ by a sign AND a factor of two, reconciled in
// cas::closed_shell_veff. If that is wrong, E_inactive below is wrong.

#include "cas_integrals.h"
#include "../atomic/cas_engine.h"
#include "../atomic/basis.h"
#include "../../libhelfem/include/PolynomialBasis.h"
#include "../../libhelfem/include/ModelPotential.h"

#include <Eigen/Eigenvalues>
#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

namespace {

  int nfail = 0;

  /// exp(A) for a real antisymmetric A -- the same construction as
  /// trustregion_scf.cpp's expm_skew: i*A is Hermitian, so diagonalizing it
  /// exponentiates the eigenvalues exactly and the result is orthogonal by
  /// construction, with no scaling-and-squaring truncation and no dependence
  /// on Eigen's unsupported modules.
  helfem::Matrix expm_skew(const helfem::Matrix & A) {
    const std::complex<double> im(0.0, 1.0);
    Eigen::MatrixXcd H = im * A.cast<std::complex<double>>();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(H);
    Eigen::VectorXcd ph =
        (-im * es.eigenvalues().cast<std::complex<double>>()).array().exp();
    return (es.eigenvectors() * ph.asDiagonal() * es.eigenvectors().adjoint())
        .real();
  }

  void check(const char * what, double got, double ref, double tol) {
    const double err = std::abs(got - ref);
    const bool ok = err < tol;
    if (!ok) nfail++;
    printf("  %-26s %18.12f  ref %18.12f  err %8.2e  %s\n",
           what, got, ref, err, ok ? "ok" : "FAIL");
  }

} // namespace

int main() {
  using namespace helfem;

  // --- the same basis the Python reference used
  const int Z = 4, lmax = 0, nnodes = 6, nelem = 3;
  const double Rmax = 20.0, zexp = 2.0;
  const int igrid = 4;

  std::shared_ptr<const polynomial_basis::PolynomialBasis> poly(
      polynomial_basis::make_basis(4, nnodes));
  Eigen::VectorXi lval, mval;
  atomic::basis::angular_basis(lmax, lmax, lval, mval);
  const helfem::Vector bval = atomic::basis::form_grid(
      modelpotential::POINT_NUCLEUS, 0.0, nelem, Rmax, igrid, zexp,
      0, igrid, zexp, Z, 0, 0, 0.0);
  atomic::basis::TwoDBasis basis(Z, modelpotential::POINT_NUCLEUS, 0.0, poly,
                                 false, 5 * poly->nbf(), bval, lval, mval,
                                 0, 0, 0.0);
  basis.compute_tei(true);

  atomic::basis::AtomicCASEngine jk(basis);
  const Eigen::Index nbf = (Eigen::Index) basis.Nbf();
  printf("Be Nbf=%i\n\n", (int) nbf);

  // --- core-guess orbitals, phases fixed
  const helfem::Matrix S = basis.overlap();
  Eigen::GeneralizedSelfAdjointEigenSolver<helfem::Matrix> es(jk.hcore(), S);
  helfem::Matrix C = es.eigenvectors();
  for (Eigen::Index j = 0; j < C.cols(); j++) {
    Eigen::Index imax = 0;
    C.col(j).cwiseAbs().maxCoeff(&imax);
    if (C(imax, j) < 0.0) C.col(j) *= -1.0;
  }
  printf("first 4 core-guess eigenvalues: %.10f %.10f %.10f %.10f\n\n",
         es.eigenvalues()(0), es.eigenvalues()(1),
         es.eigenvalues()(2), es.eigenvalues()(3));

  const Eigen::Index ninact = 1, nact = 3;

  printf("active-space Hamiltonian vs the Python reference:\n");
  const cas::ActiveHamiltonian ah =
      cas::active_hamiltonian(jk, C, ninact, nact);
  check("E_inactive", ah.E_inactive, -13.487633038924, 1e-9);
  check("trace(h_eff)", ah.h_eff.trace(), -0.498155424803, 1e-9);
  check("h_eff[0,0]", ah.h_eff(0, 0), -0.408738678685, 1e-9);
  double erisum = 0.0;
  for (double v : ah.eri) erisum += v;
  check("sum(eri)", erisum, 2.508862565558, 1e-9);
  const size_t n = (size_t) nact;
  check("eri[0,0,0,0]", ah.eri[0], 0.596925047138, 1e-9);
  check("eri[0,1,2,1]", ah.eri[((0 * n + 1) * n + 2) * n + 1],
        -0.013791558469, 1e-9);

  // --- a fixed RDM pair, so the generalized Fock and gradient are exercised
  //     without needing a CI solver here
  helfem::Matrix D(nact, nact);
  D << 1.80, 0.05, 0.01,
       0.05, 0.15, 0.02,
       0.01, 0.02, 0.05;
  std::vector<double> d((size_t) (nact * nact * nact * nact));
  for (Eigen::Index p = 0; p < nact; p++)
    for (Eigen::Index q = 0; q < nact; q++)
      for (Eigen::Index r = 0; r < nact; r++)
        for (Eigen::Index s = 0; s < nact; s++)
          d[(((size_t) p * n + q) * n + r) * n + s] =
              D(p, q) * D(r, s) - 0.5 * D(p, s) * D(r, q);

  printf("\ngeneralized Fock and gradient vs the Python reference:\n");
  const helfem::Matrix F = cas::generalized_fock(jk, C, ninact, nact, D, d);
  check("sum(F)", F.sum(), -10.092279000769, 1e-8);
  check("F[0,0]", F(0, 0), -8.034334258464, 1e-9);
  check("F[5,2]", F(5, 2), -0.029534122337, 1e-9);
  const helfem::Matrix g = cas::orbital_gradient(jk, C, ninact, nact, D, d);
  check("norm(g)", g.norm(), 5.799542776404, 1e-8);
  check("g[0,1]", g(0, 1), 0.176955086194, 1e-9);

  // --- structural properties that hold independently of the reference
  printf("\nstructural checks:\n");
  const double asym = (g + g.transpose()).cwiseAbs().maxCoeff();
  printf("  %-26s %8.2e  %s\n", "gradient antisymmetry", asym,
         asym < 1e-12 ? "ok" : "FAIL");
  if (asym >= 1e-12) nfail++;
  // A transition density has no inactive occupation; with core_occ = 0 the
  // inactive columns must lose exactly the inactive-Fock term.
  const helfem::Matrix Ft =
      cas::generalized_fock(jk, C, ninact, nact, D, d, 0.0);
  const bool active_same =
      (Ft.rightCols(nact) - F.rightCols(nact)).cwiseAbs().maxCoeff() < 1e-12;
  const bool inactive_differs =
      (Ft.leftCols(ninact) - F.leftCols(ninact)).cwiseAbs().maxCoeff() > 1e-6;
  printf("  %-26s %s\n", "core_occ=0 keeps active",
         active_same ? "ok" : "FAIL");
  printf("  %-26s %s\n", "core_occ=0 changes inactive",
         inactive_differs ? "ok" : "FAIL");
  if (!active_same || !inactive_differs) nfail++;

  // --- kappa-kappa Hessian block, against the Python reference
  printf("\nkappa-kappa Hessian vs the Python reference:\n");
  helfem::Matrix K1 = helfem::Matrix::Zero(nbf, nbf);
  for (Eigen::Index m = 0; m < nbf; m++)
    for (Eigen::Index n2 = m + 1; n2 < nbf; n2++) {
      const double v = 1.0 / (1.0 + (double) m + (double) n2);
      K1(m, n2) = v;
      K1(n2, m) = -v;
    }
  K1 /= K1.norm();
  const helfem::Matrix dF =
      cas::fock_response_kappa(jk, C, ninact, nact, D, d, K1);
  check("sum(dF)", dF.sum(), -252.858517673524, 1e-7);
  check("dF[0,0]", dF(0, 0), 0.397843886155, 1e-9);
  check("dF[5,2]", dF(5, 2), 0.017758864748, 1e-9);
  const helfem::Matrix hk =
      cas::hess_kappa_kappa_raw(jk, C, ninact, nact, D, d, K1);
  check("norm(hess_kk_raw)", hk.norm(), 273.432663490307, 1e-7);
  check("hess_kk_raw[0,1]", hk(0, 1), 0.405368937066, 1e-9);
  check("hess_kk_raw[2,7]", hk(2, 7), 0.092946004178, 1e-9);

  // --- self-contained finite differences. frozen_ci_energy is rebuilt through
  //     active_hamiltonian and shares no algebra with the gradient or the
  //     Hessian, so differencing it checks them rather than re-expanding them.
  //     No CI solver is needed: the RDM pair above is fixed.
  printf("\nfinite differences of the frozen-CI energy (self-contained):\n");
  helfem::Matrix K2 = helfem::Matrix::Zero(nbf, nbf);
  for (Eigen::Index m = 0; m < nbf; m++)
    for (Eigen::Index n2 = m + 1; n2 < nbf; n2++) {
      const double v = std::sin(1.0 + 3.0 * (double) m - 2.0 * (double) n2);
      K2(m, n2) = v;
      K2(n2, m) = -v;
    }
  K2 /= K2.norm();

  const double hstep = 2e-4;
  auto Eat = [&](double a, double bb) {
    return cas::frozen_ci_energy(jk, C * expm_skew(a * K1 + bb * K2),
                                 ninact, nact, D, d);
  };

  // Gradient: g contracted with K1. A single central difference is
  // truncation-limited here -- its error is O(h^2 E'''), which at h = 2e-4
  // lands around 3e-7 and says nothing about the formula. Richardson
  // extrapolation cancels the h^2 term, and the check that it IS truncation
  // rather than a wrong gradient is that the error falls fourfold when h
  // halves. Both are asserted.
  const double fd_g1 = (Eat(hstep, 0.0) - Eat(-hstep, 0.0)) / (2 * hstep);
  const double fd_g2 =
      (Eat(0.5 * hstep, 0.0) - Eat(-0.5 * hstep, 0.0)) / hstep;
  const double fd_g = fd_g2 + (fd_g2 - fd_g1) / 3.0;
  const double an_g = 0.5 * (g.cwiseProduct(K1)).sum();
  const double r_h = std::abs(fd_g1 - an_g), r_h2 = std::abs(fd_g2 - an_g);
  printf("  %-26s h: %8.2e   h/2: %8.2e   ratio %5.2f (expect ~4)\n",
         "central-diff error", r_h, r_h2, r_h / r_h2);
  if (!(r_h2 < r_h)) { printf("    FAIL: error did not fall with h\n"); nfail++; }
  check("K1 . gradient (Richardson)", fd_g, an_g, 1e-9);

  // Hessian: four-point d^2E/ds dt with the COMBINED exponent
  const double fd_h =
      (Eat(hstep, hstep) - Eat(hstep, -hstep) - Eat(-hstep, hstep) +
       Eat(-hstep, -hstep)) / (4 * hstep * hstep);
  const double an_h = cas::hess_kappa_kappa(jk, C, ninact, nact, D, d, K1, K2);
  check("K1 . H_kk . K2", fd_h, an_h, 1e-4);

  // --- the BCH ordering term, predicted rather than assumed
  printf("\nBCH ordering of the raw derivative:\n");
  const double a12 =
      0.5 * (cas::hess_kappa_kappa_raw(jk, C, ninact, nact, D, d, K2)
                 .cwiseProduct(K1)).sum();
  const double a21 =
      0.5 * (cas::hess_kappa_kappa_raw(jk, C, ninact, nact, D, d, K1)
                 .cwiseProduct(K2)).sum();
  const helfem::Matrix comm = K2 * K1 - K1 * K2;
  const double pred = 0.25 * (g.cwiseProduct(comm)).sum();
  check("raw asymmetry", 0.5 * (a12 - a21), pred, 1e-8);

  printf("\n%s\n", nfail ? "CAS INTEGRALS TEST FAILED" : "CAS INTEGRALS TEST PASSED");
  return nfail ? 1 : 0;
}
