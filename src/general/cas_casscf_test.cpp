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
// CASSCF on OpenTrustRegion, checked against references that do not
// go through the optimizer.
//
//   1. the Hessian-vector product in parameter space against
//      cas::hess_kappa_kappa on the full antisymmetric matrices, and its
//      symmetry;
//   2. the gradient against a central difference of the CAS energy with the
//      CI re-solved at every point -- the energy the optimizer minimizes;
//   3. CASSCF(2,1) over a frozen 1s is a closed-shell determinant, so it must
//      reach the RHF energy of a separate Roothaan iteration;
//   4. CASSCF(2,2) ends at a minimum of the CI-relaxed energy -- checked on
//      a finite-difference Hessian -- below RHF, and a perturbed restart
//      returns to it;
//   5. the one-index-transformed active Hamiltonian against central
//      differences of active_hamiltonian;
//   6. the coupled (kappa, c) gradient and Hessian against differences of the
//      energy the coupled optimizer minimizes;
//   7. coupled and two-step CASSCF(2,2) reach the same energy from the same
//      start;
//   8. equivalent blocks (the 2p of Be at lmax=1) are found, get identical
//      orbitals and tied rotations, the tied derivatives agree with the
//      energy, and the copies stay identical through the optimization.

#include "cas_ci.h"
#include "cas_integrals.h"
#include "cas_optimizer.h"
#include "scf_helpers.h"
#include "trustregion_scf.h"
#include "../atomic/cas_engine.h"
#include "../atomic/basis.h"
#include <helfem/PolynomialBasis.h>
#include <helfem/ModelPotential.h>

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

  int nfail = 0;

  void check(const char * what, double got, double ref, double tol) {
    const double diff = std::abs(got - ref);
    const bool ok = diff <= tol;
    if (!ok) nfail++;
    printf("  %-52s %s  got % .12f  ref % .12f  diff %.1e\n", what,
           ok ? "ok  " : "FAIL", got, ref, diff);
    fflush(stdout);
  }

  double dot(const std::vector<double> & a, const std::vector<double> & b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); i++) s += a[i] * b[i];
    return s;
  }

  std::vector<double> direction(size_t n, double seed) {
    std::vector<double> x(n);
    double nrm = 0.0;
    for (size_t i = 0; i < n; i++) {
      x[i] = std::sin(seed + 1.7 * (double) i);
      nrm += x[i] * x[i];
    }
    for (double & v : x) v /= std::sqrt(nrm);
    return x;
  }

} // namespace

int main() {
  using namespace helfem;
  cas::KokkosScope kokkos;

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
  const helfem::Matrix S = basis.overlap();

  Eigen::GeneralizedSelfAdjointEigenSolver<helfem::Matrix> es(jk.hcore(), S);
  const helfem::Matrix Ccore = es.eigenvectors();
  printf("Be lmax=0, Nbf=%i, core-guess orbitals\n", (int) basis.Nbf());

  // RHF by plain Roothaan iteration, sharing nothing with the optimizer.
  helfem::Matrix Chf = Ccore;
  double Ehf = 0.0;
  for (int it = 0; it < 500; it++) {
    double E;
    const helfem::Matrix F = cas::inactive_fock(jk, Chf, 2, E);
    Eigen::GeneralizedSelfAdjointEigenSolver<helfem::Matrix> f(F, S);
    Chf = f.eigenvectors();
    if (std::abs(E - Ehf) < 1e-13) break;
    Ehf = E;
  }
  printf("RHF by Roothaan iteration: %.12f\n", Ehf);

  cas::CASSCFOptions opt;
  // An energy of 15 Eh resolves gradients down to sqrt(eps |E|) ~ 5e-8 and
  // no further: below that a trust-region step cannot be validated.
  opt.otr.conv_tol = 1e-7;

  printf("\n1  Hessian-vector product, CAS(2,2) at the core guess\n");
  {
    cas::CASCI ci(2, 1, 1);
    cas::CASSCF cs(jk, S, ci, Ccore, 1, 2);
    const size_t n = cs.nparam();
    const std::vector<double> x = direction(n, 0.3), y = direction(n, 2.1);
    const std::vector<double> Hx = cs.hessian_vector(x), Hy = cs.hessian_vector(y);
    const std::vector<cas::Rotation> pairs = cas::rotation_pairs(Ccore.cols(), 1, 2, {});
    auto K = [&](const std::vector<double> & v) {
      helfem::Matrix M = helfem::Matrix::Zero(Ccore.cols(), Ccore.cols());
      for (size_t k = 0; k < n; k++) {
        M(pairs[k][0].p, pairs[k][0].q) = v[k];
        M(pairs[k][0].q, pairs[k][0].p) = -v[k];
      }
      return M;
    };
    cas::CASCI ci0(2, 1, 1);
    ci0.set_hamiltonian(cas::active_hamiltonian(jk, Ccore, 1, 2));
    std::vector<double> c;
    ci0.solve_ground(c);
    const cas::RDMs r = ci0.rdms(c, c);
    const double ref = cas::hess_kappa_kappa(jk, Ccore, 1, 2, r.D, r.d, K(y), K(x));
    printf("  %zu parameters\n", n);
    check("y . H x against hess_kappa_kappa", dot(y, Hx), ref, 1e-10 * std::abs(ref));
    check("y . H x = x . H y", dot(y, Hx), dot(x, Hy), 1e-10 * std::abs(ref));

    printf("\n2  gradient against the CI-relaxed energy\n");
    const std::vector<double> g = cs.gradient();
    auto Eat = [&](double t) {
      std::vector<double> v(x);
      for (double & e : v) e *= t;
      return cs.energy(v);
    };
    const double h = 1e-3;
    const double d1 = (Eat(h) - Eat(-h)) / (2 * h);
    const double d2 = (Eat(0.5 * h) - Eat(-0.5 * h)) / h;
    check("x . g (Richardson)", d2 + (d2 - d1) / 3.0, dot(x, g), 1e-9);
  }

  printf("\n3  CASSCF(2,1) is RHF\n");
  {
    cas::CASCI ci(1, 1, 1);
    cas::CASSCF cs(jk, S, ci, Ccore, 1, 1);
    const cas::CASSCFResult res = cs.run(opt);
    check("energy against Roothaan RHF", res.energy, Ehf, 1e-9);
  }

  printf("\n4  CASSCF(2,2): a minimum, and a perturbed restart returns to it\n");
  {
    // Be has more than one CAS(2,2) minimum over s orbitals -- the second
    // active orbital can correlate the 2s or stay a diffuse 3s -- so two
    // unrelated starts need not agree, and here they do not (by 11 mEh,
    // both ends positive definite). The checks are therefore that the
    // endpoint IS a minimum of the energy the optimizer minimizes, and that
    // a perturbation of it comes back.
    cas::CASCI ci(2, 1, 1);
    cas::CASSCF a(jk, S, ci, Ccore, 1, 2);
    const cas::CASSCFResult ra = a.run(opt);
    check("RMS gradient at the end", ra.grad_rms, 0.0, opt.otr.conv_tol);
    check("below RHF", ra.energy < Ehf - 1e-6 ? 1.0 : 0.0, 1.0, 0.0);

    // CI-relaxed Hessian by central differences of the gradient. Each
    // gradient is in its own rotated frame, which differs from the
    // reference one by O(h g) -- nothing at a stationary point.
    const size_t n = a.nparam();
    const std::vector<cas::Rotation> pairs = cas::rotation_pairs(Ccore.cols(), 1, 2, {});
    helfem::Matrix H(n, n);
    const double h = 1e-4;
    for (size_t k = 0; k < n; k++) {
      helfem::Matrix K = helfem::Matrix::Zero(Ccore.cols(), Ccore.cols());
      K(pairs[k][0].p, pairs[k][0].q) = h;
      K(pairs[k][0].q, pairs[k][0].p) = -h;
      cas::CASSCF up(jk, S, ci, a.orbitals() * trscf::expm_skew(K), 1, 2);
      cas::CASSCF dn(jk, S, ci, a.orbitals() * trscf::expm_skew(-K), 1, 2);
      const std::vector<double> gu = up.gradient(), gd = dn.gradient();
      for (size_t l = 0; l < n; l++) H(l, k) = (gu[l] - gd[l]) / (2 * h);
    }
    Eigen::SelfAdjointEigenSolver<helfem::Matrix> eh(0.5 * (H + H.transpose()));
    printf("  lowest CI-relaxed Hessian eigenvalue %.4e\n", eh.eigenvalues()(0));
    check("the endpoint is a minimum", eh.eigenvalues()(0) > 1e-3 ? 1.0 : 0.0, 1.0, 0.0);

    const std::vector<double> x = direction(n, 0.7);
    helfem::Matrix K = helfem::Matrix::Zero(Ccore.cols(), Ccore.cols());
    for (size_t k = 0; k < n; k++) {
      K(pairs[k][0].p, pairs[k][0].q) = 0.05 * x[k];
      K(pairs[k][0].q, pairs[k][0].p) = -0.05 * x[k];
    }
    cas::CASSCF b(jk, S, ci, a.orbitals() * trscf::expm_skew(K), 1, 2);
    const cas::CASSCFResult rb = b.run(opt);
    check("a perturbed restart returns", rb.energy, ra.energy, 1e-9);
  }

  printf("\n5  one-index-transformed Hamiltonian against central differences\n");
  {
    const Eigen::Index nb = Ccore.cols();
    helfem::Matrix K = helfem::Matrix::Zero(nb, nb);
    for (Eigen::Index p = 0; p < nb; p++)
      for (Eigen::Index q = 0; q < p; q++) {
        K(p, q) = std::sin(0.5 + 1.3 * (double) p - 0.7 * (double) q);
        K(q, p) = -K(p, q);
      }
    K /= K.norm();
    const double h = 1e-4;
    const cas::ActiveHamiltonian an = cas::active_hamiltonian_response(jk, Ccore, 1, 3, K);
    const cas::ActiveHamiltonian up = cas::active_hamiltonian(jk, Ccore * trscf::expm_skew(h * K), 1, 3);
    const cas::ActiveHamiltonian dn = cas::active_hamiltonian(jk, Ccore * trscf::expm_skew(-h * K), 1, 3);
    double de = 0.0;
    for (size_t i = 0; i < an.eri.size(); i++)
      de = std::max(de, std::abs((up.eri[i] - dn.eri[i]) / (2 * h) - an.eri[i]));
    check("E_inactive'", an.E_inactive, (up.E_inactive - dn.E_inactive) / (2 * h), 1e-6);
    check("h_eff' (max deviation)",
          ((up.h_eff - dn.h_eff) / (2 * h) - an.h_eff).cwiseAbs().maxCoeff(), 0.0, 1e-6);
    check("eri' (max deviation)", de, 0.0, 1e-6);
  }

  printf("\n6  coupled Hessian against four-point differences of the energy\n");
  {
    cas::CASCI ci(2, 1, 1);
    cas::CASSCF cs(jk, S, ci, Ccore, 1, 2, {}, {}, cas::Coupling::Coupled);
    const size_t n = cs.nparam();
    printf("  %zu rotations, %zu CI coefficients\n", cs.nrot(), n - cs.nrot());
    const std::vector<double> x = direction(n, 0.3), y = direction(n, 2.1);
    const std::vector<double> g = cs.gradient(), Hx = cs.hessian_vector(x),
                              Hy = cs.hessian_vector(y);
    auto Eat = [&](double s1, double s2) {
      std::vector<double> v(n);
      for (size_t i = 0; i < n; i++) v[i] = s1 * x[i] + s2 * y[i];
      return cs.energy(v);
    };
    const double h = 1e-3;
    const double d1 = (Eat(h, 0) - Eat(-h, 0)) / (2 * h);
    const double d2 = (Eat(0.5 * h, 0) - Eat(-0.5 * h, 0)) / h;
    check("x . g (Richardson)", d2 + (d2 - d1) / 3.0, dot(x, g), 1e-9);
    // Four-point differences, Richardson-extrapolated: the h^2 truncation
    // is ~1e-4 here, far above what the formula has to be checked to.
    auto fxy = [&](double e) {
      return (Eat(e, e) - Eat(e, -e) - Eat(-e, e) + Eat(-e, -e)) / (4 * e * e);
    };
    auto fxx = [&](double e) { return (Eat(e, 0) - 2 * Eat(0, 0) + Eat(-e, 0)) / (e * e); };
    const double hh = 2e-3;
    const double rxy = fxy(0.5 * hh) + (fxy(0.5 * hh) - fxy(hh)) / 3.0;
    const double rxx = fxx(0.5 * hh) + (fxx(0.5 * hh) - fxx(hh)) / 3.0;
    check("y . H x (Richardson)", dot(y, Hx), rxy, 1e-7 * std::max(1.0, std::abs(rxy)));
    check("x . H x (Richardson)", dot(x, Hx), rxx, 1e-7 * std::max(1.0, std::abs(rxx)));
    check("y . H x = x . H y", dot(y, Hx), dot(x, Hy), 1e-10 * std::max(1.0, std::abs(rxy)));
  }

  printf("\n7  coupled CASSCF(2,2) against two-step, from common starts\n");
  {
    // Each start lies in the basin of one of the two minima, so both
    // optimizers must agree on it.
    cas::CASCI ci(2, 1, 1);
    cas::CASSCF ref(jk, S, ci, Ccore, 1, 2);
    ref.run(opt);
    const std::vector<cas::Rotation> pairs = cas::rotation_pairs(Ccore.cols(), 1, 2, {});
    const std::vector<double> x = direction(pairs.size(), 0.7);
    helfem::Matrix K = helfem::Matrix::Zero(Ccore.cols(), Ccore.cols());
    for (size_t k = 0; k < pairs.size(); k++) {
      K(pairs[k][0].p, pairs[k][0].q) = 0.05 * x[k];
      K(pairs[k][0].q, pairs[k][0].p) = -0.05 * x[k];
    }
    const helfem::Matrix Cpert = ref.orbitals() * trscf::expm_skew(K);
    for (const helfem::Matrix * C0 : {&Cpert, static_cast<const helfem::Matrix *>(&Chf)}) {
      const char * what = C0 == &Cpert ? "perturbed minimum" : "RHF orbitals";
      cas::CASSCF two(jk, S, ci, *C0, 1, 2);
      const cas::CASSCFResult r2 = two.run(opt);
      cas::CASSCF one(jk, S, ci, *C0, 1, 2, {}, {}, cas::Coupling::Coupled);
      const cas::CASSCFResult r1 = one.run(opt);
      printf("  from the %s: Hessian-vector products coupled %zu, two-step %zu\n",
             what, r1.n_hessian, r2.n_hessian);
      check("  same energy", r1.energy, r2.energy, 1e-9);
    }
  }

  printf("\n8  equivalent orbitals: Be at lmax=1, per-(l,m) blocks, CAS(2,4)\n");
  {
    // The three 2p blocks are equivalent, so canonical_orbitals must find
    // them, give them identical orbitals, and tie their rotations. The
    // tied gradient and Hessian are checked against the energy, and the
    // orbitals must still be copies of one another after a tied step.
    Eigen::VectorXi lval1, mval1;
    atomic::basis::angular_basis(1, 1, lval1, mval1);
    atomic::basis::TwoDBasis basis1(Z, modelpotential::POINT_NUCLEUS, 0.0, poly,
                                    false, 5 * poly->nbf(), bval, lval1, mval1,
                                    0, 0, 0.0);
    basis1.compute_tei(true);
    atomic::basis::AtomicCASEngine jk1(basis1);
    const helfem::Matrix S1 = basis1.overlap();
    const std::vector<std::vector<Eigen::Index>> dsym = basis1.sym_idx(2);
    std::vector<helfem::Matrix> Sinvh;
    for (const auto & idx : dsym) Sinvh.push_back(scf::form_Sinvh(helfem::Matrix(S1(idx, idx)), false));

    helfem::Matrix C;
    helfem::Vector e;
    std::vector<int> label, shell;
    const helfem::Matrix P0 = helfem::Matrix::Zero(S1.rows(), S1.cols());
    cas::canonical_orbitals(jk1, S1, P0, dsym, Sinvh, C, e, label, shell);
    // The lowest five orbitals are 1s, 2s and the 2p shell's three copies,
    // in some order: with P = 0 the operator is hydrogenic, and 2s and 2p
    // are degenerate.
    std::vector<int> copies;
    for (size_t i = 0; i < 5; i++) {
      int nc = 0;
      for (size_t j = 0; j < shell.size(); j++) nc += shell[j] == shell[i];
      copies.push_back(nc);
    }
    std::sort(copies.begin(), copies.end());
    check("1s, 2s and one 2p shell of three copies",
          copies == std::vector<int>{1, 1, 3, 3, 3} ? 1.0 : 0.0, 1.0, 0.0);

    cas::CASCI ci(4, 1, 1);
    cas::CASSCF cs(jk1, S1, ci, C, 1, 4, label, shell, cas::Coupling::Coupled);
    const size_t n = cs.nparam();
    const size_t nuntied = cas::rotation_pairs(C.cols(), 1, 4, label).size();
    printf("  %zu tied rotations (%zu untied), %zu CI coefficients\n",
           cs.nrot(), nuntied, n - cs.nrot());
    const std::vector<double> x = direction(n, 0.9), y = direction(n, 1.4);
    const std::vector<double> g = cs.gradient(), Hx = cs.hessian_vector(x);
    auto Eat = [&](double s1, double s2) {
      std::vector<double> v(n);
      for (size_t i = 0; i < n; i++) v[i] = s1 * x[i] + s2 * y[i];
      return cs.energy(v);
    };
    const double h = 1e-3;
    const double d1 = (Eat(h, 0) - Eat(-h, 0)) / (2 * h);
    const double d2 = (Eat(0.5 * h, 0) - Eat(-0.5 * h, 0)) / h;
    check("x . g (Richardson)", d2 + (d2 - d1) / 3.0, dot(x, g), 1e-9);
    auto fxy = [&](double t) {
      return (Eat(t, t) - Eat(t, -t) - Eat(-t, t) + Eat(-t, -t)) / (4 * t * t);
    };
    const double hh = 2e-3;
    const double rxy = fxy(0.5 * hh) + (fxy(0.5 * hh) - fxy(hh)) / 3.0;
    check("y . H x (Richardson)", dot(y, Hx), rxy, 1e-7 * std::max(1.0, std::abs(rxy)));

    const cas::CASSCFResult r = cs.run(opt);
    // Copies of a shell: identical coefficients on their own blocks.
    double dev = 0.0;
    for (size_t i = 0; i < shell.size(); i++)
      for (size_t j = i + 1; j < shell.size(); j++)
        if (shell[i] == shell[j]) {
          const helfem::Vector ci_ = cs.orbitals().col((Eigen::Index) i)(dsym[(size_t) label[i]]);
          const helfem::Vector cj_ = cs.orbitals().col((Eigen::Index) j)(dsym[(size_t) label[j]]);
          dev = std::max(dev, (ci_ - cj_).cwiseAbs().maxCoeff());
        }
    check("copies stay identical through the optimization", dev, 0.0, 1e-10);
    check("RMS gradient at the end", r.grad_rms, 0.0, opt.otr.conv_tol);
  }

  printf("\n%s\n", nfail ? "CASSCF TEST FAILED" : "CASSCF TEST PASSED");
  return nfail ? 1 : 0;
}
