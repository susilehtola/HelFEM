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
// Two-step CASSCF on OpenTrustRegion, checked against references that do not
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
//      returns to it.

#include "cas_ci.h"
#include "cas_integrals.h"
#include "cas_optimizer.h"
#include "trustregion_scf.h"
#include "../atomic/cas_engine.h"
#include "../atomic/basis.h"
#include <helfem/PolynomialBasis.h>
#include <helfem/ModelPotential.h>

#include <Eigen/Eigenvalues>
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
    const std::vector<cas::RotationPair> pairs = cas::rotation_pairs(Ccore.cols(), 1, 2, {});
    auto K = [&](const std::vector<double> & v) {
      helfem::Matrix M = helfem::Matrix::Zero(Ccore.cols(), Ccore.cols());
      for (size_t k = 0; k < n; k++) {
        M(pairs[k].p, pairs[k].q) = v[k];
        M(pairs[k].q, pairs[k].p) = -v[k];
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
    const std::vector<cas::RotationPair> pairs = cas::rotation_pairs(Ccore.cols(), 1, 2, {});
    helfem::Matrix H(n, n);
    const double h = 1e-4;
    for (size_t k = 0; k < n; k++) {
      helfem::Matrix K = helfem::Matrix::Zero(Ccore.cols(), Ccore.cols());
      K(pairs[k].p, pairs[k].q) = h;
      K(pairs[k].q, pairs[k].p) = -h;
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
      K(pairs[k].p, pairs[k].q) = 0.05 * x[k];
      K(pairs[k].q, pairs[k].p) = -0.05 * x[k];
    }
    cas::CASSCF b(jk, S, ci, a.orbitals() * trscf::expm_skew(K), 1, 2);
    const cas::CASSCFResult rb = b.run(opt);
    check("a perturbed restart returns", rb.energy, ra.energy, 1e-9);
  }

  printf("\n%s\n", nfail ? "CASSCF TEST FAILED" : "CASSCF TEST PASSED");
  return nfail ? 1 : 0;
}
