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

// CASCI through reci, checked against references that share no code with it.
//
// Every check below either reproduces a limit computed without reci, or ties
// reci's output back to cas_integrals.cpp's own energy expression. The last
// kind is the one that guards the convention swap in cas_ci.cpp: the
// integrals go IN to reci in physicist order and the RDMs come OUT in it, so
// a wrong swap on either side gives an eigenvalue that is right but RDMs
// that do not reproduce it -- or the reverse.
//
//   1. single-determinant limit: CAS(2,1) over the 2s orbital equals the
//      closed-shell energy of 1s^2 2s^2 at the same orbitals;
//   2. full CI of two electrons: Be with a frozen 1s, both remaining
//      electrons over EVERY remaining orbital, against a direct
//      diagonalization in the two-electron product basis. Singlet and
//      triplet separately, so the spin adaptation is checked as well;
//   3. RDMs: trace = electron count, and frozen_ci_energy(D, d) reproduces
//      the eigenvalue;
//   4. transition RDMs: <L|H|R> from them equals L . sigma(R) for two
//      unrelated vectors;
//   5. the real-orbital guard refuses a tensor with (tu|vw) != (ut|vw).
//
// The orbitals are the core guess: none of these identities needs SCF
// orbitals, and the core guess is deterministic.

#include "cas_ci.h"
#include "cas_integrals.h"
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
  }

  /// Lowest singlet and triplet of two electrons over n orbitals, by direct
  /// diagonalization in the product basis |p q>, n^2-dimensional:
  ///   H[(pq),(rs)] = E0 d_pr d_qs + h_pr d_qs + d_pr h_qs + (pr|qs)
  /// A spatial function symmetric under exchange pairs with the singlet,
  /// an antisymmetric one with the triplet; each is projected out by
  /// shifting the other far up.
  void two_electron_fci(const helfem::cas::ActiveHamiltonian & ah,
                        double & Esinglet, double & Etriplet) {
    const Eigen::Index n = ah.h_eff.rows();
    const Eigen::Index N = n * n;
    auto eri = [&](Eigen::Index t, Eigen::Index u, Eigen::Index v, Eigen::Index w) {
      return ah.eri[(size_t) (((t * n + u) * n + v) * n + w)];
    };
    helfem::Matrix H = helfem::Matrix::Zero(N, N), Pswap = helfem::Matrix::Zero(N, N);
    for (Eigen::Index p = 0; p < n; p++)
      for (Eigen::Index q = 0; q < n; q++) {
        Pswap(q * n + p, p * n + q) = 1.0;
        for (Eigen::Index r = 0; r < n; r++)
          for (Eigen::Index s = 0; s < n; s++) {
            double v = eri(p, r, q, s);
            if (q == s) v += ah.h_eff(p, r);
            if (p == r) v += ah.h_eff(q, s);
            if (p == r && q == s) v += ah.E_inactive;
            H(p * n + q, r * n + s) = v;
          }
      }
    const helfem::Matrix I = helfem::Matrix::Identity(N, N);
    const double shift = 1e4;
    Eigen::SelfAdjointEigenSolver<helfem::Matrix> s(H + shift * 0.5 * (I - Pswap));
    Eigen::SelfAdjointEigenSolver<helfem::Matrix> t(H + shift * 0.5 * (I + Pswap));
    Esinglet = s.eigenvalues()(0);
    Etriplet = t.eigenvalues()(0);
  }

  double dot(const std::vector<double> & a, const std::vector<double> & b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); i++) s += a[i] * b[i];
    return s;
  }

} // namespace

int main() {
  using namespace helfem;
  cas::KokkosScope kokkos;

  // --- Be at lmax = 0: every orbital is real, as reci requires
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

  Eigen::GeneralizedSelfAdjointEigenSolver<helfem::Matrix> es(jk.hcore(), basis.overlap());
  const helfem::Matrix C = es.eigenvectors();
  printf("Be lmax=0, Nbf=%i, core-guess orbitals\n", (int) nbf);

  printf("\n1  single-determinant limit\n");
  {
    const auto closed = cas::active_hamiltonian(jk, C, 2, 1);
    cas::CASCI ci(1, 1, 1);
    ci.set_hamiltonian(cas::active_hamiltonian(jk, C, 1, 1));
    std::vector<double> c;
    const double E = ci.solve_ground(c);
    check("CAS(2,1) over 2s == closed-shell 1s2 2s2", E, closed.E_inactive, 1e-10);
  }

  printf("\n2  full CI of two electrons over all %i remaining orbitals\n", (int) nbf - 1);
  const Eigen::Index nact = nbf - 1;
  const auto ah = cas::active_hamiltonian(jk, C, 1, nact);
  double Es_ref, Et_ref;
  two_electron_fci(ah, Es_ref, Et_ref);

  cas::CASCI singlet((int) nact, 1, 1);
  singlet.set_hamiltonian(ah);
  std::vector<double> c0;
  const double Es = singlet.solve_ground(c0);
  printf("  singlet: %zu determinants, %zu CSFs\n", singlet.ndet(), singlet.ncsf());
  check("singlet energy vs product-basis FCI", Es, Es_ref, 1e-10);

  cas::CASCI triplet((int) nact, 2, 0);
  triplet.set_hamiltonian(ah);
  std::vector<double> ct;
  const double Et = triplet.solve_ground(ct);
  check("triplet energy vs product-basis FCI", Et, Et_ref, 1e-10);

  printf("\n3  state RDMs\n");
  {
    check("|c| = 1", std::sqrt(dot(c0, c0)), 1.0, 1e-12);
    check("c . H c == eigenvalue", dot(c0, singlet.sigma(c0)), Es, 1e-10);
    const cas::RDMs r = singlet.rdms(c0, c0);
    check("trace D == 2 active electrons", r.D.trace(), 2.0, 1e-12);
    check("frozen_ci_energy(D, d) == eigenvalue",
          cas::frozen_ci_energy(jk, C, 1, nact, r.D, r.d), Es, 1e-10);
    const cas::RDMs rt = triplet.rdms(ct, ct);
    check("triplet: frozen_ci_energy(D, d) == eigenvalue",
          cas::frozen_ci_energy(jk, C, 1, nact, rt.D, rt.d), Et, 1e-10);
  }

  printf("\n4  transition RDMs\n");
  {
    // Two unrelated, unnormalized vectors. <L|H|R> = E_inactive <L|R>
    //   + sum D_tu h_tu + 1/2 sum d_tuvw (tu|vw), with D, d the TRANSITION
    // RDMs -- linear in each of L and R, so nothing assumes normalization.
    const size_t nd = singlet.ndet();
    std::vector<double> L(nd), R(nd);
    for (size_t i = 0; i < nd; i++) {
      L[i] = std::sin(1.0 + 0.7 * i);
      R[i] = std::cos(0.3 + 1.3 * i);
    }
    // sigma is the determinant-basis H, which contains every spin; L and R
    // have components of all spins, and the identity holds for each.
    const cas::RDMs r = singlet.rdms(L, R);
    double E = ah.E_inactive * dot(L, R) + ah.h_eff.cwiseProduct(r.D).sum();
    for (size_t i = 0; i < r.d.size(); i++) E += 0.5 * ah.eri[i] * r.d[i];
    check("<L|H|R> from transition RDMs == L . sigma(R)", E, dot(L, singlet.sigma(R)), 1e-9);
  }

  printf("\n5  conventions and guard\n");
  {
    const size_t n = 3;
    std::vector<double> X(n * n * n * n);
    for (size_t i = 0; i < X.size(); i++) X[i] = std::sin(0.1 + i);
    const auto Y = cas::swap_middle(cas::swap_middle(X, n), n);
    double d = 0.0;
    for (size_t i = 0; i < X.size(); i++) d = std::max(d, std::abs(X[i] - Y[i]));
    check("swap_middle is an involution", d, 0.0, 0.0);

    bool refused = false;
    try {
      cas::require_real_eri(X, n);   // a generic tensor has no (tu|vw)=(ut|vw)
    } catch (const cas::ComplexOrbitalError &) {
      refused = true;
    }
    check("a tensor without (tu|vw) = (ut|vw) is refused", refused ? 1.0 : 0.0, 1.0, 0.0);

    bool accepted = true;
    try {
      cas::require_real_eri(ah.eri, (size_t) nact);
    } catch (const cas::ComplexOrbitalError &) {
      accepted = false;
    }
    check("the real lmax=0 tensor is accepted", accepted ? 1.0 : 0.0, 1.0, 0.0);
  }

  printf("\n%s\n", nfail ? "CASCI TEST FAILED" : "CASCI TEST PASSED");
  return nfail ? 1 : 0;
}
