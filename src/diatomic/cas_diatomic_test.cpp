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

// The CAS integral engine on the diatomic (prolate spheroidal) basis.
//
// The algebra is shared with the atomic path and already checked against the
// Python/PySCF reference there, so what this file tests is what is specific to
// diatomic: that the adapter is wired correctly, and that the one thing which
// can silently go wrong actually does go wrong when provoked.
//
// That one thing is TwoDBasis::absm_symmetric. With it set, exchange() builds
// only the m >= 0 half of K and mirrors the rest -- exact for a density
// symmetric under m -> -m, wrong otherwise. --symmetry=3 sets it
// (diatomic/main.cpp), and it is tempting to assume a CAS can simply inherit
// such a reference. It cannot: a CAS forms PAIR densities C_t C_u^T between
// individual active orbitals, and one between the +m and -m partners of a pi
// shell carries dm = 2. Test 3 below builds exactly that density and shows the
// two K matrices differ by an amount nothing would flag at runtime, which is
// why DiatomicCASEngine refuses the flag outright.
//
// Note the scope of that: it is a statement about EXCHANGE, not about
// --symmetry. coulomb() never consults the flag, and the AO basis is identical
// for every --symmetry value, so the CAS integrals themselves are
// symmetry-agnostic.

#include "cas_engine.h"
#include "../general/cas_integrals.h"
#include "../../libhelfem/include/PolynomialBasis.h"

#include <Eigen/Eigenvalues>
#include <complex>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

  int nfail = 0;

  void report(const char * what, bool ok, const char * detail = "") {
    if (!ok) nfail++;
    printf("  %-44s %s %s\n", what, ok ? "ok" : "FAIL", detail);
  }

  void close(const char * what, double got, double ref, double tol) {
    const double err = std::abs(got - ref);
    const bool ok = err < tol;
    if (!ok) nfail++;
    printf("  %-44s %16.10f vs %16.10f  err %8.2e  %s\n",
           what, got, ref, err, ok ? "ok" : "FAIL");
  }

  /// exp(A) for a real antisymmetric A, as in trustregion_scf.cpp: i*A is
  /// Hermitian, so diagonalising exponentiates the eigenvalues exactly.
  helfem::Matrix expm_skew(const helfem::Matrix & A) {
    const std::complex<double> im(0.0, 1.0);
    Eigen::MatrixXcd H = im * A.cast<std::complex<double>>();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(H);
    Eigen::VectorXcd ph =
        (-im * es.eigenvalues().cast<std::complex<double>>()).array().exp();
    return (es.eigenvectors() * ph.asDiagonal() * es.eigenvectors().adjoint())
        .real();
  }

} // namespace

int main() {
  using namespace helfem;

  // H2 at the equilibrium distance, small enough to stay quick.
  const int Z1 = 1, Z2 = 1, lmax = 2, mmax = 1, nnodes = 6, nelem = 3;
  const double Rhalf = 0.7, Rmax = 20.0;

  std::shared_ptr<const polynomial_basis::PolynomialBasis> poly(
      polynomial_basis::make_basis(4, nnodes));
  // lm_to_l_m takes the per-|m| lmax vector, so entry k is lmax for |m| = k.
  Eigen::VectorXi lmmax(mmax + 1);
  for (int k = 0; k <= mmax; k++) lmmax(k) = lmax;
  Eigen::VectorXi lval, mval;
  diatomic::basis::lm_to_l_m(lmmax, lval, mval);
  const helfem::Vector bval =
      helfem::Vector::LinSpaced(nelem + 1, 0.0, std::acosh(Rmax / Rhalf));
  diatomic::basis::TwoDBasis basis(Z1, Z2, Rhalf, poly, 5 * poly->nbf(),
                                   bval, lval, mval);
  basis.compute_tei(true);

  const Eigen::Index nbf = (Eigen::Index) basis.Nbf();
  printf("H2  Nbf=%i  lmax=%i mmax=%i\n\n", (int) nbf, lmax, mmax);

  printf("1  the adapter refuses an absm_symmetric basis\n");
  {
    basis.set_absm_symmetric(true);
    bool threw = false;
    try {
      diatomic::basis::DiatomicCASEngine reject(basis);
    } catch (const std::logic_error &) {
      threw = true;
    }
    report("constructing on absm_symmetric basis throws", threw);
    basis.set_absm_symmetric(false);
  }

  diatomic::basis::DiatomicCASEngine jk(basis);

  printf("\n2  the engine reproduces the shared algebra\n");
  const helfem::Matrix S = basis.overlap();
  Eigen::GeneralizedSelfAdjointEigenSolver<helfem::Matrix> es(jk.hcore(), S);
  helfem::Matrix C = es.eigenvectors();
  for (Eigen::Index j = 0; j < C.cols(); j++) {
    Eigen::Index imax = 0;
    C.col(j).cwiseAbs().maxCoeff(&imax);
    if (C(imax, j) < 0.0) C.col(j) *= -1.0;
  }
  const Eigen::Index ninact = 0, nact = 3;
  const cas::ActiveHamiltonian ah =
      cas::active_hamiltonian(jk, C, ninact, nact);
  // With no inactive orbitals the core energy must vanish identically and
  // h_eff must be the bare one-electron Hamiltonian in the active space.
  report("E_inactive == 0 at ninact=0",
         std::abs(ah.E_inactive) < 1e-12);
  const helfem::Matrix Ca = C.middleCols(ninact, nact);
  const double dh = (ah.h_eff - Ca.transpose() * jk.hcore() * Ca)
                        .cwiseAbs().maxCoeff();
  report("h_eff == C^T h C at ninact=0", dh < 1e-12);
  // Which permutational symmetries (tu|vw) carries depends on whether the
  // orbitals are real. These two hold in ANY basis:
  //     (ab|cd) = (cd|ab)   electron swap
  //     (ab|cd) = (ba|dc)   conjugation, the tensor being real
  // The remaining halves of the familiar 8-fold set -- (ab|cd) = (ba|cd) and
  // (ab|cd) = (ab|dc) -- are REAL-orbital symmetries and must NOT hold here,
  // where the pi shells carry m = +-1. They held in an earlier version of this
  // engine only because active_eri symmetrised the pair density, which is the
  // bug this file now guards against: asserting them would re-impose it.
  const size_t n = (size_t) nact;
  double perm = 0.0, realonly = 0.0;
  for (size_t t = 0; t < n; t++)
    for (size_t u = 0; u < n; u++)
      for (size_t v = 0; v < n; v++)
        for (size_t w = 0; w < n; w++) {
          const double x = ah.eri[((t * n + u) * n + v) * n + w];
          perm = std::max(perm, std::abs(x - ah.eri[((v * n + w) * n + t) * n + u]));
          perm = std::max(perm, std::abs(x - ah.eri[((u * n + t) * n + w) * n + v]));
          realonly = std::max(realonly,
                              std::abs(x - ah.eri[((u * n + t) * n + v) * n + w]));
        }
  printf("  %-44s %8.2e %s\n", "(ab|cd)=(cd|ab) and (ab|cd)=(ba|dc)", perm,
         perm < 1e-10 ? "ok" : "FAIL");
  if (perm >= 1e-10) nfail++;
  printf("  %-44s %8.2e %s\n", "(ab|cd)=(ba|cd) is ABSENT, as it must be",
         realonly, realonly > 1e-10 ? "ok" : "FAIL");
  if (realonly <= 1e-10) nfail++;

  printf("\n3  the extracted tensor rebuilds J and K\n");
  {
    // The structural checks above would pass on a tensor that is
    // self-consistently wrong. This one does not: it rebuilds J and K from
    // (tu|vw) and compares against coulomb()/exchange() called directly, which
    // share no code with active_eri's pair-density extraction.
    //
    // It works on a SUBSET of orbitals because coulomb and exchange are linear
    // and the density is confined to the active space, so the active-block
    // identity is exact for any nact -- no nbf^4 tensor needed.
    const Eigen::Index na = 6;
    const helfem::Matrix Ca = C.middleCols(0, na);
    const std::vector<double> eri = cas::active_eri(jk, Ca);

    // A symmetric active-space density, deterministic so a failure reproduces.
    helfem::Matrix Pmo(na, na);
    for(Eigen::Index t = 0; t < na; t++)
      for(Eigen::Index u = 0; u <= t; u++) {
        const double v = std::sin(0.7 * (double) (t + 1) + 0.3 * (double) (u + 1));
        Pmo(t, u) = v;
        Pmo(u, t) = v;
      }
    const helfem::Matrix Pao = Ca * Pmo * Ca.transpose();

    const size_t nn = (size_t) na;
    helfem::Matrix Jmo = helfem::Matrix::Zero(na, na);
    helfem::Matrix Kmo = helfem::Matrix::Zero(na, na);
    for(Eigen::Index t = 0; t < na; t++)
      for(Eigen::Index u = 0; u < na; u++)
        for(Eigen::Index v = 0; v < na; v++)
          for(Eigen::Index w = 0; w < na; w++) {
            // J_tu = sum_vw (tu|vw) P_vw ; K_tu = sum_vw (tv|uw) P_vw
            Jmo(t, u) += eri[(((size_t) t * nn + (size_t) u) * nn + (size_t) v) * nn
                             + (size_t) w] * Pmo(v, w);
            // (tw|vu) -- see the note below on why this ordering.
            Kmo(t, u) += eri[(((size_t) t * nn + (size_t) w) * nn + (size_t) v) * nn
                             + (size_t) u] * Pmo(v, w);
          }

    const helfem::Matrix Jref = Ca.transpose() * jk.coulomb(Pao) * Ca;
    const double dJ = (Jmo - Jref).cwiseAbs().maxCoeff();
    printf("  %-44s %8.2e %s\n", "J from (tu|vw) vs coulomb()", dJ,
           dJ < 1e-10 ? "ok" : "FAIL");
    if(dJ >= 1e-10) nfail++;

    // exchange() returns the SIGNED contribution to a spin channel's Fock
    // matrix, so the positive K the tensor builds is -exchange(P) -- the same
    // convention cas::closed_shell_veff reconciles, checked here directly.
    const helfem::Matrix Kref =
        -(Ca.transpose() * jk.exchange(Pao) * Ca);
    // K, with the ordering complex orbitals actually require:
    //     K_tu = sum_vw (tw|vu) P_vw,
    // NOT the familiar (tv|uw), which is a REAL-orbital form. The two coincide
    // at m = 0 and diverge as soon as any orbital carries m != 0 -- measured
    // 2e-1 on an atomic lmax=1 basis. This is the check that pins the
    // antisymmetric channel: it fails by ~5e-2 here if active_eri symmetrises
    // the pair density, while J stays exact at 1e-16 either way.
    const double dK = (Kmo - Kref).cwiseAbs().maxCoeff();
    printf("  %-44s %8.2e %s\n", "K from (tw|vu) vs -exchange()", dK,
           dK < 1e-10 ? "ok" : "FAIL");
    if(dK >= 1e-10) nfail++;
  }

  printf("\n4  why the flag is refused: a pi+/pi- pair density\n");
  {
    // Find one +m and one -m basis function of the same |m| and build the pair
    // density between them. This is the shape a CAS active-orbital pair takes,
    // and it is NOT symmetric under m -> -m.
    Eigen::Index ip = -1, im = -1;
    const Eigen::VectorXi mv = basis.mval();
    for (Eigen::Index i = 0; i < mv.size() && (ip < 0 || im < 0); i++) {
      if (ip < 0 && mv(i) == 1) ip = i;
      if (im < 0 && mv(i) == -1) im = i;
    }
    report("found a +1 and a -1 angular shell", ip >= 0 && im >= 0);
    if (ip >= 0 && im >= 0) {
      // Index of a shell's first RETAINED radial function. The blocks are not
      // uniform: an m != 0 shell drops its first radial function, which must
      // vanish on the axis, so Nbf is not Nang*Nrad and a fixed stride lands
      // in the wrong shell (Nbf = 101, not 7*15 = 105, for this basis).
      const Eigen::Index Nrad = (Eigen::Index) basis.Nrad();
      auto shell_offset = [&](Eigen::Index shell) {
        Eigen::Index off = 0;
        for (Eigen::Index i = 0; i < shell; i++)
          off += Nrad - (mv(i) != 0 ? 1 : 0);
        return off;
      };
      const Eigen::Index bp = shell_offset(ip), bm = shell_offset(im);

      helfem::Matrix P = helfem::Matrix::Zero(nbf, nbf);
      P(bp, bm) = 1.0;
      P(bm, bp) = 1.0;                    // symmetric matrix, dm = 2 character

      basis.set_absm_symmetric(false);
      const helfem::Matrix K_general = basis.exchange(P);
      basis.set_absm_symmetric(true);
      const helfem::Matrix K_mirrored = basis.exchange(P);
      basis.set_absm_symmetric(false);

      const double diff = (K_general - K_mirrored).cwiseAbs().maxCoeff();
      const double scale = K_general.cwiseAbs().maxCoeff();
      printf("  |K_general - K_mirrored|max = %.3e   (|K|max = %.3e)\n",
             diff, scale);
      report("the mirrored shortcut IS wrong for this density",
             diff > 1e-10 * std::max(scale, 1e-30),
             "-- so refusing the flag is necessary, not defensive");

      // And the converse: on a genuinely m-symmetric density the shortcut is
      // exact, which is why --symmetry=3 is sound for the SCF it was built for.
      helfem::Matrix Q = helfem::Matrix::Zero(nbf, nbf);
      Q(bp, bp) = 1.0;
      Q(bm, bm) = 1.0;
      basis.set_absm_symmetric(false);
      const helfem::Matrix Kq_general = basis.exchange(Q);
      basis.set_absm_symmetric(true);
      const helfem::Matrix Kq_mirror = basis.exchange(Q);
      basis.set_absm_symmetric(false);
      const double dq = (Kq_general - Kq_mirror).cwiseAbs().maxCoeff();
      printf("  on an m-symmetric density instead:  %.3e\n", dq);
      report("the shortcut is exact when the density IS m-symmetric",
             dq < 1e-10 * std::max(Kq_general.cwiseAbs().maxCoeff(), 1e-30));
    }
  }

  printf("\n5  orbital gradient against finite differences, in a COMPLEX basis\n");
  {
    // The atomic gradient/Hessian checks run at lmax = 0, where the orbitals
    // are real and index ordering, conjugation and symmetrisation all
    // coincide -- so they cannot see a real-orbital assumption. Here the pi
    // shells carry m = +-1, which is the point.
    //
    // The RDM pair is fixed, so no CI solver is needed, and
    //     d[p][q][r][s] = D(p,q) D(r,s) - D(p,s) D(r,q) / 2
    // is deliberately NOT symmetric in (r, s): that asymmetry is exactly what
    // a symmetrised contraction argument would throw away.
    const Eigen::Index nact2 = 3, ninact2 = 1;
    helfem::Matrix D(nact2, nact2);
    D << 1.80, 0.05, 0.01,
         0.05, 0.15, 0.02,
         0.01, 0.02, 0.05;
    const size_t na2 = (size_t) nact2;
    std::vector<double> dr(na2*na2*na2*na2);
    for (Eigen::Index p = 0; p < nact2; p++)
      for (Eigen::Index q = 0; q < nact2; q++)
        for (Eigen::Index r = 0; r < nact2; r++)
          for (Eigen::Index t2 = 0; t2 < nact2; t2++)
            dr[(((size_t)p*na2 + (size_t)q)*na2 + (size_t)r)*na2 + (size_t)t2] =
                D(p,q)*D(r,t2) - 0.5*D(p,t2)*D(r,q);

    const helfem::Matrix g =
        cas::orbital_gradient(jk, C, ninact2, nact2, D, dr);

    helfem::Matrix K1 = helfem::Matrix::Zero(nbf, nbf);
    for (Eigen::Index a = 0; a < nbf; a++)
      for (Eigen::Index b2 = a + 1; b2 < nbf; b2++) {
        const double v = std::sin(0.4 + 2.0*(double)a - 1.3*(double)b2);
        K1(a, b2) = v; K1(b2, a) = -v;
      }
    K1 /= K1.norm();

    const double hstep = 2e-4;
    auto Eat = [&](double x) {
      return cas::frozen_ci_energy(jk, C * expm_skew(x * K1),
                                   ninact2, nact2, D, dr);
    };
    const double fd1 = (Eat(hstep) - Eat(-hstep)) / (2*hstep);
    const double fd2 = (Eat(0.5*hstep) - Eat(-0.5*hstep)) / hstep;
    const double fd  = fd2 + (fd2 - fd1)/3.0;            // Richardson
    const double an  = 0.5 * (g.cwiseProduct(K1)).sum();
    const double e1 = std::abs(fd1 - an), e2 = std::abs(fd2 - an);
    printf("  central-diff error   h: %8.2e   h/2: %8.2e   ratio %5.2f (expect ~4)\n",
           e1, e2, e2 > 0.0 ? e1/e2 : 0.0);
    report("error falls with h (so it is truncation)", e2 < e1);
    const double err = std::abs(fd - an);
    printf("  %-44s %8.2e %s\n", "K . gradient, Richardson vs analytic", err,
           err < 1e-8 ? "ok" : "FAIL");
    if (err >= 1e-8) nfail++;
  }

  printf("\n%s\n", nfail ? "DIATOMIC CAS TEST FAILED" : "DIATOMIC CAS TEST PASSED");
  return nfail ? 1 : 0;
}
