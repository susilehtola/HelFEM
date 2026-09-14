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
  // (tu|vw) must carry the 8-fold permutational symmetry.
  const size_t n = (size_t) nact;
  double perm = 0.0;
  for (size_t t = 0; t < n; t++)
    for (size_t u = 0; u < n; u++)
      for (size_t v = 0; v < n; v++)
        for (size_t w = 0; w < n; w++) {
          const double x = ah.eri[((t * n + u) * n + v) * n + w];
          perm = std::max(perm, std::abs(x - ah.eri[((u * n + t) * n + v) * n + w]));
          perm = std::max(perm, std::abs(x - ah.eri[((v * n + w) * n + t) * n + u]));
        }
  printf("  %-44s %8.2e %s\n", "8-fold permutational symmetry of eri", perm,
         perm < 1e-10 ? "ok" : "FAIL");
  if (perm >= 1e-10) nfail++;

  printf("\n3  why the flag is refused: a pi+/pi- pair density\n");
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

  printf("\n%s\n", nfail ? "DIATOMIC CAS TEST FAILED" : "DIATOMIC CAS TEST PASSED");
  return nfail ? 1 : 0;
}
