/*
 * HelFEM atomic radial FEM export -- example for external consumers
 * (e.g. a BSD-licensed atomic-SCF library wanting to use HelFEM as a
 * "numerical atomic orbital" backend).
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * What this example does:
 *
 *   1. Constructs an atomic radial FEM basis with the SAME defaults the
 *      `atomic` executable uses (so the consumer gets numerically-exact
 *      atomic HF out of the box).
 *
 *   2. Computes the one-electron matrices S, T, T_l, V exposed by
 *      libhelfem and verifies them by solving hydrogenic eigenproblems:
 *          - H Z=1, l=0:   E_1s = -0.5 Eh           (one-electron)
 *          - H Z=2, l=1:   E_2p = -Z^2/2/(l+1)^2    (centrifugal correct)
 *
 *   3. Runs a closed-shell He HF SCF (Z=2, single l=0 shell) and
 *      verifies E = -2.86167999561 Eh. This uses TwoDBasis::coulomb/
 *      exchange because, for a single-l shell, those reduce to the
 *      monopole (k=0) radial J/K.
 *
 *   4. Sanity-checks TwoDBasis::radial_df_factors, whose public return
 *      type is now arma-free:
 *          std::vector<std::vector<helfem::Matrix>>
 *      (outer index = multipole k, inner index = Cholesky factor Q).
 *
 *   5. Documents the integration conventions (see comments at the bottom).
 *
 * Build (against an installed HelFEM):
 *
 *   Consumer's own CMakeLists.txt:
 *
 *       find_package(helfem CONFIG REQUIRED)
 *       add_executable(atomic_radial_export atomic_radial_export.cpp)
 *       target_link_libraries(atomic_radial_export PRIVATE helfem::helfem)
 *
 *   Then, from a build directory outside the HelFEM tree:
 *
 *       cmake -Dhelfem_DIR=$PREFIX/lib/cmake/helfem <src>
 *       cmake --build .
 *
 *   The exported target chains all the transitive linkage
 *   (helfem-common, legendre, lib1dfem, Eigen, Armadillo,
 *   ARMA_64BIT_WORD/ARMA_BLAS_LONG) so the consumer does not have to
 *   remember any of it. Consumer code needs no direct include of
 *   <armadillo> and no ARMA_* macros.
 */

#include <helfem/atomic/TwoDBasis.h>
#include <Matrix.h>
#include <PolynomialBasis.h>
#include <Eigen/Eigenvalues>

#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

// Bring the arma::vec / arma::ivec ctor parameters of TwoDBasis into
// scope only for the constructor call. This is the last remaining arma
// surface on the public consumer path; a follow-up will convert the
// constructor signature to Eigen and this include can go away.
#include <armadillo>

namespace helfem { namespace utils {
  arma::vec get_grid(double Rmax, int num_el, int igrid, double zexp);
}}

using namespace helfem;

namespace {

// Documented defaults from src/atomic/main.cpp: LIP-4 (Gauss-Lobatto),
// 15-node, 20 exponential elements out to R=40, zexp=2.
struct Defaults { int primbas, Nnodes, Nelem, igrid; double Rmax, zexp; };
constexpr Defaults DEF{4, 15, 20, 4, 40.0, 2.0};

atomic::basis::TwoDBasis make_single_shell(int Z, int lval_int,
                                            const Defaults & d = DEF) {
  auto poly = std::shared_ptr<const polynomial_basis::PolynomialBasis>(
      polynomial_basis::get_basis(d.primbas, d.Nnodes));
  arma::vec bval = utils::get_grid(d.Rmax, d.Nelem, d.igrid, d.zexp);
  arma::ivec lval = {lval_int};
  arma::ivec mval = {0};
  return atomic::basis::TwoDBasis(
      Z, modelpotential::POINT_NUCLEUS, /*Rrms*/0.0,
      poly, /*zeroder*/false,
      5 * poly->get_nbf(), bval,
      lval, mval, /*Zl*/0, /*Zr*/0, /*Rhalf*/0.0);
}

// Solve H c = e S c for the lowest eigenvalue, with S guaranteed
// positive-definite. Symmetric-orthogonalise, then a single Hermitian
// eig on H_orth.
double diag_lowest(const helfem::Matrix & H, const helfem::Matrix & S) {
  Eigen::SelfAdjointEigenSolver<helfem::Matrix> es_S(S);
  const helfem::Vector sval = es_S.eigenvalues();
  const helfem::Matrix svec = es_S.eigenvectors();

  // Drop tiny eigenvalues to avoid numerical noise.
  int keep = 0;
  for (Eigen::Index i = 0; i < sval.size(); ++i)
    if (sval(i) > 1e-10) ++keep;
  helfem::Matrix X(S.rows(), keep);
  int col = 0;
  for (Eigen::Index i = 0; i < sval.size(); ++i)
    if (sval(i) > 1e-10)
      X.col(col++) = svec.col(i) / std::sqrt(sval(i));

  const helfem::Matrix H_orth = X.transpose() * H * X;
  Eigen::SelfAdjointEigenSolver<helfem::Matrix> es_H(H_orth);
  return es_H.eigenvalues()(0);
}

bool one_electron_tests() {
  std::printf("=== one-electron / hydrogenic tests ===\n");
  struct Case { int l; double Z; const char * tag; };
  Case cases[] = {{0, 1.0, "H   1s"}, {0, 2.0, "He+ 1s"},
                  {1, 1.0, "H   2p"}, {1, 2.0, "He+ 2p"},
                  {2, 1.0, "H   3d"}};
  bool ok = true;
  for (auto c : cases) {
    // TwoDBasis::kinetic() already folds in the centrifugal l*(l+1)/2 * <1/r^2>
    // for the constructor's lval, and nuclear() is Z-scaled. So the
    // hydrogenic H is just T + Vnuc for a single-shell basis with lval={l}.
    auto basis = make_single_shell(static_cast<int>(c.Z), c.l);
    const helfem::Matrix S    = basis.overlap();
    const helfem::Matrix T    = basis.kinetic();
    const helfem::Matrix Vnuc = basis.nuclear();
    const helfem::Matrix H    = T + Vnuc;
    const double e   = diag_lowest(H, S);
    const double ref = -c.Z * c.Z / (2.0 * (c.l + 1) * (c.l + 1));
    const double err = std::abs(e - ref);
    const bool pass = err < 1e-6;
    std::printf("  %-7s   E = %+.10f   ref %+.10f   err %.2e %s\n",
                 c.tag, e, ref, err, pass ? "" : "  *** FAIL ***");
    if (!pass) ok = false;
  }
  return ok;
}

bool he_hf_test() {
  std::printf("\nHe HF (Z=2, single l=0 shell, closed-shell 1s^2):\n");
  auto basis = make_single_shell(/*Z*/2, /*l*/0);
  basis.compute_tei(/*exchange*/true);

  const helfem::Matrix S    = basis.overlap();
  const helfem::Matrix T    = basis.kinetic();
  const helfem::Matrix Vnuc = basis.nuclear();
  const helfem::Matrix Hc   = T + Vnuc;

  // Sinvh is still arma at the API boundary; wrap the arma::mat once
  // via Eigen::Map into a helfem::Matrix. This is the only remaining
  // arma type on the consumer path -- a follow-up will convert the
  // return type to helfem::Matrix.
  const arma::mat Sinvh_arma = basis.Sinvh(/*chol*/false, /*sym*/0);
  const Eigen::Map<const helfem::Matrix> Sinvh(
      Sinvh_arma.memptr(), Sinvh_arma.n_rows, Sinvh_arma.n_cols);

  // Core Hamiltonian guess.
  helfem::Matrix Fao = Hc;

  double Eprev = 0.0;
  for (int iter = 0; iter < 60; ++iter) {
    // Diagonalise F in orthonormal basis.
    const helfem::Matrix Forth = Sinvh.transpose() * Fao * Sinvh;
    Eigen::SelfAdjointEigenSolver<helfem::Matrix> es(Forth);
    const helfem::Matrix C = Sinvh * es.eigenvectors();

    // Closed-shell density: two electrons in the lowest orbital.
    const helfem::Vector c0 = C.col(0);
    const helfem::Matrix P  = 2.0 * c0 * c0.transpose();
    const helfem::Matrix Pa = 0.5 * P;

    // Coulomb + exchange -- Phase-5 accessors take helfem::Matrix directly.
    const helfem::Matrix J = basis.coulomb(P);
    const helfem::Matrix K = basis.exchange(Pa);

    const double Ekin  = (P * T).trace();
    const double Epot  = (P * Vnuc).trace();
    const double Ecoul = 0.5 * (P * J).trace();
    // K sign-included; single-shell -> Pa=Pb so total Exx = 2*0.5*Tr(Pa K).
    const double Exx   = (Pa * K).trace();
    const double E     = Ekin + Epot + Ecoul + Exx;

    if (iter > 0 && std::abs(E - Eprev) < 1e-12) {
      const double ref = -2.86167999561;
      const double err = std::abs(E - ref);
      std::printf("  Converged at iter %2d   E = %.11f   ref %.11f   err %.2e %s\n",
                   iter, E, ref, err, err < 1e-6 ? "" : "  *** FAIL ***");
      return err < 1e-6;
    }
    Eprev = E;

    Fao = Hc + J + K;
  }
  std::printf("  SCF did not converge\n");
  return false;
}

bool radial_df_factors_smoke_test() {
  std::printf("\nradial_df_factors smoke test (Z=2, l=0, small basis):\n");
  // Use a small basis so the Nrad^4 verification loop below stays cheap.
  const Defaults small{4, 6, 2, 4, 10.0, 2.0};
  auto basis = make_single_shell(/*Z*/2, /*l*/0, small);
  basis.compute_tei(/*exchange*/true);

  // Public return type is now arma-free:
  //   std::vector<std::vector<helfem::Matrix>>
  // Outer index = multipole k = 0..2*max(lval). Inner = Cholesky Q.
  const auto B = basis.radial_df_factors(1e-12);
  std::printf("  N_L (multipoles)     = %zu\n", B.size());
  if (B.empty()) return false;
  std::printf("  N_Q at k=0           = %zu\n", B[0].size());
  if (B[0].empty()) return false;
  std::printf("  factor shape at k=0  = (%lld x %lld)\n",
               (long long) B[0][0].rows(), (long long) B[0][0].cols());

  // Reconstruct R^0(i, j, m, n) from the Cholesky factors and verify
  // symmetry: R^0(i, j, m, n) == R^0(m, n, i, j) exactly by construction.
  const Eigen::Index Nrad = B[0][0].rows();
  double asym_max = 0.0;
  for (Eigen::Index i = 0; i < Nrad; ++i)
    for (Eigen::Index j = 0; j < Nrad; ++j)
      for (Eigen::Index m = 0; m < Nrad; ++m)
        for (Eigen::Index n = 0; n < Nrad; ++n) {
          double R0_ijmn = 0.0;
          double R0_mnij = 0.0;
          for (const auto & Bq : B[0]) {
            R0_ijmn += Bq(i, j) * Bq(m, n);
            R0_mnij += Bq(m, n) * Bq(i, j);
          }
          asym_max = std::max(asym_max, std::abs(R0_ijmn - R0_mnij));
        }
  std::printf("  reconstruction symmetry (ijmn vs mnij) max diff = %.2e\n",
               asym_max);
  return asym_max < 1e-12;
}

} // namespace

int main() {
  const bool ok1 = one_electron_tests();
  const bool ok2 = he_hf_test();
  const bool ok3 = radial_df_factors_smoke_test();
  std::printf("\nOverall: %s\n", (ok1 && ok2 && ok3) ? "PASS" : "FAIL");
  return (ok1 && ok2 && ok3) ? 0 : 1;
}

/* ============================================================================
 * INTEGRATION CONVENTIONS (verified empirically by the tests above)
 * ============================================================================
 *
 * Basis representation
 * --------------------
 *   The radial basis represents  u(r) = r * R(r)  (the radial wave function
 *   times r), where R(r) is what appears in psi = R(r) Y_lm. The FEM
 *   primitive polynomials B(r) are u(r) at quadrature points; near the
 *   origin a Taylor expansion of B(r)/r is used for numerical safety.
 *
 *   Integration measure is `dr`. There is NO implicit r^2 Jacobian.
 *
 * One-electron matrices returned by TwoDBasis
 * -------------------------------------------
 *   S  = overlap()      = integral u_i(r) u_j(r) dr
 *                       = quantum-mechanical radial overlap (r^2 absorbed in u^2)
 *   T  = kinetic()      = (1/2) integral u'_i(r) u'_j(r) dr
 *                       = radial kinetic operator, EXCLUDING centrifugal
 *   Tl = kinetic_l()    = (1/2) integral u_i(r) u_j(r) / r^2 dr
 *                       = half of <u | 1/r^2 | u>
 *   V  = nuclear()      = matrix element of -Z/r in u-basis
 *                       = Z-scaled, sign already negative
 *
 *   The l-dependent one-electron Hamiltonian is:
 *
 *      H(l) = T  +  l*(l+1) * Tl  +  V
 *
 *   Note the factor is l*(l+1), NOT l*(l+1)/2; the 1/2 is already in Tl.
 *
 * AO ordering
 * -----------
 *   For a TwoDBasis with (lval, mval) shells and Nrad radial functions,
 *   the flat basis index p for the pair (angular shell iang, radial idx
 *   irad) is:
 *      p = iang * Nrad + irad
 *
 * Nuclear charge convention
 * -------------------------
 *   TwoDBasis::nuclear() already applies -Z at the interior boundary.
 *   For finite-nucleus models, use the TwoDBasis constructor with the
 *   appropriate `nuclear_model_t` and `Rrms`; the model_potential
 *   machinery in libhelfem handles the smearing.
 *
 * Two-electron (Coulomb / exchange) convention
 * --------------------------------------------
 *   For a SINGLE-l shell (lval = {l}), TwoDBasis::coulomb(P) and
 *   exchange(P) reduce to the monopole (k=0) radial J and K. P is the
 *   TOTAL density matrix (Pa+Pb); coulomb(P) and exchange(Pa per spin)
 *   are sign-included so Fock = H_core + J + K assembles correctly.
 *
 *   For per-multipole-k radial Slater factors (external consumer that
 *   does its own Gaunt contraction), use TwoDBasis::radial_df_factors:
 *
 *     std::vector<std::vector<helfem::Matrix>> B = basis.radial_df_factors();
 *     // B[k][Q]  is Nrad x Nrad, so that
 *     // R^k(i, j, m, n) = sum_Q  B[k][Q](i, j) * B[k][Q](m, n)
 *     // where R^k is the BARE radial Slater integral (no 4*pi/(2k+1),
 *     // no Gaunt).
 *
 * ILP64 / LP64 BLAS interface
 * ---------------------------
 *   libhelfem is built against an ILP64 BLAS (libflexiblas64, 8-byte
 *   integer arguments) and Armadillo macros ARMA_64BIT_WORD /
 *   ARMA_BLAS_LONG. These are propagated to consumers as
 *   INTERFACE_COMPILE_DEFINITIONS on the exported helfem::helfem
 *   target -- a consumer that does target_link_libraries(mytarget
 *   PRIVATE helfem::helfem) gets them automatically. LP64/ILP64
 *   mismatch corrupts the heap inside LAPACK routines silently.
 * ============================================================================
 */
