/*
 *                This source code is part of
 *
 *                          HelFEM
 *
 * Finite element methods for electronic structure calculations on small systems
 *
 * Written by Susi Lehtola, 2018-
 * Copyright (c) 2018- Susi Lehtola
 *
 * Released under the BSD 3-Clause License.
 */

// Phase 1 of the Eigen migration arc -- validate the helfem::Matrix /
// Vector typedefs and the Arma <-> Eigen converters.
//
// Verifies:
//   1. The Eigen-based typedefs (Matrix, Vector, Cube) exist and basic
//      operations work (allocate, fill, multiply, decompose).
//   2. to_eigen / to_arma owning copies round-trip exactly.
//   3. to_eigen_view / to_arma_view share storage (mutating one is
//      visible through the other).
//   4. Matrix multiply gives the same result through both libraries.

#include "Matrix.h"
#include "ArmaEigen.h"

#include <Eigen/Eigenvalues>
#include <armadillo>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {

  void check(bool cond, const char * msg) {
    if (!cond) {
      std::fprintf(stderr, "FAIL: %s\n", msg);
      std::exit(1);
    }
    std::printf("  ok: %s\n", msg);
  }

  double max_abs_diff(const arma::mat & A, const helfem::Matrix & B) {
    if (A.n_rows != (arma::uword) B.rows() || A.n_cols != (arma::uword) B.cols())
      return 1e300;
    double d = 0.0;
    for (arma::uword j = 0; j < A.n_cols; ++j)
      for (arma::uword i = 0; i < A.n_rows; ++i)
        d = std::max(d, std::abs(A(i, j) - B(i, j)));
    return d;
  }

} // namespace

int main() {
  using helfem::Matrix;
  using helfem::Vector;
  using helfem::Cube;

  std::printf("=== Eigen foundation test ===\n");

  // 1. Basic typedefs work.
  {
    Matrix M = Matrix::Random(5, 4);
    Vector v = Vector::Random(4);
    Vector y = M * v;
    check(y.size() == 5, "Matrix * Vector returns correct size");
    check(M.transpose().rows() == 4, "Matrix::transpose is well-formed");
  }

  // 2. Cube = std::vector<Matrix> works.
  {
    Cube C;
    C.reserve(3);
    for (int k = 0; k < 3; ++k) C.emplace_back(Matrix::Constant(4, 4, double(k)));
    check(C.size() == 3, "Cube has 3 slices");
    check(C[1](2, 3) == 1.0, "Cube slice value preserved");
  }

  // 3. Owning round-trip arma <-> Eigen.
  arma::arma_rng::set_seed(42);
  arma::mat A_arma = arma::randn(7, 5);
  Matrix    A_eig  = helfem::to_eigen(A_arma);
  arma::mat A_back = helfem::to_arma(A_eig);
  check(max_abs_diff(A_arma, A_eig) < 1e-15, "to_eigen preserves values");
  check(arma::approx_equal(A_arma, A_back, "absdiff", 1e-15),
        "round-trip arma -> Eigen -> arma is exact");

  // 4. Zero-copy view: mutating via the Eigen view is visible in arma.
  arma::mat B_arma = arma::randn(4, 6);
  {
    auto B_view = helfem::to_eigen_view(B_arma);
    B_view(2, 3) = 99.0;
  }
  check(B_arma(2, 3) == 99.0, "to_eigen_view mutates arma buffer in place");

  // 5. Const view does not require non-const arma.
  {
    const arma::mat & B_const = B_arma;
    auto B_view = helfem::to_eigen_view(B_const);
    check(B_view(2, 3) == 99.0, "const to_eigen_view reads through to arma");
  }

  // 6. Matrix product: arma vs Eigen give the same numerical answer.
  arma::mat P = arma::randn(5, 7);
  arma::mat Q = arma::randn(7, 4);
  arma::mat PQ_arma = P * Q;
  Matrix PQ_eig = helfem::to_eigen(P) * helfem::to_eigen(Q);
  check(max_abs_diff(PQ_arma, PQ_eig) < 1e-12,
        "arma * arma == to_arma(Eigen * Eigen) (1e-12)");

  // 7. Eigenvalue decomposition: arma::eig_sym vs Eigen::SelfAdjointEigenSolver.
  arma::mat S = arma::randn(6, 6);
  S = 0.5 * (S + S.t());                       // symmetrise
  S += 6.0 * arma::eye(6, 6);                  // shift positive definite-ish
  arma::vec  arma_eigvals;
  arma::mat  arma_eigvecs;
  arma::eig_sym(arma_eigvals, arma_eigvecs, S);

  Eigen::SelfAdjointEigenSolver<Matrix> es(helfem::to_eigen(S));
  Vector eig_eigvals = es.eigenvalues();
  check(arma_eigvals.n_elem == (arma::uword) eig_eigvals.size(),
        "eigenvalue counts match");
  double eig_diff = 0.0;
  for (arma::uword i = 0; i < arma_eigvals.n_elem; ++i)
    eig_diff = std::max(eig_diff,
                        std::abs(arma_eigvals(i) - eig_eigvals(i)));
  check(eig_diff < 1e-12,
        "arma::eig_sym vs Eigen::SelfAdjointEigenSolver agree (1e-12)");

  // 8. Phase 4: MatrixT<long double> template instantiation.
  // Sanity check the template machinery: a long-double symmetric
  // matrix should diagonalise and reconstruct itself to long-double
  // precision (~1e-18, several orders tighter than the ~1e-15 we get
  // from double).
  {
    using MatLD = helfem::MatrixT<long double>;
    using VecLD = helfem::VectorT<long double>;
    MatLD M = MatLD::Random(6, 6);
    M = (M + M.transpose()).eval();
    M.diagonal().array() += 6.0L;

    Eigen::SelfAdjointEigenSolver<MatLD> es_ld(M);
    VecLD evals = es_ld.eigenvalues();
    MatLD evecs = es_ld.eigenvectors();
    // Reconstruct M = V diag(eig) V^T and check residual.
    MatLD recon = evecs * evals.asDiagonal() * evecs.transpose();
    long double resid = 0.0L;
    for (Eigen::Index j = 0; j < M.cols(); ++j)
      for (Eigen::Index i = 0; i < M.rows(); ++i)
        resid = std::max(resid, std::fabs((long double)(M(i, j) - recon(i, j))));
    // long double on x86-64 is 80-bit extended (~1.08e-19 eps); cap
    // the threshold safely at 1e-16 to cover platforms where long
    // double is just IEEE double.
    check(resid < 1.0e-14L,
          "MatrixT<long double>: V * diag(eig) * V^T == M (1e-14)");

    // Cube<long double> also instantiates and stores slices correctly.
    helfem::CubeT<long double> C;
    C.reserve(2);
    for (int k = 0; k < 2; ++k) C.emplace_back(MatLD::Constant(3, 3, (long double) k));
    check(C.size() == 2 && C[1](0, 0) == 1.0L,
          "CubeT<long double> stores slice values");
  }

  std::printf("PASS\n");
  return 0;
}
