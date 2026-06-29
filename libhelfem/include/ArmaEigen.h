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
#ifndef HELFEM_ARMA_EIGEN_H
#define HELFEM_ARMA_EIGEN_H

#include "Matrix.h"
#include <armadillo>

namespace helfem {

  // Arma <-> Eigen converters for the migration boundary.
  //
  // Both libraries default to column-major dense storage, so a zero-copy
  // Map is safe as long as both sides agree on shape and lifetime.
  //
  // - to_eigen_view / to_arma_view : zero-copy views. The data is
  //   shared; the underlying buffer must outlive the view. Use when
  //   the source object lives in the same scope and is not resized.
  //
  // - to_eigen / to_arma           : owning copy. Use at API boundaries
  //   where the caller wants its own buffer (return values from
  //   migrated leaf classes, etc.).
  //
  // Once a leaf class has been fully migrated to helfem::Matrix, its
  // existing arma-returning methods can wrap a single line:
  //     arma::mat foo_arma() const { return to_arma(foo_eigen()); }
  // to keep arma-consuming callers working unchanged. When the LAST
  // caller is migrated, the arma wrapper is deleted.

  // ---- Owning copies --------------------------------------------------

  inline Matrix to_eigen(const arma::mat & A) {
    Matrix out(A.n_rows, A.n_cols);
    if (A.n_rows == 0 || A.n_cols == 0)
      return out;
    // arma::mat memory is column-major; so is Eigen::Matrix by default.
    std::copy(A.memptr(), A.memptr() + A.n_rows * A.n_cols, out.data());
    return out;
  }

  inline Vector to_eigen(const arma::vec & v) {
    Vector out(v.n_elem);
    if (v.n_elem == 0) return out;
    std::copy(v.memptr(), v.memptr() + v.n_elem, out.data());
    return out;
  }

  inline arma::mat to_arma(const Matrix & A) {
    arma::mat out(A.rows(), A.cols());
    if (A.rows() == 0 || A.cols() == 0)
      return out;
    std::copy(A.data(), A.data() + A.rows() * A.cols(), out.memptr());
    return out;
  }

  inline arma::vec to_arma(const Vector & v) {
    arma::vec out(v.size());
    if (v.size() == 0) return out;
    std::copy(v.data(), v.data() + v.size(), out.memptr());
    return out;
  }

  // ---- Zero-copy views ------------------------------------------------
  //
  // Cast away const on arma::mat::memptr() for the const overloads: arma
  // exposes a const-correct memptr() only for non-const objects (memptr()
  // is non-const). We need the const-correct mapping at the language
  // level even though the underlying buffer is read-only by contract.

  inline Eigen::Map<Matrix> to_eigen_view(arma::mat & A) {
    return Eigen::Map<Matrix>(A.memptr(),
                              static_cast<Eigen::Index>(A.n_rows),
                              static_cast<Eigen::Index>(A.n_cols));
  }

  inline Eigen::Map<const Matrix> to_eigen_view(const arma::mat & A) {
    return Eigen::Map<const Matrix>(A.memptr(),
                                    static_cast<Eigen::Index>(A.n_rows),
                                    static_cast<Eigen::Index>(A.n_cols));
  }

  inline Eigen::Map<Vector> to_eigen_view(arma::vec & v) {
    return Eigen::Map<Vector>(v.memptr(),
                              static_cast<Eigen::Index>(v.n_elem));
  }

  inline Eigen::Map<const Vector> to_eigen_view(const arma::vec & v) {
    return Eigen::Map<const Vector>(v.memptr(),
                                    static_cast<Eigen::Index>(v.n_elem));
  }

  inline arma::mat to_arma_view(Matrix & A) {
    // arma::mat ctor (ptr, n_rows, n_cols, copy_aux_mem=true, strict=false)
    // -- pass copy_aux_mem=false to get a view onto the Eigen buffer.
    return arma::mat(A.data(),
                     static_cast<arma::uword>(A.rows()),
                     static_cast<arma::uword>(A.cols()),
                     /*copy_aux_mem=*/false,
                     /*strict=*/false);
  }

  inline arma::vec to_arma_view(Vector & v) {
    return arma::vec(v.data(),
                     static_cast<arma::uword>(v.size()),
                     /*copy_aux_mem=*/false,
                     /*strict=*/false);
  }

} // namespace helfem

#endif // HELFEM_ARMA_EIGEN_H
