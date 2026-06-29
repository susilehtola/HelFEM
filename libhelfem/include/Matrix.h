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
#ifndef HELFEM_MATRIX_H
#define HELFEM_MATRIX_H

#include <Eigen/Core>
#include <vector>

namespace helfem {

  // Phase 1 of the Eigen migration arc.
  //
  // Scope: introduce a tiny set of Eigen-based matrix / vector typedefs
  // that subsequent phases can adopt leaf-by-leaf. NO existing code is
  // migrated in this phase. The typedefs live alongside Armadillo so
  // both can coexist during the multi-PR transition.
  //
  // Convention:
  //   - Scalar (default 'double') is the floating-point type. Phase 4
  //     will templatise on this for arbitrary precision (long double,
  //     __float128, boost::multiprecision, ...).
  //   - Matrix / Vector are dynamic-size column-major Eigen types,
  //     matching arma::mat / arma::vec storage order. The migration
  //     can therefore use ArmaEigen.h's Map-based zero-copy converters
  //     at the interface boundary without reallocating data.
  //   - Cube is std::vector<Matrix>, mirroring how arma::cube is used
  //     in HelFEM (per-slice access via cube.slice(i); no heavy use of
  //     3-D-contiguous tensor operations). Eigen's tensor module is in
  //     unsupported/ and brings extra complexity we don't need.

  using Scalar = double;

  using Matrix    = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using Vector    = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using RowVector = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;
  using IVector   = Eigen::Matrix<long long, Eigen::Dynamic, 1>;
  using UVector   = Eigen::Matrix<unsigned long long, Eigen::Dynamic, 1>;

  /// "Cube" semantics matching arma::cube usage in HelFEM: a stack of
  /// equally-sized matrices indexed by a slice integer. NOT a contiguous
  /// 3-D tensor; for genuine tensor work, callers can hold a single
  /// Matrix viewed as a 2-D flattening (this is what the radial DF
  /// factors do today via arma::cube).
  using Cube = std::vector<Matrix>;

} // namespace helfem

#endif // HELFEM_MATRIX_H
