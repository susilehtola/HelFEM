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

  // Phase 4 of the Eigen migration arc.
  //
  // The matrix / vector / cube types are now templates on the scalar
  // type T (default 'double'), with back-compat aliases so all the
  // double-precision code from Phases 1..3 continues to compile and
  // run unchanged.
  //
  // To use arbitrary precision later:
  //   helfem::MatrixT<long double> M(n, n);
  //   helfem::MatrixT<boost::multiprecision::mpfr_float> Mhp(n, n);
  // and pair with Eigen::SelfAdjointEigenSolver<MatrixT<T>>, etc.
  //
  // Note: this PR introduces the type machinery only. Individual
  // methods on TwoDBasis / FEMRadialBasis still accept and return
  // MatrixT<double> (i.e. the unqualified `Matrix` alias). Templating
  // those methods would require migrating their internals away from
  // arma (which is double-only) -- a separate, larger effort.

  // Default scalar -- the previous Phase-1 type-alias name kept for
  // header compatibility (some places used helfem::Scalar).
  using Scalar = double;

  /// Templated matrix type. Default precision is double; specialise
  /// the parameter for arbitrary precision (long double,
  /// boost::multiprecision::mpfr_float, ...). Storage order is
  /// column-major to match arma::mat, so ArmaEigen.h's Map-based
  /// converters stay zero-copy at the double-precision boundary.
  template <typename T = double>
  using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

  /// Templated column vector.
  template <typename T = double>
  using VectorT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  /// Templated row vector.
  template <typename T = double>
  using RowVectorT = Eigen::Matrix<T, 1, Eigen::Dynamic>;

  // Short aliases matching the lib1dfem::Vec<T> / Mat<T> / RowVec<T>
  // shorthand. Use these in new templated code -- the verbose
  // Eigen::Matrix<T, Eigen::Dynamic, ...> spelling hurts readability.
  template <typename T = double>
  using Vec    = VectorT<T>;
  template <typename T = double>
  using Mat    = MatrixT<T>;
  template <typename T = double>
  using RowVec = RowVectorT<T>;

  // Back-compat aliases: all code written in Phases 1..3 uses these
  // unqualified names, which keep meaning Matrix<double> etc.
  using Matrix    = MatrixT<double>;
  using Vector    = VectorT<double>;
  using RowVector = RowVectorT<double>;

  /// Integer vectors stay scalar-free; no template parameter needed.
  using IVector   = Eigen::Matrix<long long, Eigen::Dynamic, 1>;
  using UVector   = Eigen::Matrix<unsigned long long, Eigen::Dynamic, 1>;

  /// "Cube" semantics matching arma::cube usage in HelFEM: a stack of
  /// equally-sized matrices indexed by a slice integer. NOT a
  /// contiguous 3-D tensor.
  template <typename T = double>
  using CubeT = std::vector<MatrixT<T>>;

  using Cube = CubeT<double>;

} // namespace helfem

#endif // HELFEM_MATRIX_H
