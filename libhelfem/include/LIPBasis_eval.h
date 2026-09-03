/*
 *                This source code is part of
 *                          HelFEM
 *
 * Written by Susi Lehtola, 2018-
 * Copyright (c) 2018- Susi Lehtola
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#ifndef HELFEM_FEM_LIPBASIS_EVAL_H
#define HELFEM_FEM_LIPBASIS_EVAL_H

#include <types.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace helfem {
namespace polynomial_basis {
namespace detail {

// Evaluation of Lagrange interpolating polynomials and their derivatives on
// the reference element.
//
// Values come from the second (true) barycentric form, which is backward
// stable for node families with a slowly growing Lebesgue constant --
// Gauss-Lobatto among them -- and costs O(n) per point instead of O(n^2).
//   N. J. Higham, IMA J. Numer. Anal. 24 (2004) 547.
//
// Derivatives are NOT obtained by differentiating that expression. l_j^(k) is
// a polynomial of degree n-1-k <= n-1, so it is fixed exactly by its values
// at the n nodes, and
//
//     l_j^(k)(x) = sum_i l_i(x) D^(k)_{ij},   D^(k)_{ij} = l_j^(k)(x0_i)
//
// is an identity. The nested product-sums the previous implementation used --
// O(n^(k+2)) per point -- collapse into one matrix product against a
// differentiation matrix that depends only on the node set.
//
// D^(k) is built by the usual recurrence, but its diagonal comes from the
// negative sum trick rather than its own closed form: rows of D^(k) sum to
// zero exactly, because differentiating a constant gives zero, and imposing
// that identity removes the cancellation that otherwise dominates at high
// order.
//   R. Baltensperger and M. R. Trummer, SIAM J. Sci. Comput. 24 (2003) 1465.

/// Barycentric weights w_j = 1 / prod_{k != j} (x0_j - x0_k).
///
/// Only RATIOS of the weights enter every formula below, so any common factor
/// is free to remove. Two are: the differences are scaled to keep the running
/// product near unity as n grows, and the result is normalised by max |w|.
template <typename T>
void lip_bary_weights(const Vec<T> & x0, Vec<T> & w) {
  const Eigen::Index n = x0.size();
  w.resize(n);
  if (n == 0)
    return;

  // 4 / (span) maps a typical node separation onto O(1), which keeps the
  // (n-1)-fold product representable for large node counts.
  const T span = x0(n - 1) - x0(0);
  const T scale = (span > T(0)) ? T(4) / span : T(1);

  for (Eigen::Index j = 0; j < n; j++) {
    T p = T(1);
    for (Eigen::Index k = 0; k < n; k++)
      if (k != j)
        p *= (x0(j) - x0(k)) * scale;
    w(j) = T(1) / p;
  }

  T big = T(0);
  for (Eigen::Index j = 0; j < n; j++)
    big = std::max(big, std::abs(w(j)));
  if (big > T(0))
    for (Eigen::Index j = 0; j < n; j++)
      w(j) /= big;
}

/// Values l_j(x) by the second barycentric form; f is (npoints, nnodes).
template <typename T>
void lip_values_bary(const Vec<T> & x, const Vec<T> & x0, const Vec<T> & w,
                     Mat<T> & f) {
  const Eigen::Index np = x.size(), n = x0.size();
  f.setZero(np, n);
  for (Eigen::Index ix = 0; ix < np; ix++) {
    // A point sitting exactly on a node: the interpolant is the Kronecker
    // delta there, and the general form would divide by zero.
    Eigen::Index hit = -1;
    for (Eigen::Index j = 0; j < n; j++)
      if (x(ix) == x0(j)) {
        hit = j;
        break;
      }
    if (hit >= 0) {
      f(ix, hit) = T(1);
      continue;
    }
    T denom = T(0);
    for (Eigen::Index j = 0; j < n; j++) {
      const T t = w(j) / (x(ix) - x0(j));
      f(ix, j) = t;
      denom += t;
    }
    for (Eigen::Index j = 0; j < n; j++)
      f(ix, j) /= denom;
  }
}

/// Differentiation matrices D[k](i, j) = l_j^(k)(x0_i), for k = 0 .. nmax.
template <typename T>
void lip_diff_matrices(const Vec<T> & x0, const Vec<T> & w, int nmax,
                       std::vector<Mat<T>> & D) {
  const Eigen::Index n = x0.size();
  D.assign((size_t) nmax + 1, Mat<T>::Zero(n, n));
  D[0] = Mat<T>::Identity(n, n);
  for (int k = 1; k <= nmax; k++)
    for (Eigen::Index i = 0; i < n; i++) {
      T rowsum = T(0);
      for (Eigen::Index j = 0; j < n; j++) {
        if (i == j)
          continue;
        D[(size_t) k](i, j) = T(k) / (x0(i) - x0(j)) *
                              ((w(j) / w(i)) * D[(size_t) k - 1](i, i) -
                               D[(size_t) k - 1](i, j));
        rowsum += D[(size_t) k](i, j);
      }
      // Negative sum trick: the diagonal is whatever makes the row sum vanish.
      D[(size_t) k](i, i) = -rowsum;
    }
}

/// Cached evaluator for a fixed node set.
///
/// The weights and the low-order differentiation matrices depend only on x0,
/// which is fixed for the lifetime of a basis, so they are built once here and
/// only lip_values_bary stays on the hot path.
///
/// The cache is filled eagerly at construction and never mutated afterwards:
/// basis objects are shared across threads inside the OpenMP grid loops, so a
/// lazily grown cache would be a data race. Orders above the cached range are
/// served by building the matrices locally in eval(), which is thread-safe and
/// costs O(k n^2) against the O(np n^2) matrix product it feeds.
template <typename T>
class LIPEvaluator {
  Vec<T> x0_;
  Vec<T> w_;
  std::vector<Mat<T>> D_;

 public:
  /// Derivative orders cached. The radial code asks for values, first and
  /// second derivatives; the Hermite bases add the third at construction.
  static constexpr int kmax_cache = 4;

  LIPEvaluator() = default;

  explicit LIPEvaluator(const Vec<T> & x0) : x0_(x0) {
    lip_bary_weights<T>(x0_, w_);
    const int kmax = std::min<int>(kmax_cache, (int) x0_.size() - 1);
    if (kmax >= 0)
      lip_diff_matrices<T>(x0_, w_, kmax, D_);
  }

  const Vec<T> & nodes() const { return x0_; }
  const Vec<T> & weights() const { return w_; }

  void eval(const Vec<T> & x, Mat<T> & dnf, int n) const {
    const Eigen::Index nn = x0_.size();
    // l_j has degree nn-1, so every derivative of order nn or above vanishes
    // identically. Saying so is exact and avoids a pointless recurrence.
    if (n >= (int) nn) {
      dnf.setZero(x.size(), nn);
      return;
    }

    Mat<T> f;
    lip_values_bary<T>(x, x0_, w_, f);
    if (n == 0) {
      dnf = f;
      return;
    }

    if ((size_t) n < D_.size()) {
      dnf.noalias() = f * D_[(size_t) n];
    } else {
      std::vector<Mat<T>> D;
      lip_diff_matrices<T>(x0_, w_, n, D);
      dnf.noalias() = f * D[(size_t) n];
    }
  }
};

/// Evaluate the n-th derivative of every LIP polynomial on the reference
/// element [-1, 1] at the given points x, given the control-node vector
/// x0. Fills dnf (size x.size() x x0.size()) with d^n L_i(x_j)/dx^n in
/// column-major (point, polynomial) layout. element_length is unused
/// at this layer (the chain-rule scaling lives in PolynomialBasis::eval_dnf).
///
/// This form builds the node-dependent data on every call. Callers holding a
/// fixed node set should keep a LIPEvaluator instead; this entry point exists
/// for the generated Hermite headers, which are handed x0 rather than a basis.
template <typename T>
void eval_lip_prim_dnf(const Vec<T> & x, const Vec<T> & x0, Mat<T> & dnf,
                       int n) {
  if (n < 0)
    throw std::logic_error("Negative derivative order requested!\n");
  LIPEvaluator<T>(x0).eval(x, dnf, n);
}

} // namespace detail
} // namespace polynomial_basis
} // namespace helfem

#endif
