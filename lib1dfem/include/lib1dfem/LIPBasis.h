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
#ifndef LIB1DFEM_LIPBASIS_H
#define LIB1DFEM_LIPBASIS_H

#include <lib1dfem/PolynomialBasis.h>
#include <lib1dfem/LIPBasis_eval.h>
#include <armadillo>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace helfem {
namespace lib1dfem {
namespace polynomial_basis {

/// Lagrange interpolating polynomial basis on the reference element [-1, 1]
/// with caller-supplied control nodes x0 (must include the endpoints).
/// Templated on the scalar type T.
template <typename T>
class LIPBasis : public PolynomialBasis<T> {
 protected:
  /// Control nodes (sorted ascending; must span the reference element).
  arma::Col<T> x0;

 public:
  LIPBasis() = default;

  LIPBasis(const arma::Col<T> & x, int id_ = 4) {
    x0 = arma::sort(x, "ascend");

    const T sqrteps = std::sqrt(std::numeric_limits<T>::epsilon());
    if (std::abs(x0(0) + T(1)) >= sqrteps)
      throw std::logic_error("LIP leftmost node is not at -1!\n");
    if (std::abs(x0(x0.n_elem - 1) - T(1)) >= sqrteps)
      throw std::logic_error("LIP rightmost node is not at -1!\n");

    this->noverlap = 1;
    this->nprim    = static_cast<int>(x0.n_elem);
    this->enabled  = arma::linspace<arma::uvec>(0, x0.n_elem - 1, x0.n_elem);
    this->id       = id_;
    this->nnodes   = static_cast<int>(this->enabled.n_elem);
  }

  ~LIPBasis() override = default;

  LIPBasis<T> * copy() const override {
    return new LIPBasis<T>(*this);
  }

  void drop_first(bool func, bool deriv) override {
    (void)deriv;
    if (func)
      this->enabled = this->enabled.subvec(1, this->enabled.n_elem - 1);
  }
  void drop_last(bool func, bool deriv) override {
    (void)deriv;
    if (func)
      this->enabled = this->enabled.subvec(0, this->enabled.n_elem - 2);
  }

  void eval_prim_dnf(const arma::Col<T> & x, arma::Mat<T> & dnf, int n,
                     T element_length) const override {
    (void)element_length;
    detail::eval_lip_prim_dnf<T>(x, x0, dnf, n);
  }

  // Pull the base's 3-arg matrix-returning eval_over_r overload back into
  // scope (overriding the 4-arg virtual below would otherwise hide it).
  using PolynomialBasis<T>::eval_over_r;

  /// Analytic B_u(r)/r for the surviving (post-drop_first) LIP shape
  /// functions on the first element. Exploits the Dirichlet-induced
  /// factorisation
  ///     L_i(x) = ((x+1)/(x_i+1)) * L_i^{(0)}(x)
  /// where L_i^{(0)} is the Lagrange polynomial over the reduced node set
  /// {x_1, ..., x_{n-1}}, so for r = (Delta/2)(x+1):
  ///     B_i(r)/r       = (2/Delta) * L_i^{(0)}(x) / (x_i + 1)
  ///     d^n[B_i/r]/dr^n = (2/Delta)^(n+1) * L_i^{(0)(n)}(x) / (x_i + 1)
  ///
  /// L_i^{(0)} is computed by reusing the templated LIP evaluator on the
  /// reduced node set. Output columns are indexed by `enabled` (just like
  /// eval_dnf).
  void eval_over_r(const arma::Col<T> & x, arma::Mat<T> & dnf_over_r, int n,
                   T element_length) const override {
    if (this->enabled.n_elem == 0)
      throw std::logic_error("eval_over_r: no surviving basis functions.\n");
    if (this->enabled(0) == 0)
      throw std::logic_error(
          "eval_over_r requires drop_first(func=true) on LIPBasis: the shape "
          "function at x=-1 has B(-1) != 0 and B/r is singular at the origin.\n");

    const arma::Col<T> x0_reduced = x0.subvec(1, x0.n_elem - 1);
    arma::Mat<T> dnf_reduced;
    detail::eval_lip_prim_dnf<T>(x, x0_reduced, dnf_reduced, n);
    // dnf_reduced column j_red ∈ [0, n-2] corresponds to full index j_full = j_red + 1.

    const T scale = std::pow(T(2) / element_length, n + 1);
    dnf_over_r.set_size(x.n_elem, this->enabled.n_elem);
    for (arma::uword k = 0; k < this->enabled.n_elem; ++k) {
      const arma::uword i_full = this->enabled(k);
      const arma::uword i_red  = i_full - 1;
      const T denom = x0(i_full) + T(1);
      for (arma::uword ix = 0; ix < x.n_elem; ++ix) {
        dnf_over_r(ix, k) = scale * dnf_reduced(ix, i_red) / denom;
      }
    }
  }

  arma::Col<T> get_nodes() const override { return x0; }
};

} // namespace polynomial_basis
} // namespace lib1dfem
} // namespace helfem

#endif
