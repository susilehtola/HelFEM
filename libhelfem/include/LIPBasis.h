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
#ifndef HELFEM_FEM_LIPBASIS_H
#define HELFEM_FEM_LIPBASIS_H

#include <PolynomialBasisT.h>
#include <LIPBasis_eval.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace helfem {
namespace polynomial_basis {

/// Lagrange interpolating polynomial basis on the reference element [-1, 1]
/// with caller-supplied control nodes x0 (must include the endpoints).
/// Templated on the scalar type T. Phase 5.2: Eigen-typed.
template <typename T>
class LIPBasisT : public PolynomialBasisT<T> {
 protected:
  /// Control nodes (sorted ascending; must span the reference element).
  Vec<T> x0;

 public:
  LIPBasisT() = default;

  LIPBasisT(const Vec<T> & x, int id_ = 4) {
    x0 = x;
    std::sort(x0.data(), x0.data() + x0.size());

    const T sqrteps = std::sqrt(std::numeric_limits<T>::epsilon());
    if (std::abs(x0(0) + T(1)) >= sqrteps)
      throw std::logic_error("LIP leftmost node is not at -1!\n");
    if (std::abs(x0(x0.size() - 1) - T(1)) >= sqrteps)
      throw std::logic_error("LIP rightmost node is not at -1!\n");

    this->noverlap = 1;
    this->nprim    = static_cast<int>(x0.size());
    this->enabled  = IVec::LinSpaced(x0.size(), 0, x0.size() - 1);
    this->id       = id_;
    this->nnodes   = static_cast<int>(this->enabled.size());
  }

  ~LIPBasisT() override = default;

  LIPBasisT<T> * copy() const override {
    return new LIPBasisT<T>(*this);
  }

  void drop_first(bool func, bool deriv) override {
    (void)deriv;
    if (func)
      this->enabled = this->enabled.segment(1, this->enabled.size() - 1).eval();
  }
  void drop_last(bool func, bool deriv) override {
    (void)deriv;
    if (func)
      this->enabled = this->enabled.segment(0, this->enabled.size() - 1).eval();
  }

  void eval_prim_dnf(const Vec<T> & x, Mat<T> & dnf, int n,
                     T element_length) const override {
    (void)element_length;
    detail::eval_lip_prim_dnf<T>(x, x0, dnf, n);
  }

  // Pull the base's 3-arg matrix-returning eval_over_r overload back into
  // scope (overriding the 4-arg virtual below would otherwise hide it).
  using PolynomialBasisT<T>::eval_over_r;

  /// Analytic B_u(r)/r for the surviving (post-drop_first) LIP shape
  /// functions on the first element. See LIPBasis.h header comment in
  /// the v1 code for the deflation derivation.
  void eval_over_r(const Vec<T> & x, Mat<T> & dnf_over_r, int n,
                   T element_length) const override {
    if (this->enabled.size() == 0)
      throw std::logic_error("eval_over_r: no surviving basis functions.\n");
    if (this->enabled(0) == 0)
      throw std::logic_error(
          "eval_over_r requires drop_first(func=true) on LIPBasisT: the shape "
          "function at x=-1 has B(-1) != 0 and B/r is singular at the origin.\n");

    const Vec<T> x0_reduced = x0.segment(1, x0.size() - 1);
    Mat<T> dnf_reduced;
    detail::eval_lip_prim_dnf<T>(x, x0_reduced, dnf_reduced, n);
    // dnf_reduced column j_red ∈ [0, n-2] corresponds to full index j_full = j_red + 1.

    const T scale = T(1) / std::pow(element_length, n + 1);
    dnf_over_r.resize(x.size(), this->enabled.size());
    for (Eigen::Index k = 0; k < this->enabled.size(); ++k) {
      const Eigen::Index i_full = this->enabled(k);
      const Eigen::Index i_red  = i_full - 1;
      const T denom = x0(i_full) + T(1);
      for (Eigen::Index ix = 0; ix < x.size(); ++ix) {
        dnf_over_r(ix, k) = scale * dnf_reduced(ix, i_red) / denom;
      }
    }
  }

  Vec<T> get_nodes() const override { return x0; }
};

} // namespace polynomial_basis
} // namespace helfem

#endif
