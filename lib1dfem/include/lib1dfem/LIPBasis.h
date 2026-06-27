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

  arma::Col<T> get_nodes() const override { return x0; }
};

} // namespace polynomial_basis
} // namespace lib1dfem
} // namespace helfem

#endif
