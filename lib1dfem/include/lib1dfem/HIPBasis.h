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
#ifndef LIB1DFEM_HIPBASIS_H
#define LIB1DFEM_HIPBASIS_H

#include <lib1dfem/LIPBasis.h>
#include <lib1dfem/HIPBasis_eval.h>
#include <armadillo>

namespace helfem {
namespace lib1dfem {
namespace polynomial_basis {

/// Hermite interpolating polynomial basis (order 1: interpolates value
/// and first derivative at each node). Derives from LIPBasis<T> because
/// the HIP shape functions are constructed from L_i(x)^2 and need the
/// LIP derivative table.
template <typename T>
class HIPBasis : public LIPBasis<T> {
 protected:
  /// L_i'(x_i) at each node (precomputed at construction)
  arma::Col<T> lipxi;

 public:
  HIPBasis(const arma::Col<T> & x, int id_ = 5) : LIPBasis<T>(x, id_) {
    // Two overlapping functions: function + derivative
    this->noverlap = 2;
    this->nprim    = 2 * static_cast<int>(this->x0.n_elem);
    this->enabled  = arma::linspace<arma::uvec>(0, this->nprim - 1, this->nprim);
    this->nnodes   = static_cast<int>(this->x0.n_elem);

    // L'_i(x_i): use the LIP-derivative evaluator at the node positions.
    arma::Mat<T> dlip;
    detail::eval_lip_prim_dnf<T>(this->x0, this->x0, dlip, 1);
    lipxi = arma::diagvec(dlip);
  }

  ~HIPBasis() override = default;

  HIPBasis<T> * copy() const override {
    return new HIPBasis<T>(*this);
  }

  void drop_first(bool func, bool deriv) override {
    if (func && deriv) {
      this->enabled = this->enabled.subvec(2, this->enabled.n_elem - 1);
    } else if (func) {
      this->enabled = this->enabled.subvec(1, this->enabled.n_elem - 1);
    } else if (deriv) {
      arma::uvec new_enabled(this->enabled.n_elem - 1);
      new_enabled(0) = this->enabled(0);
      new_enabled.subvec(1, new_enabled.n_elem - 1) =
          this->enabled.subvec(2, this->enabled.n_elem - 1);
      this->enabled = new_enabled;
    }
  }

  void drop_last(bool func, bool deriv) override {
    if (func && deriv) {
      this->enabled = this->enabled.subvec(0, this->enabled.n_elem - 3);
    } else if (deriv) {
      this->enabled = this->enabled.subvec(0, this->enabled.n_elem - 2);
    } else {
      arma::uvec new_enabled(this->enabled.n_elem - 1);
      new_enabled.subvec(0, this->enabled.n_elem - 3) =
          this->enabled.subvec(0, this->enabled.n_elem - 3);
      new_enabled(this->enabled.n_elem - 2) = this->enabled(this->enabled.n_elem - 1);
      this->enabled = new_enabled;
    }
  }

  void eval_prim_dnf(const arma::Col<T> & x, arma::Mat<T> & dnf, int n,
                     T element_length) const override {
    detail::eval_hip_prim_dnf<T>(x, this->x0, lipxi, dnf, n, element_length);
  }
};

} // namespace polynomial_basis
} // namespace lib1dfem
} // namespace helfem

#endif
