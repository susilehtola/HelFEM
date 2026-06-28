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
#ifndef LIB1DFEM_HIP2BASIS_H
#define LIB1DFEM_HIP2BASIS_H

#include <lib1dfem/LIPBasis.h>
#include <lib1dfem/HIP2Basis_eval.h>
#include <armadillo>

namespace helfem {
namespace lib1dfem {
namespace polynomial_basis {

/// Analytic second-order Hermite interpolating polynomial basis: at each
/// node x_i the basis carries three shape functions h_{i,0}, h_{i,1},
/// h_{i,2} that interpolate value, first r-derivative, and second
/// r-derivative respectively. Replaces the runtime-inverted
/// GeneralHIPBasis(nder=2) path with closed-form polynomials derived by
/// libhelfem/src/generate_hip2_code.py.
///
/// Templated on the scalar type T.
template <typename T>
class HIP2Basis : public LIPBasis<T> {
 protected:
  /// L_i'(x_i) at every node (precomputed at construction).
  arma::Col<T> lipxi;
  /// L_i''(x_i) at every node (precomputed at construction).
  arma::Col<T> lipxi2;

 public:
  /// id_ is just an identifier echoed via get_id(); the libhelfem
  /// factory passes the primbas value (8 for HIP2).
  HIP2Basis(const arma::Col<T> & x, int id_ = 8) : LIPBasis<T>(x, id_) {
    // Three overlapping functions per node (value + 1st deriv + 2nd deriv).
    this->noverlap = 3;
    this->nprim    = 3 * static_cast<int>(this->x0.n_elem);
    this->enabled  = arma::linspace<arma::uvec>(0, this->nprim - 1, this->nprim);
    this->nnodes   = static_cast<int>(this->x0.n_elem);

    // L_i'(x_i) and L_i''(x_i) at every node, via the existing LIP evaluator.
    arma::Mat<T> dlip, ddlip;
    detail::eval_lip_prim_dnf<T>(this->x0, this->x0, dlip,  1);
    detail::eval_lip_prim_dnf<T>(this->x0, this->x0, ddlip, 2);
    lipxi  = arma::diagvec(dlip);
    lipxi2 = arma::diagvec(ddlip);
  }

  ~HIP2Basis() override = default;

  HIP2Basis<T> * copy() const override {
    return new HIP2Basis<T>(*this);
  }

  void eval_prim_dnf(const arma::Col<T> & x, arma::Mat<T> & dnf, int n,
                     T element_length) const override {
    detail::eval_hip2_prim_dnf<T>(x, this->x0, lipxi, lipxi2, dnf, n,
                                  element_length);
  }

  /// Drop first node's shape functions. Same convention as the existing
  /// GeneralHIPBasis it replaces:
  ///   func=true,  deriv=true  : drop value + both derivative interpolants
  ///   func=true,  deriv=false : drop only the value interpolant
  ///                             (Dirichlet at r=0 for atomic radial bases)
  ///   func=false, deriv=true  : drop the two derivative interpolants
  ///   func=false, deriv=false : drop nothing
  void drop_first(bool func, bool deriv) override {
    const arma::uvec first(this->enabled.subvec(0, 2));
    const arma::uvec rest(this->enabled.subvec(3, this->enabled.n_elem - 1));
    arma::uvec keep;
    keep.set_size((func ? 0 : 1) + (deriv ? 0 : 2) + rest.n_elem);
    arma::uword idx = 0;
    if (!func)  keep(idx++) = first(0);
    if (!deriv) { keep(idx++) = first(1); keep(idx++) = first(2); }
    if (rest.n_elem)
      keep.subvec(idx, idx + rest.n_elem - 1) = rest;
    this->enabled = keep;
  }

  void drop_last(bool func, bool deriv) override {
    const arma::uvec head(this->enabled.subvec(0, this->enabled.n_elem - 4));
    const arma::uvec last(this->enabled.subvec(this->enabled.n_elem - 3,
                                               this->enabled.n_elem - 1));
    arma::uvec keep;
    keep.set_size(head.n_elem + (func ? 0 : 1) + (deriv ? 0 : 2));
    arma::uword idx = 0;
    if (head.n_elem) {
      keep.subvec(0, head.n_elem - 1) = head;
      idx = head.n_elem;
    }
    if (!func)  keep(idx++) = last(0);
    if (!deriv) { keep(idx++) = last(1); keep(idx++) = last(2); }
    this->enabled = keep;
  }
};

} // namespace polynomial_basis
} // namespace lib1dfem
} // namespace helfem

#endif
