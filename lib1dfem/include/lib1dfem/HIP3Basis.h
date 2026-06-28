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
#ifndef LIB1DFEM_HIP3BASIS_H
#define LIB1DFEM_HIP3BASIS_H

#include <lib1dfem/LIPBasis.h>
#include <lib1dfem/HIP3Basis_eval.h>
#include <armadillo>

namespace helfem {
namespace lib1dfem {
namespace polynomial_basis {

/// Analytic third-order Hermite interpolating polynomial basis: at each
/// node x_i the basis carries four shape functions h_{i,0..3} that
/// interpolate value, first, second, and third r-derivative respectively.
/// Replaces the runtime-inverted GeneralHIPBasis(nder=3) path with
/// closed-form polynomials derived by libhelfem/src/generate_hip3_code.py.
///
/// Templated on the scalar type T.
template <typename T>
class HIP3Basis : public LIPBasis<T> {
 protected:
  /// L_i^(1)(x_i), L_i^(2)(x_i), L_i^(3)(x_i) at every node (precomputed).
  arma::Col<T> lipxi, lipxi2, lipxi3;

 public:
  /// id_ is the primbas value echoed via get_id() (typically 9 for HIP3).
  HIP3Basis(const arma::Col<T> & x, int id_ = 9) : LIPBasis<T>(x, id_) {
    // Four overlapping functions per node (value + 1st + 2nd + 3rd deriv).
    this->noverlap = 4;
    this->nprim    = 4 * static_cast<int>(this->x0.n_elem);
    this->enabled  = arma::linspace<arma::uvec>(0, this->nprim - 1, this->nprim);
    this->nnodes   = static_cast<int>(this->x0.n_elem);

    arma::Mat<T> dlip, ddlip, dddlip;
    detail::eval_lip_prim_dnf<T>(this->x0, this->x0, dlip,   1);
    detail::eval_lip_prim_dnf<T>(this->x0, this->x0, ddlip,  2);
    detail::eval_lip_prim_dnf<T>(this->x0, this->x0, dddlip, 3);
    lipxi  = arma::diagvec(dlip);
    lipxi2 = arma::diagvec(ddlip);
    lipxi3 = arma::diagvec(dddlip);
  }

  ~HIP3Basis() override = default;

  HIP3Basis<T> * copy() const override {
    return new HIP3Basis<T>(*this);
  }

  void eval_prim_dnf(const arma::Col<T> & x, arma::Mat<T> & dnf, int n,
                     T element_length) const override {
    detail::eval_hip3_prim_dnf<T>(x, this->x0, lipxi, lipxi2, lipxi3, dnf, n,
                                  element_length);
  }

  /// Drop first node's shape functions. Same convention as GeneralHIPBasis:
  ///   func=true,  deriv=true  : drop value + all three derivative interpolants
  ///   func=true,  deriv=false : drop only the value interpolant
  ///                             (Dirichlet at r=0 for atomic radial bases)
  ///   func=false, deriv=true  : drop only the three derivative interpolants
  ///   func=false, deriv=false : drop nothing
  void drop_first(bool func, bool deriv) override {
    const arma::uvec first(this->enabled.subvec(0, 3));
    const arma::uvec rest(this->enabled.subvec(4, this->enabled.n_elem - 1));
    arma::uvec keep;
    keep.set_size((func ? 0 : 1) + (deriv ? 0 : 3) + rest.n_elem);
    arma::uword idx = 0;
    if (!func)  keep(idx++) = first(0);
    if (!deriv) { keep(idx++) = first(1); keep(idx++) = first(2); keep(idx++) = first(3); }
    if (rest.n_elem)
      keep.subvec(idx, idx + rest.n_elem - 1) = rest;
    this->enabled = keep;
  }

  void drop_last(bool func, bool deriv) override {
    const arma::uvec head(this->enabled.subvec(0, this->enabled.n_elem - 5));
    const arma::uvec last(this->enabled.subvec(this->enabled.n_elem - 4,
                                               this->enabled.n_elem - 1));
    arma::uvec keep;
    keep.set_size(head.n_elem + (func ? 0 : 1) + (deriv ? 0 : 3));
    arma::uword idx = 0;
    if (head.n_elem) {
      keep.subvec(0, head.n_elem - 1) = head;
      idx = head.n_elem;
    }
    if (!func)  keep(idx++) = last(0);
    if (!deriv) { keep(idx++) = last(1); keep(idx++) = last(2); keep(idx++) = last(3); }
    this->enabled = keep;
  }
};

} // namespace polynomial_basis
} // namespace lib1dfem
} // namespace helfem

#endif
