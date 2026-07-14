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
#include <lib1dfem/HIP2Basis_over_r.h>
#include <cmath>
#include <sstream>
#include <stdexcept>

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
  Vec<T> lipxi;
  /// L_i''(x_i) at every node (precomputed at construction).
  Vec<T> lipxi2;

 public:
  /// id_ is just an identifier echoed via get_id(); the libhelfem
  /// factory passes the primbas value (8 for HIP2).
  HIP2Basis(const Vec<T> & x, int id_ = 8) : LIPBasis<T>(x, id_) {
    // Three overlapping functions per node (value + 1st deriv + 2nd deriv).
    this->noverlap = 3;
    this->nprim    = 3 * static_cast<int>(this->x0.size());
    this->enabled  = IVec::LinSpaced(this->nprim, 0, this->nprim - 1);
    this->nnodes   = static_cast<int>(this->x0.size());

    // L_i'(x_i) and L_i''(x_i) at every node, via the existing LIP evaluator.
    Mat<T> dlip, ddlip;
    detail::eval_lip_prim_dnf<T>(this->x0, this->x0, dlip,  1);
    detail::eval_lip_prim_dnf<T>(this->x0, this->x0, ddlip, 2);
    lipxi  = dlip.diagonal();
    lipxi2 = ddlip.diagonal();
  }

  ~HIP2Basis() override = default;

  HIP2Basis<T> * copy() const override {
    return new HIP2Basis<T>(*this);
  }

  void eval_prim_dnf(const Vec<T> & x, Mat<T> & dnf, int n,
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
    const IVec first(this->enabled.segment(0, (2) - (0) + 1));
    const IVec rest(this->enabled.segment(3, (this->enabled.size() - (3) + 1) - 1));
    IVec keep;
    keep.resize((func ? 0 : 1) + (deriv ? 0 : 2) + rest.size());
    Eigen::Index idx = 0;
    if (!func)  keep(idx++) = first(0);
    if (!deriv) { keep(idx++) = first(1); keep(idx++) = first(2); }
    if (rest.size())
      keep.segment(idx, (idx + rest.size() - (idx) + 1) - 1) = rest;
    this->enabled = keep;
  }

  void drop_last(bool func, bool deriv) override {
    const IVec head(this->enabled.segment(0, (this->enabled.size() - (0) + 1) - 4));
    const IVec last(this->enabled.segment(this->enabled.size() - 3, (this->enabled.size() - (this->enabled.size() - 3) + 1) - 1));
    IVec keep;
    keep.resize(head.size() + (func ? 0 : 1) + (deriv ? 0 : 2));
    Eigen::Index idx = 0;
    if (head.size()) {
      keep.segment(0, (head.size() - (0) + 1) - 1) = head;
      idx = head.size();
    }
    if (!func)  keep(idx++) = last(0);
    if (!deriv) { keep(idx++) = last(1); keep(idx++) = last(2); }
    this->enabled = keep;
  }

  // Pull the base's 3-arg eval_over_r overload back into scope.
  using PolynomialBasis<T>::eval_over_r;

  /// Analytic R(r) = B(r)/r and its first two r-derivatives for the
  /// surviving HIP2 shape functions on the first element. Precondition:
  /// drop_first(zero_func=true, zero_deriv=false) was called so the
  /// value-shape at node 0 is dropped (it has B(-1) != 0).
  ///
  /// Derivations (with e = element_length = scaling_factor, x_i = x0(node),
  /// p1 = L_i'(x_i), p2 = L_i''(x_i), p = L_0(x) for node 0 or
  /// p = L_i^{(0)}(x) for i >= 1 with L_i = ((x+1)/(x_i+1)) L_i^{(0)}):
  ///
  ///   R_{0,1} = p^3 * (1 - 3 p1 (x+1))                               (e cancels)
  ///   R_{0,2} = (e/2)(x+1) p^3
  ///   R_{i,0} = (1/e) (x+1)^2 / (x_i+1)^3 * p^3 * q0(x-x_i)            (i >= 1)
  ///   R_{i,1} = (x-x_i)(x+1)^2 / (x_i+1)^3 * p^3 * (1 - 3 p1 (x-x_i))  (i >= 1)
  ///   R_{i,2} = (e/2)(x-x_i)^2 (x+1)^2 / (x_i+1)^3 * p^3               (i >= 1)
  ///
  /// with q0(t) = 1 - 3 p1 t + (6 p1^2 - (3/2) p2) t^2.  d^n/dr^n picks
  /// up a (1/e)^n chain-rule factor.
  ///
  /// Implemented for n in {0, 1, 2}; throws for higher orders.
  void eval_over_r(const Vec<T> & x, Mat<T> & dnf_over_r, int n,
                   T element_length) const override {
    // All of the deflation maths is generated -- see
    // libhelfem/src/generate_hip_family_code.py --order 2 --over-r
    detail::eval_hip2_prim_over_r<T>(x, this->x0, lipxi, lipxi2, this->enabled,
                                      dnf_over_r, n, element_length);
  }
};

} // namespace polynomial_basis
} // namespace lib1dfem
} // namespace helfem

#endif
