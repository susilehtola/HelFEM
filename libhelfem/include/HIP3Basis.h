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
#ifndef HELFEM_FEM_HIP3BASIS_H
#define HELFEM_FEM_HIP3BASIS_H

#include <LIPBasis.h>
#include <HIP3Basis_eval.h>
#include <HIP3Basis_over_r.h>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace helfem {
namespace polynomial_basis {

/// Analytic third-order Hermite interpolating polynomial basis: at each
/// node x_i the basis carries four shape functions h_{i,0..3} that
/// interpolate value, first, second, and third r-derivative respectively.
/// Replaces the runtime-inverted GeneralHIPBasis(nder=3) path with
/// closed-form polynomials derived by libhelfem/src/generate_hip3_code.py.
///
/// Templated on the scalar type T.
template <typename T>
class HIP3BasisT : public LIPBasisT<T> {
 protected:
  /// L_i^(1)(x_i), L_i^(2)(x_i), L_i^(3)(x_i) at every node (precomputed).
  Vec<T> lipxi, lipxi2, lipxi3;

 public:
  /// id_ is the primbas value echoed via get_id() (typically 9 for HIP3).
  HIP3BasisT(const Vec<T> & x, int id_ = 9) : LIPBasisT<T>(x, id_) {
    // Four overlapping functions per node (value + 1st + 2nd + 3rd deriv).
    this->noverlap = 4;
    this->nprim    = 4 * static_cast<int>(this->x0.size());
    this->enabled  = IVec::LinSpaced(this->nprim, 0, this->nprim - 1);
    this->nnodes   = static_cast<int>(this->x0.size());

    Mat<T> dlip, ddlip, dddlip;
    detail::eval_lip_prim_dnf<T>(this->x0, this->x0, dlip,   1);
    detail::eval_lip_prim_dnf<T>(this->x0, this->x0, ddlip,  2);
    detail::eval_lip_prim_dnf<T>(this->x0, this->x0, dddlip, 3);
    lipxi  = dlip.diagonal();
    lipxi2 = ddlip.diagonal();
    lipxi3 = dddlip.diagonal();
  }

  ~HIP3BasisT() override = default;

  HIP3BasisT<T> * copy() const override {
    return new HIP3BasisT<T>(*this);
  }

  void eval_prim_dnf(const Vec<T> & x, Mat<T> & dnf, int n,
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
    const IVec first(this->enabled.segment(0, (3) - (0) + 1));
    const IVec rest(this->enabled.segment(4, (this->enabled.size() - (4) + 1) - 1));
    IVec keep;
    keep.resize((func ? 0 : 1) + (deriv ? 0 : 3) + rest.size());
    Eigen::Index idx = 0;
    if (!func)  keep(idx++) = first(0);
    if (!deriv) { keep(idx++) = first(1); keep(idx++) = first(2); keep(idx++) = first(3); }
    if (rest.size())
      keep.segment(idx, (idx + rest.size() - (idx) + 1) - 1) = rest;
    this->enabled = keep;
  }

  void drop_last(bool func, bool deriv) override {
    const IVec head(this->enabled.segment(0, (this->enabled.size() - (0) + 1) - 5));
    const IVec last(this->enabled.segment(this->enabled.size() - 4, (this->enabled.size() - (this->enabled.size() - 4) + 1) - 1));
    IVec keep;
    keep.resize(head.size() + (func ? 0 : 1) + (deriv ? 0 : 3));
    Eigen::Index idx = 0;
    if (head.size()) {
      keep.segment(0, (head.size() - (0) + 1) - 1) = head;
      idx = head.size();
    }
    if (!func)  keep(idx++) = last(0);
    if (!deriv) { keep(idx++) = last(1); keep(idx++) = last(2); keep(idx++) = last(3); }
    this->enabled = keep;
  }

  // Pull the base's 3-arg eval_over_r overload back into scope.
  using PolynomialBasisT<T>::eval_over_r;

  /// Analytic R(r) = B(r)/r and its first two r-derivatives for the
  /// surviving HIP3 shape functions on the first element. Precondition:
  /// drop_first(zero_func=true, zero_deriv=false) so the value-shape at
  /// node 0 is dropped (B(-1) != 0 otherwise).
  ///
  /// Derivations (e = element_length = scaling_factor; p = L_0(x) for
  /// node 0 or L_i^{(0)}(x) for i >= 1; p1, p2, p3 = L_i^{(1,2,3)}(x_i)
  /// precomputed at construction):
  ///
  ///   R_{0,1} = p^4 * [1 - 4 p1 (x+1) + (10 p1^2 - 2 p2)(x+1)^2]
  ///   R_{0,2} = (e/2)(x+1) p^4 * [1 - 4 p1 (x+1)]
  ///   R_{0,3} = (e^2/6)(x+1)^2 p^4
  ///   R_{i,0} = (1/e)(x+1)^3 / (x_i+1)^4 * p^4 * q0(x-x_i)
  ///   R_{i,1} = (x-x_i)(x+1)^3 / (x_i+1)^4 * p^4 *
  ///                            [1 - 4 p1 (x-x_i) + (10 p1^2 - 2 p2)(x-x_i)^2]
  ///   R_{i,2} = (e/2)(x-x_i)^2 (x+1)^3 / (x_i+1)^4 * p^4 * [1 - 4 p1 (x-x_i)]
  ///   R_{i,3} = (e^2/6)(x-x_i)^3 (x+1)^3 / (x_i+1)^4 * p^4
  ///
  /// where q0(t) = 1 - 4 p1 t + (10 p1^2 - 2 p2) t^2 +
  ///                (-20 p1^3 + 10 p1 p2 - (2/3) p3) t^3.
  /// d^n/dr^n picks up (1/e)^n via the chain rule.
  ///
  /// Implemented for n in {0, 1, 2}; throws for higher orders.
  void eval_over_r(const Vec<T> & x, Mat<T> & dnf_over_r, int n,
                   T element_length) const override {
    // All of the deflation maths is generated -- see
    // libhelfem/src/generate_hip_family_code.py --order 3 --over-r
    detail::eval_hip3_prim_over_r<T>(x, this->x0, lipxi, lipxi2, lipxi3, this->enabled,
                                      dnf_over_r, n, element_length);
  }
};

} // namespace polynomial_basis
} // namespace helfem

#endif
