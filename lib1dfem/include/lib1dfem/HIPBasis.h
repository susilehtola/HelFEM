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
#include <lib1dfem/HIPBasis_over_r.h>

namespace helfem {
namespace lib1dfem {
namespace polynomial_basis {

/// Hermite interpolating polynomial basis (order 1: interpolates value
/// and first derivative at each node). Derives from LIPBasis<T> because
/// the HIP shape functions are constructed from L_i(x)^2 and need the
/// LIP derivative table. Phase 5.2: Eigen-typed.
template <typename T>
class HIPBasis : public LIPBasis<T> {
 protected:
  /// L_i'(x_i) at each node (precomputed at construction)
  Vec<T> lipxi;

 public:
  HIPBasis(const Vec<T> & x, int id_ = 5) : LIPBasis<T>(x, id_) {
    // Two overlapping functions: function + derivative
    this->noverlap = 2;
    this->nprim    = 2 * static_cast<int>(this->x0.size());
    this->enabled  = IVec::LinSpaced(this->nprim, 0, this->nprim - 1);
    this->nnodes   = static_cast<int>(this->x0.size());

    // L'_i(x_i): use the LIP-derivative evaluator at the node positions.
    Mat<T> dlip;
    detail::eval_lip_prim_dnf<T>(this->x0, this->x0, dlip, 1);
    lipxi = dlip.diagonal();
  }

  ~HIPBasis() override = default;

  HIPBasis<T> * copy() const override {
    return new HIPBasis<T>(*this);
  }

  void drop_first(bool func, bool deriv) override {
    if (func && deriv) {
      this->enabled = this->enabled.segment(2, this->enabled.size() - 2).eval();
    } else if (func) {
      this->enabled = this->enabled.segment(1, this->enabled.size() - 1).eval();
    } else if (deriv) {
      IVec new_enabled(this->enabled.size() - 1);
      new_enabled(0) = this->enabled(0);
      new_enabled.segment(1, new_enabled.size() - 1)
          = this->enabled.segment(2, this->enabled.size() - 2);
      this->enabled = new_enabled;
    }
  }

  void drop_last(bool func, bool deriv) override {
    if (func && deriv) {
      this->enabled = this->enabled.segment(0, this->enabled.size() - 2).eval();
    } else if (deriv) {
      this->enabled = this->enabled.segment(0, this->enabled.size() - 1).eval();
    } else {
      IVec new_enabled(this->enabled.size() - 1);
      new_enabled.segment(0, this->enabled.size() - 2)
          = this->enabled.segment(0, this->enabled.size() - 2);
      new_enabled(this->enabled.size() - 2) = this->enabled(this->enabled.size() - 1);
      this->enabled = new_enabled;
    }
  }

  void eval_prim_dnf(const Vec<T> & x, Mat<T> & dnf, int n,
                     T element_length) const override {
    detail::eval_hip_prim_dnf<T>(x, this->x0, lipxi, dnf, n, element_length);
  }

  // Pull the base's 3-arg matrix-returning eval_over_r overload back into
  // scope (overriding the 4-arg virtual below would otherwise hide it).
  using PolynomialBasis<T>::eval_over_r;

  /// Analytic B_u(r)/r for the surviving (post-drop_first(true,false)) HIP
  /// shape functions on the first element. See header comment in v1 code
  /// for the deflation derivation.
  void eval_over_r(const Vec<T> & x, Mat<T> & dnf_over_r, int n,
                   T element_length) const override {
    // All of the deflation maths is generated -- see
    // libhelfem/src/generate_hip_family_code.py --order 1 --over-r
    detail::eval_hip_prim_over_r<T>(x, this->x0, lipxi, this->enabled,
                                      dnf_over_r, n, element_length);
  }
};

} // namespace polynomial_basis
} // namespace lib1dfem
} // namespace helfem

#endif
