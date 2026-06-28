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
#include <cmath>
#include <sstream>
#include <stdexcept>

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

  // Pull the base's 3-arg eval_over_r overload back into scope.
  using PolynomialBasis<T>::eval_over_r;

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
  void eval_over_r(const arma::Col<T> & x, arma::Mat<T> & dnf_over_r, int n,
                   T element_length) const override {
    if (n < 0 || n > 2) {
      std::ostringstream oss;
      oss << "HIP3Basis::eval_over_r: derivative order " << n
          << " not implemented (only 0, 1, 2 supported).\n";
      throw std::logic_error(oss.str());
    }
    if (this->enabled.n_elem == 0)
      throw std::logic_error("HIP3Basis::eval_over_r: no surviving basis functions.\n");
    if (this->enabled(0) == 0)
      throw std::logic_error(
          "HIP3Basis::eval_over_r requires drop_first(func=true, deriv=false): "
          "the value-shape at node 0 has B(-1) != 0 and B/r is singular.\n");

    const arma::Col<T> & x0_full = this->x0;
    const arma::Col<T> x0_red    = x0_full.subvec(1, x0_full.n_elem - 1);

    arma::Mat<T> Lr0, Lr1, Lr2;
    detail::eval_lip_prim_dnf<T>(x, x0_red, Lr0, 0);
    if (n >= 1) detail::eval_lip_prim_dnf<T>(x, x0_red, Lr1, 1);
    if (n >= 2) detail::eval_lip_prim_dnf<T>(x, x0_red, Lr2, 2);

    arma::Mat<T> Lf0, Lf1, Lf2;
    detail::eval_lip_prim_dnf<T>(x, x0_full, Lf0, 0);
    if (n >= 1) detail::eval_lip_prim_dnf<T>(x, x0_full, Lf1, 1);
    if (n >= 2) detail::eval_lip_prim_dnf<T>(x, x0_full, Lf2, 2);

    const T inv_e   = T(1) / element_length;
    const T chain_n = std::pow(inv_e, n);

    dnf_over_r.set_size(x.n_elem, this->enabled.n_elem);
    for (arma::uword k = 0; k < this->enabled.n_elem; ++k) {
      const arma::uword idx  = this->enabled(k);
      const arma::uword node = idx / 4;
      const arma::uword kind = idx % 4;

      for (arma::uword ix = 0; ix < x.n_elem; ++ix) {
        const T xv   = x(ix);
        const T xpx1 = xv + T(1);
        const T xi   = x0_full(node);
        const T xmxi = xv - xi;
        const T p1   = lipxi(node);
        const T p2   = lipxi2(node);
        const T p3   = lipxi3(node);
        const T ev   = element_length;
        T R;

        // Pick the LIP value to use: full L_0 for node==0, reduced L_i^{(0)} for i>=1.
        T p, pd, pdd;
        if (node == 0) {
          p   = Lf0(ix, 0);
          pd  = (n >= 1) ? Lf1(ix, 0) : T(0);
          pdd = (n >= 2) ? Lf2(ix, 0) : T(0);
        } else {
          const arma::uword i_red = node - 1;
          p   = Lr0(ix, i_red);
          pd  = (n >= 1) ? Lr1(ix, i_red) : T(0);
          pdd = (n >= 2) ? Lr2(ix, i_red) : T(0);
        }
        // F = p^4 and its first two x-derivatives.
        const T p2_  = p * p;
        const T p3_  = p2_ * p;
        const T p4   = p2_ * p2_;
        const T F    = p4;
        const T Fd   = T(4) * p3_ * pd;
        const T Fdd  = T(12) * p2_ * pd * pd + T(4) * p3_ * pdd;

        // Helper closures via inline lambdas for "F * P" derivatives.
        // Each branch computes R via the (F, Fd, Fdd) * (P, Pd, Pdd) Leibniz combo.
        auto combine = [&](const T P, const T Pd, const T Pdd) -> T {
          if (n == 0) return F * P;
          if (n == 1) return Fd * P + F * Pd;
          return Fdd * P + T(2) * Fd * Pd + F * Pdd;
        };

        if (node == 0) {
          if (kind == 1) {
            // R = p^4 * [1 - 4 p1 (x+1) + (10 p1^2 - 2 p2)(x+1)^2]
            // P(x) = 1 - 4 p1 (x+1) + c2 (x+1)^2, with c2 = 10 p1^2 - 2 p2
            const T c2 = T(10) * p1 * p1 - T(2) * p2;
            const T P   = T(1) - T(4) * p1 * xpx1 + c2 * xpx1 * xpx1;
            const T Pd  = -T(4) * p1 + T(2) * c2 * xpx1;
            const T Pdd = T(2) * c2;
            R = combine(P, Pd, Pdd);
          } else if (kind == 2) {
            // R = (e/2)(x+1) * p^4 * [1 - 4 p1 (x+1)]
            // P(x) = (e/2)(x+1) (1 - 4 p1 (x+1))
            const T half_e = ev / T(2);
            const T A   = T(1) - T(4) * p1 * xpx1;
            const T Ap  = -T(4) * p1;
            // P = half_e * xpx1 * A
            const T P   = half_e * xpx1 * A;
            const T Pd  = half_e * (A + xpx1 * Ap);
            const T Pdd = half_e * (T(2) * Ap);            // A'' = 0
            R = combine(P, Pd, Pdd);
          } else if (kind == 3) {
            // R = (e^2/6)(x+1)^2 * p^4
            const T pre = ev * ev / T(6);
            const T P   = pre * xpx1 * xpx1;
            const T Pd  = pre * T(2) * xpx1;
            const T Pdd = pre * T(2);
            R = combine(P, Pd, Pdd);
          } else {
            throw std::logic_error("HIP3Basis::eval_over_r internal error: node-0 kind=0 in enabled.\n");
          }
        } else {
          // node >= 1
          const T y  = xi + T(1);
          const T y4 = y * y * y * y;

          if (kind == 0) {
            // R = (1/e) (x+1)^3 / y^4 * p^4 * q0(x - x_i)
            // q0(t) = 1 - 4 p1 t + (10 p1^2 - 2 p2) t^2 + (-20 p1^3 + 10 p1 p2 - (2/3) p3) t^3
            const T a = T(1);
            const T b = -T(4) * p1;
            const T c = T(10) * p1 * p1 - T(2) * p2;
            const T d = -T(20) * p1 * p1 * p1 + T(10) * p1 * p2 - (T(2)/T(3)) * p3;
            const T t1 = xmxi;
            const T t2 = t1 * t1;
            const T q   = a + b * t1 + c * t2 + d * t1 * t2;
            const T qp  = b + T(2) * c * t1 + T(3) * d * t2;
            const T qpp = T(2) * c + T(6) * d * t1;
            const T xp3   = xpx1 * xpx1 * xpx1;
            const T xp3p  = T(3) * xpx1 * xpx1;
            const T xp3pp = T(6) * xpx1;
            // P(x) = xpx1^3 * q(t)
            const T P   = xp3 * q;
            const T Pd  = xp3p * q + xp3 * qp;
            const T Pdd = xp3pp * q + T(2) * xp3p * qp + xp3 * qpp;
            const T scale_kind = T(1) / (ev * y4);
            if (n == 0)      R = scale_kind * F * P;
            else if (n == 1) R = scale_kind * (Fd * P + F * Pd);
            else             R = scale_kind * (Fdd * P + T(2) * Fd * Pd + F * Pdd);
          } else if (kind == 1) {
            // R = (1/y^4) (x-xi)(x+1)^3 * p^4 * [1 - 4 p1 (x-xi) + (10 p1^2 - 2 p2)(x-xi)^2]
            const T c2  = T(10) * p1 * p1 - T(2) * p2;
            const T t1  = xmxi;
            const T t2  = t1 * t1;
            const T A   = T(1) - T(4) * p1 * t1 + c2 * t2;
            const T Ap  = -T(4) * p1 + T(2) * c2 * t1;
            const T App = T(2) * c2;
            // tA = t * A
            const T tA   = t1 * A;
            const T tAp  = A + t1 * Ap;
            const T tApp = T(2) * Ap + t1 * App;
            const T xp3   = xpx1 * xpx1 * xpx1;
            const T xp3p  = T(3) * xpx1 * xpx1;
            const T xp3pp = T(6) * xpx1;
            // P(x) = xp3 * tA
            const T P   = xp3 * tA;
            const T Pd  = xp3p * tA + xp3 * tAp;
            const T Pdd = xp3pp * tA + T(2) * xp3p * tAp + xp3 * tApp;
            const T scale_kind = T(1) / y4;
            if (n == 0)      R = scale_kind * F * P;
            else if (n == 1) R = scale_kind * (Fd * P + F * Pd);
            else             R = scale_kind * (Fdd * P + T(2) * Fd * Pd + F * Pdd);
          } else if (kind == 2) {
            // R = (e/2) (x-xi)^2 (x+1)^3 / y^4 * p^4 * [1 - 4 p1 (x-xi)]
            const T t1  = xmxi;
            const T t2  = t1 * t1;
            const T A   = T(1) - T(4) * p1 * t1;
            const T Ap  = -T(4) * p1;
            // U = t^2 A
            const T U   = t2 * A;
            const T Up  = T(2) * t1 * A + t2 * Ap;
            const T Upp = T(2) * A + T(4) * t1 * Ap;
            const T xp3   = xpx1 * xpx1 * xpx1;
            const T xp3p  = T(3) * xpx1 * xpx1;
            const T xp3pp = T(6) * xpx1;
            const T P   = xp3 * U;
            const T Pd  = xp3p * U + xp3 * Up;
            const T Pdd = xp3pp * U + T(2) * xp3p * Up + xp3 * Upp;
            const T scale_kind = ev / (T(2) * y4);
            if (n == 0)      R = scale_kind * F * P;
            else if (n == 1) R = scale_kind * (Fd * P + F * Pd);
            else             R = scale_kind * (Fdd * P + T(2) * Fd * Pd + F * Pdd);
          } else /* kind == 3 */ {
            // R = (e^2/6) (x-xi)^3 (x+1)^3 / y^4 * p^4
            const T t1   = xmxi;
            const T t3   = t1 * t1 * t1;
            const T t3p  = T(3) * t1 * t1;
            const T t3pp = T(6) * t1;
            const T xp3   = xpx1 * xpx1 * xpx1;
            const T xp3p  = T(3) * xpx1 * xpx1;
            const T xp3pp = T(6) * xpx1;
            // P = xp3 * t3
            const T P   = xp3 * t3;
            const T Pd  = xp3p * t3 + xp3 * t3p;
            const T Pdd = xp3pp * t3 + T(2) * xp3p * t3p + xp3 * t3pp;
            const T scale_kind = ev * ev / (T(6) * y4);
            if (n == 0)      R = scale_kind * F * P;
            else if (n == 1) R = scale_kind * (Fd * P + F * Pd);
            else             R = scale_kind * (Fdd * P + T(2) * Fd * Pd + F * Pdd);
          }
        }
        dnf_over_r(ix, k) = chain_n * R;
      }
    }
  }
};

} // namespace polynomial_basis
} // namespace lib1dfem
} // namespace helfem

#endif
