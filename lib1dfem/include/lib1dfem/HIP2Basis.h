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
  void eval_over_r(const arma::Col<T> & x, arma::Mat<T> & dnf_over_r, int n,
                   T element_length) const override {
    if (n < 0 || n > 2) {
      std::ostringstream oss;
      oss << "HIP2Basis::eval_over_r: derivative order " << n
          << " not implemented (only 0, 1, 2 supported).\n";
      throw std::logic_error(oss.str());
    }
    if (this->enabled.n_elem == 0)
      throw std::logic_error("HIP2Basis::eval_over_r: no surviving basis functions.\n");
    if (this->enabled(0) == 0)
      throw std::logic_error(
          "HIP2Basis::eval_over_r requires drop_first(func=true, deriv=false): "
          "the value-shape at node 0 has B(-1) != 0 and B/r is singular.\n");

    const arma::Col<T> & x0_full = this->x0;
    const arma::Col<T> x0_red    = x0_full.subvec(1, x0_full.n_elem - 1);

    // Reduced L_i^{(0)}(x) and derivatives for i >= 1.
    arma::Mat<T> Lr0, Lr1, Lr2;
    detail::eval_lip_prim_dnf<T>(x, x0_red, Lr0, 0);
    if (n >= 1) detail::eval_lip_prim_dnf<T>(x, x0_red, Lr1, 1);
    if (n >= 2) detail::eval_lip_prim_dnf<T>(x, x0_red, Lr2, 2);

    // Full L_0(x) (col 0) for the surviving node-0 shapes.
    arma::Mat<T> Lf0, Lf1, Lf2;
    detail::eval_lip_prim_dnf<T>(x, x0_full, Lf0, 0);
    if (n >= 1) detail::eval_lip_prim_dnf<T>(x, x0_full, Lf1, 1);
    if (n >= 2) detail::eval_lip_prim_dnf<T>(x, x0_full, Lf2, 2);

    const T inv_e   = T(1) / element_length;
    const T chain_n = std::pow(inv_e, n);

    dnf_over_r.set_size(x.n_elem, this->enabled.n_elem);
    for (arma::uword k = 0; k < this->enabled.n_elem; ++k) {
      const arma::uword idx  = this->enabled(k);
      const arma::uword node = idx / 3;
      const arma::uword kind = idx % 3;

      for (arma::uword ix = 0; ix < x.n_elem; ++ix) {
        const T xv  = x(ix);
        const T xpx1 = xv + T(1);
        const T xi   = x0_full(node);
        const T xmxi = xv - xi;
        const T p1   = lipxi(node);
        const T p2   = lipxi2(node);
        const T ev   = element_length;
        T R;

        if (node == 0) {
          const T p   = Lf0(ix, 0);
          const T pd  = (n >= 1) ? Lf1(ix, 0) : T(0);
          const T pdd = (n >= 2) ? Lf2(ix, 0) : T(0);
          const T p2_ = p * p;
          const T p3  = p2_ * p;
          if (kind == 1) {
            // R = p^3 (1 - 3 p1 (x+1)); A(x) = 1 - 3 p1 (x+1), A' = -3 p1, A'' = 0
            const T A   = T(1) - T(3) * p1 * xpx1;
            if (n == 0) {
              R = p3 * A;
            } else if (n == 1) {
              // R' = 3 p^2 p' A + p^3 A'
              R = T(3) * p2_ * pd * A - T(3) * p1 * p3;
            } else {
              // R'' = 6 p (p')^2 A + 3 p^2 p'' A + 6 p^2 p' A'
              R = T(6) * p * pd * pd * A
                + T(3) * p2_ * pdd * A
                - T(18) * p1 * p2_ * pd;
            }
          } else if (kind == 2) {
            // R = (e/2)(x+1) p^3
            const T half_e = ev / T(2);
            if (n == 0) {
              R = half_e * xpx1 * p3;
            } else if (n == 1) {
              // R' = (e/2)[p^3 + 3(x+1) p^2 p']
              R = half_e * (p3 + T(3) * xpx1 * p2_ * pd);
            } else {
              // R'' = (e/2)[6 p^2 p' + 6(x+1) p (p')^2 + 3(x+1) p^2 p'']
              R = half_e * (T(6) * p2_ * pd
                          + T(6) * xpx1 * p * pd * pd
                          + T(3) * xpx1 * p2_ * pdd);
            }
          } else {
            // kind == 0 dropped by precondition above.
            throw std::logic_error("HIP2Basis::eval_over_r internal error: node-0 kind=0 in enabled.\n");
          }
        } else {
          // node >= 1; use reduced LIP at col (node - 1).
          const arma::uword i_red = node - 1;
          const T p   = Lr0(ix, i_red);
          const T pd  = (n >= 1) ? Lr1(ix, i_red) : T(0);
          const T pdd = (n >= 2) ? Lr2(ix, i_red) : T(0);
          const T p2_ = p * p;
          const T p3  = p2_ * p;
          const T y   = xi + T(1);                      // x_i + 1
          const T y3  = y * y * y;
          // Common factors for (Lr^3 derivatives):
          //   F   = p^3 ,  F' = 3 p^2 p' ,  F'' = 6 p (p')^2 + 3 p^2 p''
          const T F   = p3;
          const T Fd  = T(3) * p2_ * pd;
          const T Fdd = T(6) * p * pd * pd + T(3) * p2_ * pdd;

          if (kind == 0) {
            // R = (1/e) (x+1)^2 / y^3 * F(x) * q0(x - x_i)
            // q0(t) = 1 - 3 p1 t + (6 p1^2 - 3 p2/2) t^2
            // G(x) = (x+1)^2 q0(x - x_i)
            const T a = T(1), b = -T(3) * p1, c = T(6)*p1*p1 - T(3)*p2/T(2);
            const T q0   = a + b * xmxi + c * xmxi * xmxi;
            const T q0p  = b + T(2) * c * xmxi;       // q0'(t) wrt t = wrt x (dt/dx=1)
            const T q0pp = T(2) * c;
            const T G   = xpx1 * xpx1 * q0;
            const T Gp  = T(2) * xpx1 * q0 + xpx1 * xpx1 * q0p;
            const T Gpp = T(2) * q0 + T(4) * xpx1 * q0p + xpx1 * xpx1 * q0pp;
            const T scale_kind = T(1) / (ev * y3);
            if (n == 0) {
              R = scale_kind * F * G;
            } else if (n == 1) {
              R = scale_kind * (Fd * G + F * Gp);
            } else {
              R = scale_kind * (Fdd * G + T(2) * Fd * Gp + F * Gpp);
            }
          } else if (kind == 1) {
            // R = (1/y^3) (x+1)^2 (x-xi) [1 - 3 p1 (x-xi)] * F(x)  (no e factor; cancels)
            // Let H(x) = (x+1)^2 (x-xi) (1 - 3 p1 (x-xi)) / y^3
            const T A   = T(1) - T(3) * p1 * xmxi;       // A(x) = 1 - 3 p1 t
            const T Ap  = -T(3) * p1;
            // tA = t * A, with t = x - xi
            const T tA   = xmxi * A;
            const T tAp  = A + xmxi * Ap;                // (tA)' = A + t A'
            const T tApp = T(2) * Ap;                    // (tA)'' = 2 A' + t A'' (A''=0)
            // (x+1)^2 derivatives
            const T u   = xpx1 * xpx1;
            const T up  = T(2) * xpx1;
            const T upp = T(2);
            // H = u * tA / y^3
            const T H   = u * tA;
            const T Hp  = up * tA + u * tAp;
            const T Hpp = upp * tA + T(2) * up * tAp + u * tApp;
            const T scale_kind = T(1) / y3;
            if (n == 0) {
              R = scale_kind * F * H;
            } else if (n == 1) {
              R = scale_kind * (Fd * H + F * Hp);
            } else {
              R = scale_kind * (Fdd * H + T(2) * Fd * Hp + F * Hpp);
            }
          } else /* kind == 2 */ {
            // R = (e/2) (x-xi)^2 (x+1)^2 / y^3 * F(x)
            const T t2   = xmxi * xmxi;
            const T t2p  = T(2) * xmxi;
            const T t2pp = T(2);
            const T u    = xpx1 * xpx1;
            const T up   = T(2) * xpx1;
            const T upp  = T(2);
            // K(x) = t^2 u
            const T K    = t2 * u;
            const T Kp   = t2p * u + t2 * up;
            const T Kpp  = t2pp * u + T(2) * t2p * up + t2 * upp;
            const T scale_kind = ev / (T(2) * y3);
            if (n == 0) {
              R = scale_kind * F * K;
            } else if (n == 1) {
              R = scale_kind * (Fd * K + F * Kp);
            } else {
              R = scale_kind * (Fdd * K + T(2) * Fd * Kp + F * Kpp);
            }
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
