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

  // Pull the base's 3-arg matrix-returning eval_over_r overload back into
  // scope (overriding the 4-arg virtual below would otherwise hide it).
  using PolynomialBasis<T>::eval_over_r;

  /// Analytic B_u(r)/r for the surviving (post-drop_first(true,false)) HIP
  /// shape functions on the first element.
  ///
  /// The element_length parameter is the scaling_factor = half the full
  /// element width, matching the eval_dnf convention; r = element_length *
  /// (x+1) on the first element.
  ///
  /// Surviving shapes after dropping only the function at node 0:
  ///   - df_0 = (x+1) L_0(x)^2 * element_length   (derivative shape at node 0)
  ///   - f_i, df_i for i >= 1                     (both shapes at remaining nodes)
  ///
  /// For i >= 1, factor L_i(x) = ((x+1)/(x_i+1)) * L_i^{(0)}(x), where
  /// L_i^{(0)} is the LIP over the reduced node set {x_1, ..., x_{n-1}}.
  /// Plug into the HIP shape formulas and divide by r = element_length*(x+1):
  ///
  ///   df_0(r)/r  = L_0(x)^2                                                          (n=0)
  ///   f_i(r)/r   = (1/element_length) * (x+1) * [1 - 2(x-x_i) L_i'(x_i)]
  ///                * L_i^{(0)}(x)^2 / (x_i+1)^2                                       (n=0, i>=1)
  ///   df_i(r)/r  = (x+1) * (x-x_i) * L_i^{(0)}(x)^2 / (x_i+1)^2                       (n=0, i>=1)
  ///
  /// (the element_length factor on the derivative shape cancels with the
  /// 1/r division for df_0 and df_i, but f_i picks up a 1/element_length.)
  /// r-derivatives pull in (1/element_length)^n chain-rule factors.
  ///
  /// Implemented for n in {0, 1, 2}; throws for higher orders (this matches
  /// the planned 'no Taylor pipeline' design where RadialBasis only needs
  /// these three orders).
  void eval_over_r(const arma::Col<T> & x, arma::Mat<T> & dnf_over_r, int n,
                   T element_length) const override {
    if (n < 0 || n > 2) {
      std::ostringstream oss;
      oss << "HIPBasis::eval_over_r: derivative order " << n
          << " not implemented (only 0, 1, 2 supported).\n";
      throw std::logic_error(oss.str());
    }
    if (this->enabled.n_elem == 0)
      throw std::logic_error("HIPBasis::eval_over_r: no surviving basis functions.\n");
    if (this->enabled(0) == 0)
      throw std::logic_error(
          "HIPBasis::eval_over_r requires drop_first(func=true, deriv=false): the "
          "function shape at x=-1 has B(-1) != 0 and B/r is singular at the origin.\n");

    const arma::Col<T> & x0_full = this->x0;
    const arma::Col<T> x0_red    = x0_full.subvec(1, x0_full.n_elem - 1);

    // Full LIP values+derivatives (for L_0)
    arma::Mat<T> Lf0, Lf1, Lf2;
    detail::eval_lip_prim_dnf<T>(x, x0_full, Lf0, 0);
    if (n >= 1) detail::eval_lip_prim_dnf<T>(x, x0_full, Lf1, 1);
    if (n >= 2) detail::eval_lip_prim_dnf<T>(x, x0_full, Lf2, 2);

    // Reduced LIP values+derivatives (for L_i^{(0)}, i >= 1)
    arma::Mat<T> Lr0, Lr1, Lr2;
    detail::eval_lip_prim_dnf<T>(x, x0_red, Lr0, 0);
    if (n >= 1) detail::eval_lip_prim_dnf<T>(x, x0_red, Lr1, 1);
    if (n >= 2) detail::eval_lip_prim_dnf<T>(x, x0_red, Lr2, 2);

    const T inv_h = T(1) / element_length;       // 1 / scaling_factor
    const T chain = std::pow(inv_h, n);          // (1/element_length)^n

    dnf_over_r.set_size(x.n_elem, this->enabled.n_elem);

    for (arma::uword k = 0; k < this->enabled.n_elem; ++k) {
      const arma::uword idx  = this->enabled(k);
      const arma::uword node = idx / 2;
      const bool        deriv_shape = (idx % 2) == 1;

      for (arma::uword ix = 0; ix < x.n_elem; ++ix) {
        const T xv = x(ix);
        T q;  // value of Q(x) for this shape (so that R(r) = prefactor * Q(x))

        if (node == 0 && deriv_shape) {
          // df_0(r)/r = L_0(x)^2.  Q(x) = L_0^2; prefactor (1/element_length)^n.
          const T L  = Lf0(ix, 0);
          if (n == 0) {
            q = L * L;
          } else if (n == 1) {
            const T Lp = Lf1(ix, 0);
            // d/dx [L^2] = 2 L L'
            q = T(2) * L * Lp;
          } else { // n == 2
            const T Lp  = Lf1(ix, 0);
            const T Lpp = Lf2(ix, 0);
            // d^2/dx^2 [L^2] = 2 (L')^2 + 2 L L''
            q = T(2) * Lp * Lp + T(2) * L * Lpp;
          }
          dnf_over_r(ix, k) = chain * q;
        } else if (node >= 1 && !deriv_shape) {
          // f_i(r)/r = (1/element_length) * (x+1) * B(x) * C(x) / (x_i+1)^2
          //   B(x) = 1 - 2 (x - x_i) lipxi_i,  B' = -2 lipxi_i,  B'' = 0
          //   C(x) = L_i^{(0)}(x)^2,  C' = 2 L L',  C'' = 2 (L')^2 + 2 L L''
          // Prefactor: (1/element_length)^(n+1)
          const arma::uword i_red = node - 1;
          const T xi_p1 = x0_full(node) + T(1);
          const T inv_xi_p1_sq = T(1) / (xi_p1 * xi_p1);
          const T xpx1 = xv + T(1);
          const T xmxi = xv - x0_full(node);
          const T lipxi_i = lipxi(node);
          const T B  = T(1) - T(2) * xmxi * lipxi_i;
          const T Bp = -T(2) * lipxi_i;
          const T L  = Lr0(ix, i_red);
          if (n == 0) {
            const T C = L * L;
            q = xpx1 * B * C;
          } else if (n == 1) {
            const T Lp = Lr1(ix, i_red);
            const T C  = L * L;
            const T Cp = T(2) * L * Lp;
            // d/dx [(x+1) B C] = 1*B*C + (x+1)*Bp*C + (x+1)*B*Cp
            q = B * C + xpx1 * Bp * C + xpx1 * B * Cp;
          } else { // n == 2
            const T Lp  = Lr1(ix, i_red);
            const T Lpp = Lr2(ix, i_red);
            const T C   = L * L;
            const T Cp  = T(2) * L * Lp;
            const T Cpp = T(2) * Lp * Lp + T(2) * L * Lpp;
            // d^2/dx^2 [(x+1) B C] with A=x+1, A''=0, A'=1, B''=0:
            //   = 2*A'*Bp*C + 2*A'*B*Cp + (x+1)*0*C + 2*(x+1)*Bp*Cp + (x+1)*B*Cpp
            //   = 2 Bp C + 2 B Cp + 2 (x+1) Bp Cp + (x+1) B Cpp
            q = T(2) * Bp * C + T(2) * B * Cp + T(2) * xpx1 * Bp * Cp + xpx1 * B * Cpp;
          }
          dnf_over_r(ix, k) = inv_h * chain * q * inv_xi_p1_sq;
        } else if (node >= 1 && deriv_shape) {
          // df_i(r)/r = (x+1)(x-x_i) * C(x) / (x_i+1)^2
          //   D(x) = (x+1)(x-x_i),  D' = 2x + 1 - x_i,  D'' = 2
          //   C(x) = L_i^{(0)}(x)^2 as above
          // Prefactor: (1/element_length)^n
          const arma::uword i_red = node - 1;
          const T xi    = x0_full(node);
          const T xi_p1 = xi + T(1);
          const T inv_xi_p1_sq = T(1) / (xi_p1 * xi_p1);
          const T xpx1 = xv + T(1);
          const T xmxi = xv - xi;
          const T D  = xpx1 * xmxi;
          const T Dp = T(2) * xv + T(1) - xi;
          const T Dpp = T(2);
          const T L  = Lr0(ix, i_red);
          if (n == 0) {
            q = D * L * L;
          } else if (n == 1) {
            const T Lp = Lr1(ix, i_red);
            const T C  = L * L;
            const T Cp = T(2) * L * Lp;
            q = Dp * C + D * Cp;
          } else { // n == 2
            const T Lp  = Lr1(ix, i_red);
            const T Lpp = Lr2(ix, i_red);
            const T C   = L * L;
            const T Cp  = T(2) * L * Lp;
            const T Cpp = T(2) * Lp * Lp + T(2) * L * Lpp;
            q = Dpp * C + T(2) * Dp * Cp + D * Cpp;
          }
          dnf_over_r(ix, k) = chain * q * inv_xi_p1_sq;
        } else {
          // node == 0 && !deriv_shape: function shape at node 0 was supposed to
          // be dropped (the enabled(0)==0 check above guards this).
          throw std::logic_error(
              "HIPBasis::eval_over_r: encountered the function shape at node 0 "
              "in enabled set; drop_first(true, false) must be called.\n");
        }
      }
    }
  }
};

} // namespace polynomial_basis
} // namespace lib1dfem
} // namespace helfem

#endif
