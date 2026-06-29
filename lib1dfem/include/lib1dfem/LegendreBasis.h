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
#ifndef LIB1DFEM_LEGENDREBASIS_H
#define LIB1DFEM_LEGENDREBASIS_H

#include <lib1dfem/PolynomialBasis.h>
#include <lib1dfem/legendre_poly.h>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace helfem {
namespace lib1dfem {
namespace polynomial_basis {

/// Legendre spectral element basis. Phase 5.2: Eigen-typed.
template <typename T>
class LegendreBasis : public PolynomialBasis<T> {
 protected:
  /// Maximum polynomial order
  int lmax;
  /// (lmax+1) x (lmax+1) transformation from raw P_l to the shape-function basis
  Mat<T> Tmat;

  Mat<T> f_eval(const Vec<T> & x) const {
    return helfem::lib1dfem::legendre::legendre_batch<T>(lmax, x);
  }
  Mat<T> df_eval(const Vec<T> & x) const {
    return helfem::lib1dfem::legendre::dlegendre_batch<T>(lmax, x);
  }
  Mat<T> d2f_eval(const Vec<T> & x) const {
    return helfem::lib1dfem::legendre::d2legendre_batch<T>(lmax, x);
  }

 public:
  LegendreBasis(int n_nodes, int id_ = 3) {
    lmax = n_nodes - 1;
    Tmat = Mat<T>::Zero(lmax + 1, lmax + 1);
    // First shape function: (P_0 - P_1) / 2
    Tmat(0, 0)    = T(1) / T(2);
    Tmat(1, 0)    = -T(1) / T(2);
    // Last shape function: (P_0 + P_1) / 2
    Tmat(0, lmax) = T(1) / T(2);
    Tmat(1, lmax) = T(1) / T(2);
    // Interior shape functions
    for (int j = 1; j < lmax; ++j) {
      const T sqfac = T(1) / std::sqrt(T(4 * j + 2));
      Tmat(j + 1, j) = sqfac;
      Tmat(j - 1, j) = -sqfac;
    }
    this->noverlap = 1;
    this->nprim    = static_cast<int>(Tmat.cols());
    this->nnodes   = static_cast<int>(Tmat.cols());
    this->enabled  = IVec::LinSpaced(Tmat.cols(), 0, Tmat.cols() - 1);
    this->id       = id_;
  }

  ~LegendreBasis() override = default;

  LegendreBasis<T> * copy() const override {
    return new LegendreBasis<T>(*this);
  }

  void drop_first(bool func, bool deriv) override {
    (void)deriv;
    if (func)
      this->enabled = this->enabled.segment(1, this->enabled.size() - 1).eval();
  }
  void drop_last(bool func, bool deriv) override {
    (void)deriv;
    if (func)
      this->enabled = this->enabled.segment(0, this->enabled.size() - 1).eval();
  }

  void eval_prim_dnf(const Vec<T> & x, Mat<T> & dnf, int n,
                     T element_length) const override {
    (void)element_length;
    switch (n) {
      case 0: dnf = f_eval(x)  * Tmat; break;
      case 1: dnf = df_eval(x) * Tmat; break;
      case 2: dnf = d2f_eval(x) * Tmat; break;
      default: {
        std::ostringstream oss;
        oss << n << "th order derivatives not implemented for Legendre basis functions!\n";
        throw std::logic_error(oss.str());
      }
    }
  }

  // Pull the base's 3-arg eval_over_r overload back into scope.
  using PolynomialBasis<T>::eval_over_r;

  /// Analytic B_u(r)/r and r-derivatives for the surviving Legendre shape
  /// functions on the first element (r = element_length * (x+1)).
  /// See v1 code header comment for the deflation derivation.
  void eval_over_r(const Vec<T> & x, Mat<T> & dnf_over_r, int n,
                   T element_length) const override {
    if (n < 0 || n > 2) {
      std::ostringstream oss;
      oss << "LegendreBasis::eval_over_r: derivative order " << n
          << " not implemented (only 0, 1, 2 supported).\n";
      throw std::logic_error(oss.str());
    }
    if (this->enabled.size() == 0)
      throw std::logic_error("LegendreBasis::eval_over_r: no surviving basis functions.\n");
    if (this->enabled(0) == 0)
      throw std::logic_error(
          "LegendreBasis::eval_over_r requires drop_first(zero_func=true): "
          "the value-shape (1-x)/2 has B(-1)=1 and B/r is singular at the origin.\n");

    // Build Q_n, Q'_n, Q''_n for n = 0..lmax at every integration point.
    Mat<T> Q   = Mat<T>::Zero(x.size(), lmax + 1);
    Mat<T> Qp  = Mat<T>::Zero(x.size(), lmax + 1);
    Mat<T> Qpp = Mat<T>::Zero(x.size(), lmax + 1);
    if (lmax >= 1)
      for (Eigen::Index ix = 0; ix < x.size(); ++ix)
        Q(ix, 1) = T(1);
    for (int k = 1; k < lmax; ++k) {
      const T inv_kp1  = T(1) / T(k + 1);
      const T two_k_p1 = T(2 * k + 1);
      const T k_T      = T(k);
      const T sign_k   = (k % 2 == 0) ? T(1) : T(-1);
      for (Eigen::Index ix = 0; ix < x.size(); ++ix) {
        const T xv = x(ix);
        Q  (ix, k + 1) = (two_k_p1 * xv * Q  (ix, k) - k_T * Q  (ix, k - 1)
                          + sign_k * two_k_p1) * inv_kp1;
        Qp (ix, k + 1) = (two_k_p1 * (Q (ix, k) + xv * Qp (ix, k))
                          - k_T * Qp (ix, k - 1)) * inv_kp1;
        Qpp(ix, k + 1) = (two_k_p1 * (T(2) * Qp(ix, k) + xv * Qpp(ix, k))
                          - k_T * Qpp(ix, k - 1)) * inv_kp1;
      }
    }

    const T inv_e   = T(1) / element_length;
    const T chain_n = std::pow(inv_e, n);

    dnf_over_r.resize(x.size(), this->enabled.size());
    for (Eigen::Index k = 0; k < this->enabled.size(); ++k) {
      const Eigen::Index j = this->enabled(k);
      if (j == static_cast<Eigen::Index>(lmax)) {
        // Last shape: R(r) = 1/(2 e); derivatives vanish.
        const T val0 = T(1) / (T(2) * element_length);
        for (Eigen::Index ix = 0; ix < x.size(); ++ix)
          dnf_over_r(ix, k) = (n == 0) ? val0 : T(0);
      } else {
        // Interior j in [1, lmax-1].
        const T inv_norm = T(1) / std::sqrt(T(4) * T(j) + T(2));
        for (Eigen::Index ix = 0; ix < x.size(); ++ix) {
          T R;
          if      (n == 0) R = (Q  (ix, j + 1) - Q  (ix, j - 1)) * inv_norm * inv_e;
          else if (n == 1) R = (Qp (ix, j + 1) - Qp (ix, j - 1)) * inv_norm * inv_e;
          else             R = (Qpp(ix, j + 1) - Qpp(ix, j - 1)) * inv_norm * inv_e;
          dnf_over_r(ix, k) = chain_n * R;
        }
      }
    }
  }
};

} // namespace polynomial_basis
} // namespace lib1dfem
} // namespace helfem

#endif
