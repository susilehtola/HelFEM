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
#include <armadillo>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace helfem {
namespace lib1dfem {
namespace polynomial_basis {

/// Legendre spectral element basis. The first and last shape functions
/// are linear blends of P_0 and P_1 (the standard nodal bubble functions);
/// interior shape functions are the orthogonal combinations
///     (P_{j+1} - P_{j-1}) / sqrt(4 j + 2)
/// from Flores, Clementi & Sonnad, Chem. Phys. Lett. 163, 198 (1989).
///
/// Templated on the scalar type T.
template <typename T>
class LegendreBasis : public PolynomialBasis<T> {
 protected:
  /// Maximum polynomial order
  int lmax;
  /// (lmax+1) x (lmax+1) transformation from raw P_l to the shape-function basis
  arma::Mat<T> Tmat;

  arma::Mat<T> f_eval(const arma::Col<T> & x) const {
    return helfem::lib1dfem::legendre::legendre_batch<T>(lmax, x);
  }
  arma::Mat<T> df_eval(const arma::Col<T> & x) const {
    return helfem::lib1dfem::legendre::dlegendre_batch<T>(lmax, x);
  }
  arma::Mat<T> d2f_eval(const arma::Col<T> & x) const {
    return helfem::lib1dfem::legendre::d2legendre_batch<T>(lmax, x);
  }

 public:
  /// `n_nodes` here is the number of shape functions (= polynomial order + 1).
  /// `id_` is just an identifier echoed back through get_id(); the libhelfem
  /// factory passes the primbas value (default 3).
  LegendreBasis(int n_nodes, int id_ = 3) {
    lmax = n_nodes - 1;
    Tmat.zeros(lmax + 1, lmax + 1);
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
    this->nprim    = static_cast<int>(Tmat.n_cols);
    this->nnodes   = static_cast<int>(Tmat.n_cols);
    this->enabled  = arma::linspace<arma::uvec>(0, Tmat.n_cols - 1, Tmat.n_cols);
    this->id       = id_;
  }

  ~LegendreBasis() override = default;

  LegendreBasis<T> * copy() const override {
    return new LegendreBasis<T>(*this);
  }

  void drop_first(bool func, bool deriv) override {
    (void)deriv;
    if (func)
      this->enabled = this->enabled.subvec(1, this->enabled.n_elem - 1);
  }
  void drop_last(bool func, bool deriv) override {
    (void)deriv;
    if (func)
      this->enabled = this->enabled.subvec(0, this->enabled.n_elem - 2);
  }

  void eval_prim_dnf(const arma::Col<T> & x, arma::Mat<T> & dnf, int n,
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
};

} // namespace polynomial_basis
} // namespace lib1dfem
} // namespace helfem

#endif
