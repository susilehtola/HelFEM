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
#ifndef LIB1DFEM_POLYNOMIALBASIS_H
#define LIB1DFEM_POLYNOMIALBASIS_H

#include <lib1dfem/types.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include <fstream>

namespace helfem {
namespace lib1dfem {
namespace polynomial_basis {

/// Abstract template for a primitive polynomial basis defined on the
/// reference element [-1, 1]. Concrete subclasses (LIPBasis, HIPBasis,
/// GeneralHIPBasis, LegendreBasis) implement eval_prim_dnf, copy,
/// drop_first, drop_last and may override get_nodes.
///
/// Templated on the scalar type T (Phase 5.2: lib1dfem migrated arma -> Eigen).
template <typename T>
class PolynomialBasis {
 protected:
  /// Number of primitive functions
  int nprim = 0;
  /// List of enabled functions (column indices into the primitive set).
  IVec enabled;
  /// Number of overlapping functions (between adjacent elements)
  int noverlap = 0;
  /// Identifier (used by factories and downstream metadata)
  int id = 0;
  /// Number of nodes
  int nnodes = 0;

  /// Evaluate nth derivatives of primitive polynomials at given points.
  /// Default throws -- concrete subclasses must override.
  virtual void eval_prim_dnf(const Vec<T> & x, Mat<T> & dnf,
                             int n, T element_length) const {
    (void)x; (void)dnf; (void)n; (void)element_length;
    throw std::logic_error(
        "Values haven't been implemented for the used family of basis polynomials.\n");
  }

 public:
  PolynomialBasis() = default;
  virtual ~PolynomialBasis() = default;

  /// Polymorphic clone.
  virtual PolynomialBasis * copy() const = 0;

  int get_nprim()    const { return nprim; }
  int get_nbf()      const { return static_cast<int>(enabled.size()); }
  int get_noverlap() const { return noverlap; }
  int get_id()       const { return id; }
  int get_nnodes()   const { return nnodes; }

  /// Default node set is the reference element boundary {-1, +1};
  /// concrete subclasses (LIP/HIP) override with the actual node set.
  virtual Vec<T> get_nodes() const {
    Vec<T> n(2);
    n(0) = T(-1);
    n(1) = T(1);
    return n;
  }

  IVec get_enabled() const { return enabled; }

  /// Drop first function(s); zero_deriv: also set derivatives to zero.
  virtual void drop_first(bool zero_func, bool zero_deriv) = 0;
  /// Drop last function(s); zero_deriv: also set derivatives to zero.
  virtual void drop_last(bool zero_func, bool zero_deriv) = 0;

  /// Evaluate nth derivatives of polynomials at given points; applies the
  /// element_length^-n chain-rule scaling and restricts to enabled
  /// functions.
  void eval_dnf(const Vec<T> & x, Mat<T> & dnf, int n,
                T element_length) const {
    Mat<T> prim;
    eval_prim_dnf(x, prim, n, element_length);
    // Column subset using Eigen 3.4 indexing: prim(all, enabled).
    dnf = prim(Eigen::all, enabled) / std::pow(element_length, n);
  }

  Mat<T> eval_dnf(const Vec<T> & x, int n, T element_length) const {
    Mat<T> dnf;
    eval_dnf(x, dnf, n, element_length);
    return dnf;
  }

  /// Evaluate n-th derivative (w.r.t. r) of B_u(r)/r for every enabled
  /// shape function on the FIRST element [0, element_length], where x is in
  /// reference coords [-1, +1] and r = (element_length/2) * (x+1).
  ///
  /// Precondition: drop_first(zero_func=true, ...) must have been called so
  /// that every surviving shape function B_u satisfies B_u(x=-1) = 0; without
  /// that, B_u(r)/r has a 1/r singularity at the origin and this routine is
  /// ill-defined.
  ///
  /// This routine replaces the Taylor-cutoff machinery in RadialBasis for
  /// the small-r region: it computes B(r)/r analytically by deflating the
  /// (x+1) factor that the Dirichlet BC guarantees.
  ///
  /// Default implementation throws; concrete subclasses provide the analytic
  /// deflation when they support it (LIP and HIP today).
  virtual void eval_over_r(const Vec<T> & x, Mat<T> & dnf_over_r,
                           int n, T element_length) const {
    (void)x; (void)dnf_over_r; (void)n; (void)element_length;
    throw std::logic_error(
        "eval_over_r is not implemented for this PolynomialBasis subclass.\n");
  }

  Mat<T> eval_over_r(const Vec<T> & x, int n, T element_length) const {
    Mat<T> dnf_over_r;
    eval_over_r(x, dnf_over_r, n, element_length);
    return dnf_over_r;
  }

  /// Diagnostic dump of basis functions (and first derivatives) sampled
  /// on a fine grid; writes "bf<str>.dat" / "df<str>.dat" as plain text.
  void print(const std::string & str = "") const {
    Vec<T> x = Vec<T>::LinSpaced(1001, T(-1), T(1));
    Mat<T> bf, df;
    eval_dnf(x, bf, 0, T(1));
    eval_dnf(x, df, 1, T(1));
    auto dump = [&](const std::string & fname, const Mat<T> & m) {
      std::ofstream os(fname);
      for (Eigen::Index i = 0; i < x.size(); ++i) {
        os << x(i);
        for (Eigen::Index j = 0; j < m.cols(); ++j) os << " " << m(i, j);
        os << "\n";
      }
    };
    dump("bf" + str + ".dat", bf);
    dump("df" + str + ".dat", df);
  }
};

} // namespace polynomial_basis
} // namespace lib1dfem
} // namespace helfem

#endif
