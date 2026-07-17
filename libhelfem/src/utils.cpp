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
#include "utils.h"
#include <math.h>
// Scalar formatter that prints a T value at its own precision (no truncation
// to double). Header-only, needs only Matrix.h + std; see src/general/eigen_io.h.
#include "../../src/general/eigen_io.h"
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <cmath>
#include <type_traits>

namespace helfem {
  namespace utils {
    // Thin wrappers around the templated FEM math helpers.
    // arcosh is only ever used from the diatomic driver at double, so it
    // stays a scalar function; the Bessel functions are called from the
    // templated radial layer and follow the scalar type.
    double arcosh(double x) {
      return helfem::math::arcosh<double>(x);
    }
    template <typename T> T bessel_il(T r, int L) {
      return helfem::math::bessel_il<T>(r, L);
    }
    template <typename T> T bessel_kl(T r, int L) {
      return helfem::math::bessel_kl<T>(r, L);
    }

    template double      bessel_il<double>     (double, int);
    template long double bessel_il<long double>(long double, int);
#ifdef HELFEM_HAVE_FLOAT128
    template _Float128 bessel_il<_Float128>(_Float128, int);
#endif
    template double      bessel_kl<double>     (double, int);
    template long double bessel_kl<long double>(long double, int);
#ifdef HELFEM_HAVE_FLOAT128
    template _Float128 bessel_kl<_Float128>(_Float128, int);
#endif

    template <typename T>
    helfem::Mat<T> exchange_tei(const helfem::Mat<T> & tei,
                                size_t Ni, size_t Nj, size_t Nk, size_t Nl) {
      // Eigen overload of the same scalar-by-scalar (ij|kl) -> (jk|il)
      // permutation. Both arma and Eigen are column-major so the
      // packed-pair index layout (a-fast, b-slow) is identical. Pure
      // index shuffling -- no arithmetic -- so it is exact at any T.
      if (static_cast<size_t>(tei.rows()) != Ni*Nj) {
        std::ostringstream oss;
        oss << "Invalid input tei: was supposed to get " << Ni*Nj
            << " rows but got " << tei.rows() << "!\n";
        throw std::logic_error(oss.str());
      }
      if (static_cast<size_t>(tei.cols()) != Nk*Nl) {
        std::ostringstream oss;
        oss << "Invalid input tei: was supposed to get " << Nk*Nl
            << " cols but got " << tei.cols() << "!\n";
        throw std::logic_error(oss.str());
      }
      helfem::Mat<T> ktei = helfem::Mat<T>::Zero(Nj*Nk, Ni*Nl);
      for (size_t ii = 0; ii < Ni; ++ii)
        for (size_t jj = 0; jj < Nj; ++jj)
          for (size_t kk = 0; kk < Nk; ++kk)
            for (size_t ll = 0; ll < Nl; ++ll)
              ktei(kk*Nj+jj, ll*Ni+ii) = tei(jj*Ni+ii, ll*Nk+kk);
      return ktei;
    }

    template helfem::Mat<double>      exchange_tei<double>     (const helfem::Mat<double> &,      size_t, size_t, size_t, size_t);
    template helfem::Mat<long double> exchange_tei<long double>(const helfem::Mat<long double> &, size_t, size_t, size_t, size_t);
#ifdef HELFEM_HAVE_FLOAT128
    template helfem::Mat<_Float128>   exchange_tei<_Float128>  (const helfem::Mat<_Float128> &,   size_t, size_t, size_t, size_t);
#endif

    int stricmp(const std::string & str1, const std::string & str2) {
      return strcasecmp(str1.c_str(),str2.c_str());
    }

    namespace {
      // Elementwise x^(-1/2).
      //
      // Eigen's Array::pow(scalar) is SFINAE-gated on Eigen's OWN
      // internal::is_arithmetic trait, whose list of floating types does not
      // include _Float128, so the templated invh below cannot spell it that
      // way for every T. double keeps the original .pow(-0.5) expression
      // verbatim -- std::pow(x, -0.5) and 1/std::sqrt(x) may differ in the
      // last ulp, and the point of this whole exercise is that the double
      // results stay bit-for-bit what they were.
      template <typename T>
      helfem::Vec<T> inv_sqrt(const helfem::Vec<T> & v) {
        if constexpr (std::is_same_v<T, double>) {
          return v.array().pow(-0.5).matrix();
        } else {
          return v.unaryExpr([](T x) { return T(1) / std::sqrt(x); });
        }
      }
    }

    // Phase 5.10: invh migrated to Eigen. Now templated on the scalar type:
    // the symmetric orthonormalization sits directly on the SCF path, so
    // pinning it to double would cap an otherwise higher-precision
    // calculation right where it matters. Eigen's LLT and
    // SelfAdjointEigenSolver are themselves generic in the scalar, so the
    // body is unchanged apart from the types -- and the T = double
    // instantiation is bit-identical to the pre-template code (see inv_sqrt
    // above, which preserves the original .pow(-0.5) spelling at double).
    template <typename T>
    helfem::Mat<T> invh(helfem::Mat<T> S, bool chol) {
      // Basis function norms: 1 / sqrt(diag(S))
      const helfem::Vec<T> Sdiag = S.diagonal();
      const helfem::Vec<T> bfnormlz = inv_sqrt<T>(Sdiag);

      // Go to normalized basis: S -> diag(bfnormlz) S diag(bfnormlz)
      S = bfnormlz.asDiagonal() * S * bfnormlz.asDiagonal();

      helfem::Mat<T> Sinvh;
      if (chol) {
        // Sinvh = inv(chol(S))  -- upper-triangular L from LLT, inverted.
        Eigen::LLT<helfem::Mat<T>> llt(S);
        if (llt.info() != Eigen::Success)
          throw std::logic_error("Cholesky decomposition of overlap matrix failed\n");
        // arma::chol(S) returns the upper triangular U with U^T U = S;
        // Eigen LLT stores L (lower) with L L^T = S. To mirror arma we
        // take L^T then invert.
        const helfem::Mat<T> U = llt.matrixL().transpose();
        Sinvh = U.inverse();
      } else {
        Eigen::SelfAdjointEigenSolver<helfem::Mat<T>> es(S);
        if (es.info() != Eigen::Success)
          throw std::logic_error("Diagonalization of overlap matrix failed\n");
        const helfem::Vec<T> Sval = es.eigenvalues();
        const helfem::Mat<T> Svec = es.eigenvectors();
        // Format the T eigenvalues at their own precision (no truncation to
        // double). Sval(0) is the smallest eigenvalue of the (SPD) overlap
        // matrix and hence positive, so the leading " " reproduces the old
        // "% e" space flag exactly.
        printf("Smallest eigenvalue of overlap matrix is %s, condition number %s\n",
               (" " + helfem::io::fmt_sci(Sval(0))).c_str(),
               helfem::io::fmt_sci(Sval(Sval.size() - 1) / Sval(0)).c_str());
        Sinvh = Svec * inv_sqrt<T>(Sval).asDiagonal() * Svec.transpose();
      }

      Sinvh = bfnormlz.asDiagonal() * Sinvh;
      return Sinvh;
    }

    template helfem::Mat<double>      invh<double>     (helfem::Mat<double>,      bool);
    template helfem::Mat<long double> invh<long double>(helfem::Mat<long double>, bool);
#ifdef HELFEM_HAVE_FLOAT128
    template helfem::Mat<_Float128>   invh<_Float128>  (helfem::Mat<_Float128>,   bool);
#endif
  }
}
