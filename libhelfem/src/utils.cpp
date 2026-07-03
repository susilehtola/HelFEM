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
#include <lib1dfem/math.h>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <cmath>

namespace helfem {
  namespace utils {
    // Scalar-only wrappers around the templated lib1dfem math helpers.
    double arcosh(double x) {
      return helfem::lib1dfem::math::arcosh<double>(x);
    }
    double bessel_il(double r, int L) {
      return helfem::lib1dfem::math::bessel_il<double>(r, L);
    }
    double bessel_kl(double r, int L) {
      return helfem::lib1dfem::math::bessel_kl<double>(r, L);
    }

    helfem::Matrix exchange_tei(const helfem::Matrix & tei,
                                 size_t Ni, size_t Nj, size_t Nk, size_t Nl) {
      // Eigen overload of the same scalar-by-scalar (ij|kl) -> (jk|il)
      // permutation. Both arma and Eigen are column-major so the
      // packed-pair index layout (a-fast, b-slow) is identical.
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
      helfem::Matrix ktei = helfem::Matrix::Zero(Nj*Nk, Ni*Nl);
      for (size_t ii = 0; ii < Ni; ++ii)
        for (size_t jj = 0; jj < Nj; ++jj)
          for (size_t kk = 0; kk < Nk; ++kk)
            for (size_t ll = 0; ll < Nl; ++ll)
              ktei(kk*Nj+jj, ll*Ni+ii) = tei(jj*Ni+ii, ll*Nk+kk);
      return ktei;
    }

    int stricmp(const std::string & str1, const std::string & str2) {
      return strcasecmp(str1.c_str(),str2.c_str());
    }

    // Phase 5.10: invh migrated to Eigen.
    helfem::Matrix invh(helfem::Matrix S, bool chol) {
      // Basis function norms: 1 / sqrt(diag(S))
      const helfem::Vector bfnormlz = S.diagonal().array().pow(-0.5).matrix();

      // Go to normalized basis: S -> diag(bfnormlz) S diag(bfnormlz)
      S = bfnormlz.asDiagonal() * S * bfnormlz.asDiagonal();

      helfem::Matrix Sinvh;
      if (chol) {
        // Sinvh = inv(chol(S))  -- upper-triangular L from LLT, inverted.
        Eigen::LLT<helfem::Matrix> llt(S);
        if (llt.info() != Eigen::Success)
          throw std::logic_error("Cholesky decomposition of overlap matrix failed\n");
        // arma::chol(S) returns the upper triangular U with U^T U = S;
        // Eigen LLT stores L (lower) with L L^T = S. To mirror arma we
        // take L^T then invert.
        const helfem::Matrix U = llt.matrixL().transpose();
        Sinvh = U.inverse();
      } else {
        Eigen::SelfAdjointEigenSolver<helfem::Matrix> es(S);
        if (es.info() != Eigen::Success)
          throw std::logic_error("Diagonalization of overlap matrix failed\n");
        const helfem::Vector Sval = es.eigenvalues();
        const helfem::Matrix Svec = es.eigenvectors();
        printf("Smallest eigenvalue of overlap matrix is % e, condition number %e\n",
               Sval(0), Sval(Sval.size() - 1) / Sval(0));
        Sinvh = Svec * Sval.array().pow(-0.5).matrix().asDiagonal() * Svec.transpose();
      }

      Sinvh = bfnormlz.asDiagonal() * Sinvh;
      return Sinvh;
    }
  }
}
