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
#ifndef HELFEM_SCF_DRIVER_COMMON_H
#define HELFEM_SCF_DRIVER_COMMON_H

// Shared building blocks for the atomic and diatomic SCF drivers.
// Extracted from src/atomic/main.cpp and src/diatomic/main.cpp
// (both files kept byte-identical copies of these).

#include <armadillo>
#include <cstdio>
#include <stdexcept>

namespace helfem {
  namespace scf_driver {
    /// In-place symmetric scaling M(i, j) *= norm(i) * norm(j),
    /// used by both drivers to renormalise Fock / density blocks.
    inline void normalize_matrix(arma::mat & M, const arma::vec & norm) {
      if (M.n_rows != norm.n_elem) throw std::logic_error("Incompatible dimensions!\n");
      if (M.n_cols != norm.n_elem) throw std::logic_error("Incompatible dimensions!\n");
      for (size_t i = 0; i < M.n_rows; ++i)
        for (size_t j = 0; j < M.n_cols; ++j)
          M(i, j) *= norm(i) * norm(j);
    }

    /// Gram-Schmidt reorthonormalise the first nocc columns of C
    /// against the overlap S. Used after re-loading orbitals from a
    /// checkpoint or projecting between bases so subsequent SCF
    /// iterations start from a strictly orthonormal set.
    inline void gram_schmidt(arma::mat & C, const arma::mat & S, int nocc) {
      for (int i = 0; i < nocc; ++i) {
        for (int j = 0; j < i; ++j)
          C.col(i) -= C.col(j) * (arma::trans(C.col(j)) * S * C.col(i));
        C.col(i) /= std::sqrt(arma::as_scalar(arma::trans(C.col(i)) * S * C.col(i)));
      }
    }

    /// Report orbital-orthonormality deviation ||Sinvh^T S Sinvh - I||_F.
    /// Both drivers print this immediately after Sinvh is formed.
    inline void report_ortho_deviation(const arma::mat & S, const arma::mat & Sinvh) {
      arma::mat Smo(Sinvh.t() * S * Sinvh);
      Smo -= arma::eye<arma::mat>(Smo.n_rows, Smo.n_cols);
      printf("Orbital orthonormality deviation is %e\n", arma::norm(Smo, "fro"));
    }

    /// Report half-overlap consistency ||Sh^T Sinvh - I||_F. Both
    /// drivers print this immediately after Sh is formed.
    inline void report_halfoverlap_error(const arma::mat & Sh, const arma::mat & Sinvh) {
      arma::mat Smo(Sh.t() * Sinvh);
      Smo -= arma::eye<arma::mat>(Smo.n_rows, Smo.n_cols);
      printf("Half-overlap error is %e\n", arma::norm(Smo, "fro"));
    }
  } // namespace scf_driver
} // namespace helfem

#endif
