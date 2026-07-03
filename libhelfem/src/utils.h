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
#ifndef UTILS_H
#define UTILS_H

#include <armadillo>
#include "../include/Matrix.h"

namespace helfem {
  namespace utils {
    /// inverse cosh
    double arcosh(double x);

    /// Modified Bessel function
    double bessel_il(double x, int L);
    /// Modified Bessel function
    double bessel_kl(double x, int L);

    /// Form two-electron integrals from product of large-r and small-r radial moment matrices
    arma::mat product_tei(const arma::mat & big, const arma::mat & small);

    /// Check that the two-electron integral has proper symmetry i<->j and k<->l
    void check_tei_symmetry(const arma::mat & tei, size_t Ni, size_t Nj, size_t Nk, size_t Nl);

    /// Permute indices (ij|kl) -> (jk|il)
    arma::mat exchange_tei(const arma::mat & tei, size_t Ni, size_t Nj, size_t Nk, size_t Nl);
    /// Eigen overload of the same permutation (Phase 2c cleanup -- lets
    /// the libhelfem 2e helpers skip the to_eigen(to_arma(...)) round-
    /// trip that bridged the old arma-only function).
    helfem::Matrix exchange_tei(const helfem::Matrix & tei,
                                 size_t Ni, size_t Nj, size_t Nk, size_t Nl);

    /// Case independent string comparison
    int stricmp(const std::string & str1, const std::string & str2);
  }
}

#endif
