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

namespace helfem {
  namespace utils {
    /// inverse cosh
    double arcosh(double x);
    /// inverse cosh
    arma::vec arcosh(const arma::vec & x);
    /// inverse sinh
    double arsinh(double x);
    /// inverse sinh
    arma::vec arsinh(const arma::vec & x);

    /// Modified Bessel function
    double bessel_il(double x, int L);
    /// Modified Bessel function
    arma::vec bessel_il(const arma::vec & x, int L);
    /// Modified Bessel function
    double bessel_kl(double x, int L);
    /// Modified Bessel function
    arma::vec bessel_kl(const arma::vec & x, int L);

    /// Form two-electron integrals from product of large-r and small-r radial moment matrices
    arma::mat product_tei(const arma::mat & big, const arma::mat & small);

    /// Check that the two-electron integral has proper symmetry i<->j and k<->l
    void check_tei_symmetry(const arma::mat & tei, size_t Ni, size_t Nj, size_t Nk, size_t Nl);

    /// Permute indices (ij|kl) -> (jk|il)
    arma::mat exchange_tei(const arma::mat & tei, size_t Ni, size_t Nj, size_t Nk, size_t Nl);

    /// Case independent string comparison
    int stricmp(const std::string & str1, const std::string & str2);
  }
}

#endif
