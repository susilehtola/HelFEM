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

#include "../include/Matrix.h"
#include <string>

namespace helfem {
  namespace utils {
    /// inverse cosh
    double arcosh(double x);

    /// Modified Bessel function
    double bessel_il(double x, int L);
    /// Modified Bessel function
    double bessel_kl(double x, int L);

    /// Permute indices (ij|kl) -> (jk|il).
    helfem::Matrix exchange_tei(const helfem::Matrix & tei,
                                 size_t Ni, size_t Nj, size_t Nk, size_t Nl);

    /// Case independent string comparison
    int stricmp(const std::string & str1, const std::string & str2);
  }
}

#endif
