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

    // Mathematical constants at the working precision. <cmath>'s M_PI /
    // M_2_SQRTPI are double-precision macros, which would silently pin a
    // long double computation to double accuracy. The long-double literals
    // below round to exactly M_PI / M_2_SQRTPI at T = double, so the
    // double instantiation is bit-identical to the pre-template code.
    /// pi
    template <typename T> inline T pi() {
      return T(3.14159265358979323846264338327950288419716939937510L);
    }
    /// 2/sqrt(pi)
    template <typename T> inline T two_over_sqrtpi() {
      return T(1.12837916709551257389615890312154517168810125865800L);
    }

    /// Modified Bessel function
    template <typename T> T bessel_il(T x, int L);
    /// Modified Bessel function
    template <typename T> T bessel_kl(T x, int L);

    /// Permute indices (ij|kl) -> (jk|il).
    helfem::Matrix exchange_tei(const helfem::Matrix & tei,
                                 size_t Ni, size_t Nj, size_t Nk, size_t Nl);

    /// Case independent string comparison
    int stricmp(const std::string & str1, const std::string & str2);
  }
}

#endif
