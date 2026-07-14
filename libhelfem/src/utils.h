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
    //
    // NOTE the `L` suffix: the literal is rounded to LONG DOUBLE (64-bit
    // mantissa, ~19 digits) BEFORE being converted to T. That is exactly right
    // for T = double and T = long double, but at T = _Float128 it throws away
    // 15 of the 34 digits the type can hold -- pi would come out wrong by
    // 1.6e-20 relative. These constants are physical (the 4*pi/(2L+1) of the
    // multipole expansion, the Gaussian-nucleus normalisation, the Gaunt
    // prefactors), not basis choices, so that error would land straight in the
    // Hamiltonian and cap the whole quad path at ~1e-19 no matter how good the
    // basis or the arithmetic. Hence the explicit _Float128 specialisations
    // below, which spell the SAME digits with the C++23 `f128` suffix so they
    // survive to the full 113-bit mantissa.
    /// pi
    template <typename T> inline T pi() {
      return T(3.14159265358979323846264338327950288419716939937510L);
    }
    /// 2/sqrt(pi)
    template <typename T> inline T two_over_sqrtpi() {
      return T(1.12837916709551257389615890312154517168810125865800L);
    }

#ifdef HELFEM_HAVE_FLOAT128
    template <> inline _Float128 pi<_Float128>() {
      return 3.14159265358979323846264338327950288419716939937510f128;
    }
    template <> inline _Float128 two_over_sqrtpi<_Float128>() {
      return 1.12837916709551257389615890312154517168810125865800f128;
    }
#endif

    /// Modified Bessel function
    template <typename T> T bessel_il(T x, int L);
    /// Modified Bessel function
    template <typename T> T bessel_kl(T x, int L);

    /// Permute indices (ij|kl) -> (jk|il).
    /// Templated on the scalar type: the in-element two-electron tensors
    /// follow FEMRadialBasisT<T>, so the exchange permutation must too.
    template <typename T>
    helfem::Mat<T> exchange_tei(const helfem::Mat<T> & tei,
                                size_t Ni, size_t Nj, size_t Nk, size_t Nl);

    /// Case independent string comparison
    int stricmp(const std::string & str1, const std::string & str2);
  }
}

#endif
