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
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

// Adapter to instantiate helfem::legendre::plm/qlm with GCC's __float128.
//
// As of GCC 16, libstdc++ does not provide unambiguous std::log /sqrt /cosh
// /abs overloads for __float128 (or for _Float128) — std overload resolution
// either matches multiple candidates or none. We sidestep that by including
// this header BEFORE Legendre.h in any TU that wants the __float128
// instantiation; this header specialises the per-type math wrappers in
// helfem::legendre::detail to libquadmath equivalents.
//
// std::numeric_limits<__float128> is already specialised by libstdc++.
//
// Linking the TU requires -lquadmath.

#ifndef HELFEM_LEGENDRE_FLOAT128_H
#define HELFEM_LEGENDRE_FLOAT128_H

#if !defined(__SIZEOF_FLOAT128__)
#  error "__float128 is not available on this compiler"
#endif

#include <quadmath.h>
#include "Legendre.h"

namespace helfem {
namespace legendre {
namespace detail {

template <> inline __float128 m_log <__float128>(__float128 x) { return logq (x); }
template <> inline __float128 m_sqrt<__float128>(__float128 x) { return sqrtq(x); }
template <> inline __float128 m_cosh<__float128>(__float128 x) { return coshq(x); }
template <> inline __float128 m_abs <__float128>(__float128 x) { return fabsq(x); }

}  // namespace detail
}  // namespace legendre
}  // namespace helfem

#endif
