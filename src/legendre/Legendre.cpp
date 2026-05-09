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

// Explicit instantiations of the templated implementation in Legendre.h.
// All the algorithmic content lives in the header so users can instantiate
// for further numeric types (boost::multiprecision, etc.) just by including
// it, without rebuilding the library.

#include "Legendre.h"

namespace helfem {
namespace legendre {

template void plm<double>(double *, int, int, double);
template void qlm<double>(double *, int, int, double);
template void plm<long double>(long double *, int, int, long double);
template void qlm<long double>(long double *, int, int, long double);

}  // namespace legendre
}  // namespace helfem
