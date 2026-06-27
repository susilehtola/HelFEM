/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Susi Lehtola
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * See the LICENSE file at the root of this source distribution
 * for the full license text.
 */

#ifndef ERKALE_SPHHARM
#define ERKALE_SPHHARM

#include <complex>

/// Calculate value of \f$ Y_{l}^{m} (\cos \theta, \phi) = (-1)^m \sqrt{ \frac {2l +1} {4 \pi} \frac {(l-m)!} {(l+m)!} } P_l^m (\cos \theta) e^{i m \phi} \f$
std::complex<double> spherical_harmonics(int l, int m, double cth, double phi);

#endif
