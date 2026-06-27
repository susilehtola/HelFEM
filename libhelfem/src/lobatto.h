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
#ifndef LOBATTO_H
#define LOBATTO_H

#include <armadillo>

/// Compute a Gauss-Lobatto quadrature rule for \f$ \int_{-1}^1 f(x)dx \approx \frac 2 {n(n-1)} \left[ f(-1) + f(1) \right] + \sum_{i=2}^{n-1} w_i f(x_i) \f$
void lobatto_compute ( int n, arma::vec & x, arma::vec & w);

#endif
