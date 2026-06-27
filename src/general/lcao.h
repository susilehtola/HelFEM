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
#ifndef LCAO_H
#define LCAO_H

#include <armadillo>

namespace helfem {
  namespace lcao {
    /// Evaluate radial GTO
    double radial_GTO(double r, int l, double alpha);
    /// Evaluate radial GTO
    arma::mat radial_GTO(const arma::vec & r, int l, const arma::vec & alpha);
    /// Evaluate radial STO
    double radial_STO(double r, int l, double zeta);
    /// Evaluate radial STO
    arma::mat radial_STO(const arma::vec & r, int l, const arma::vec & zeta);
  }
}

#endif
