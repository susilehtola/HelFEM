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
