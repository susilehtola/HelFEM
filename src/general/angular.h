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
#ifndef ANGULAR_H
#define ANGULAR_H

#include <armadillo>

namespace helfem {
  namespace angular {
    /// Form compound rule
    void compound_rule(const arma::vec & xth0, const arma::vec & wth0, int nphi, arma::vec & th, arma::vec & phi, arma::vec & w);

    /// Angular quadrature rule of order (l,l)
    void angular_lobatto(int l, arma::vec & cth, arma::vec & phi, arma::vec & w);
    /// Angular quadrature rule of order (l,m)
    void angular_lobatto(int l, int m, arma::vec & cth, arma::vec & phi, arma::vec & w);

    /// Angular quadrature rule of order (l,l)
    void angular_chebyshev(int l, arma::vec & cth, arma::vec & phi, arma::vec & w);
    /// Angular quadrature rule of order (l,m)
    void angular_chebyshev(int l, int m, arma::vec & cth, arma::vec & phi, arma::vec & w);
  }
}

#endif
