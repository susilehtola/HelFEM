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
#ifndef CHEBYSHEV_H
#define CHEBYSHEV_H

#include <armadillo>

namespace helfem {
  namespace chebyshev {
    /**
       Modified Gauss-Chebyshev quadrature of the second kind for calculating
       \f$ \int_{-1}^{1} f(x) dx \f$
    */
    void chebyshev(int n, arma::vec & x, arma::vec & w);

    /// Modified Gauss-Chebyshev quadrature of the second kind for
    /// calculating \f$\int_{0}^{\infty} f(r) dr\f$. NB! For
    /// integration in spherical coordinates, you need to plug in the
    /// r^2 factor as well.
    void radial_chebyshev(int n, arma::vec & r, arma::vec & wr);
  }
}

#endif
