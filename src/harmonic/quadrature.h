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
#ifndef INTEGRALS_H
#define INTEGRALS_H

#include <armadillo>
#include "FiniteElementBasis.h"

namespace helfem {
  namespace quadrature {
    /**
     * Computes an integral of the type x^n B_1 (x) B_2(x) dx.
     *
     * Input
     *   xmin: start of element boundary
     *   xmax: end of element boundary
     *       x: integration nodes
     *      wx: integration weights
     *      bf: basis functions evaluated at integration nodes.
     */
    arma::mat radial_integral(const helfem::polynomial_basis::FiniteElementBasis & fem, size_t iel, int n, const arma::vec & xq, const arma::vec & wxq);
    
    /**
     * Computes a derivative matrix element of the type
     * dB_1(x)/dx dB_2/dx dx
     */
    arma::mat derivative_integral(const helfem::polynomial_basis::FiniteElementBasis & fem, size_t iel, const arma::vec & xq, const arma::vec & wxq);
  }
}

#endif
