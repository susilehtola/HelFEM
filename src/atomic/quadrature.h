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
#include "../general/polynomial_basis.h"
#include "../general/sap.h"

namespace helfem {
  namespace quadrature {
    /**
     * Computes a radial integral of the type \f$ \int_0^\infty B_1 (r) B_2(r) r^n dr \f$.
     *
     * Input
     *   rmin: start of element boundary
     *   rmax: end of element boundary
     *       x: integration nodes
     *      wx: integration weights
     *      bf: basis functions evaluated at integration nodes.
     */
    arma::mat radial_integral(double rmin, double rmax, int n, const arma::vec & x, const arma::vec & wx, const arma::mat & bf);

    /**
     * Computes a derivative matrix element of the type
     * r^2 dB_1(r)/dr dB_2/dr dr
     */
    arma::mat derivative_integral(double rmin, double rmax, const arma::vec & x, const arma::vec & wx, const arma::mat & dbf);

    /**
     * Computes a GSZ radial integral \f$ \int_0^\infty Z(r) B_1 (r) B_2(r) / r dr \f$.
     *
     * Input
     *   rmin: start of element boundary
     *   rmax: end of element boundary
     *       x: integration nodes
     *      wx: integration weights
     *      bf: basis functions evaluated at integration nodes.
     */
    arma::mat gsz_integral(double Z, double dz, double Hz, double rmin, double rmax, const arma::vec & x, const arma::vec & wx, const arma::mat & bf);

    /**
     * Computes a SAP radial integral \f$ \int_0^\infty Z(r) B_1 (r) B_2(r) / r dr \f$.
     *
     * Input
     *   rmin: start of element boundary
     *   rmax: end of element boundary
     *       x: integration nodes
     *      wx: integration weights
     *      bf: basis functions evaluated at integration nodes.
     */
    arma::mat sap_integral(int Z, double rmin, double rmax, const arma::vec & x, const arma::vec & wx, const arma::mat & bf);

    /**
     * Computes a Thomas Fermi radial integral \f$ \int_0^\infty Z(r) B_1 (r) B_2(r) / r dr \f$.
     *
     * Input
     *   rmin: start of element boundary
     *   rmax: end of element boundary
     *       x: integration nodes
     *      wx: integration weights
     *      bf: basis functions evaluated at integration nodes.
     */
    arma::mat thomasfermi_integral(int Z, double rmin, double rmax, const arma::vec & x, const arma::vec & wx, const arma::mat & bf);

    /**
     * Computes the inner in-element two-electron integral:
     * \f$ \phi(r) = \frac 1 r^{L+1} \int_0^r dr' r'^{L} B_k(r') B_l(r') \f$
     */
    arma::mat twoe_inner_integral(double rmin, double rmax, const arma::vec & x, const arma::vec & wx, const polynomial_basis::PolynomialBasis * poly, int L);

    /**
     * Computes a primitive two-electron in-element integral.
     * Cross-element integrals reduce to products of radial integrals.
     * Note that the routine needs the polynomial representation.
     */
    arma::mat twoe_integral(double rmin, double rmax, const arma::vec & x, const arma::vec & wx, const polynomial_basis::PolynomialBasis * poly, int L);

    /**
     * Computes the spherically symmetric potential V(r).
     */
    arma::mat spherical_potential(double rmin, double rmax, const arma::vec & x, const arma::vec & wx, const polynomial_basis::PolynomialBasis * poly);
  }
}

#endif
