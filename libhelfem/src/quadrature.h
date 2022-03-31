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
#ifndef QUADRATURE_H
#define QUADRATURE_H

#include <armadillo>
#include "polynomial_basis.h"
#include <helfem/ModelPotential.h>

namespace helfem {
  namespace quadrature {
    /**
     * Computes a radial integral of the type \f$ \int_0^\infty B_1 (r) B_2(r) r^n dr \f$.
     *
     * Input
     *   rmin: start of element boundary
     *   rmax: end of element boundary
     *      x: integration nodes
     *     wx: integration weights
     *     bf: basis functions evaluated at integration nodes.
     */
    arma::mat radial_integral(double rmin, double rmax, int n, const arma::vec & x, const arma::vec & wx, const arma::mat & bf);

    /**
     * Computes a derivative matrix element of the type
     * r^2 dB_1(r)/dr dB_2/dr dr
     */
    arma::mat derivative_integral(double rmin, double rmax, const arma::vec & x, const arma::vec & wx, const arma::mat & dbf);

    /**
     * Computes a radial integral of the type \f$ \int_0^\infty B_1 (r) B_2(r) Vnuc(r) dr \f$.
     *
     * Input
     *   rmin: start of element boundary
     *   rmax: end of element boundary
     *      x: integration nodes
     *     wx: integration weights
     *     bf: basis functions evaluated at integration nodes.
     */
    arma::mat model_potential_integral(double rmin, double rmax, const modelpotential::ModelPotential * nuc, const arma::vec & x, const arma::vec & wx, const arma::mat & bf);

    /**
     * Computes a radial integral of the type \f$ \int_0^\infty B_1 (r) B_2(r) i_L(\lambda r) dr \f$.
     *
     * Input
     *   rmin: start of element boundary
     *   rmax: end of element boundary
     *      L: Bessel function order
     *      x: integration nodes
     *     wx: integration weights
     *     bf: basis functions evaluated at integration nodes.
     */
    arma::mat bessel_il_integral(double rmin, double rmax, int L, double lambda, const arma::vec & x, const arma::vec & wx, const arma::mat & bf);

    /**
     * Computes a radial integral of the type \f$ \int_0^\infty B_1 (r) B_2(r) k_L(\lambda r) dr \f$.
     *
     * Input
     *   rmin: start of element boundary
     *   rmax: end of element boundary
     *      L: Bessel function order
     *      x: integration nodes
     *     wx: integration weights
     *     bf: basis functions evaluated at integration nodes.
     */
    arma::mat bessel_kl_integral(double rmin, double rmax, int L, double lambda, const arma::vec & x, const arma::vec & wx, const arma::mat & bf);

    /**
     * Computes the inner in-element two-electron integral:
     * \f$ \phi(r) = \frac 1 r^{L+1} \int_0^r dr' r'^{L} B_k(r') B_l(r') \f$
     */
    arma::mat twoe_inner_integral(double rmin, double rmax, const arma::vec & x, const arma::vec & wx, const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly, int L);

    /**
     * Computes a primitive two-electron in-element integral.
     * Cross-element integrals reduce to products of radial integrals.
     * Note that the routine needs the polynomial representation.
     */
    arma::mat twoe_integral(double rmin, double rmax, const arma::vec & x, const arma::vec & wx, const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly, int L);

    /**
     * Computes the inner in-element two-electron Yukawa integral:
     * \f$ \phi(r) = \frac 1 r^{L+1} \int_0^r dr' r'^{L} B_k(r') B_l(r') \f$
     */
    arma::mat yukawa_inner_integral(double rmin, double rmax, const arma::vec & x, const arma::vec & wx, const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly, int L, double lambda);

    /**
     * Computes a primitive two-electron in-element Yukawa integral.
     * Cross-element integrals reduce to products of radial integrals.
     * Note that the routine needs the polynomial representation.
     */
    arma::mat yukawa_integral(double rmin, double rmax, const arma::vec & x, const arma::vec & wx, const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly, int L, double lambda);

    /**
     * Computes a primitive two-electron complementary error function
     * integral. Note that these integrals do not factorize.
     */
    arma::mat erfc_integral(double rmini, double rmaxi, const arma::mat & bfi, const arma::vec & xi, const arma::vec & wi, double rmink, double rmaxk, const arma::mat & bfk, const arma::vec & xk, const arma::vec & wk, int L, double mu);

    /**
     * Computes the spherically symmetric potential V(r).
     */
    arma::mat spherical_potential(double rmin, double rmax, const arma::vec & x, const arma::vec & wx, const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly);
  }
}

#endif
