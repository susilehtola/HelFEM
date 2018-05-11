#ifndef INTEGRALS_H
#define INTEGRALS_H

#include <armadillo>

namespace helfem {
  namespace quadrature {
    /**
     * Computes a radial integral of the type B_1 (r) B_2(r) r^n dr.
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
     * Computes the inner in-element integral.
     */
    arma::mat twoe_inner_integral(double rmin, double rmax, const arma::vec & x, const arma::vec & wx, const arma::mat & bf, int L);

    /**
     * Computes a primitive two-electron in-element integral.
     * Cross-element integrals reduce to products of radial integrals.
     */
    arma::mat twoe_integral(double rmin, double rmax, const arma::vec & x, const arma::vec & wx, const arma::mat & bf, int L);
  }
}

#endif
