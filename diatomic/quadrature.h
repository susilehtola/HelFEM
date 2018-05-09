#ifndef INTEGRALS_H
#define INTEGRALS_H

#include <armadillo>

namespace helfem {
  namespace quadrature {
    /**
     * Computes a radial integral of the type B_1 (mu) B_2(mu) sinh^k
     * mu cosh^l mu dmu.
     *
     * Input
     *   mumin: start of element boundary
     *   mumax: end of element boundary
     *       x: integration nodes
     *      wx: integration weights
     *      bf: basis functions evaluated at integration nodes.
     */
    arma::mat radial_integral(double mumin, double mumax, int k, int l, const arma::vec & x, const arma::vec & wx, const arma::mat & bf);
    
    /**
     * Computes the radial Laplacian element of the first kind,
     * i.e. sinh mu (d2B/dmu2) + cosh mu (dB/dmu) dmu.
     */
    arma::mat derivative_integral(double mumin, double mumax, const arma::vec & x, const arma::vec & wx, const arma::mat & bf, const arma::mat & dbf, const arma::mat & hbf);
  }
}

#endif
