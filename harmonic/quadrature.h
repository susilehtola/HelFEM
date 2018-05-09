#ifndef INTEGRALS_H
#define INTEGRALS_H

#include <armadillo>

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
    arma::mat radial_integral(double xmin, double xmax, int n, const arma::vec & x, const arma::vec & wx, const arma::mat & bf);
    
    /**
     * Computes a derivative matrix element of the type
     * dB_1(x)/dx dB_2/dx dx
     */
    arma::mat derivative_integral(double xmin, double xmax, const arma::vec & x, const arma::vec & wx, const arma::mat & dbf);
  }
}

#endif
