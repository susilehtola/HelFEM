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
  }
}

#endif
