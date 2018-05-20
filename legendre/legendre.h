#ifndef LEGENDRE_H
#define LEGENDRE_H

#include <armadillo>

namespace helfem {
  namespace legendre {
    /* Computes associated Legendre function of the first kind P_l^m(x), |x|<=1 */
    arma::vec legendreP(int l, int m, const arma::vec & x);
    /* Computes associated Legendre function of the second kind Q_l^m(x), |x|<=1 */
    arma::vec legendreQ(int l, int m, const arma::vec & x);
    /* Computes associated Legendre function of the first kind P_l^m(x), |x|>1 */
    arma::vec legendreP_prolate(int l, int m, const arma::vec & x);
    /* Computes associated Legendre function of the second kind Q_l^m(x), |x|>1 */
    arma::vec legendreQ_prolate(int l, int m, const arma::vec & x);
  }
}

#endif
