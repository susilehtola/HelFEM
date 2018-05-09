#ifndef POLYNOMIAL_H
#define POLYNOMIAL_H

#include <armadillo>

namespace helfem {
  namespace polynomial {
    /**
     * Evaluates the polynomial expansion
     *   f(x)  = c_0 + c_1 x + c_2 x^2 + ... + c_n x^n
     * at a given point
     */
    double polyval(const arma::vec & c, double x);
    /**
     * Evaluates the polynomial expansion
     *   f(x)  = c_0 + c_1 x + c_2 x^2 + ... + c_n x^n
     * at given point(s) x
     */
    arma::mat polyval(const arma::mat & c, const arma::vec & x);
    
    /**
     * Given the polynomial expansion
     *   f(x)  = c_0 + c_1 x + c_2 x^2 + ... + c_n x^n
     * calculates the expansion for the derivative
     *   f'(x) = c_1 + 2 c_2 x + 3 c_3 x^2 + ... + n c_n x^{n-1}
     * of arbitrary order.
     *
     * The coefficients are assumed to be stored column-wise.
     */
    arma::mat derivative_coeffs(const arma::mat & c, int der_order);
    
    /**
     * Get the coefficient matrix for a Lagrange (der_order = 0) or
     * Hermite (der_order>0) interpolating polynomial basis.
     */
    arma::mat hermite_coeffs(int n_nodes, int der_order);
  }
}

#endif

