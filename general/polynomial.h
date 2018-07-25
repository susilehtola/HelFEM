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
     * Evaluates the Lagrange interpolating polynomial
     *   f(x)  = \prod_{k=1}^{n-1} \frac {x - x_k} {x_0 - x_k}
     * at a given point x.
     */
    double lipval(const arma::vec & x0, double x);
    /**
     * Evaluates Lagrange interpolating polynomials at wanted points
     * x.
     */
    arma::mat lipval(const arma::mat & x0, const arma::vec & x);

    /**
     * Calculate factorial n! = n*(n-1)*...*2*1
     */
    double factorial(int n);

    /**
     * Calculate ratio of factorials n!/(n-m!)
     */
    double factorial_ratio(int n, int m);

    /**
     * Calculate binomial coefficient
     */
    double choose(int n, int k);

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

    /**
     * Generates the transformation matrix from
     *   f(x)  = c_0 + c_1 x + c_2 x^2 + ... + c_n x^n
     * to
     *   f(r)  = c_0 + c_1 r + c_2 r^2 + ... + c_n r^n
     */
    arma::mat conversion_matrix(size_t xmax, double rmin, double rmax);

    /**
     * Convert an expansion in [-1,1] given as
     *   f(x)  = c_0 + c_1 x + c_2 x^2 + ... + c_n x^n
     * into an expansion in an element
     *   f(r)  = c_0 + c_1 r + c_2 r^2 + ... + c_n r^n
     */
    arma::mat convert_coeffs(const arma::mat & c, double rmin, double rmax);
  }
}

#endif

