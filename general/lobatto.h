#ifndef LOBATTO
#define LOBATTO

#include <armadillo>

/// Compute a Gauss-Lobatto quadrature rule for \f$ \int_{-1}^1 f(x)dx \approx \frac 2 {n(n-1)} \left[ f(-1) + f(1) \right] + \sum_{i=2}^{n-1} w_i f(x_i) \f$
void lobatto_compute ( int n, arma::vec & x, arma::vec & w);

#endif
