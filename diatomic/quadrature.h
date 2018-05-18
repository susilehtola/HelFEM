#ifndef INTEGRALS_H
#define INTEGRALS_H

#include <armadillo>

namespace helfem {
  namespace diatomic {
    namespace quadrature {
      /**
       * Computes a radial integral of the type \f$ \int_0^\infty B_1 (\mu) B_2(\mu) \sinh^m (\mu) \cosh^n (\mu) d\mu \f$.
       *
       * Input
       *   mumin: start of element boundary
       *   mumax: end of element boundary
       *       x: integration nodes
       *      wx: integration weights
       *      bf: basis functions evaluated at integration nodes.
       */
      arma::mat radial_integral(double mumin, double mumax, int m, int n, const arma::vec & x, const arma::vec & wx, const arma::mat & bf);

      /**
       * Computes the inner in-element two-electron integral:
       * \f$ \phi(r) = \frac 1 r^{L+1} \int_0^r dr' r'^{L} B_k(r') B_l(r') \f$
       */
      arma::mat twoe_inner_integral(double rmin, double rmax, const arma::vec & x, const arma::vec & wx, const arma::mat & bf_C, int L);

      /**
       * Computes a primitive two-electron in-element integral.
       * Cross-element integrals reduce to products of radial integrals.
       * Note that the routine needs the polynomial representation.
       */
      arma::mat twoe_integral(double rmin, double rmax, const arma::vec & x, const arma::vec & wx, const arma::mat & bf_C, int L);
    }
  }
}

#endif
