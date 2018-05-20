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
       * Computes a radial integral of the type \f$ \int_0^\infty B_1 (\mu) B_2(\mu) \cosh^m (\mu) P_L^M (\mu) d\mu \f$.
       *
       * Input
       *   mumin: start of element boundary
       *   mumax: end of element boundary
       *       x: integration nodes
       *      wx: integration weights
       *      bf: basis functions evaluated at integration nodes.
       */
      arma::mat Plm_radial_integral(double mumin, double mumax, int m, const arma::vec & x, const arma::vec & wx, const arma::mat & bf, int L, int M);

      /**
       * Computes a radial integral of the type \f$ \int_0^\infty B_1 (\mu) B_2(\mu) \cosh^m (\mu) Q_L^M (\mu) d\mu \f$.
       *
       * Input
       *   mumin: start of element boundary
       *   mumax: end of element boundary
       *       x: integration nodes
       *      wx: integration weights
       *      bf: basis functions evaluated at integration nodes.
       */
      arma::mat Qlm_radial_integral(double mumin, double mumax, int m, const arma::vec & x, const arma::vec & wx, const arma::mat & bf, int L, int M);

      /**
       * Computes the inner in-element two-electron integral:
       * \f$ \phi^{l,LM}(\mu) = \int_{0}^{\mu}d\mu'\cosh^{l}\mu'\sinh\mu'B_{\gamma}(\mu')B_{\delta}(\mu')P_{L,|M|}(\cosh\mu') \f$
       */
      arma::mat twoe_inner_integral(double mumin, double mumax, int l, const arma::vec & x, const arma::vec & wx, const arma::mat & bf_C, int L, int M);

      /**
       * Computes a primitive two-electron in-element integral.
       * Cross-element integrals reduce to products of radial integrals.
       * Note that the routine needs the polynomial representation.
       */
      arma::mat twoe_integral(double rmin, double rmax, int k, int l, const arma::vec & x, const arma::vec & wx, const arma::mat & bf_C, int L, int M);
    }
  }
}

#endif
