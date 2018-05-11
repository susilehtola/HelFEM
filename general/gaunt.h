#ifndef GAUNT
#define GAUNT

#include <armadillo>

namespace helfem {
  namespace gaunt {
/**
 * Computes Gaunt coefficient \f$ G^{M m m'}_{L l l'} \f$ in the expansion
 * \f$ Y_l^m (\Omega) Y_{l'}^{m'} (\Omega) = \sum_{L,M} G^{M m m'}_{L l l'} Y_L^M (\Omega) \f$
 */
    double gaunt_coefficient(int L, int M, int l, int m, int lp, int mp);

    /// Table of Gaunt coefficients
    class Gaunt {
      /// Table of coefficients
      arma::cube table;
    public:
      /// Dummy constructor
      Gaunt();
      /// Constructor
      Gaunt(int Lmax, int lmax, int lpmax);
      /// Destructor
      ~Gaunt();

      /// Get Gaunt coefficient
      double coeff(int L, int M, int l, int m, int lp, int mp) const;
    };
  }
}

#endif
