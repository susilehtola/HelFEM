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

    /// Get "modified" Gaunt coefficient (interim coupling through cos^2)
    double modified_gaunt_coefficient(int L, int M, int l, int m, int lp, int mp);

    /// Table of Gaunt coefficients
    class Gaunt {
      /// Table of coefficients
      arma::cube table;
      /// Limited m set
      bool mlimit;
      /// Maximum m values
      int Mmax, mmax, mpmax;

    public:
      /// Dummy constructor
      Gaunt();
      /// Constructor
      Gaunt(int Lmax, int lmax, int lpmax);
      /// Fine grained constructor
      Gaunt(int Lmax, int Mmax, int lmax, int mmax, int lpmax, int mpmax);
      /// Destructor
      ~Gaunt();

      /// Get Gaunt coefficient
      double coeff(int L, int M, int l, int m, int lp, int mp) const;
      /// Get "modified" Gaunt coefficient (interim coupling through cos^2)
      double mod_coeff(int L, int M, int l, int m, int lp, int mp) const;

      /// Get cosine type coupling
      double cosine_coupling(int lj, int mj, int li, int mi) const;
      /// Get cosine^2 type coupling
      double cosine2_coupling(int lj, int mj, int li, int mi) const;
      /// Get cosine^3 type coupling
      double cosine3_coupling(int lj, int mj, int li, int mi) const;
      /// Get cosine^4 type coupling
      double cosine4_coupling(int lj, int mj, int li, int mi) const;
      /// Get cosine^5 type coupling
      double cosine5_coupling(int lj, int mj, int li, int mi) const;

      /// Get sine^2 type coupling
      double sine2_coupling(int lj, int mj, int li, int mi) const;
      /// Get cosine^2 sine^2 type coupling
      double cosine2_sine2_coupling(int lj, int mj, int li, int mi) const;
    };
  }
}

#endif
