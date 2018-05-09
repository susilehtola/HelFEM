#ifndef BASIS_H
#define BASIS_H

#include <armadillo>

namespace helfem {
  namespace basis {
    class MuBasis {
      // Primitive polynomial basis in mu direction
      arma::mat Cpoly;
      // .. and its derivatives
      arma::mat Dpoly;
      // .. and its second derivatives
      arma::mat Hpoly;

      // Number of bordering functions (= der_order+1)
      int n_border;
      // Mu element boundaries
      arma::vec muval;

      // Nuclear distance
      double Rnuc;

    public:
      // Construct mu basis
      MuBasis(int n_nodes, int der_order, int num_el, double mumax, double R);
      // Destructor
      ~MuBasis();
    };
  }
}


#endif
