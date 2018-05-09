#include "basis.h"
#include "../general/polynomial.h"

namespace helfem {
  namespace basis {
    MuBasis::MuBasis(int n_nodes, int der_order, int num_el, double mumax, double R) {
      // Get primitive polynomial representation
      Cpoly=polynomial::hermite_coeffs(n_nodes, der_order);
      Dpoly=polynomial::derivative_coeffs(Cpoly, 1);

      // Number of bordering elements is
      n_border=der_order+1;
      // Get mu grid
      muval=arma::linspace<arma::vec>(0.0,mumax,num_el+1);
      
      // R value
      Rnuc=R;
    }

    MuBasis::~MuBasis() {
    }
  }
}
