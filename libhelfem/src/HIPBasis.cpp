#include "HIPBasis.h"

namespace helfem {
  namespace polynomial_basis {

    HIPBasis::HIPBasis(const arma::vec & x, int id_) : LIPBasis(x, id_) {
      // Two overlapping functions
      noverlap=2;
      nprim=2*x0.n_elem;
      // All functions are enabled
      enabled=arma::linspace<arma::uvec>(0,2*x0.n_elem-1,2*x0.n_elem);
      /// Number of nodes is
      nnodes=x0.n_elem;

      // Evaluate derivatives at nodes
      arma::mat dlip;
      double dummy_length = 1.0;
      LIPBasis::eval_prim_dnf(x, dlip, 1, dummy_length);
      lipxi = arma::diagvec(dlip);
    }

    HIPBasis::~HIPBasis() {
    }

    HIPBasis * HIPBasis::copy() const {
      return new HIPBasis(*this);
    }

    void HIPBasis::drop_first(bool func, bool deriv) {
      if(func && deriv) {
        // Drop both function and derivative
        enabled=enabled.subvec(2,enabled.n_elem-1);
      } else if(func) {
        // Only drop function
        enabled=enabled.subvec(1,enabled.n_elem-1);
      } else if(deriv) {
        // Only drop derivative
        arma::uvec new_enabled(enabled.n_elem-1);
        new_enabled(0) = enabled(0);
        new_enabled.subvec(1,new_enabled.n_elem-1) = enabled.subvec(2,enabled.n_elem-1);
        enabled = new_enabled;
      }
    }

    void HIPBasis::drop_last(bool func, bool deriv) {
      if(func && deriv) {
        // Drop both function and derivative
        enabled=enabled.subvec(0,enabled.n_elem-3);
      } else if(deriv) {
        // Only drop derivative
        enabled=enabled.subvec(0,enabled.n_elem-2);
      } else {
        // Only drop function
        arma::uvec new_enabled(enabled.n_elem-1);
        new_enabled.subvec(0,enabled.n_elem-3) = enabled.subvec(0,enabled.n_elem-3);
        new_enabled(enabled.n_elem-2) = enabled(enabled.n_elem-1);
        enabled = new_enabled;
      }
    }
  }
}
