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
      LIPBasis::eval_df_raw(x, dlip);
      lipxi = arma::diagvec(dlip);
    }

    HIPBasis::~HIPBasis() {
    }

    HIPBasis * HIPBasis::copy() const {
      return new HIPBasis(*this);
    }

    void HIPBasis::eval_prim_f(const arma::vec & x, arma::mat & f, double element_length) const {
      // Evaluate LIP basis
      arma::mat lip, dlip;
      LIPBasis::eval_f_raw(x, lip);
      LIPBasis::eval_df_raw(x, dlip);

      // Basis function values
      f.zeros(x.n_elem, 2*x0.n_elem);
      for(size_t ix=0;ix<x.n_elem;ix++) {
        for(size_t fi=0;fi<x0.n_elem;fi++) {
          double dx = x(ix)-x0(fi);
          f(ix,2*fi)   = (1.0 - 2.0*dx*lipxi(fi)) * std::pow(lip(ix,fi),2);
          f(ix,2*fi+1) = dx * std::pow(lip(ix,fi),2) * element_length;
        }
      }
    }

    void HIPBasis::eval_prim_df(const arma::vec & x, arma::mat & df, double element_length) const {
      // Evaluate LIP basis
      arma::mat lip, dlip;
      LIPBasis::eval_f_raw(x, lip);
      LIPBasis::eval_df_raw(x, dlip);

      df.zeros(x.n_elem, 2*x0.n_elem);
      for(size_t ix=0;ix<x.n_elem;ix++) {
        for(size_t fi=0;fi<x0.n_elem;fi++) {
          df(ix,2*fi)   = 2.0*dlip(ix,fi)*lip(ix,fi)*(1.0 - 2.0*(x(ix)-x0(fi))*lipxi(fi)) - 2.0*lipxi(fi)*std::pow(lip(ix,fi),2);
          df(ix,2*fi+1) = (std::pow(lip(ix,fi),2) + 2.0*(x(ix)-x0(fi))*lip(ix,fi)*dlip(ix,fi)) * element_length;
        }
      }
    }

    void HIPBasis::eval_prim_d2f(const arma::vec & x, arma::mat & d2f, double element_length) const {
      // Evaluate LIP basis
      arma::mat lip, dlip, d2lip, d3lip;
      LIPBasis::eval_f_raw(x, lip);
      LIPBasis::eval_df_raw(x, dlip);
      LIPBasis::eval_d2f_raw(x, d2lip);
      LIPBasis::eval_d3f_raw(x, d3lip);

      d2f.zeros(x.n_elem, 2*x0.n_elem);
      for(size_t ix=0;ix<x.n_elem;ix++) {
        for(size_t fi=0;fi<x0.n_elem;fi++) {
          d2f(ix,2*fi)   = 2.0*(d2lip(ix,fi)*lip(ix,fi) + std::pow(dlip(ix,fi),2))*(1.0 - 2.0*(x(ix)-x0(fi))*lipxi(fi)) - 8.0*lip(ix,fi)*dlip(ix,fi)*lipxi(fi);
          d2f(ix,2*fi+1) = (4.0*lip(ix,fi)*dlip(ix,fi) + 2.0*(x(ix)-x0(fi))*(d2lip(ix,fi)*lip(ix,fi) + std::pow(dlip(ix,fi),2))) * element_length;
        }
      }
    }

    void HIPBasis::drop_first(bool zero_deriv) {
      if(zero_deriv) {
        enabled=enabled.subvec(2,enabled.n_elem-1);
      } else {
        enabled=enabled.subvec(1,enabled.n_elem-1);
      }
    }

    void HIPBasis::drop_last(bool zero_deriv) {
      if(zero_deriv) {
        enabled=enabled.subvec(0,enabled.n_elem-3);
      } else {
        arma::uvec new_enabled(enabled.n_elem-2);
        new_enabled.subvec(0,enabled.n_elem-3) = enabled.subvec(0,enabled.n_elem-3);
        new_enabled(enabled.n_elem-2) = enabled(enabled.n_elem-1);
        enabled = new_enabled;
      }
    }
  }
}
