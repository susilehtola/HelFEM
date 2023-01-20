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
          /* First function is [1 - 2(x-xi)*lipxi(fi)] [l_i(x)]^2 = f1 * f2.
             Second function is (x-xi) * [l_i(x)]^2 = f3 * f2
          */

          double f1 = 1.0 - 2.0*(x(ix)-x0(fi))*lipxi(fi);
          double f2 = std::pow(lip(ix, fi), 2);
          double f3 = x(ix)-x0(fi);

          f(ix,2*fi)   = f1*f2;
          f(ix,2*fi+1) = f3*f2 * element_length;
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
          /* First function is [1 - 2(x-xi)*lipxi(fi)] [l_i(x)]^2 = f1 * f2.
             Second function is (x-xi) * [l_i(x)]^2 = f3 * f2
          */

          double f1 = 1.0 - 2.0*(x(ix)-x0(fi))*lipxi(fi);
          double df1 = -2.0*lipxi(fi);

          double f2 = std::pow(lip(ix, fi), 2);
          double df2 = 2*lip(ix,fi)*dlip(ix,fi);

          double f3 = x(ix)-x0(fi);
          double df3 = 1;

          df(ix,2*fi)   = df1*f2 + f1*df2;
          df(ix,2*fi+1) = (df3*f2 + f3*df2) * element_length;
        }
      }
    }

    void HIPBasis::eval_prim_d2f(const arma::vec & x, arma::mat & d2f, double element_length) const {
      // Evaluate LIP basis
      arma::mat lip, dlip, d2lip;
      LIPBasis::eval_f_raw(x, lip);
      LIPBasis::eval_df_raw(x, dlip);
      LIPBasis::eval_d2f_raw(x, d2lip);

      d2f.zeros(x.n_elem, 2*x0.n_elem);
      for(size_t ix=0;ix<x.n_elem;ix++) {
        for(size_t fi=0;fi<x0.n_elem;fi++) {
          double f1 = 1.0 - 2.0*(x(ix)-x0(fi))*lipxi(fi);
          double df1 = -2.0*lipxi(fi);

          double df2 = 2*lip(ix,fi)*dlip(ix,fi);
          double d2f2 = 2*dlip(ix,fi)*dlip(ix,fi) + 2*lip(ix,fi)*d2lip(ix,fi);

          double f3 = x(ix)-x0(fi);
          double df3 = 1.0;

          d2f(ix,2*fi)   = 2*df1*df2 + f1*d2f2;
          d2f(ix,2*fi+1) = (2*df3*df2 + f3*d2f2) * element_length;
        }
      }
    }

    void HIPBasis::eval_prim_d3f(const arma::vec & x, arma::mat & d3f, double element_length) const {
      // Evaluate LIP basis
      arma::mat lip, dlip, d2lip, d3lip;
      LIPBasis::eval_f_raw(x, lip);
      LIPBasis::eval_df_raw(x, dlip);
      LIPBasis::eval_d2f_raw(x, d2lip);
      LIPBasis::eval_d3f_raw(x, d3lip);

      d3f.zeros(x.n_elem, 2*x0.n_elem);
      for(size_t ix=0;ix<x.n_elem;ix++) {
        for(size_t fi=0;fi<x0.n_elem;fi++) {
          double f1 = 1.0 - 2.0*(x(ix)-x0(fi))*lipxi(fi);
          double df1 = -2.0*lipxi(fi);

          double d2f2 = 2*dlip(ix,fi)*dlip(ix,fi) + 2*lip(ix,fi)*d2lip(ix,fi);
          double d3f2 = 6*dlip(ix,fi)*d2lip(ix,fi) + 2*lip(ix,fi)*d3lip(ix,fi);

          double f3 = x(ix)-x0(fi);
          double df3 = 1.0;

          d3f(ix,2*fi)   = 3*df1*d2f2 + f1*d3f2;
          d3f(ix,2*fi+1) = (3*df3*d2f2 + f3*d3f2) * element_length;
        }
      }
    }

    void HIPBasis::eval_prim_d4f(const arma::vec & x, arma::mat & d4f, double element_length) const {
      // Evaluate LIP basis
      arma::mat lip, dlip, d2lip, d3lip, d4lip;
      LIPBasis::eval_f_raw(x, lip);
      LIPBasis::eval_df_raw(x, dlip);
      LIPBasis::eval_d2f_raw(x, d2lip);
      LIPBasis::eval_d3f_raw(x, d3lip);
      LIPBasis::eval_d4f_raw(x, d4lip);

      d4f.zeros(x.n_elem, 2*x0.n_elem);
      for(size_t ix=0;ix<x.n_elem;ix++) {
        for(size_t fi=0;fi<x0.n_elem;fi++) {
          double f1 = 1.0 - 2.0*(x(ix)-x0(fi))*lipxi(fi);
          double df1 = -2.0*lipxi(fi);

          double d3f2 = 6*dlip(ix,fi)*d2lip(ix,fi) + 2*lip(ix,fi)*d3lip(ix,fi);
          double d4f2 = 6*d2lip(ix,fi)*d2lip(ix,fi) + 8*dlip(ix,fi)*d3lip(ix,fi) + 2*lip(ix,fi)*d4lip(ix,fi);

          double f3 = x(ix)-x0(fi);
          double df3 = 1.0;

          d4f(ix,2*fi)   = 4*df1*d3f2 + f1*d4f2;
          d4f(ix,2*fi+1) = (4*df3*d3f2 + f3*d4f2) * element_length;
        }
      }
    }

    void HIPBasis::eval_prim_d5f(const arma::vec & x, arma::mat & d5f, double element_length) const {
      // Evaluate LIP basis
      arma::mat lip, dlip, d2lip, d3lip, d4lip, d5lip;
      LIPBasis::eval_f_raw(x, lip);
      LIPBasis::eval_df_raw(x, dlip);
      LIPBasis::eval_d2f_raw(x, d2lip);
      LIPBasis::eval_d3f_raw(x, d3lip);
      LIPBasis::eval_d4f_raw(x, d4lip);
      LIPBasis::eval_d5f_raw(x, d5lip);

      d5f.zeros(x.n_elem, 2*x0.n_elem);
      for(size_t ix=0;ix<x.n_elem;ix++) {
        for(size_t fi=0;fi<x0.n_elem;fi++) {
          double f1 = 1.0 - 2.0*(x(ix)-x0(fi))*lipxi(fi);
          double df1 = -2.0*lipxi(fi);

          double d4f2 = 6*d2lip(ix,fi)*d2lip(ix,fi) + 8*dlip(ix,fi)*d3lip(ix,fi) + 2*lip(ix,fi)*d4lip(ix,fi);
          double d5f2 = 20*d2lip(ix,fi)*d3lip(ix,fi) + 10*dlip(ix,fi)*d4lip(ix,fi) + 2*lip(ix,fi)*d5lip(ix,fi);

          double f3 = x(ix)-x0(fi);
          double df3 = 1.0;

          d5f(ix,2*fi)   = 5*df1*d4f2 + f1*d5f2;
          d5f(ix,2*fi+1) = (5*df3*d4f2 + f3*d5f2) * element_length;
        }
      }
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
