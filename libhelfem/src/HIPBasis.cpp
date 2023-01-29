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

    void HIPBasis::eval_prim_f(const arma::vec & x, arma::mat & f, double element_length) const {
      // Evaluate LIP basis
      arma::mat lip;
      double dummy_length = 1.0;
      LIPBasis::eval_prim_dnf(x, lip, 0, dummy_length);

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
      double dummy_length = 1.0;
      LIPBasis::eval_prim_dnf(x, lip, 0, dummy_length);
      LIPBasis::eval_prim_dnf(x, dlip, 1, dummy_length);

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
      double dummy_length = 1.0;
      LIPBasis::eval_prim_dnf(x, lip, 0, dummy_length);
      LIPBasis::eval_prim_dnf(x, dlip, 1, dummy_length);
      LIPBasis::eval_prim_dnf(x, d2lip, 2, dummy_length);

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
      double dummy_length = 1.0;
      LIPBasis::eval_prim_dnf(x, lip, 0, dummy_length);
      LIPBasis::eval_prim_dnf(x, dlip, 1, dummy_length);
      LIPBasis::eval_prim_dnf(x, d2lip, 2, dummy_length);
      LIPBasis::eval_prim_dnf(x, d3lip, 3, dummy_length);

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
      double dummy_length = 1.0;
      LIPBasis::eval_prim_dnf(x, lip, 0, dummy_length);
      LIPBasis::eval_prim_dnf(x, dlip, 1, dummy_length);
      LIPBasis::eval_prim_dnf(x, d2lip, 2, dummy_length);
      LIPBasis::eval_prim_dnf(x, d3lip, 3, dummy_length);
      LIPBasis::eval_prim_dnf(x, d4lip, 4, dummy_length);

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
      double dummy_length = 1.0;
      LIPBasis::eval_prim_dnf(x, lip, 0, dummy_length);
      LIPBasis::eval_prim_dnf(x, dlip, 1, dummy_length);
      LIPBasis::eval_prim_dnf(x, d2lip, 2, dummy_length);
      LIPBasis::eval_prim_dnf(x, d3lip, 3, dummy_length);
      LIPBasis::eval_prim_dnf(x, d4lip, 4, dummy_length);
      LIPBasis::eval_prim_dnf(x, d5lip, 5, dummy_length);

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

    void HIPBasis::eval_prim_d6f(const arma::vec & x, arma::mat & d6f, double element_length) const {
      // Evaluate LIP basis
      arma::mat lip, dlip, d2lip, d3lip, d4lip, d5lip, d6lip;
      double dummy_length = 1.0;
      LIPBasis::eval_prim_dnf(x, lip, 0, dummy_length);
      LIPBasis::eval_prim_dnf(x, dlip, 1, dummy_length);
      LIPBasis::eval_prim_dnf(x, d2lip, 2, dummy_length);
      LIPBasis::eval_prim_dnf(x, d3lip, 3, dummy_length);
      LIPBasis::eval_prim_dnf(x, d4lip, 4, dummy_length);
      LIPBasis::eval_prim_dnf(x, d5lip, 5, dummy_length);
      LIPBasis::eval_prim_dnf(x, d6lip, 6, dummy_length);

      d6f.zeros(x.n_elem, 2*x0.n_elem);
      for(size_t ix=0;ix<x.n_elem;ix++) {
        for(size_t fi=0;fi<x0.n_elem;fi++) {
          double f1 = 1.0 - 2.0*(x(ix)-x0(fi))*lipxi(fi);
          double df1 = -2.0*lipxi(fi);

          double d5f2 = 20*d2lip(ix,fi)*d3lip(ix,fi) + 10*dlip(ix,fi)*d4lip(ix,fi) + 2*lip(ix,fi)*d5lip(ix,fi);
          double d6f2 = 20*d3lip(ix,fi)*d3lip(ix,fi) + 30*d2lip(ix,fi)*d4lip(ix,fi) + 12*dlip(ix,fi)*d5lip(ix,fi) + 2*lip(ix,fi)*d6lip(ix,fi);

          double f3 = x(ix)-x0(fi);
          double df3 = 1.0;

          d6f(ix,2*fi)   = 6*df1*d5f2 + f1*d6f2;
          d6f(ix,2*fi+1) = (6*df3*d5f2 + f3*d6f2) * element_length;
        }
      }
    }

    void HIPBasis::eval_prim_d7f(const arma::vec & x, arma::mat & d7f, double element_length) const {
      // Evaluate LIP basis
      arma::mat lip, dlip, d2lip, d3lip, d4lip, d5lip, d6lip, d7lip;
      double dummy_length = 1.0;
      LIPBasis::eval_prim_dnf(x, lip, 0, dummy_length);
      LIPBasis::eval_prim_dnf(x, dlip, 1, dummy_length);
      LIPBasis::eval_prim_dnf(x, d2lip, 2, dummy_length);
      LIPBasis::eval_prim_dnf(x, d3lip, 3, dummy_length);
      LIPBasis::eval_prim_dnf(x, d4lip, 4, dummy_length);
      LIPBasis::eval_prim_dnf(x, d5lip, 5, dummy_length);
      LIPBasis::eval_prim_dnf(x, d6lip, 6, dummy_length);
      LIPBasis::eval_prim_dnf(x, d7lip, 7, dummy_length);

      d7f.zeros(x.n_elem, 2*x0.n_elem);
      for(size_t ix=0;ix<x.n_elem;ix++) {
        for(size_t fi=0;fi<x0.n_elem;fi++) {
          double f1 = 1.0 - 2.0*(x(ix)-x0(fi))*lipxi(fi);
          double df1 = -2.0*lipxi(fi);

          double d6f2 = 20*d3lip(ix,fi)*d3lip(ix,fi) + 30*d2lip(ix,fi)*d4lip(ix,fi) + 12*dlip(ix,fi)*d5lip(ix,fi) + 2*lip(ix,fi)*d6lip(ix,fi);
          double d7f2 = 70*d3lip(ix,fi)*d4lip(ix,fi) + 42*d2lip(ix,fi)*d5lip(ix,fi) + 14*dlip(ix,fi)*d6lip(ix,fi) + 2*lip(ix,fi)*d7lip(ix,fi);

          double f3 = x(ix)-x0(fi);
          double df3 = 1.0;

          d7f(ix,2*fi)   = 7*df1*d6f2 + f1*d7f2;
          d7f(ix,2*fi+1) = (7*df3*d6f2 + f3*d7f2) * element_length;
        }
      }
    }

    void HIPBasis::eval_prim_dnf(const arma::vec & x, arma::mat & dnf, int n, double element_length) const {
      switch(n) {
      case(0):
        eval_prim_f(x,dnf,element_length);
        break;
      case(1):
        eval_prim_df(x,dnf,element_length);
        break;
      case(2):
        eval_prim_d2f(x,dnf,element_length);
        break;
      case(3):
        eval_prim_d3f(x,dnf,element_length);
        break;
      case(4):
        eval_prim_d4f(x,dnf,element_length);
        break;
      case(5):
        eval_prim_d5f(x,dnf,element_length);
        break;
      case(6):
        eval_prim_d6f(x,dnf,element_length);
        break;
      case(7):
        eval_prim_d7f(x,dnf,element_length);
        break;
      default:
        std::ostringstream oss;
        oss << n << "th order derivatives not implemented for analytic first-order HIP basis functions!\n";
        throw std::logic_error(oss.str());
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
