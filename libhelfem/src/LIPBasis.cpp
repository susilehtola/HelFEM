#include "LIPBasis.h"
#include <cfloat>

namespace helfem {
  namespace polynomial_basis {

    LIPBasis::LIPBasis() {
    }

    LIPBasis::LIPBasis(const arma::vec & x, int id_) {
      // Make sure nodes are in order
      x0=arma::sort(x,"ascend");

      // Sanity check
      if(std::abs(x(0)+1)>=sqrt(DBL_EPSILON))
        throw std::logic_error("LIP leftmost node is not at -1!\n");
      if(std::abs(x(x.n_elem-1)-1)>=sqrt(DBL_EPSILON))
        throw std::logic_error("LIP rightmost node is not at -1!\n");

      // One overlapping function
      noverlap=1;
      nprim=x0.n_elem;
      // All functions are enabled
      enabled=arma::linspace<arma::uvec>(0,x0.n_elem-1,x0.n_elem);

      /// Identifier is
      id=id_;
      /// Number of nodes is
      nnodes=enabled.n_elem;
    }

    LIPBasis::~LIPBasis() {
    }

    LIPBasis * LIPBasis::copy() const {
      return new LIPBasis(*this);
    }

    void LIPBasis::eval_prim_f(const arma::vec & x, arma::mat & f, double element_length) const {
      (void) element_length;
      eval_f_raw(x, f);
    }

    void LIPBasis::eval_prim_df(const arma::vec & x, arma::mat & df, double element_length) const {
      (void) element_length;
      eval_df_raw(x, df);
    }

    void LIPBasis::eval_prim_d2f(const arma::vec & x, arma::mat & d2f, double element_length) const {
      (void) element_length;
      eval_d2f_raw(x, d2f);
    }

    void LIPBasis::eval_prim_d3f(const arma::vec & x, arma::mat & d3f, double element_length) const {
      (void) element_length;
      eval_d3f_raw(x, d3f);
    }

    void LIPBasis::eval_prim_d4f(const arma::vec & x, arma::mat & d4f, double element_length) const {
      (void) element_length;
      eval_d4f_raw(x, d4f);
    }

    void LIPBasis::eval_prim_d5f(const arma::vec & x, arma::mat & d5f, double element_length) const {
      (void) element_length;
      eval_d5f_raw(x, d5f);
    }

    void LIPBasis::drop_first(bool func, bool deriv) {
      (void) deriv;
      if(func)
        enabled=enabled.subvec(1,enabled.n_elem-1);
    }

    void LIPBasis::drop_last(bool func, bool deriv) {
      (void) deriv;
      if(func)
        enabled=enabled.subvec(0,enabled.n_elem-2);
    }

  }
}
