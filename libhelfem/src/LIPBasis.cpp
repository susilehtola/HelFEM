#include "LIPBasis.h"
#include <cfloat>

namespace helfem {
  namespace polynomial_basis {

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

    void LIPBasis::eval_f_raw(const arma::vec & x, arma::mat & bf) const {
      // Memory for values
      bf.zeros(x.n_elem,x0.n_elem);

      // Fill in array
      for(size_t ix=0;ix<x.n_elem;ix++) {
        // Loop over polynomials: x_i term excluded
        for(size_t fi=0;fi<x0.n_elem;fi++) {
          // Evaluate the l_i polynomial
          double fval=1.0;
          for(size_t fj=0;fj<x0.n_elem;fj++) {
            // Term not included
            if(fi==fj)
              continue;
            // Compute ratio
            fval *= (x(ix)-x0(fj))/(x0(fi)-x0(fj));
          }
          // Store value
          bf(ix,fi)=fval;
        }
      }
    }

    void LIPBasis::eval_df_raw(const arma::vec & x, arma::mat & df) const {
      // Derivative
      df.zeros(x.n_elem,x0.n_elem);
      for(size_t ix=0;ix<x.n_elem;ix++) {
        // Loop over polynomials
        for(size_t fi=0;fi<x0.n_elem;fi++) {
          // Derivative yields a sum over one of the indices
          for(size_t fj=0;fj<x0.n_elem;fj++) {
            if(fi==fj)
              continue;

            double fval=1.0;
            for(size_t fk=0;fk<x0.n_elem;fk++) {
              // Term not included
              if(fi==fk)
                continue;
              if(fj==fk)
                continue;
              // Compute ratio
              fval *= (x(ix)-x0(fk))/(x0(fi)-x0(fk));
            }
            // Increment derivative
            df(ix,fi)+=fval/(x0(fi)-x0(fj));
          }
        }
      }
    }

    void LIPBasis::eval_d2f_raw(const arma::vec & x, arma::mat & d2f) const {
      // Second derivative
      d2f.zeros(x.n_elem,x0.n_elem);
      for(size_t ix=0;ix<x.n_elem;ix++) {
        // Loop over polynomials
        for(size_t fi=0;fi<x0.n_elem;fi++) {
          // Derivative yields a sum over one of the indices
          for(size_t fj=0;fj<x0.n_elem;fj++) {
            if(fi==fj)
              continue;
            // Second derivative yields another sum over the indices
            for(size_t fk=0;fk<x0.n_elem;fk++) {
              if(fi==fk)
                continue;
              if(fj==fk)
                continue;

              double fval=1.0;
              for(size_t fl=0;fl<x0.n_elem;fl++) {
                // Term not included
                if(fi==fl)
                  continue;
                if(fj==fl)
                  continue;
                if(fk==fl)
                  continue;
                // Compute ratio
                fval *= (x(ix)-x0(fl))/(x0(fi)-x0(fl));
              }
              // Increment second derivative
              d2f(ix,fi)+=fval/((x0(fi)-x0(fj))*(x0(fi)-x0(fk)));
            }
          }
        }
      }
    }

    void LIPBasis::eval_d3f_raw(const arma::vec & x, arma::mat & d3f) const {
      // Third derivative
      d3f.zeros(x.n_elem,x0.n_elem);
      for(size_t ix=0;ix<x.n_elem;ix++) {
        // Loop over polynomials
        for(size_t fi=0;fi<x0.n_elem;fi++) {
          // Derivative yields a sum over one of the indices
          for(size_t fj=0;fj<x0.n_elem;fj++) {
            if(fi==fj)
              continue;
            // Second derivative yields another sum over the indices
            for(size_t fk=0;fk<x0.n_elem;fk++) {
              if(fi==fk)
                continue;
              if(fj==fk)
                continue;
              // Third derivative yields yet another sum over the indices
              for(size_t fl=0;fl<x0.n_elem;fl++) {
                if(fi==fl)
                  continue;
                if(fj==fl)
                  continue;
                if(fk==fl)
                  continue;

                double fval=1.0;
                for(size_t fm=0;fm<x0.n_elem;fm++) {
                  // Term not included
                  if(fi==fm)
                    continue;
                  if(fj==fm)
                    continue;
                  if(fk==fm)
                    continue;
                  if(fl==fm)
                    continue;
                  // Compute ratio
                  fval *= (x(ix)-x0(fm))/(x0(fi)-x0(fm));
                }
                // Increment third derivative
                d3f(ix,fi)+=fval/((x0(fi)-x0(fj))*(x0(fi)-x0(fk))*(x0(fi)-x0(fl)));
              }
            }
          }
        }
      }
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
