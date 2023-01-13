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

    void LIPBasis::eval_f_raw(const arma::vec &x, arma::mat &f) const {
      // Allocate memory
      f.zeros(x.n_elem, x0.n_elem);
      // Loop over points
      for (size_t ix = 0; ix < x.n_elem; ix++) {
        // Loop over polynomials
        for (size_t fi = 0; fi < x0.n_elem; fi++) {
          // Form the LIP product
          double fval = 1.0;
          for (size_t ip = 0; ip < x0.n_elem; ip++) {
            // Skip terms which have been acted upon by a derivative
            if (ip == fi)
              continue;
            fval *= (x(ix) - x0(ip)) / (x0(fi) - x0(ip));
          }
          // Store the computed value
          f(ix, fi) += fval;
        }
      }
    }

    void LIPBasis::eval_df_raw(const arma::vec &x, arma::mat &df) const {
      // Allocate memory
      df.zeros(x.n_elem, x0.n_elem);
      // Loop over points
      for (size_t ix = 0; ix < x.n_elem; ix++) {
        // Loop over polynomials
        for (size_t fi = 0; fi < x0.n_elem; fi++) {
          // Derivative 1 acting on index
          for (size_t d1 = 0; d1 < x0.n_elem; d1++) {
            if (d1 == fi)
              continue;
            // Form the LIP product
            double dfval = 1.0;
            for (size_t ip = 0; ip < x0.n_elem; ip++) {
              // Skip terms which have been acted upon by a derivative
              if (ip == d1)
                continue;
              if (ip == fi)
                continue;
              dfval *= (x(ix) - x0(ip)) / (x0(fi) - x0(ip));
            }
            // Apply derivative denominators
            dfval /= (x0(fi) - x0(d1));
            // Store the computed value
            df(ix, fi) += dfval;
          }
        }
      }
    }

    void LIPBasis::eval_d2f_raw(const arma::vec &x, arma::mat &d2f) const {
      // Allocate memory
      d2f.zeros(x.n_elem, x0.n_elem);
      // Loop over points
      for (size_t ix = 0; ix < x.n_elem; ix++) {
        // Loop over polynomials
        for (size_t fi = 0; fi < x0.n_elem; fi++) {
          // Derivative 1 acting on index
          for (size_t d1 = 0; d1 < x0.n_elem; d1++) {
            if (d1 == fi)
              continue;
            // Derivative 2 acting on index
            for (size_t d2 = 0; d2 < x0.n_elem; d2++) {
              if (d2 == d1)
                continue;
              if (d2 == fi)
                continue;
              // Form the LIP product
              double d2fval = 1.0;
              for (size_t ip = 0; ip < x0.n_elem; ip++) {
                // Skip terms which have been acted upon by a derivative
                if (ip == d1)
                  continue;
                if (ip == d2)
                  continue;
                if (ip == fi)
                  continue;
                d2fval *= (x(ix) - x0(ip)) / (x0(fi) - x0(ip));
              }
              // Apply derivative denominators
              d2fval /= (x0(fi) - x0(d1)) * (x0(fi) - x0(d2));
              // Store the computed value
              d2f(ix, fi) += d2fval;
            }
          }
        }
      }
    }

    void LIPBasis::eval_d3f_raw(const arma::vec &x, arma::mat &d3f) const {
      // Allocate memory
      d3f.zeros(x.n_elem, x0.n_elem);
      // Loop over points
      for (size_t ix = 0; ix < x.n_elem; ix++) {
        // Loop over polynomials
        for (size_t fi = 0; fi < x0.n_elem; fi++) {
          // Derivative 1 acting on index
          for (size_t d1 = 0; d1 < x0.n_elem; d1++) {
            if (d1 == fi)
              continue;
            // Derivative 2 acting on index
            for (size_t d2 = 0; d2 < x0.n_elem; d2++) {
              if (d2 == d1)
                continue;
              if (d2 == fi)
                continue;
              // Derivative 3 acting on index
              for (size_t d3 = 0; d3 < x0.n_elem; d3++) {
                if (d3 == d1)
                  continue;
                if (d3 == d2)
                  continue;
                if (d3 == fi)
                  continue;
                // Form the LIP product
                double d3fval = 1.0;
                for (size_t ip = 0; ip < x0.n_elem; ip++) {
                  // Skip terms which have been acted upon by a derivative
                  if (ip == d1)
                    continue;
                  if (ip == d2)
                    continue;
                  if (ip == d3)
                    continue;
                  if (ip == fi)
                    continue;
                  d3fval *= (x(ix) - x0(ip)) / (x0(fi) - x0(ip));
                }
                // Apply derivative denominators
                d3fval /= (x0(fi) - x0(d1)) * (x0(fi) - x0(d2)) * (x0(fi) - x0(d3));
                // Store the computed value
                d3f(ix, fi) += d3fval;
              }
            }
          }
        }
      }
    }

    void LIPBasis::eval_d4f_raw(const arma::vec &x, arma::mat &d4f) const {
      // Allocate memory
      d4f.zeros(x.n_elem, x0.n_elem);
      // Loop over points
      for (size_t ix = 0; ix < x.n_elem; ix++) {
        // Loop over polynomials
        for (size_t fi = 0; fi < x0.n_elem; fi++) {
          // Derivative 1 acting on index
          for (size_t d1 = 0; d1 < x0.n_elem; d1++) {
            if (d1 == fi)
              continue;
            // Derivative 2 acting on index
            for (size_t d2 = 0; d2 < x0.n_elem; d2++) {
              if (d2 == d1)
                continue;
              if (d2 == fi)
                continue;
              // Derivative 3 acting on index
              for (size_t d3 = 0; d3 < x0.n_elem; d3++) {
                if (d3 == d1)
                  continue;
                if (d3 == d2)
                  continue;
                if (d3 == fi)
                  continue;
                // Derivative 4 acting on index
                for (size_t d4 = 0; d4 < x0.n_elem; d4++) {
                  if (d4 == d1)
                    continue;
                  if (d4 == d2)
                    continue;
                  if (d4 == d3)
                    continue;
                  if (d4 == fi)
                    continue;
                  // Form the LIP product
                  double d4fval = 1.0;
                  for (size_t ip = 0; ip < x0.n_elem; ip++) {
                    // Skip terms which have been acted upon by a derivative
                    if (ip == d1)
                      continue;
                    if (ip == d2)
                      continue;
                    if (ip == d3)
                      continue;
                    if (ip == d4)
                      continue;
                    if (ip == fi)
                      continue;
                    d4fval *= (x(ix) - x0(ip)) / (x0(fi) - x0(ip));
                  }
                  // Apply derivative denominators
                  d4fval /= (x0(fi) - x0(d1)) * (x0(fi) - x0(d2)) *
                    (x0(fi) - x0(d3)) * (x0(fi) - x0(d4));
                  // Store the computed value
                  d4f(ix, fi) += d4fval;
                }
              }
            }
          }
        }
      }
    }

    void LIPBasis::eval_d5f_raw(const arma::vec &x, arma::mat &d5f) const {
      // Allocate memory
      d5f.zeros(x.n_elem, x0.n_elem);
      // Loop over points
      for (size_t ix = 0; ix < x.n_elem; ix++) {
        // Loop over polynomials
        for (size_t fi = 0; fi < x0.n_elem; fi++) {
          // Derivative 1 acting on index
          for (size_t d1 = 0; d1 < x0.n_elem; d1++) {
            if (d1 == fi)
              continue;
            // Derivative 2 acting on index
            for (size_t d2 = 0; d2 < x0.n_elem; d2++) {
              if (d2 == d1)
                continue;
              if (d2 == fi)
                continue;
              // Derivative 3 acting on index
              for (size_t d3 = 0; d3 < x0.n_elem; d3++) {
                if (d3 == d1)
                  continue;
                if (d3 == d2)
                  continue;
                if (d3 == fi)
                  continue;
                // Derivative 4 acting on index
                for (size_t d4 = 0; d4 < x0.n_elem; d4++) {
                  if (d4 == d1)
                    continue;
                  if (d4 == d2)
                    continue;
                  if (d4 == d3)
                    continue;
                  if (d4 == fi)
                    continue;
                  // Derivative 5 acting on index
                  for (size_t d5 = 0; d5 < x0.n_elem; d5++) {
                    if (d5 == d1)
                      continue;
                    if (d5 == d2)
                      continue;
                    if (d5 == d3)
                      continue;
                    if (d5 == d4)
                      continue;
                    if (d5 == fi)
                      continue;
                    // Form the LIP product
                    double d5fval = 1.0;
                    for (size_t ip = 0; ip < x0.n_elem; ip++) {
                      // Skip terms which have been acted upon by a derivative
                      if (ip == d1)
                        continue;
                      if (ip == d2)
                        continue;
                      if (ip == d3)
                        continue;
                      if (ip == d4)
                        continue;
                      if (ip == d5)
                        continue;
                      if (ip == fi)
                        continue;
                      d5fval *= (x(ix) - x0(ip)) / (x0(fi) - x0(ip));
                    }
                    // Apply derivative denominators
                    d5fval /= (x0(fi) - x0(d1)) * (x0(fi) - x0(d2)) *
                      (x0(fi) - x0(d3)) * (x0(fi) - x0(d4)) *
                      (x0(fi) - x0(d5));
                    // Store the computed value
                    d5f(ix, fi) += d5fval;
                  }
                }
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
