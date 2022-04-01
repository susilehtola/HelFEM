/*
 *                This source code is part of
 *
 *                          HelFEM
 *                             -
 * Finite element methods for electronic structure calculations on small systems
 *
 * Written by Susi Lehtola, 2018-
 * Copyright (c) 2018- Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */
#ifndef POLYNOMIAL_BASIS_LIPBASIS_H
#define POLYNOMIAL_BASIS_LIPBASIS_H

#include "helfem/PolynomialBasis.h"
#include <armadillo>

namespace helfem {
  namespace polynomial_basis {
    /// Lagrange interpolating polynomials
    class LIPBasis: public PolynomialBasis {
    protected:
      /// Control nodes
      arma::vec x0;

      /// Evaluate functions
      void eval_f_raw(const arma::vec & x, arma::mat & f) const;
      /// Evaluate derivatives
      void eval_df_raw(const arma::vec & x, arma::mat & f) const;
      /// Evaluate second derivatives
      void eval_d2f_raw(const arma::vec & x, arma::mat & f) const;
      /// Evaluate third derivatives
      void eval_d3f_raw(const arma::vec & x, arma::mat & f) const;
    public:
      /// Constructor
      LIPBasis(const arma::vec & x0, int id);
      /// Destructor
      ~LIPBasis();
      /// Get a copy
      LIPBasis * copy() const override;

      /// Drop first function
      void drop_first(bool func, bool deriv) override;
      /// Drop last function
      void drop_last(bool func, bool deriv) override;

      /// Evaluate polynomials at given points
      void eval_prim_f(const arma::vec & x, arma::mat & f, double element_length) const override;
      /// Evaluate derivatives of polynomials at given points
      void eval_prim_df(const arma::vec & x, arma::mat & df, double element_length) const override;
      /// Evaluate second derivatives of polynomials at given points
      void eval_prim_d2f(const arma::vec & x, arma::mat & d2f, double element_length) const override;
      /// Evaluate third derivatives of polynomials at given points
      void eval_prim_d3f(const arma::vec & x, arma::mat & d3f, double element_length) const override;
    };
  }
}
#endif
