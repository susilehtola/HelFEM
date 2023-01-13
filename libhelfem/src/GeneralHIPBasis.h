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
#ifndef POLYNOMIAL_BASIS_GeneralHIPBASIS_H
#define POLYNOMIAL_BASIS_GeneralHIPBASIS_H

#include "helfem/PolynomialBasis.h"
#include "LIPBasis.h"
#include <armadillo>

namespace helfem {
  namespace polynomial_basis {
    /// Hermite interpolating polynomials
    class GeneralHIPBasis: public PolynomialBasis {
    protected:
      /// Underlying LIP basis set
      polynomial_basis::LIPBasis lip;
      /// Transformation matrix
      arma::mat T;
      /// Number of derivatives
      int nder;

      /// Scale the derivatives
      void scale_derivatives(arma::mat & f, double element_length) const;

    public:
      /// Constructor
      GeneralHIPBasis(const arma::vec & x0, int id, int nder);
      /// Destructor
      ~GeneralHIPBasis();
      /// Get a copy
      GeneralHIPBasis * copy() const override;

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
      /// Evaluate fourth derivatives of polynomials at given points
      void eval_prim_d4f(const arma::vec & x, arma::mat & d4f, double element_length) const override;
      /// Evaluate fifth derivatives of polynomials at given points
      void eval_prim_d5f(const arma::vec & x, arma::mat & d5f, double element_length) const override;
    };
  }
}
#endif
