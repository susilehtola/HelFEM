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
#ifndef POLYNOMIAL_BASIS_LEGENDREBASIS_H
#define POLYNOMIAL_BASIS_LEGENDREBASIS_H

#include "helfem/PolynomialBasis.h"
#include <armadillo>

namespace helfem {
  namespace polynomial_basis {/// Legendre functions
    class LegendreBasis: public PolynomialBasis {
      /// Maximum order
      int lmax;
      /// Transformation matrix
      arma::mat T;

      /// Evaluate Legendre polynomials
      arma::mat f_eval(const arma::vec & x) const;
      /// Evaluate Legendre polynomials' derivatives
      arma::mat df_eval(const arma::vec & x) const;
      /// Evaluate Legendre polynomials' second derivatives
      arma::mat lf_eval(const arma::vec & x) const;
    public:
      /// Constructor
      LegendreBasis(int nfuncs, int id);
      /// Destructor
      ~LegendreBasis();
      /// Get a copy
      LegendreBasis * copy() const override;

      /// Drop first function
      void drop_first() override;
      /// Drop last function
      void drop_last() override;

      /// Evaluate polynomials at given points
      arma::mat eval(const arma::vec & x) const override;
      /// Evaluate polynomials and derivatives at given points
      void eval(const arma::vec & x, arma::mat & f, arma::mat & df) const override;
      /// Evaluate second derivatives at given points
      void eval_lapl(const arma::vec & x, arma::mat & lf) const override;
    };
  }
}
#endif
