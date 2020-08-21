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
#ifndef POLYNOMIAL_BASIS_POLYNOMIALBASIS_H
#define POLYNOMIAL_BASIS_POLYNOMIALBASIS_H

#include <armadillo>

namespace helfem {
  namespace polynomial_basis {
    /// Template for a primitive basis
    class PolynomialBasis {
    protected:
      /// Number of basis functions
      int nbf;
      /// Number of overlapping functions
      int noverlap;
      /// Identifier
      int id;
      /// Order
      int order;
    public:
      /// Constructor
      PolynomialBasis();
      /// Destructor
      virtual ~PolynomialBasis();
      /// Get a copy
      virtual PolynomialBasis * copy() const=0;

      /// Get number of basis functions
      int get_nbf() const;
      /// Get number of overlapping functions
      int get_noverlap() const;

      /// Get identifier
      int get_id() const;
      /// Get order
      int get_order() const;

      /// Drop first function
      virtual void drop_first()=0;
      /// Drop last function
      virtual void drop_last()=0;

      /// Evaluate polynomials at given points
      virtual arma::mat eval(const arma::vec & x) const=0;
      /// Evaluate polynomials and derivatives at given points
      virtual void eval(const arma::vec & x, arma::mat & f, arma::mat & df) const=0;
      /// Evaluate second derivatives at given point
      virtual void eval_lapl(const arma::vec & x, arma::mat & lf) const;

      /// Print out the basis functions
      void print(const std::string & str="") const;
    };
  }
}
#endif
