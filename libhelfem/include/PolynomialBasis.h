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
      /// Number of primitive functions
      int nprim;
      /// List of enabled functions
      arma::uvec enabled;
      /// Number of overlapping functions
      int noverlap;
      /// Identifier
      int id;
      /// Number of nodes
      int nnodes;

      /// Evaluate nth derivatives of primitive polynomials at given points
      virtual void eval_prim_dnf(const arma::vec & x, arma::mat & dnf, int n, double element_length) const;

    public:
      /// Constructor
      PolynomialBasis();
      /// Destructor
      virtual ~PolynomialBasis();
      /// Get a copy
      virtual PolynomialBasis * copy() const=0;

      /// Get number of primitive functions
      int get_nprim() const;
      /// Get number of basis functions
      int get_nbf() const;
      /// Get number of overlapping functions
      int get_noverlap() const;

      /// Get identifier
      int get_id() const;
      /// Get number of nodes
      int get_nnodes() const;

      /// Get list of enabled primitives
      arma::uvec get_enabled() const;
      /// Drop first function(s); zero_deriv: also set derivatives to zero
      virtual void drop_first(bool zero_func, bool zero_deriv)=0;
      /// Drop last function(s); zero_deriv: also set derivatives to zero
      virtual void drop_last(bool zero_func, bool zero_deriv)=0;

      /// Evaluate nth derivatives of polynomials at given points
      void eval_dnf(const arma::vec & x, arma::mat & dnf, int n, double element_length) const;
      /// Evaluate nth derivatives of polynomials at given points
      arma::mat eval_dnf(const arma::vec & x, int n, double element_length) const;

      /// Print out the basis functions
      void print(const std::string & str="") const;
    };

    /// Get the wanted polynomial basis
    PolynomialBasis * get_basis(int primbas, int Nnodes);
  }
}
#endif
