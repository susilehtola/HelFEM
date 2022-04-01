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
#ifndef POLYNOMIAL_BASIS_FINITEELEMENTBASIS_H
#define POLYNOMIAL_BASIS_FINITEELEMENTBASIS_H

#include <armadillo>
#include <memory>
#include "PolynomialBasis.h"

namespace helfem {
  namespace polynomial_basis {
    /// Finite element basis set
    class FiniteElementBasis {
    protected:
      /// Polynomial basis
      std::shared_ptr<const polynomial_basis::PolynomialBasis> poly;

      /// Element boundary values
      arma::vec bval;
      /// Zero out derivatives at left end?
      bool zero_deriv_left;
      /// Zero out derivatives at right end?
      bool zero_deriv_right;

      /// First basis function in element
      arma::uvec first_func_in_element;
      /// Last basis function in element
      arma::uvec last_func_in_element;
      /// Update the above list of basis functions
      void update_bf_list();

      /// Used basis function indices in element
      arma::uvec basis_indices(size_t iel) const;

    public:
      /// Dummy constructor
      FiniteElementBasis();
      /// Constructor
      FiniteElementBasis(const std::shared_ptr<const polynomial_basis::PolynomialBasis> &poly,
                         const arma::vec &bval, bool zero_deriv_left=true, bool zero_deriv_right=true);
      /// Destructor
      ~FiniteElementBasis();

      /// Add an element boundary
      void add_boundary(double r);

      /// Get the polynomial basis
      std::shared_ptr<polynomial_basis::PolynomialBasis>  get_poly() const;
      /// Get basis functions in element
      std::shared_ptr<polynomial_basis::PolynomialBasis>
      get_basis(size_t iel) const;

      /// Get the numerical id of the polynomial basis
      int get_poly_id() const;
      /// Get the polynomial order of the polynomial basis
      int get_poly_order() const;

      /// Get the used subset of primitives in the element
      arma::mat get_basis(const arma::mat &bas, size_t iel) const;

      /// Get the element boundaries
      arma::vec get_bval() const;
      /// Element begins at
      double element_begin(size_t iel) const;
      /// Element ends at
      double element_end(size_t iel) const;
      /// Element midpoint is at
      double element_midpoint(size_t iel) const;
      /// Element length
      double element_length(size_t iel) const;

      /// Evaluate real coordinate values from primitive coordinates
      arma::vec eval_coord(const arma::vec & xprim, size_t iel) const;
      /// Evaluate primitive coordinate values from real coordinates
      arma::vec eval_prim(const arma::vec & xreal, size_t iel) const;
      /// Element size scaling factor
      double scaling_factor(size_t iel) const;

      /// Get the consecutive index range of the basis functions in the element
      void get_idx(size_t iel, size_t &ifirst, size_t &ilast) const;

      /// Get maximum possible number of primitives
      size_t get_max_nprim() const;
      /// Get number of functions in element
      size_t get_nprim(size_t iel) const;
      /// Get number of basis functions
      size_t get_nbf() const;
      /// Get number of elements
      size_t get_nelem() const;

      /// Evaluate polynomials at given points
      void eval_f(const arma::vec & x, arma::mat & f, size_t iel) const;
      /// Evaluate derivatives of polynomials at given points
      void eval_df(const arma::vec & x, arma::mat & df, size_t iel) const;
      /// Evaluate second derivatives of polynomials at given points
      void eval_d2f(const arma::vec & x, arma::mat & d2f, size_t iel) const;

      /// Evaluate polynomials at given points
      arma::mat eval_f(const arma::vec & x, size_t iel) const;
      /// Evaluate derivatives of polynomials at given points
      arma::mat eval_df(const arma::vec & x, size_t iel) const;
      /// Evaluate second derivatives of polynomials at given points
      arma::mat eval_d2f(const arma::vec & x, size_t iel) const;

      /**
       * Compute matrix elements in the finite element basis <lh|f|rh>
       *
       * lhder: instead of basis function, use basis function derivative
       * rhder: instead of basis function, use basis function derivative
       * xq:    quadrature nodes
       * wq:    quadrature weights
       * f(r):  additional weight function, use nullptr for unit weight
       */
      arma::mat matrix_element(bool lhder, bool rhder, const arma::vec & xq, const arma::vec & wq, double (*f)(double)) const;
      /// Same as above, but only in a single element
      arma::mat matrix_element(size_t iel, bool lhder, bool rhder, const arma::vec & xq, const arma::vec & wq, double (*f)(double)) const;

      /// Print out the basis functions
      void print(const std::string & str="") const;
    };
  }
}
#endif
