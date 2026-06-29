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
 * SPDX-License-Identifier: BSD-3-Clause
 * See the LICENSE file at the root of this source distribution
 * for the full license text.
 */
#ifndef POLYNOMIAL_BASIS_FINITEELEMENTBASIS_H
#define POLYNOMIAL_BASIS_FINITEELEMENTBASIS_H

#include <functional>
#include <armadillo>
#include <memory>
#include "PolynomialBasis.h"
#include "Matrix.h"

namespace helfem {
  namespace polynomial_basis {
    /// Finite element basis set
    class FiniteElementBasis {
    protected:
      /// Polynomial basis
      std::shared_ptr<const polynomial_basis::PolynomialBasis> poly;

      /// Element boundary values
      arma::vec bval;
      /// Zero out function at left end?
      bool zero_func_left;
      /// Zero out derivative at left end?
      bool zero_deriv_left;
      /// Zero out function at right end?
      bool zero_func_right;
      /// Zero out derivatives at right end?
      bool zero_deriv_right;

      /// First basis function in element
      arma::uvec first_func_in_element;
      /// Last basis function in element
      arma::uvec last_func_in_element;
      /// Update the above list of basis functions
      void update_bf_list();

      /// Check that basis functions are continuous
      void check_bf_continuity() const;

      /// Used basis function indices in element
      arma::uvec basis_indices(size_t iel) const;

    public:
      /// Dummy constructor
      FiniteElementBasis();
      /// Constructor
      FiniteElementBasis(const std::shared_ptr<const polynomial_basis::PolynomialBasis> &poly,
                         const arma::vec &bval, bool zero_func_left, bool zero_deriv_left, bool zero_func_right, bool zero_deriv_right);
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
      /// Get the number of nodes in the polynomial basis
      int get_poly_nnodes() const;

      /// Get the used subset of primitives in the element
      arma::mat get_basis(const arma::mat &bas, size_t iel) const;

      /// Get the element boundaries
      helfem::Vector get_bval() const;
      /// Element begins at
      double element_begin(size_t iel) const;
      /// Element ends at
      double element_end(size_t iel) const;
      /// Element midpoint is at
      double element_midpoint(size_t iel) const;
      /// Element length
      double element_length(size_t iel) const;

      /// Find the element the point is at
      size_t find_element(double x) const;

      // Phase 5.3: per-element evaluators migrated arma -> Eigen.
      /// Evaluate real coordinate values from primitive coordinates
      helfem::Vector eval_coord(const helfem::Vector & xprim, size_t iel) const;
      /// Evaluate real coordinate values from primitive coordinates
      double eval_coord(double xprim, size_t iel) const;
      /// Evaluate full set of coordinate valuess from primitive coordinates
      helfem::Vector eval_coord(const helfem::Vector & xq) const;

      /// Evaluate primitive coordinate values from real coordinates
      helfem::Vector eval_prim(const helfem::Vector & xreal, size_t iel) const;

      /// Evaluate full set of weights for primitive weights
      helfem::Vector eval_weights(const helfem::Vector & wq) const;

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
      void eval_f  (const helfem::Vector & x, helfem::Matrix & f,   size_t iel) const;
      /// Evaluate derivatives of polynomials at given points
      void eval_df (const helfem::Vector & x, helfem::Matrix & df,  size_t iel) const;
      /// Evaluate second derivatives of polynomials at given points
      void eval_d2f(const helfem::Vector & x, helfem::Matrix & d2f, size_t iel) const;
      /// Evaluate third derivatives of polynomials at given points
      void eval_d3f(const helfem::Vector & x, helfem::Matrix & d3f, size_t iel) const;
      /// Evaluate fourth derivatives of polynomials at given points
      void eval_d4f(const helfem::Vector & x, helfem::Matrix & d4f, size_t iel) const;
      /// Evaluate fifth derivatives of polynomials at given points
      void eval_d5f(const helfem::Vector & x, helfem::Matrix & d5f, size_t iel) const;
      /// Evaluate nth derivative
      void eval_dnf(const helfem::Vector & x, helfem::Matrix & dnf, int n, size_t iel) const;

      /// Evaluate polynomials at given points
      helfem::Matrix eval_f  (const helfem::Vector & x, size_t iel) const;
      /// Evaluate derivatives of polynomials at given points
      helfem::Matrix eval_df (const helfem::Vector & x, size_t iel) const;
      /// Evaluate second derivatives of polynomials at given points
      helfem::Matrix eval_d2f(const helfem::Vector & x, size_t iel) const;
      /// Evaluate third derivatives of polynomials at given points
      helfem::Matrix eval_d3f(const helfem::Vector & x, size_t iel) const;
      /// Evaluate fourth derivatives of polynomials at given points
      helfem::Matrix eval_d4f(const helfem::Vector & x, size_t iel) const;
      /// Evaluate fifth derivatives of polynomials at given points
      helfem::Matrix eval_d5f(const helfem::Vector & x, size_t iel) const;
      /// Evaluate nth derivative
      helfem::Matrix eval_dnf(const helfem::Vector & x, int n, size_t iel) const;

      /// Evaluate the nth derivative at all quadrature points
      helfem::Matrix eval_dnf(const helfem::Vector & xq, int n) const;

      /// Evaluate the n-th r-derivative of B_u(r)/r for the surviving shape
      /// functions on element iel. Only valid when element iel starts at
      /// r=0 (the Dirichlet-induced (x+1) factor in each B_u is what makes
      /// the deflation work). Delegates to PolynomialBasis::eval_over_r
      /// with the element's scaling_factor as the size argument.
      helfem::Matrix eval_over_r(const helfem::Vector & x, int n, size_t iel) const;

      /**
       * Compute matrix elements in the finite element basis <lh|f|rh>
       *
       * lhder: use lhder derivative of lh basis function
       * rhder: use rhder derivative of rh basis function
       * xq:    quadrature nodes
       * wq:    quadrature weights
       * f(r):  additional weight function, use nullptr for unit weight
       */
      // Phase 5.4: matrix_element / vector_element overloads migrated to
      // Eigen. The fn-pointer overload's lambda signature is now
      // helfem::Matrix(helfem::Vector, size_t).
      helfem::Matrix matrix_element(int lhder, int rhder, const helfem::Vector & xq, const helfem::Vector & wq, const std::function<double(double)> & f) const;
      /// Same as above, but only in a single element
      helfem::Matrix matrix_element(size_t iel, int lhder, int rhder, const helfem::Vector & xq, const helfem::Vector & wq, const std::function<double(double)> & f) const;

      /**
       * Compute vector elements in the finite element basis <lh|f|rh>
       *
       * der:   use der derivative of basis function
       * xq:    quadrature nodes
       * wq:    quadrature weights
       * f(r):  additional weight function, use nullptr for unit weight
       */
      helfem::Vector vector_element(int der, const helfem::Vector & xq, const helfem::Vector & wq, const std::function<double(double)> & f) const;
      /// Same as above, but only in a single element
      helfem::Vector vector_element(size_t iel, int der, const helfem::Vector & xq, const helfem::Vector & wq, const std::function<double(double)> & f) const;

      /**
       * Compute matrix elements in the finite element basis <lh|f|rh>
       *
       * eval_lh: function to evaluate lh basis functions
       * eval_rh: function to evaluate rh basis functions
       * xq:    quadrature nodes
       * wq:    quadrature weights
       * f(r):  additional weight function, use nullptr for unit weight
       */
      helfem::Matrix matrix_element(const std::function<helfem::Matrix(helfem::Vector,size_t)> & eval_lh, const std::function<helfem::Matrix(helfem::Vector,size_t)> & eval_rh, const helfem::Vector & xq, const helfem::Vector & wq, const std::function<double(double)> & f) const;
      /// The driver function
      helfem::Matrix matrix_element(size_t iel, const std::function<helfem::Matrix(helfem::Vector,size_t)> & eval_lh, const std::function<helfem::Matrix(helfem::Vector,size_t)> & eval_rh, const helfem::Vector & xq, const helfem::Vector & wq, const std::function<double(double)> & f, double x_left = -1.0, double x_right = 1.0) const;


      /**
       * Compute vector elements in the finite element basis <bf|f>
       *
       * eval_bf: function to evaluate basis functions
       * xq:    quadrature nodes
       * wq:    quadrature weights
       * f(r):  additional weight function, use nullptr for unit weight
       */
      helfem::Vector vector_element(const std::function<helfem::Matrix(helfem::Vector,size_t)> & eval_bf, const helfem::Vector & xq, const helfem::Vector & wq, const std::function<double(double)> & f) const;
      /// The driver function
      helfem::Vector vector_element(size_t iel, const std::function<helfem::Matrix(helfem::Vector,size_t)> & eval_bf, const helfem::Vector & xq, const helfem::Vector & wq, const std::function<double(double)> & f) const;

      /// Print out the basis functions
      void print(const std::string & str="") const;
    };
  }
}
#endif
