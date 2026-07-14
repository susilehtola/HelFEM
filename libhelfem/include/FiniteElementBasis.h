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
#include <memory>
#include "PolynomialBasis.h"
#include "Matrix.h"
#include <lib1dfem/types.h>

namespace helfem {
  namespace polynomial_basis {
    /// Finite element basis set.
    ///
    /// Templated on the scalar type. Everything below this class -- lib1dfem's
    /// PolynomialBasis<T>, LIPBasis<T>, HIPBasis<T>, LegendreBasis<T>,
    /// lobatto_compute<T> -- was already generic; libhelfem was the only layer
    /// pinning T = double, which is what blocked running HelFEM in higher
    /// precision. Instantiated below for double, long double and __float128.
    template<typename T>
    class FiniteElementBasisT {
    protected:
      /// Polynomial basis
      std::shared_ptr<const helfem::lib1dfem::polynomial_basis::PolynomialBasis<T>> poly;

      /// Element boundary values
      // Phase 5.5: members migrated to Eigen.
      helfem::Vec<T> bval;
      /// Zero out function at left end?
      bool zero_func_left;
      /// Zero out derivative at left end?
      bool zero_deriv_left;
      /// Zero out function at right end?
      bool zero_func_right;
      /// Zero out derivatives at right end?
      bool zero_deriv_right;

      /// First basis function in element
      helfem::lib1dfem::IVec first_func_in_element;
      /// Last basis function in element
      helfem::lib1dfem::IVec last_func_in_element;
      /// Update the above list of basis functions
      void update_bf_list();

      /// Check that basis functions are continuous
      void check_bf_continuity() const;

      /// Used basis function indices in element
      helfem::lib1dfem::IVec basis_indices(size_t iel) const;

    public:
      /// Dummy constructor
      FiniteElementBasisT();
      /// Constructor. Phase 5.26: bval is Eigen at the public boundary
      /// (matches the internal helfem::Vec<T> storage introduced in
      /// Phase 5.5).
      FiniteElementBasisT(const std::shared_ptr<const helfem::lib1dfem::polynomial_basis::PolynomialBasis<T>> &poly,
                         const helfem::Vec<T> &bval, bool zero_func_left, bool zero_deriv_left, bool zero_func_right, bool zero_deriv_right);
      /// Destructor
      ~FiniteElementBasisT();

      /// Add an element boundary
      void add_boundary(T r);

      /// Get the polynomial basis
      std::shared_ptr<helfem::lib1dfem::polynomial_basis::PolynomialBasis<T>>  get_poly() const;
      /// Get basis functions in element
      std::shared_ptr<helfem::lib1dfem::polynomial_basis::PolynomialBasis<T>>
      get_basis(size_t iel) const;

      /// Get the numerical id of the polynomial basis
      int get_poly_id() const;
      /// Get the number of nodes in the polynomial basis
      int get_poly_nnodes() const;

      /// Get the element boundaries
      helfem::Vec<T> get_bval() const;
      /// Element begins at
      T element_begin(size_t iel) const;
      /// Element ends at
      T element_end(size_t iel) const;
      /// Element midpoint is at
      T element_midpoint(size_t iel) const;
      /// Element length
      T element_length(size_t iel) const;

      /// Find the element the point is at
      size_t find_element(T x) const;

      /// Evaluate real coordinate values from primitive coordinates
      helfem::Vec<T> eval_coord(const helfem::Vec<T> & xprim, size_t iel) const;
      /// Evaluate real coordinate values from primitive coordinates
      T eval_coord(T xprim, size_t iel) const;
      /// Evaluate full set of coordinate valuess from primitive coordinates
      helfem::Vec<T> eval_coord(const helfem::Vec<T> & xq) const;

      /// Evaluate primitive coordinate values from real coordinates
      helfem::Vec<T> eval_prim(const helfem::Vec<T> & xreal, size_t iel) const;

      /// Evaluate full set of weights for primitive weights
      helfem::Vec<T> eval_weights(const helfem::Vec<T> & wq) const;

      /// Element size scaling factor
      T scaling_factor(size_t iel) const;

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
      void eval_f  (const helfem::Vec<T> & x, helfem::Mat<T> & f,   size_t iel) const;
      /// Evaluate derivatives of polynomials at given points
      void eval_df (const helfem::Vec<T> & x, helfem::Mat<T> & df,  size_t iel) const;
      /// Evaluate second derivatives of polynomials at given points
      void eval_d2f(const helfem::Vec<T> & x, helfem::Mat<T> & d2f, size_t iel) const;
      /// Evaluate third derivatives of polynomials at given points
      void eval_d3f(const helfem::Vec<T> & x, helfem::Mat<T> & d3f, size_t iel) const;
      /// Evaluate fourth derivatives of polynomials at given points
      void eval_d4f(const helfem::Vec<T> & x, helfem::Mat<T> & d4f, size_t iel) const;
      /// Evaluate fifth derivatives of polynomials at given points
      void eval_d5f(const helfem::Vec<T> & x, helfem::Mat<T> & d5f, size_t iel) const;
      /// Evaluate nth derivative
      void eval_dnf(const helfem::Vec<T> & x, helfem::Mat<T> & dnf, int n, size_t iel) const;

      /// Evaluate polynomials at given points
      helfem::Mat<T> eval_f  (const helfem::Vec<T> & x, size_t iel) const;
      /// Evaluate derivatives of polynomials at given points
      helfem::Mat<T> eval_df (const helfem::Vec<T> & x, size_t iel) const;
      /// Evaluate second derivatives of polynomials at given points
      helfem::Mat<T> eval_d2f(const helfem::Vec<T> & x, size_t iel) const;
      /// Evaluate third derivatives of polynomials at given points
      helfem::Mat<T> eval_d3f(const helfem::Vec<T> & x, size_t iel) const;
      /// Evaluate fourth derivatives of polynomials at given points
      helfem::Mat<T> eval_d4f(const helfem::Vec<T> & x, size_t iel) const;
      /// Evaluate fifth derivatives of polynomials at given points
      helfem::Mat<T> eval_d5f(const helfem::Vec<T> & x, size_t iel) const;
      /// Evaluate nth derivative
      helfem::Mat<T> eval_dnf(const helfem::Vec<T> & x, int n, size_t iel) const;

      /// Evaluate the nth derivative at all quadrature points
      helfem::Mat<T> eval_dnf(const helfem::Vec<T> & xq, int n) const;

      /// Evaluate the n-th r-derivative of B_u(r)/r for the surviving shape
      /// functions on element iel. Only valid when element iel starts at
      /// r=0 (the Dirichlet-induced (x+1) factor in each B_u is what makes
      /// the deflation work). Delegates to PolynomialBasis::eval_over_r
      /// with the element's scaling_factor as the size argument.
      helfem::Mat<T> eval_over_r(const helfem::Vec<T> & x, int n, size_t iel) const;

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
      // helfem::Mat<T>(helfem::Vec<T>, size_t).
      helfem::Mat<T> matrix_element(int lhder, int rhder, const helfem::Vec<T> & xq, const helfem::Vec<T> & wq, const std::function<T(T)> & f) const;
      /// Same as above, but only in a single element
      helfem::Mat<T> matrix_element(size_t iel, int lhder, int rhder, const helfem::Vec<T> & xq, const helfem::Vec<T> & wq, const std::function<T(T)> & f) const;

      /**
       * Compute vector elements in the finite element basis <lh|f|rh>
       *
       * der:   use der derivative of basis function
       * xq:    quadrature nodes
       * wq:    quadrature weights
       * f(r):  additional weight function, use nullptr for unit weight
       */
      helfem::Vec<T> vector_element(int der, const helfem::Vec<T> & xq, const helfem::Vec<T> & wq, const std::function<T(T)> & f) const;
      /// Same as above, but only in a single element
      helfem::Vec<T> vector_element(size_t iel, int der, const helfem::Vec<T> & xq, const helfem::Vec<T> & wq, const std::function<T(T)> & f) const;

      /**
       * Compute matrix elements in the finite element basis <lh|f|rh>
       *
       * eval_lh: function to evaluate lh basis functions
       * eval_rh: function to evaluate rh basis functions
       * xq:    quadrature nodes
       * wq:    quadrature weights
       * f(r):  additional weight function, use nullptr for unit weight
       */
      helfem::Mat<T> matrix_element(const std::function<helfem::Mat<T>(helfem::Vec<T>,size_t)> & eval_lh, const std::function<helfem::Mat<T>(helfem::Vec<T>,size_t)> & eval_rh, const helfem::Vec<T> & xq, const helfem::Vec<T> & wq, const std::function<T(T)> & f) const;
      /// The driver function
      helfem::Mat<T> matrix_element(size_t iel, const std::function<helfem::Mat<T>(helfem::Vec<T>,size_t)> & eval_lh, const std::function<helfem::Mat<T>(helfem::Vec<T>,size_t)> & eval_rh, const helfem::Vec<T> & xq, const helfem::Vec<T> & wq, const std::function<T(T)> & f, T x_left = T(-1.0), T x_right = T(1.0)) const;


      /**
       * Compute vector elements in the finite element basis <bf|f>
       *
       * eval_bf: function to evaluate basis functions
       * xq:    quadrature nodes
       * wq:    quadrature weights
       * f(r):  additional weight function, use nullptr for unit weight
       */
      helfem::Vec<T> vector_element(const std::function<helfem::Mat<T>(helfem::Vec<T>,size_t)> & eval_bf, const helfem::Vec<T> & xq, const helfem::Vec<T> & wq, const std::function<T(T)> & f) const;
      /// The driver function
      helfem::Vec<T> vector_element(size_t iel, const std::function<helfem::Mat<T>(helfem::Vec<T>,size_t)> & eval_bf, const helfem::Vec<T> & xq, const helfem::Vec<T> & wq, const std::function<T(T)> & f) const;

      /// Print out the basis functions
      void print(const std::string & str="") const;
    };

    /// The double instantiation, which every existing caller uses.
    using FiniteElementBasis = FiniteElementBasisT<double>;
  }
}
#endif
