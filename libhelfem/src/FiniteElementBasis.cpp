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

#include "helfem/FiniteElementBasis.h"

namespace helfem {
  namespace polynomial_basis {
    FiniteElementBasis::FiniteElementBasis() {
    }

    FiniteElementBasis::FiniteElementBasis(const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly_,
                                           const arma::vec &bval_, bool zero_deriv_left_, bool zero_deriv_right_) : bval(bval_), zero_deriv_left(zero_deriv_left_), zero_deriv_right(zero_deriv_right_) {
      poly = std::shared_ptr<const polynomial_basis::PolynomialBasis>(poly_->copy());
      // Update list of basis functions
      update_bf_list();
    }

    FiniteElementBasis::~FiniteElementBasis() {
    }

    void FiniteElementBasis::update_bf_list() {
      if(bval.n_elem==0)
        throw std::logic_error("Can't update basis function list since there are no elements!\n");

      // Form list of element boundaries
      first_func_in_element.zeros(bval.n_elem-1);
      last_func_in_element.zeros(bval.n_elem-1);
      for(size_t iel=0; iel<first_func_in_element.n_elem; iel++) {
        // First func is
        first_func_in_element[iel] = (iel == 0) ? 0 : last_func_in_element[iel-1] - poly->get_noverlap() + 1;
        // Last func is
        last_func_in_element[iel] = first_func_in_element[iel] + basis_indices(iel).n_elem - 1;
      }
    }

    void FiniteElementBasis::get_idx(size_t iel, size_t &ifirst, size_t &ilast) const {
      ifirst = first_func_in_element[iel];
      ilast = last_func_in_element[iel];
    }

    void FiniteElementBasis::add_boundary(double r) {
      // Check that r is not in bval
      bool in_bval = false;
      for (size_t i = 0; i < bval.n_elem; i++)
        if (bval(i) == r)
          in_bval = true;

      // Add
      if (!in_bval) {
        arma::vec newbval(bval.n_elem + 1);
        newbval.subvec(0, bval.n_elem - 1) = bval;
        newbval(bval.n_elem) = r;
        bval = arma::sort(newbval, "ascend");
        update_bf_list();
      }
    }

    std::shared_ptr<polynomial_basis::PolynomialBasis> FiniteElementBasis::get_poly() const { return std::shared_ptr<polynomial_basis::PolynomialBasis>(poly->copy()); }

    double FiniteElementBasis::scaling_factor(size_t iel) const {
      // The primitive range is [-1, 1] leading to the factor 2
      return element_length(iel)/2;
    }

    double FiniteElementBasis::element_length(size_t iel) const {
      if(iel>=get_nelem()) {
        std::ostringstream oss;
        oss << "Trying to access length of element " << iel << " but only have " << get_nelem() << "!\n";
        throw std::logic_error(oss.str());
      }
      return bval(iel+1)-bval(iel);
    }

    double FiniteElementBasis::element_begin(size_t iel) const {
      if(iel>=get_nelem()) {
        std::ostringstream oss;
        oss << "Trying to access length of element " << iel << " but only have " << get_nelem() << "!\n";
        throw std::logic_error(oss.str());
      }
      return bval(iel);
    }

    double FiniteElementBasis::element_end(size_t iel) const {
      if(iel>=get_nelem()) {
        std::ostringstream oss;
        oss << "Trying to access length of element " << iel << " but only have " << get_nelem() << "!\n";
        throw std::logic_error(oss.str());
      }
      return bval(iel+1);
    }

    double FiniteElementBasis::element_midpoint(size_t iel) const {
      return 0.5*(element_begin(iel) + element_end(iel));
    }

    arma::vec FiniteElementBasis::get_bval() const {
      return bval;
    }

    arma::vec FiniteElementBasis::eval_coord(const arma::vec & x, size_t iel) const {
      // The coordinates are
      return element_midpoint(iel) * arma::ones<arma::vec>(x.n_elem) + scaling_factor(iel) * x;
    }

    arma::vec FiniteElementBasis::eval_prim(const arma::vec & y, size_t iel) const {
      if(arma::min(y) < element_begin(iel) || arma::max(y) > element_end(iel)) {
        throw std::logic_error("coordinates don't correspond to this element!\n");
      }
      // The primitive coordinates are thus
      return ((y - element_midpoint(iel) * arma::ones<arma::vec>(y.n_elem)) / scaling_factor(iel));
    }

    int FiniteElementBasis::get_poly_id() const {
      return poly->get_id();
    }

    int FiniteElementBasis::get_poly_nnodes() const {
      return poly->get_nnodes();
    }

    arma::uvec FiniteElementBasis::basis_indices(size_t iel) const {
      std::shared_ptr<polynomial_basis::PolynomialBasis> p(get_basis(iel));
      return p->get_enabled();
    }

    arma::mat FiniteElementBasis::get_basis(const arma::mat &bas, size_t iel) const {
      arma::uvec idx(basis_indices(iel));
      return bas.cols(idx);
    }

    std::shared_ptr<polynomial_basis::PolynomialBasis>
    FiniteElementBasis::get_basis(size_t iel) const {
      std::shared_ptr<polynomial_basis::PolynomialBasis> p(poly->copy());
      if (iel == 0)
        p->drop_first(zero_deriv_left);
      if (iel == bval.n_elem - 2)
        p->drop_last(zero_deriv_right);

      return p;
    }

    size_t FiniteElementBasis::get_nbf() const {
      // The number of function is just the index of the last function + 1
      if(last_func_in_element.n_elem==0)
        throw std::logic_error("Basis function list has not been filled\n");
      return last_func_in_element[last_func_in_element.n_elem-1]+1;
    }

    size_t FiniteElementBasis::get_nelem() const {
      return bval.n_elem-1;
    }

    size_t FiniteElementBasis::get_max_nprim() const {
      return poly->get_nprim();
    }

    size_t FiniteElementBasis::get_nprim(size_t iel) const {
      return get_basis(iel)->get_nbf();
    }

    void FiniteElementBasis::eval_f(const arma::vec & x, arma::mat & f, size_t iel) const {
      std::shared_ptr<polynomial_basis::PolynomialBasis> p(get_basis(iel));
      p->eval_f(x,f,scaling_factor(iel));
   }

    void FiniteElementBasis::eval_df(const arma::vec & x, arma::mat & df, size_t iel) const {
      std::shared_ptr<polynomial_basis::PolynomialBasis> p(get_basis(iel));
      p->eval_df(x,df,scaling_factor(iel));
    }

    void FiniteElementBasis::eval_d2f(const arma::vec & x, arma::mat & d2f, size_t iel) const {
      std::shared_ptr<polynomial_basis::PolynomialBasis> p(get_basis(iel));
      p->eval_d2f(x,d2f,scaling_factor(iel));
    }

    arma::mat FiniteElementBasis::eval_f(const arma::vec & x, size_t iel) const {
      std::shared_ptr<polynomial_basis::PolynomialBasis> p(get_basis(iel));
      return p->eval_f(x,scaling_factor(iel));
    }

    arma::mat FiniteElementBasis::eval_df(const arma::vec & x, size_t iel) const {
      std::shared_ptr<polynomial_basis::PolynomialBasis> p(get_basis(iel));
      return p->eval_df(x,scaling_factor(iel));
    }

    arma::mat FiniteElementBasis::eval_d2f(const arma::vec & x, size_t iel) const {
      std::shared_ptr<polynomial_basis::PolynomialBasis> p(get_basis(iel));
      return p->eval_d2f(x,scaling_factor(iel));
    }

    arma::mat FiniteElementBasis::matrix_element(bool lhder, bool rhder, const arma::vec & xq, const arma::vec & wq, const std::function<double(double)> & f) const {
      arma::mat M(get_nbf(),get_nbf(),arma::fill::zeros);

      for(size_t iel=0; iel<get_nelem(); iel++) {
        // Indices in matrix
        size_t ifirst, ilast;
        get_idx(iel, ifirst, ilast);

        // Accumulate
        M.submat(ifirst, ifirst, ilast, ilast) += matrix_element(iel, lhder, rhder, xq, wq, f);
      }

      return M;
    }

    arma::mat FiniteElementBasis::matrix_element(size_t iel, bool lhder, bool rhder, const arma::vec & xq, const arma::vec & wq, const std::function<double(double)> & f) const {
      // Get coordinate values
      arma::vec r(eval_coord(xq, iel));
      // Calculate total weight per point
      arma::vec wp(wq*scaling_factor(iel));
      // Include the function
      if(f) {
          for(size_t i=0; i<wp.n_elem; i++)
            wp(i)*=f(r(i));
      }

      // Operands
      arma::mat lhbf = lhder ? eval_df(xq, iel) : eval_f(xq, iel);
      arma::mat rhbf = rhder ? eval_df(xq, iel) : eval_f(xq, iel);

      // Include weight in the lh operand
      for(size_t i=0;i<lhbf.n_cols;i++)
        lhbf.col(i)%=wp;

      return arma::trans(lhbf)*rhbf;
    }
  }
}
