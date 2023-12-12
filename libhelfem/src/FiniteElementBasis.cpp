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

#include "FiniteElementBasis.h"
#include <cfloat>

namespace helfem {
  namespace polynomial_basis {
    FiniteElementBasis::FiniteElementBasis() {
    }

    FiniteElementBasis::FiniteElementBasis(const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly_,
                                           const arma::vec &bval_, bool zero_func_left_, bool zero_deriv_left_, bool zero_func_right_, bool zero_deriv_right_) : bval(bval_), zero_func_left(zero_func_left_), zero_deriv_left(zero_deriv_left_), zero_func_right(zero_func_right_), zero_deriv_right(zero_deriv_right_) {
      poly = std::shared_ptr<const polynomial_basis::PolynomialBasis>(poly_->copy());
      // Update list of basis functions
      update_bf_list();
      // Check that basis functions are continuous
      check_bf_continuity();
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

    void FiniteElementBasis::check_bf_continuity() const {
      if(get_nelem()==1)
        return;
      int noverlap(poly->get_noverlap());

      arma::vec dnorm(get_nelem()-1);
      for(size_t iel=0; iel+1<get_nelem(); iel++) {
        // Points that correspond to lh and rh elements
        arma::vec xrh(1), xlh(1);
        xlh(0)=1.0;  // right-most point of left element
        xrh(0)=-1.0; // should equal left-most point of right element

        /// Check that coordinates match
        arma::vec rlh(eval_coord(xlh, iel));
        arma::vec rrh(eval_coord(xrh, iel+1));
        double dr(arma::norm(rlh-rrh,2));
        if(dr > 10*DBL_EPSILON*arma::norm(rlh,2)) {
          rlh.print("rlh");
          rrh.print("rrh");
          std::ostringstream oss;
          oss << "Coordinates do not match between elements " << iel << " and " << iel+1 << "!\n";
          throw std::logic_error(oss.str());
        }

        // Evaluate bordering value in lh element
        arma::mat lh(noverlap, noverlap);
        for(int ider=0;ider<noverlap;ider++) {
          // We want the last noverlap functions evaluated at the r
          arma::rowvec fval(eval_dnf(xlh, ider, iel));
          lh.col(ider) = fval.subvec(fval.n_elem-noverlap, fval.n_elem-1).t();
        }

        // Evaluate bordering value in rh element
        arma::mat rh(noverlap, noverlap);
        for(int ider=0;ider<noverlap;ider++) {
          // We want the first noverlap functions
          arma::rowvec fval(eval_dnf(xrh, ider, iel+1));
          rh.col(ider) = fval.subvec(0, noverlap-1).t();
        }

        // The function values should go to zero at the boundaries,
        // except the overlaid functions. The derivatives should also
        // go to zero, except the overlaid ones. The scaling does not
        // matter.
        arma::mat diff(lh-rh);
        dnorm(iel) = arma::norm(diff,2);
        if(dnorm(iel) > sqrt(DBL_EPSILON)) {
          printf("Discontinuity between elements %i and %i (C indexing)\n",(int) iel,(int) iel+1);
          lh.print("lh values");
          rh.print("rh values");
          diff.print("difference");
          printf("Difference norm %e\n",arma::norm(diff,2));
        }
      }
      //dnorm.t().print("Difference norms");
      arma::uword imax;
      dnorm.max(imax);
      printf("Finite element basis set max discontinuity %e between elements %i and %i\n",dnorm(imax),(int) imax,(int) imax+1);
      fflush(stdout);
      if(dnorm(imax) > sqrt(DBL_EPSILON)) {
        throw std::logic_error("Finite element basis set is not continuous\n");
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

    size_t FiniteElementBasis::find_element(double x) const {
      // Find the element x is in
      size_t element_left = 0;
      size_t element_right = get_nelem()-1;

      if(x <= element_end(element_left))
        return element_left;
      if(x >= element_begin(element_right))
        return element_right;

      size_t element_middle;
      while(true) {
        element_middle = (element_right + element_left) / 2;
        if(x >= element_begin(element_middle) && x <= element_end(element_middle))
          return element_middle;
        else if(x < element_begin(element_middle))
          element_right = element_middle;
        else
          element_left = element_middle;
      }
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
        p->drop_first(zero_func_left, zero_deriv_left);
      if (iel == bval.n_elem - 2)
        p->drop_last(zero_func_right, zero_deriv_right);

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
      eval_dnf(x,f,0,iel);
   }

    void FiniteElementBasis::eval_df(const arma::vec & x, arma::mat & df, size_t iel) const {
      eval_dnf(x,df,1,iel);
    }

    void FiniteElementBasis::eval_d2f(const arma::vec & x, arma::mat & d2f, size_t iel) const {
      eval_dnf(x,d2f,2,iel);
    }

    void FiniteElementBasis::eval_dnf(const arma::vec & x, arma::mat & dnf, int n, size_t iel) const {
      std::shared_ptr<polynomial_basis::PolynomialBasis> p(get_basis(iel));
      p->eval_dnf(x,dnf,n,scaling_factor(iel));
    }

    arma::mat FiniteElementBasis::eval_f(const arma::vec & x, size_t iel) const {
      return eval_dnf(x,0,iel);
    }

    arma::mat FiniteElementBasis::eval_df(const arma::vec & x, size_t iel) const {
      return eval_dnf(x,1,iel);
    }

    arma::mat FiniteElementBasis::eval_d2f(const arma::vec & x, size_t iel) const {
      return eval_dnf(x,2,iel);
    }

    arma::mat FiniteElementBasis::eval_dnf(const arma::vec & x, int n, size_t iel) const {
      std::shared_ptr<polynomial_basis::PolynomialBasis> p(get_basis(iel));
      return p->eval_dnf(x,n,scaling_factor(iel));
    }

    arma::mat FiniteElementBasis::matrix_element(const std::function<arma::mat(arma::vec,size_t)> & eval_lh, const std::function<arma::mat(arma::vec,size_t)> & eval_rh, const arma::vec & xq, const arma::vec & wq, const std::function<double(double)> & f) const {
      // Compute matrix elements in parallel
      std::vector<arma::mat> matel(get_nelem());
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for(size_t iel=0; iel<get_nelem(); iel++) {
        matel[iel] = matrix_element(iel, eval_lh, eval_rh, xq, wq, f);
      }

      // Fill in the global matrix
      arma::mat M(get_nbf(),get_nbf(),arma::fill::zeros);
      for(size_t iel=0; iel<get_nelem(); iel++) {
        // Indices in matrix
        size_t ifirst, ilast;
        get_idx(iel, ifirst, ilast);

        // Accumulate
        M.submat(ifirst, ifirst, ilast, ilast) += matel[iel];
      }

      return M;
    }

    arma::vec FiniteElementBasis::vector_element(const std::function<arma::mat(arma::vec,size_t)> & eval, const arma::vec & xq, const arma::vec & wq, const std::function<double(double)> & f) const {
      // Compute matrix elements in parallel
      std::vector<arma::vec> vecel(get_nelem());
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for(size_t iel=0; iel<get_nelem(); iel++) {
        vecel[iel] = vector_element(iel, eval, xq, wq, f);
      }

      // Fill in the global vector
      arma::vec V(get_nbf(),arma::fill::zeros);
      for(size_t iel=0; iel<get_nelem(); iel++) {
        // Indices in matrix
        size_t ifirst, ilast;
        get_idx(iel, ifirst, ilast);

        // Accumulate
        V.subvec(ifirst, ilast) += vecel[iel];
      }

      return V;
    }

    arma::mat FiniteElementBasis::matrix_element(int lhder, int rhder, const arma::vec & xq, const arma::vec & wq, const std::function<double(double)> & f) const {
      std::function<arma::mat(const arma::vec &,size_t)> eval_lh = [this,lhder](const arma::vec & x, size_t iel) {return this->eval_dnf(x, lhder, iel);};
      std::function<arma::mat(const arma::vec &,size_t)> eval_rh = [this,rhder](const arma::vec & x, size_t iel) {return this->eval_dnf(x, rhder, iel);};
      return matrix_element(eval_lh, eval_rh, xq, wq, f);
    }

    arma::mat FiniteElementBasis::matrix_element(size_t iel, int lhder, int rhder, const arma::vec & xq, const arma::vec & wq, const std::function<double(double)> & f) const {
      std::function<arma::mat(const arma::vec &,size_t)> eval_lh = [this,lhder](const arma::vec & x, size_t iel) {return this->eval_dnf(x, lhder, iel);};
      std::function<arma::mat(const arma::vec &,size_t)> eval_rh = [this,rhder](const arma::vec & x, size_t iel) {return this->eval_dnf(x, rhder, iel);};
      return matrix_element(iel, eval_lh, eval_rh, xq, wq, f);
    }

    arma::mat FiniteElementBasis::matrix_element(size_t iel, const std::function<arma::mat(arma::vec,size_t)> & eval_lh, const std::function<arma::mat(arma::vec,size_t)> & eval_rh, const arma::vec & xq, const arma::vec & wq, const std::function<double(double)> & f) const {
      // Get coordinate values
      arma::vec r(eval_coord(xq, iel));
      // Calculate total weight per point
      arma::vec wp(wq*scaling_factor(iel));
      // Include the function
      if(f) {
          for(size_t i=0; i<wp.n_elem; i++)
            wp(i)*=f(r(i));
      }

      // Evaluate basis functions
      if(!eval_lh)
        throw std::logic_error("Need function for evaluating left-hand basis functions!\n");
      arma::mat lhbf = eval_lh(xq, iel);
      if(!eval_rh)
        throw std::logic_error("Need function for evaluating right-hand basis functions!\n");
      arma::mat rhbf = eval_rh(xq, iel);

      // Include weight in the lh operand
      for(size_t i=0;i<lhbf.n_cols;i++)
        lhbf.col(i)%=wp;

      return arma::trans(lhbf)*rhbf;
    }

    arma::vec FiniteElementBasis::vector_element(size_t iel, const std::function<arma::mat(arma::vec,size_t)> & eval_bf, const arma::vec & xq, const arma::vec & wq, const std::function<double(double)> & f) const {
      // Get coordinate values
      arma::vec r(eval_coord(xq, iel));
      // Calculate total weight per point
      arma::vec wp(wq*scaling_factor(iel));
      // Include the function
      if(f) {
          for(size_t i=0; i<wp.n_elem; i++)
            wp(i)*=f(r(i));
      }

      // Evaluate basis functions
      if(!eval_bf)
        throw std::logic_error("Need function for evaluating basis functions!\n");
      arma::mat bf = eval_bf(xq, iel);

      return bf.t()*wp;
    }

    arma::vec FiniteElementBasis::vector_element(int der, const arma::vec & xq, const arma::vec & wq, const std::function<double(double)> & f) const {
      std::function<arma::mat(const arma::vec &,size_t)> eval_f = [this,der](const arma::vec & x, size_t iel) {return this->eval_dnf(x, der, iel);};
      return vector_element(eval_f, xq, wq, f);
    }

    arma::vec FiniteElementBasis::vector_element(size_t iel, int der, const arma::vec & xq, const arma::vec & wq, const std::function<double(double)> & f) const {
      std::function<arma::mat(const arma::vec &,size_t)> eval_f = [this,der](const arma::vec & x, size_t iel) {return this->eval_dnf(x, der, iel);};
      return vector_element(iel, eval_f, xq, wq, f);
    }

    void FiniteElementBasis::print(const std::string & str) const {
      printf("%s",str.c_str());
      bval.print("bval");
    }
  }
}
