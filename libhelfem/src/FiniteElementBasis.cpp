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

#include "FiniteElementBasis.h"
#include <cfloat>
#include <cstring>

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
        // Phase 5.3: eval_coord / eval_dnf now Eigen; build the
        // 1-element x vectors as Eigen for direct use, materialise
        // arma copies only for the existing logging code.
        helfem::Vector xlh_e(1), xrh_e(1);
        xlh_e(0) = 1.0;
        xrh_e(0) = -1.0;

        /// Check that coordinates match
        helfem::Vector rlh_e = eval_coord(xlh_e, iel);
        helfem::Vector rrh_e = eval_coord(xrh_e, iel + 1);
        arma::vec rlh(rlh_e.size()); std::memcpy(rlh.memptr(), rlh_e.data(), sizeof(double) * rlh_e.size());
        arma::vec rrh(rrh_e.size()); std::memcpy(rrh.memptr(), rrh_e.data(), sizeof(double) * rrh_e.size());
        double dr(arma::norm(rlh-rrh,2));
        if(dr > 10*DBL_EPSILON*(1+arma::norm(rlh,2))) {
          rlh.print("rlh");
          rrh.print("rrh");
          (rrh-rlh).print("rrh-rlh");
          std::ostringstream oss;
          oss << "Coordinates do not match between elements " << iel << " and " << iel+1 << ", difference " << dr << " tolerance " << 100*DBL_EPSILON*arma::norm(rlh,2) << "!\n";
          throw std::logic_error(oss.str());
        }

        // Evaluate bordering value in lh element
        arma::mat lh(noverlap, noverlap);
        for(int ider=0;ider<noverlap;ider++) {
          // We want the last noverlap functions evaluated at the r
          helfem::Matrix fval_e = eval_dnf(xlh_e, ider, iel);
          arma::rowvec fval(fval_e.cols());
          std::memcpy(fval.memptr(), fval_e.data(), sizeof(double) * fval_e.size());
          lh.col(ider) = fval.subvec(fval.n_elem-noverlap, fval.n_elem-1).t();
        }

        // Evaluate bordering value in rh element
        arma::mat rh(noverlap, noverlap);
        for(int ider=0;ider<noverlap;ider++) {
          // We want the first noverlap functions
          helfem::Matrix fval_e = eval_dnf(xrh_e, ider, iel + 1);
          arma::rowvec fval(fval_e.cols());
          std::memcpy(fval.memptr(), fval_e.data(), sizeof(double) * fval_e.size());
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

    // Phase 5.3: per-element evaluators migrated arma -> Eigen.
    helfem::Vector FiniteElementBasis::eval_coord(const helfem::Vector & x, size_t iel) const {
      return helfem::Vector::Constant(x.size(), element_midpoint(iel)) + scaling_factor(iel) * x;
    }

    double FiniteElementBasis::eval_coord(double x, size_t iel) const {
      return element_midpoint(iel) + scaling_factor(iel) * x;
    }

    helfem::Vector FiniteElementBasis::eval_coord(const helfem::Vector & x) const {
      helfem::Vector r(get_nelem() * x.size());
      for (size_t iel = 0; iel < get_nelem(); ++iel)
        r.segment(iel * x.size(), x.size()) = eval_coord(x, iel);
      return r;
    }

    arma::vec FiniteElementBasis::eval_weights(const arma::vec & w) const {
      arma::vec wr(get_nelem()*w.n_elem);
      for(size_t iel=0;iel<get_nelem();iel++)
        wr.subvec(iel*w.n_elem, (iel+1)*w.n_elem-1) = w*scaling_factor(iel);
      return wr;
    }

    helfem::Vector FiniteElementBasis::eval_prim(const helfem::Vector & y, size_t iel) const {
      if (y.minCoeff() < element_begin(iel) || y.maxCoeff() > element_end(iel)) {
        throw std::logic_error("coordinates don't correspond to this element!\n");
      }
      return (y - helfem::Vector::Constant(y.size(), element_midpoint(iel))) / scaling_factor(iel);
    }

    int FiniteElementBasis::get_poly_id() const {
      return poly->get_id();
    }

    int FiniteElementBasis::get_poly_nnodes() const {
      return poly->get_nnodes();
    }

    arma::uvec FiniteElementBasis::basis_indices(size_t iel) const {
      // Phase 5.2: lib1dfem PolynomialBasis::enabled is IVec (Eigen);
      // bridge to arma::uvec at the libhelfem boundary.
      std::shared_ptr<polynomial_basis::PolynomialBasis> p(get_basis(iel));
      const auto & ie = p->get_enabled();
      arma::uvec out(ie.size());
      for (Eigen::Index i = 0; i < ie.size(); ++i)
        out(i) = static_cast<arma::uword>(ie(i));
      return out;
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

    // Phase 5.3: eval_* migrated to Eigen; internal lib1dfem call no longer
    // bridges (its signature matches).
    void FiniteElementBasis::eval_f  (const helfem::Vector & x, helfem::Matrix & f,   size_t iel) const { eval_dnf(x, f,   0, iel); }
    void FiniteElementBasis::eval_df (const helfem::Vector & x, helfem::Matrix & df,  size_t iel) const { eval_dnf(x, df,  1, iel); }
    void FiniteElementBasis::eval_d2f(const helfem::Vector & x, helfem::Matrix & d2f, size_t iel) const { eval_dnf(x, d2f, 2, iel); }
    void FiniteElementBasis::eval_d3f(const helfem::Vector & x, helfem::Matrix & d3f, size_t iel) const { eval_dnf(x, d3f, 3, iel); }
    void FiniteElementBasis::eval_d4f(const helfem::Vector & x, helfem::Matrix & d4f, size_t iel) const { eval_dnf(x, d4f, 4, iel); }
    void FiniteElementBasis::eval_d5f(const helfem::Vector & x, helfem::Matrix & d5f, size_t iel) const { eval_dnf(x, d5f, 5, iel); }

    void FiniteElementBasis::eval_dnf(const helfem::Vector & x, helfem::Matrix & dnf, int n, size_t iel) const {
      // helfem::Vector == lib1dfem::Vec<double>, same for Matrix; no
      // conversion needed.
      std::shared_ptr<polynomial_basis::PolynomialBasis> p(get_basis(iel));
      p->eval_dnf(x, dnf, n, scaling_factor(iel));
    }

    helfem::Matrix FiniteElementBasis::eval_f  (const helfem::Vector & x, size_t iel) const { return eval_dnf(x, 0, iel); }
    helfem::Matrix FiniteElementBasis::eval_df (const helfem::Vector & x, size_t iel) const { return eval_dnf(x, 1, iel); }
    helfem::Matrix FiniteElementBasis::eval_d2f(const helfem::Vector & x, size_t iel) const { return eval_dnf(x, 2, iel); }
    helfem::Matrix FiniteElementBasis::eval_d3f(const helfem::Vector & x, size_t iel) const { return eval_dnf(x, 3, iel); }
    helfem::Matrix FiniteElementBasis::eval_d4f(const helfem::Vector & x, size_t iel) const { return eval_dnf(x, 4, iel); }
    helfem::Matrix FiniteElementBasis::eval_d5f(const helfem::Vector & x, size_t iel) const { return eval_dnf(x, 5, iel); }

    helfem::Matrix FiniteElementBasis::eval_dnf(const helfem::Vector & x, int n, size_t iel) const {
      helfem::Matrix dnf;
      eval_dnf(x, dnf, n, iel);
      return dnf;
    }

    helfem::Matrix FiniteElementBasis::eval_over_r(const helfem::Vector & x, int n, size_t iel) const {
      if (std::abs(element_begin(iel)) > 1e-14) {
        std::ostringstream oss;
        oss << "FiniteElementBasis::eval_over_r is only valid when the element starts at r=0;"
            " element " << iel << " starts at " << element_begin(iel) << ".\n";
        throw std::logic_error(oss.str());
      }
      std::shared_ptr<polynomial_basis::PolynomialBasis> p(get_basis(iel));
      helfem::Matrix dnf_over_r;
      p->eval_over_r(x, dnf_over_r, n, scaling_factor(iel));
      return dnf_over_r;
    }

    helfem::Matrix FiniteElementBasis::eval_dnf(const helfem::Vector & x, int n) const {
      helfem::Matrix f(get_nelem() * x.size(), get_nbf());
      f.setZero();
      for (size_t iel = 0; iel < get_nelem(); ++iel) {
        const Eigen::Index ifirst = static_cast<Eigen::Index>(first_func_in_element(iel));
        const Eigen::Index ilast  = static_cast<Eigen::Index>(last_func_in_element(iel));
        f.block(iel * x.size(), ifirst, x.size(), ilast - ifirst + 1) = eval_dnf(x, n, iel);
      }
      return f;
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

    namespace {
      // Phase 5.3 bridge: matrix_element's function-pointer overload takes
      // arma::mat-returning lambdas, but eval_dnf is now Eigen-typed. Wrap
      // the eval_dnf result through to_arma so the function-pointer
      // overload's signature stays unchanged.
      inline std::function<arma::mat(const arma::vec &, size_t)>
      make_arma_eval_dnf(const FiniteElementBasis * fe, int der) {
        return [fe, der](const arma::vec & xa, size_t iel) {
          helfem::Vector xe(xa.n_elem);
          std::memcpy(xe.data(), xa.memptr(), sizeof(double) * xa.n_elem);
          helfem::Matrix me = fe->eval_dnf(xe, der, iel);
          arma::mat out(me.rows(), me.cols());
          std::memcpy(out.memptr(), me.data(),
                      sizeof(double) * static_cast<size_t>(me.size()));
          return out;
        };
      }
    } // namespace

    arma::mat FiniteElementBasis::matrix_element(int lhder, int rhder, const arma::vec & xq, const arma::vec & wq, const std::function<double(double)> & f) const {
      auto eval_lh = make_arma_eval_dnf(this, lhder);
      auto eval_rh = make_arma_eval_dnf(this, rhder);
      return matrix_element(eval_lh, eval_rh, xq, wq, f);
    }

    arma::mat FiniteElementBasis::matrix_element(size_t iel, int lhder, int rhder, const arma::vec & xq, const arma::vec & wq, const std::function<double(double)> & f) const {
      auto eval_lh = make_arma_eval_dnf(this, lhder);
      auto eval_rh = make_arma_eval_dnf(this, rhder);
      return matrix_element(iel, eval_lh, eval_rh, xq, wq, f);
    }

    arma::mat FiniteElementBasis::matrix_element(size_t iel, const std::function<arma::mat(arma::vec,size_t)> & eval_lh, const std::function<arma::mat(arma::vec,size_t)> & eval_rh, const arma::vec & xq, const arma::vec & wq, const std::function<double(double)> & f, double x_left, double x_right) const {

      // todo: figure out how to transform xq from [-1,1] to [x_left, x_right] and the same transformation for wq
      arma::vec x_shifted((x_right-x_left)/2.0*xq + (x_right+x_left)/2.0*arma::ones<arma::vec>(xq.n_elem));

      // Get coordinate values (Phase 5.3: eval_coord is Eigen).
      helfem::Vector xs_e(x_shifted.n_elem);
      std::memcpy(xs_e.data(), x_shifted.memptr(), sizeof(double) * x_shifted.n_elem);
      helfem::Vector r_e = eval_coord(xs_e, iel);
      arma::vec r(r_e.size());
      std::memcpy(r.memptr(), r_e.data(), sizeof(double) * r_e.size());
      // Calculate total weight per point
      arma::vec wp(wq*scaling_factor(iel)*(x_right-x_left)/2.0);
      // Include the function
      if(f) {
          for(size_t i=0; i<wp.n_elem; i++)
            wp(i)*=f(r(i));
      }

      // Evaluate basis functions
      if(!eval_lh)
        throw std::logic_error("Need function for evaluating left-hand basis functions!\n");
      arma::mat lhbf = eval_lh(x_shifted, iel);
      if(!eval_rh)
        throw std::logic_error("Need function for evaluating right-hand basis functions!\n");
      arma::mat rhbf = eval_rh(x_shifted, iel);

      // Include weight in the lh operand
      for(size_t i=0;i<lhbf.n_cols;i++)
        lhbf.col(i)%=wp;

      return arma::trans(lhbf)*rhbf;
    }

    arma::vec FiniteElementBasis::vector_element(size_t iel, const std::function<arma::mat(arma::vec,size_t)> & eval_bf, const arma::vec & xq, const arma::vec & wq, const std::function<double(double)> & f) const {
      // Get coordinate values (Phase 5.3 bridge).
      helfem::Vector xq_e(xq.n_elem);
      std::memcpy(xq_e.data(), xq.memptr(), sizeof(double) * xq.n_elem);
      helfem::Vector r_e = eval_coord(xq_e, iel);
      arma::vec r(r_e.size());
      std::memcpy(r.memptr(), r_e.data(), sizeof(double) * r_e.size());
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
      auto eval_f = make_arma_eval_dnf(this, der);
      return vector_element(eval_f, xq, wq, f);
    }

    arma::vec FiniteElementBasis::vector_element(size_t iel, int der, const arma::vec & xq, const arma::vec & wq, const std::function<double(double)> & f) const {
      auto eval_f = make_arma_eval_dnf(this, der);
      return vector_element(iel, eval_f, xq, wq, f);
    }

    void FiniteElementBasis::print(const std::string & str) const {
      printf("%s",str.c_str());
      bval.print("bval");
    }
  }
}
