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
#include <lobatto.h>
#include <algorithm>
#include <cfloat>
#include <iostream>
// Scalar formatter that prints a T value at its own precision (no truncation
// to double). Header-only, needs only Matrix.h + std; see src/general/eigen_io.h.
#include "../../src/general/eigen_io.h"
#include <limits>
#include <vector>

namespace helfem {
  namespace polynomial_basis {
    template<typename T>
    FiniteElementBasisT<T>::FiniteElementBasisT() {
    }

    template<typename T>
    FiniteElementBasisT<T>::FiniteElementBasisT(const std::shared_ptr<const helfem::polynomial_basis::PolynomialBasisT<T>> & poly_,
                                           const helfem::Vec<T> &bval_, bool zero_func_left_, bool zero_deriv_left_, bool zero_func_right_, bool zero_deriv_right_) : zero_func_left(zero_func_left_), zero_deriv_left(zero_deriv_left_), zero_func_right(zero_func_right_), zero_deriv_right(zero_deriv_right_) {
      // Phase 5.26: bval is Eigen at both the public boundary and the
      // internal storage; direct assignment, no bridge.
      bval = bval_;
      poly = std::shared_ptr<const helfem::polynomial_basis::PolynomialBasisT<T>>(poly_->copy());
      // Update list of basis functions
      update_bf_list();
      // Check that basis functions are continuous
      check_bf_continuity();
    }

    template<typename T>
    FiniteElementBasisT<T>::~FiniteElementBasisT() {
    }

    template<typename T>
    void FiniteElementBasisT<T>::update_bf_list() {
      if (bval.size() == 0)
        throw std::logic_error("Can't update basis function list since there are no elements!\n");

      // Form list of element boundaries
      first_func_in_element = helfem::IVec::Zero(bval.size() - 1);
      last_func_in_element  = helfem::IVec::Zero(bval.size() - 1);
      for (Eigen::Index iel = 0; iel < first_func_in_element.size(); ++iel) {
        first_func_in_element(iel) = (iel == 0) ? 0
            : last_func_in_element(iel - 1) - poly->get_noverlap() + 1;
        last_func_in_element(iel) = first_func_in_element(iel) + basis_indices(iel).size() - 1;
      }
    }

    template<typename T>
    void FiniteElementBasisT<T>::check_bf_continuity() const {
      if(get_nelem()==1)
        return;
      int noverlap(poly->get_noverlap());

      helfem::Vec<T> dnorm(get_nelem()-1);
      for(size_t iel=0; iel+1<get_nelem(); iel++) {
        // Points that correspond to lh and rh elements
        helfem::Vec<T> xlh(1), xrh(1);
        xlh(0) = 1.0;
        xrh(0) = -1.0;

        /// Check that coordinates match
        const helfem::Vec<T> rlh = eval_coord(xlh, iel);
        const helfem::Vec<T> rrh = eval_coord(xrh, iel + 1);
        // Keep the comparison in T: narrowing to double here discards the
        // precision the check is made at, and for _Float128 it is a
        // greater-conversion-rank narrowing (-Wnarrowing).
        const T dr = (rlh - rrh).norm();
        // Deliberately a DBL_EPSILON-scaled sanity bound rather than eps(T):
        // this catches a mis-built grid, it is not a precision assertion.
        // Bound the tolerance once so the diagnostic below reports the same
        // value that was actually applied.
        const T drtol = T(10)*T(DBL_EPSILON)*(T(1) + rlh.norm());
        if(dr > drtol) {
          std::cout << "rlh:\n"     << rlh.template cast<double>().transpose()       << "\n";
          std::cout << "rrh:\n"     << rrh.template cast<double>().transpose()       << "\n";
          std::cout << "rrh-rlh:\n" << (rrh-rlh).template cast<double>().transpose() << "\n";
          std::ostringstream oss;
          // fmt_sci renders T at its own precision: _Float128 has no
          // unambiguous operator<< in libstdc++, so never stream T directly.
          oss << "Coordinates do not match between elements " << iel << " and " << iel+1
              << ", difference " << helfem::io::fmt_sci(dr)
              << " tolerance " << helfem::io::fmt_sci(drtol) << "!\n";
          throw std::logic_error(oss.str());
        }

        // Evaluate bordering value in lh element (last noverlap functions)
        helfem::Mat<T> lh(noverlap, noverlap);
        for(int ider=0;ider<noverlap;ider++) {
          const helfem::Mat<T> fval = eval_dnf(xlh, ider, iel);
          lh.col(ider) = fval.row(0).tail(noverlap).transpose();
        }

        // Evaluate bordering value in rh element (first noverlap functions)
        helfem::Mat<T> rh(noverlap, noverlap);
        for(int ider=0;ider<noverlap;ider++) {
          const helfem::Mat<T> fval = eval_dnf(xrh, ider, iel + 1);
          rh.col(ider) = fval.row(0).head(noverlap).transpose();
        }

        // The function values should go to zero at the boundaries,
        // except the overlaid functions. The derivatives should also
        // go to zero, except the overlaid ones. The scaling does not
        // matter.
        // Note: arma::norm(M, 2) was the spectral norm; Eigen's .norm()
        // is Frobenius. For the noverlap x noverlap matrices here the
        // two agree to a small constant factor and the threshold
        // sqrt(DBL_EPSILON) ~ 1.5e-8 is many orders looser than that,
        // so the change in norm choice is not observable.
        const helfem::Mat<T> diff = lh - rh;
        dnorm(iel) = diff.norm();
        if(dnorm(iel) > sqrt(DBL_EPSILON)) {
          printf("Discontinuity between elements %i and %i (C indexing)\n",(int) iel,(int) iel+1);
          std::cout << "lh values:\n" << lh.template cast<double>()   << "\n";
          std::cout << "rh values:\n" << rh.template cast<double>()   << "\n";
          std::cout << "difference:\n" << diff.template cast<double>() << "\n";
          // Format the T norm at its own precision (no truncation to double).
          printf("Difference norm %s\n", helfem::io::fmt_sci(dnorm(iel)).c_str());
        }
      }
      Eigen::Index imax;
      dnorm.maxCoeff(&imax);
      printf("Finite element basis set max discontinuity %s between elements %i and %i\n",helfem::io::fmt_sci(dnorm(imax)).c_str(),(int) imax,(int) imax+1);
      fflush(stdout);
      if(dnorm(imax) > sqrt(DBL_EPSILON)) {
        throw std::logic_error("Finite element basis set is not continuous\n");
      }
    }

    template<typename T>
    void FiniteElementBasisT<T>::get_idx(size_t iel, size_t &ifirst, size_t &ilast) const {
      ifirst = first_func_in_element[iel];
      ilast = last_func_in_element[iel];
    }

    template<typename T>
    void FiniteElementBasisT<T>::add_boundary(T r) {
      // Check that r is not in bval
      for (Eigen::Index i = 0; i < bval.size(); ++i)
        if (bval(i) == r)
          return;

      // Append + sort
      helfem::Vec<T> newbval(bval.size() + 1);
      newbval.head(bval.size()) = bval;
      newbval(bval.size()) = r;
      std::sort(newbval.data(), newbval.data() + newbval.size());
      bval = newbval;
      update_bf_list();
    }

    template<typename T>
    std::shared_ptr<helfem::polynomial_basis::PolynomialBasisT<T>> FiniteElementBasisT<T>::get_poly() const { return std::shared_ptr<helfem::polynomial_basis::PolynomialBasisT<T>>(poly->copy()); }

    template<typename T>
    T FiniteElementBasisT<T>::scaling_factor(size_t iel) const {
      // The primitive range is [-1, 1] leading to the factor 2
      return element_length(iel)/2;
    }

    template<typename T>
    T FiniteElementBasisT<T>::element_length(size_t iel) const {
      if(iel>=get_nelem()) {
        std::ostringstream oss;
        oss << "Trying to access length of element " << iel << " but only have " << get_nelem() << "!\n";
        throw std::logic_error(oss.str());
      }
      return bval(iel+1)-bval(iel);
    }

    template<typename T>
    T FiniteElementBasisT<T>::element_begin(size_t iel) const {
      if(iel>=get_nelem()) {
        std::ostringstream oss;
        oss << "Trying to access length of element " << iel << " but only have " << get_nelem() << "!\n";
        throw std::logic_error(oss.str());
      }
      return bval(iel);
    }

    template<typename T>
    T FiniteElementBasisT<T>::element_end(size_t iel) const {
      if(iel>=get_nelem()) {
        std::ostringstream oss;
        oss << "Trying to access length of element " << iel << " but only have " << get_nelem() << "!\n";
        throw std::logic_error(oss.str());
      }
      return bval(iel+1);
    }

    template<typename T>
    T FiniteElementBasisT<T>::element_midpoint(size_t iel) const {
      return T(0.5)*(element_begin(iel) + element_end(iel));
    }

    template<typename T>
    size_t FiniteElementBasisT<T>::find_element(T x) const {
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

    template<typename T>
    helfem::Vec<T> FiniteElementBasisT<T>::get_bval() const {
      // Phase 5.5: bval is now native Eigen; direct return.
      return bval;
    }

    template<typename T>
    helfem::Vec<T> FiniteElementBasisT<T>::eval_coord(const helfem::Vec<T> & x, size_t iel) const {
      return helfem::Vec<T>::Constant(x.size(), element_midpoint(iel)) + scaling_factor(iel) * x;
    }

    template<typename T>
    T FiniteElementBasisT<T>::eval_coord(T x, size_t iel) const {
      return element_midpoint(iel) + scaling_factor(iel) * x;
    }

    template<typename T>
    helfem::Vec<T> FiniteElementBasisT<T>::eval_coord(const helfem::Vec<T> & x) const {
      helfem::Vec<T> r(get_nelem() * x.size());
      for (size_t iel = 0; iel < get_nelem(); ++iel)
        r.segment(iel * x.size(), x.size()) = eval_coord(x, iel);
      return r;
    }

    template<typename T>
    helfem::Vec<T> FiniteElementBasisT<T>::eval_weights(const helfem::Vec<T> & w) const {
      helfem::Vec<T> wr(get_nelem() * w.size());
      for (size_t iel = 0; iel < get_nelem(); ++iel)
        wr.segment(iel * w.size(), w.size()) = w * scaling_factor(iel);
      return wr;
    }

    template<typename T>
    helfem::Vec<T> FiniteElementBasisT<T>::eval_prim(const helfem::Vec<T> & y, size_t iel) const {
      if (y.minCoeff() < element_begin(iel) || y.maxCoeff() > element_end(iel)) {
        throw std::logic_error("coordinates don't correspond to this element!\n");
      }
      return (y - helfem::Vec<T>::Constant(y.size(), element_midpoint(iel))) / scaling_factor(iel);
    }

    template<typename T>
    int FiniteElementBasisT<T>::get_poly_id() const {
      return poly->get_id();
    }

    template<typename T>
    int FiniteElementBasisT<T>::get_poly_nnodes() const {
      return poly->get_nnodes();
    }

    template<typename T>
    helfem::IVec FiniteElementBasisT<T>::basis_indices(size_t iel) const {
      // Phase 5.5: native Eigen pass-through.
      std::shared_ptr<helfem::polynomial_basis::PolynomialBasisT<T>> p(get_basis(iel));
      return p->get_enabled();
    }

    template<typename T>
    std::shared_ptr<helfem::polynomial_basis::PolynomialBasisT<T>>
    FiniteElementBasisT<T>::get_basis(size_t iel) const {
      std::shared_ptr<helfem::polynomial_basis::PolynomialBasisT<T>> p(poly->copy());
      if (iel == 0)
        p->drop_first(zero_func_left, zero_deriv_left);
      if (iel == bval.size() - 2)
        p->drop_last(zero_func_right, zero_deriv_right);

      return p;
    }

    template<typename T>
    size_t FiniteElementBasisT<T>::get_nbf() const {
      if (last_func_in_element.size() == 0)
        throw std::logic_error("Basis function list has not been filled\n");
      return static_cast<size_t>(last_func_in_element(last_func_in_element.size() - 1) + 1);
    }

    template<typename T>
    size_t FiniteElementBasisT<T>::get_nelem() const {
      return bval.size() - 1;
    }

    template<typename T>
    size_t FiniteElementBasisT<T>::get_max_nprim() const {
      return poly->get_nprim();
    }

    template<typename T>
    size_t FiniteElementBasisT<T>::get_nprim(size_t iel) const {
      return get_basis(iel)->get_nbf();
    }

    // Phase 5.3: eval_* migrated to Eigen; internal primitive-basis call no longer
    // bridges (its signature matches).

    template<typename T>
    void FiniteElementBasisT<T>::eval_dnf(const helfem::Vec<T> & x, helfem::Mat<T> & dnf, int n, size_t iel) const {
      // helfem::Vec<T> == Vec<double>, same for Matrix; no
      // conversion needed.
      std::shared_ptr<helfem::polynomial_basis::PolynomialBasisT<T>> p(get_basis(iel));
      p->eval_dnf(x, dnf, n, scaling_factor(iel));
    }


    template<typename T>
    helfem::Mat<T> FiniteElementBasisT<T>::eval_dnf(const helfem::Vec<T> & x, int n, size_t iel) const {
      helfem::Mat<T> dnf;
      eval_dnf(x, dnf, n, iel);
      return dnf;
    }

    template<typename T>
    helfem::Mat<T> FiniteElementBasisT<T>::eval_over_r(const helfem::Vec<T> & x, int n, size_t iel) const {
      if (std::abs(element_begin(iel)) > 1e-14) {
        std::ostringstream oss;
        oss << "FiniteElementBasisT<T>::eval_over_r is only valid when the element starts at r=0;"
            " element " << iel << " starts at " << (double) element_begin(iel) << ".\n";
        throw std::logic_error(oss.str());
      }
      std::shared_ptr<helfem::polynomial_basis::PolynomialBasisT<T>> p(get_basis(iel));
      helfem::Mat<T> dnf_over_r;
      p->eval_over_r(x, dnf_over_r, n, scaling_factor(iel));
      return dnf_over_r;
    }

    template<typename T>
    helfem::Mat<T> FiniteElementBasisT<T>::eval_dnf(const helfem::Vec<T> & x, int n) const {
      helfem::Mat<T> f(get_nelem() * x.size(), get_nbf());
      f.setZero();
      for (size_t iel = 0; iel < get_nelem(); ++iel) {
        const Eigen::Index ifirst = static_cast<Eigen::Index>(first_func_in_element(iel));
        const Eigen::Index ilast  = static_cast<Eigen::Index>(last_func_in_element(iel));
        f.block(iel * x.size(), ifirst, x.size(), ilast - ifirst + 1) = eval_dnf(x, n, iel);
      }
      return f;
    }

    // Phase 5.4: matrix_element / vector_element migrated to Eigen.
    // Function-pointer overload's lambdas now have signature
    // helfem::Mat<T>(helfem::Vec<T>, size_t), matching eval_dnf.
    template<typename T>
    helfem::Mat<T> FiniteElementBasisT<T>::matrix_element(const std::function<helfem::Mat<T>(helfem::Vec<T>,size_t)> & eval_lh, const std::function<helfem::Mat<T>(helfem::Vec<T>,size_t)> & eval_rh, const helfem::Vec<T> & xq, const helfem::Vec<T> & wq, const std::function<T(T)> & f) const {
      // Compute matrix elements in parallel
      std::vector<helfem::Mat<T>> matel(get_nelem());
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (size_t iel = 0; iel < get_nelem(); ++iel) {
        matel[iel] = matrix_element(iel, eval_lh, eval_rh, xq, wq, f);
      }

      const Eigen::Index N = static_cast<Eigen::Index>(get_nbf());
      helfem::Mat<T> M = helfem::Mat<T>::Zero(N, N);
      for (size_t iel = 0; iel < get_nelem(); ++iel) {
        size_t ifirst, ilast;
        get_idx(iel, ifirst, ilast);
        M.block((Eigen::Index) ifirst, (Eigen::Index) ifirst,
                ilast - ifirst + 1, ilast - ifirst + 1) += matel[iel];
      }
      return M;
    }

    template<typename T>
    helfem::Vec<T> FiniteElementBasisT<T>::vector_element(const std::function<helfem::Mat<T>(helfem::Vec<T>,size_t)> & eval, const helfem::Vec<T> & xq, const helfem::Vec<T> & wq, const std::function<T(T)> & f) const {
      std::vector<helfem::Vec<T>> vecel(get_nelem());
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (size_t iel = 0; iel < get_nelem(); ++iel) {
        vecel[iel] = vector_element(iel, eval, xq, wq, f);
      }

      helfem::Vec<T> V = helfem::Vec<T>::Zero(get_nbf());
      for (size_t iel = 0; iel < get_nelem(); ++iel) {
        size_t ifirst, ilast;
        get_idx(iel, ifirst, ilast);
        V.segment((Eigen::Index) ifirst, ilast - ifirst + 1) += vecel[iel];
      }
      return V;
    }

    template<typename T>
    helfem::Mat<T> FiniteElementBasisT<T>::matrix_element(int lhder, int rhder, const helfem::Vec<T> & xq, const helfem::Vec<T> & wq, const std::function<T(T)> & f) const {
      auto eval_lh = [this, lhder](helfem::Vec<T> x, size_t iel) { return this->eval_dnf(x, lhder, iel); };
      auto eval_rh = [this, rhder](helfem::Vec<T> x, size_t iel) { return this->eval_dnf(x, rhder, iel); };
      return matrix_element(eval_lh, eval_rh, xq, wq, f);
    }

    template<typename T>
    helfem::Mat<T> FiniteElementBasisT<T>::matrix_element(size_t iel, int lhder, int rhder, const helfem::Vec<T> & xq, const helfem::Vec<T> & wq, const std::function<T(T)> & f) const {
      auto eval_lh = [this, lhder](helfem::Vec<T> x, size_t iel_) { return this->eval_dnf(x, lhder, iel_); };
      auto eval_rh = [this, rhder](helfem::Vec<T> x, size_t iel_) { return this->eval_dnf(x, rhder, iel_); };
      return matrix_element(iel, eval_lh, eval_rh, xq, wq, f);
    }

    template<typename T>
    helfem::Mat<T> FiniteElementBasisT<T>::matrix_element(size_t iel, const std::function<helfem::Mat<T>(helfem::Vec<T>,size_t)> & eval_lh, const std::function<helfem::Mat<T>(helfem::Vec<T>,size_t)> & eval_rh, const helfem::Vec<T> & xq, const helfem::Vec<T> & wq, const std::function<T(T)> & f, T x_left, T x_right) const {
      // Transform xq from [-1,1] to [x_left, x_right] and the same for wq.
      const T a = (x_right - x_left) / T(2);
      const T b = (x_right + x_left) / T(2);
      const helfem::Vec<T> x_shifted = a * xq + helfem::Vec<T>::Constant(xq.size(), b);

      // Get coordinate values
      const helfem::Vec<T> r = eval_coord(x_shifted, iel);
      // Total weight per point
      helfem::Vec<T> wp = wq * scaling_factor(iel) * a;
      if (f) {
        for (Eigen::Index i = 0; i < wp.size(); ++i) {
          const T fv = f(r(i));
          // Auto-converging quadrature uses Gauss-Lobatto, which -- unlike the
          // old open Gauss-Chebyshev rule -- samples the element ENDPOINTS. A
          // weight with an integrable singularity at an endpoint (e.g. a bare
          // -Z/r nuclear/model potential in the B representation at r=0)
          // returns +/-inf there, while the Dirichlet basis is exactly zero at
          // that node, so the exact contribution is 0 but the naive product is
          // 0*inf = NaN. Treat a non-finite weight as a zero contribution: the
          // node carries no basis amplitude, so this is exact for the FE basis.
          wp(i) = std::isfinite(fv) ? (wp(i) * fv) : T(0);
        }
      }

      // Evaluate basis functions
      if (!eval_lh)
        throw std::logic_error("Need function for evaluating left-hand basis functions!\n");
      helfem::Mat<T> lhbf = eval_lh(x_shifted, iel);
      if (!eval_rh)
        throw std::logic_error("Need function for evaluating right-hand basis functions!\n");
      const helfem::Mat<T> rhbf = eval_rh(x_shifted, iel);

      // Include weight in lh operand (column-wise multiply by wp).
      for (Eigen::Index j = 0; j < lhbf.cols(); ++j)
        lhbf.col(j).array() *= wp.array();

      return lhbf.transpose() * rhbf;
    }

    // ----------------------------------------------------------------------
    // Phase 2: auto-converging matrix element.
    //
    // The block over one element (or a sub-panel of it) is recomputed at a
    // rising Gauss-Lobatto order until it stops changing to 8*eps(T). Because
    // the tolerance is tied to eps(T), the order required to satisfy it grows
    // on its own with the precision of T -- a double block stabilises around
    // n~8, the same integrand at _Float128 keeps refining to n~13+ and thus
    // carries digits beyond double. Non-smooth weights (kinks) are handled by
    // splitting the element at the supplied breakpoints, since plain order-
    // refinement stalls across a cusp (see docs/autoconv_prototype.cpp).
    // ----------------------------------------------------------------------
    namespace {
      // Warn at most once per process if a panel hits the order cap.
      bool auto_matel_cap_warned = false;

      // Refine one smooth sub-panel [x_left,x_right] (reference coordinates in
      // [-1,1]) until the block is stable to 8*eps(T), doubling the Gauss-
      // Lobatto order from nstart up to nmax.
      template<typename T>
      helfem::Mat<T> converge_panel(const helfem::polynomial_basis::FiniteElementBasisT<T> & fe,
                                    size_t iel,
                                    const std::function<helfem::Mat<T>(helfem::Vec<T>,size_t)> & eval_lh,
                                    const std::function<helfem::Mat<T>(helfem::Vec<T>,size_t)> & eval_rh,
                                    const std::function<T(T)> & f,
                                    T x_left, T x_right, int nstart, int nmax) {
        const T tol = T(8) * std::numeric_limits<T>::epsilon();

        helfem::Vec<T> x, w;
        helfem::Mat<T> prev, cur;
        bool have = false;
        int n = std::max(nstart, 2);
        for (;;) {
          helfem::lobatto::lobatto_compute<T>(n, x, w);
          cur = fe.matrix_element(iel, eval_lh, eval_rh, x, w, f, x_left, x_right);
          if (have) {
            const T diff  = (cur - prev).cwiseAbs().maxCoeff();
            const T scale = cur.cwiseAbs().maxCoeff();
            if (diff <= tol * (scale + tol))
              return cur;
          }
          prev = cur;
          have = true;
          if (n >= nmax) {
            if (!auto_matel_cap_warned) {
              auto_matel_cap_warned = true;
              printf("Warning: FiniteElementBasis::matrix_element hit the Gauss-Lobatto"
                     " order cap (n=%d) without converging to eps(T); using best estimate.\n", nmax);
              fflush(stdout);
            }
            return cur;
          }
          n = std::min(2 * n, nmax);
        }
      }
    }

    template<typename T>
    helfem::Mat<T> FiniteElementBasisT<T>::matrix_element_auto(
        size_t iel,
        const std::function<helfem::Mat<T>(helfem::Vec<T>,size_t)> & eval_lh,
        const std::function<helfem::Mat<T>(helfem::Vec<T>,size_t)> & eval_rh,
        const std::function<T(T)> & f,
        const std::vector<T> & breakpoints,
        int poly_degree_f,
        T x_left, T x_right) const {
      // Sub-panel boundaries in the reference [-1,1] coordinate: split at any
      // breakpoint that falls strictly inside (x_left, x_right).
      const T mid = element_midpoint(iel);
      const T sf  = scaling_factor(iel);
      std::vector<T> xb;
      xb.push_back(x_left);
      for (T bp : breakpoints) {
        const T xp = (bp - mid) / sf; // real-space breakpoint -> reference coord
        if (xp > x_left && xp < x_right)
          xb.push_back(xp);
      }
      xb.push_back(x_right);
      std::sort(xb.begin(), xb.end());

      // Starting order. n-point Gauss-Lobatto is exact to degree 2n-3, so pick
      // n that already integrates B*B*(polynomial f) exactly; the refine loop
      // then confirms (polynomial case) or extends (non-polynomial f). Using an
      // upper bound on the basis degree only affects speed, never correctness.
      const int basisdeg = std::max(0, poly->get_noverlap() * poly->get_nnodes() - 1);
      const int fdeg     = std::max(0, poly_degree_f);
      const int deg      = 2 * basisdeg + fdeg;
      const int nmax     = 512;
      int nstart = (deg + 4) / 2 + 2;
      if (nstart < 5)     nstart = 5;
      if (nstart > nmax)  nstart = nmax;

      helfem::Mat<T> total;
      bool have_total = false;
      for (size_t p = 0; p + 1 < xb.size(); ++p) {
        if (!(xb[p + 1] > xb[p])) continue; // skip degenerate panels
        const helfem::Mat<T> block =
            converge_panel<T>(*this, iel, eval_lh, eval_rh, f, xb[p], xb[p + 1], nstart, nmax);
        if (!have_total) { total = block; have_total = true; }
        else total += block;
      }
      if (!have_total)
        // Degenerate range (x_left==x_right): return a correctly sized zero block.
        total = converge_panel<T>(*this, iel, eval_lh, eval_rh, f, x_left, x_right, nstart, nmax);
      return total;
    }

    template<typename T>
    helfem::Mat<T> FiniteElementBasisT<T>::matrix_element_auto(
        const std::function<helfem::Mat<T>(helfem::Vec<T>,size_t)> & eval_lh,
        const std::function<helfem::Mat<T>(helfem::Vec<T>,size_t)> & eval_rh,
        const std::function<T(T)> & f,
        const std::vector<T> & breakpoints,
        int poly_degree_f) const {
      std::vector<helfem::Mat<T>> matel(get_nelem());
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (size_t iel = 0; iel < get_nelem(); ++iel)
        matel[iel] = matrix_element_auto(iel, eval_lh, eval_rh, f, breakpoints, poly_degree_f);

      const Eigen::Index N = static_cast<Eigen::Index>(get_nbf());
      helfem::Mat<T> M = helfem::Mat<T>::Zero(N, N);
      for (size_t iel = 0; iel < get_nelem(); ++iel) {
        size_t ifirst, ilast;
        get_idx(iel, ifirst, ilast);
        M.block((Eigen::Index) ifirst, (Eigen::Index) ifirst,
                ilast - ifirst + 1, ilast - ifirst + 1) += matel[iel];
      }
      return M;
    }

    template<typename T>
    helfem::Mat<T> FiniteElementBasisT<T>::matrix_element(
        size_t iel, int lhder, int rhder, const std::function<T(T)> & f,
        const std::vector<T> & breakpoints, int poly_degree_f) const {
      auto eval_lh = [this, lhder](helfem::Vec<T> x, size_t iel_) { return this->eval_dnf(x, lhder, iel_); };
      auto eval_rh = [this, rhder](helfem::Vec<T> x, size_t iel_) { return this->eval_dnf(x, rhder, iel_); };
      return matrix_element_auto(iel, eval_lh, eval_rh, f, breakpoints, poly_degree_f);
    }

    template<typename T>
    helfem::Mat<T> FiniteElementBasisT<T>::matrix_element(
        int lhder, int rhder, const std::function<T(T)> & f,
        const std::vector<T> & breakpoints, int poly_degree_f) const {
      auto eval_lh = [this, lhder](helfem::Vec<T> x, size_t iel_) { return this->eval_dnf(x, lhder, iel_); };
      auto eval_rh = [this, rhder](helfem::Vec<T> x, size_t iel_) { return this->eval_dnf(x, rhder, iel_); };
      return matrix_element_auto(eval_lh, eval_rh, f, breakpoints, poly_degree_f);
    }

    template<typename T>
    helfem::Vec<T> FiniteElementBasisT<T>::vector_element(size_t iel, const std::function<helfem::Mat<T>(helfem::Vec<T>,size_t)> & eval_bf, const helfem::Vec<T> & xq, const helfem::Vec<T> & wq, const std::function<T(T)> & f) const {
      const helfem::Vec<T> r = eval_coord(xq, iel);
      helfem::Vec<T> wp = wq * scaling_factor(iel);
      if (f) {
        for (Eigen::Index i = 0; i < wp.size(); ++i)
          wp(i) *= f(r(i));
      }

      if (!eval_bf)
        throw std::logic_error("Need function for evaluating basis functions!\n");
      const helfem::Mat<T> bf = eval_bf(xq, iel);

      return bf.transpose() * wp;
    }

    template<typename T>
    helfem::Vec<T> FiniteElementBasisT<T>::vector_element(int der, const helfem::Vec<T> & xq, const helfem::Vec<T> & wq, const std::function<T(T)> & f) const {
      auto eval_f = [this, der](helfem::Vec<T> x, size_t iel) { return this->eval_dnf(x, der, iel); };
      return vector_element(eval_f, xq, wq, f);
    }

    template<typename T>
    helfem::Vec<T> FiniteElementBasisT<T>::vector_element(size_t iel, int der, const helfem::Vec<T> & xq, const helfem::Vec<T> & wq, const std::function<T(T)> & f) const {
      auto eval_f = [this, der](helfem::Vec<T> x, size_t iel_) { return this->eval_dnf(x, der, iel_); };
      return vector_element(iel, eval_f, xq, wq, f);
    }

    template<typename T>
    void FiniteElementBasisT<T>::print(const std::string & str) const {
      printf("%s",str.c_str());
      printf("bval has %lld entries\n", (long long) bval.size());
    }

    // Explicit instantiations. Everything below libhelfem was already generic;
    // these are the precisions HelFEM is built for.
    template class FiniteElementBasisT<double>;
    template class FiniteElementBasisT<long double>;
#ifdef HELFEM_HAVE_FLOAT128
    template class FiniteElementBasisT<_Float128>;
#endif
  }
}
