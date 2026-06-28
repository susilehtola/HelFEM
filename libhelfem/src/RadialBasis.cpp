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
#include "RadialBasis.h"
#include "RadialPotential.h"
#include "chebyshev.h"
#include "quadrature.h"
#include "utils.h"
#include <cfloat>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace helfem {
  namespace atomic {
    namespace basis {
      FEMRadialBasis::FEMRadialBasis() {}

      FEMRadialBasis::FEMRadialBasis(const polynomial_basis::FiniteElementBasis & fem_, int n_quad, int taylor_order_) : fem(fem_) {
        // taylor_order is a vestigial parameter from the pre-eval_over_r days.
        // The Taylor expansion near r=0 has been retired in favour of the
        // analytic deflation in PolynomialBasis::eval_over_r; the value is
        // ignored but accepted for API back-compatibility. Stored as 0 so the
        // (also vestigial) get_taylor_order() accessor returns something
        // bounded.
        (void)taylor_order_;

        // Get quadrature rule
        chebyshev::chebyshev(n_quad, xq, wq);
        for (size_t i = 0; i < xq.n_elem; i++) {
          if (!std::isfinite(xq[i]))
            printf("xq[%i]=%e\n", (int)i, xq[i]);
          if (!std::isfinite(wq[i]))
            printf("wq[%i]=%e\n", (int)i, wq[i]);
        }
      }

      FEMRadialBasis::~FEMRadialBasis() {}

      int FEMRadialBasis::get_nquad() const {
        return (int)xq.n_elem;
      }

      arma::vec FEMRadialBasis::get_xq() const {
        return xq;
      }

      size_t FEMRadialBasis::Nbf() const {
        return fem.get_nbf();
      }

      size_t FEMRadialBasis::Nel() const {
        return fem.get_nelem();
      }

      size_t FEMRadialBasis::Nprim(size_t iel) const {
        return fem.get_nprim(iel);
      }

      size_t FEMRadialBasis::max_Nprim() const {
        return fem.get_max_nprim();
      }

      void FEMRadialBasis::get_idx(size_t iel, size_t &ifirst, size_t &ilast) const {
        fem.get_idx(iel, ifirst, ilast);
      }

      arma::vec FEMRadialBasis::get_bval() const {
        return fem.get_bval();
      }

      int FEMRadialBasis::get_poly_id() const {
        return fem.get_poly_id();
      }

      int FEMRadialBasis::get_poly_nnodes() const {
        return fem.get_poly_nnodes();
      }

      // The Taylor pipeline has been retired in favour of PolynomialBasis::eval_over_r.
      // These accessors stay for API back-compatibility but return 0.
      double FEMRadialBasis::get_small_r_taylor_cutoff() const { return 0.0; }
      int    FEMRadialBasis::get_taylor_order()         const { return 0;   }
      double FEMRadialBasis::get_taylor_diff()          const { return 0.0; }


      arma::mat FEMRadialBasis::radial_integral(int Rexp, size_t iel, double x_left, double x_right) const {
        // <R | r^Rexp | R> with the FE-natural dr measure means weight r^(Rexp+2):
        //   integral B(r)^2 r^(Rexp+2) / r^2  dr  =  integral R(r)^2 r^(Rexp+2)  dr
        // = QM <r^Rexp> = integral R^2 r^Rexp r^2 dr.
        const std::function<double(double)> rpowL =
            [Rexp](double r){ return std::pow(r, Rexp + 2); };
        arma::mat ret = matrix_element(iel, BasisKind::R0, BasisKind::R0,
                                       rpowL, x_left, x_right);
        if (ret.has_nan())
          printf("radial_integral(%i,%i) has NaN!\n", Rexp, (int)iel);
        return ret;
      }

      // Pre-bound evaluator for each BasisKind: a callable (x, iel) -> arma::mat
      // suitable for FiniteElementBasis::matrix_element. The evaluator captures
      // `this` only; it does not allocate per call beyond what fem.eval_*/get_*
      // already does.
      static std::function<arma::mat(const arma::vec &, size_t)>
      make_evaluator(const FEMRadialBasis * rb, FEMRadialBasis::BasisKind k) {
        using BK = FEMRadialBasis::BasisKind;
        switch (k) {
          case BK::B0: return [rb](const arma::vec & x, size_t iel) {
            return rb->get_fem().eval_f(x, iel);
          };
          case BK::B1: return [rb](const arma::vec & x, size_t iel) {
            return rb->get_fem().eval_df(x, iel);
          };
          case BK::B2: return [rb](const arma::vec & x, size_t iel) {
            return rb->get_fem().eval_d2f(x, iel);
          };
          case BK::R0: return [rb](const arma::vec & x, size_t iel) {
            return rb->get_bf(x, iel);
          };
          case BK::R1: return [rb](const arma::vec & x, size_t iel) {
            return rb->get_df(x, iel);
          };
          case BK::R2: return [rb](const arma::vec & x, size_t iel) {
            return rb->get_lf(x, iel);
          };
        }
        throw std::logic_error("FEMRadialBasis::matrix_element: unknown BasisKind\n");
      }

      arma::mat FEMRadialBasis::matrix_element(
          size_t iel, BasisKind bra, BasisKind ket,
          const std::function<double(double)> & weight) const {
        auto lhs = make_evaluator(this, bra);
        auto rhs = make_evaluator(this, ket);
        return fem.matrix_element(iel, lhs, rhs, xq, wq, weight);
      }

      arma::mat FEMRadialBasis::matrix_element(
          BasisKind bra, BasisKind ket,
          const std::function<double(double)> & weight) const {
        auto lhs = make_evaluator(this, bra);
        auto rhs = make_evaluator(this, ket);
        return fem.matrix_element(lhs, rhs, xq, wq, weight);
      }

      arma::mat FEMRadialBasis::matrix_element(
          size_t iel, BasisKind bra, BasisKind ket,
          const std::function<double(double)> & weight,
          double x_left, double x_right) const {
        auto lhs = make_evaluator(this, bra);
        auto rhs = make_evaluator(this, ket);
        return fem.matrix_element(iel, lhs, rhs, xq, wq, weight, x_left, x_right);
      }

      arma::mat FEMRadialBasis::bessel_il_integral(int L, double lambda, size_t iel) const {
        return matrix_element(iel, BasisKind::B0, BasisKind::B0,
                              [L, lambda](double r){ return utils::bessel_il(r * lambda, L); });
      }

      arma::mat FEMRadialBasis::bessel_kl_integral(int L, double lambda, size_t iel) const {
        return matrix_element(iel, BasisKind::B0, BasisKind::B0,
                              [L, lambda](double r){ return utils::bessel_kl(r * lambda, L); });
      }

      arma::mat FEMRadialBasis::radial_integral(const FEMRadialBasis &rh, int n, bool lhder,
                                             bool rhder) const {
        modelpotential::RadialPotential rad(n);
        return model_potential(rh, &rad, lhder, rhder);
      }

      arma::mat FEMRadialBasis::model_potential(const FEMRadialBasis &rh,
                                             const modelpotential::ModelPotential *model,
                                             bool lhder, bool rhder) const {
        // Use the larger number of quadrature points to assure
        // projection is computed ok
        size_t n_quad(std::max(xq.n_elem, rh.xq.n_elem));

        arma::vec xproj, wproj;
        chebyshev::chebyshev(n_quad, xproj, wproj);

        // Form list of overlapping elements
        std::vector< std::vector<size_t> > overlap(fem.get_nelem());
        for (size_t iel = 0; iel < fem.get_nelem(); iel++) {
          // Range of element i
          double istart(fem.element_begin(iel));
          double iend(fem.element_end(iel));

          for (size_t jel = 0; jel < rh.fem.get_nelem(); jel++) {
            // Range of element j
            double jstart(rh.fem.element_begin(jel));
            double jend(rh.fem.element_end(jel));

            // Is there overlap?
            if ((jstart >= istart && jstart < iend) || (istart >= jstart && istart < jend)) {
              overlap[iel].push_back(jel);
              // printf("New element %i overlaps with old element %i\n",iel,jel);
            }
          }
        }

        // Form overlap matrix
        arma::mat S(Nbf(), rh.Nbf());
        S.zeros();
        for (size_t iel = 0; iel < fem.get_nelem(); iel++) {
          // Loop over overlapping elements
          for (size_t jj = 0; jj < overlap[iel].size(); jj++) {
            // Index of element is
            size_t jel = overlap[iel][jj];

            // Because the functions are only defined within a single
            // element, the product can be very raggedy. However,
            // since we *know* where the overlap is non-zero, we can
            // restrict the quadrature to just that zone.

            // Limits
            double imin(fem.element_begin(iel));
            double imax(fem.element_end(iel));
            // Range of element
            double jmin(rh.fem.element_begin(jel));
            double jmax(rh.fem.element_end(jel));

            // Range of integral is thus
            double intstart(std::max(imin, jmin));
            double intend(std::min(imax, jmax));
            // Inteval mid-point is at
            double intmid(0.5 * (intend + intstart));
            double intlen(0.5 * (intend - intstart));

            // the r values we're going to use are then
            arma::vec r(intmid * arma::ones<arma::vec>(xproj.n_elem) + intlen * xproj);

            // Where are we in the matrix?
            size_t ifirst, ilast;
            get_idx(iel, ifirst, ilast);
            size_t jfirst, jlast;
            rh.get_idx(jel, jfirst, jlast);

            // Back-transform r values into i:th and j:th elements
            arma::vec xi(fem.eval_prim(r, iel));
            arma::vec xj(rh.fem.eval_prim(r, jel));

            // Calculate total weight per point
            arma::vec wtot(wproj * intlen);
            // Put in the potential
            wtot %= model->V(r);

            arma::mat ifunc = lhder ? fem.eval_df(xi, iel) : fem.eval_f(xi, iel);
            arma::mat jfunc = rhder ? rh.fem.eval_df(xj, jel) : rh.fem.eval_f(xj, jel);

            // Perform quadrature
            arma::mat s(arma::trans(ifunc) * arma::diagmat(wtot) * jfunc);

            // Increment overlap matrix
            S.submat(ifirst, jfirst, ilast, jlast) += s;
          }
        }

        return S;
      }

      arma::mat FEMRadialBasis::overlap(const FEMRadialBasis &rh) const {
        return radial_integral(rh, 0);
      }

      // The four named matrix elements below are the proof-of-concept
      // migration onto the BasisKind dispatcher introduced above. Math is
      // unchanged; each routine collapses to a single composition of
      // (bra evaluator, ket evaluator, weight function).
      //
      //   overlap     = integral B(r) B(r) dr                    -> B0 * B0 * 1
      //   kinetic     = (1/2) integral B'(r) B'(r) dr            -> B1 * B1 * 1
      //   kinetic_l   = (1/2) integral R(r) R(r) dr              -> R0 * R0 * 1
      //                (= (1/2) <u | 1/r^2 | u> = half-centrifugal per l(l+1))
      //   nuclear     = - integral R(r) * r * R(r) dr            -> R0 * R0 * r
      //                (= - <u | 1/r | u>; sign included, Z=1 implicit)

      static const std::function<double(double)> kNoWeight;  // default-constructed = identity
      static const std::function<double(double)> kRWeight = [](double r){ return r; };

      arma::mat FEMRadialBasis::overlap()         const { return matrix_element(BasisKind::B0, BasisKind::B0, kNoWeight); }
      arma::mat FEMRadialBasis::overlap(size_t iel) const { return matrix_element(iel, BasisKind::B0, BasisKind::B0, kNoWeight); }

      arma::mat FEMRadialBasis::kinetic()         const { return 0.5 * matrix_element(BasisKind::B1, BasisKind::B1, kNoWeight); }
      arma::mat FEMRadialBasis::kinetic(size_t iel) const { return 0.5 * matrix_element(iel, BasisKind::B1, BasisKind::B1, kNoWeight); }

      arma::mat FEMRadialBasis::kinetic_l()       const { return 0.5 * matrix_element(BasisKind::R0, BasisKind::R0, kNoWeight); }
      arma::mat FEMRadialBasis::kinetic_l(size_t iel) const { return 0.5 * matrix_element(iel, BasisKind::R0, BasisKind::R0, kNoWeight); }

      arma::mat FEMRadialBasis::nuclear()         const { return -matrix_element(BasisKind::R0, BasisKind::R0, kRWeight); }
      arma::mat FEMRadialBasis::nuclear(size_t iel) const { return -matrix_element(iel, BasisKind::R0, BasisKind::R0, kRWeight); }

      arma::mat FEMRadialBasis::polynomial_confinement(size_t iel, int N, double shift_pot) const {
        return matrix_element(iel, BasisKind::R0, BasisKind::R0,
                              [N, shift_pot](double r) {
                                return (r < shift_pot)
                                    ? 0.0
                                    : std::pow(r - shift_pot, N + 2);
                              });
      }

      arma::mat FEMRadialBasis::exponential_confinement(size_t iel, int N, double r_0, double shift_pot) const {
        return matrix_element(iel, BasisKind::B0, BasisKind::B0,
                              [r_0, N, shift_pot](double r) {
                                if (r < shift_pot) return 0.0;
                                const double r_ratio = (r - shift_pot) / r_0;
                                double fact = 1.0;
                                double V = 0.0;
                                double r_ratio_pow_k = 1.0;
                                for (int k = 0; k < N; ++k) {
                                  V -= r_ratio_pow_k / fact;
                                  fact *= k + 1;
                                  r_ratio_pow_k *= r_ratio;
                                }
                                V += std::exp(r_ratio);
                                V *= fact;
                                return V;
                              });
      }

      arma::mat FEMRadialBasis::barrier_confinement(size_t iel, double V, double shift_pot) const {
        return matrix_element(iel, BasisKind::B0, BasisKind::B0,
                              [V, shift_pot](double r) {
                                return (r < shift_pot) ? 0.0 : V;
                              });
      }

      arma::mat FEMRadialBasis::junq_confinement(size_t iel, int N, double V0, double r_c, double shift_pot) const {
        return matrix_element(iel, BasisKind::B0, BasisKind::B0,
                              [N, r_c, V0, shift_pot](double r) {
                                if (r < shift_pot) return 0.0;
                                const double denominator  = std::pow(r_c - r, N);
                                const double exponential  = std::exp(-(r_c - shift_pot) / (r - shift_pot));
                                return V0 * exponential / denominator;
                              });
      }

      arma::mat FEMRadialBasis::confinement_potential(size_t iel, int N, double r_0, int iconf, double V, double shift_pot) const {
	// Attractive potential does not make sense for shift_pot != 0

	// sign of r0 controls if the potential is attractive or repulsive
	int sign = (r_0<0) ? -1 : 1;
	r_0 = std::abs(r_0);

	if(iconf==1) {
          printf("Polynomial confinement, r_0 = %e N = %i shift = %e \n",r_0,N,shift_pot);
	  if(N<0) {
	    if(shift_pot != 0.0)
	      throw std::logic_error("Cannot have a divergent potential with a shift!\n");
	    return sign*std::pow(r_0, N)*polynomial_confinement(iel, N, shift_pot);
	  } else {
	    return sign*std::pow(r_0, -N)*polynomial_confinement(iel, N, shift_pot);
	  }

	} else if(iconf==2) {
          printf("Exponential confinement, r_0 = %e N = %i shift = %e \n",r_0,N,shift_pot);

	  if(N<0)
	    throw std::logic_error("Exponential confinement potential does not make sense with negative N!\n");
	  if(N==0)
	    throw std::logic_error("Exponential confinement potential requires N >= 1!");

	  return exponential_confinement(iel, N, r_0, shift_pot);

	} else if(iconf==3) {
	  if(V<0)
	    throw std::logic_error("Cannot have attractive barrier!\n");

          printf("Barrier confinement, V = %e shift = %e \n",V,shift_pot);
	  return barrier_confinement(iel, V, shift_pot);

	} else if(iconf==4) {
          printf("Junquera-type confinement, r_0 = %e N = %i V = %e shift = %e \n",r_0,N,V,shift_pot);
	  if(N<=0)
	    throw std::logic_error("Junquera confinement potential requires N >= 1!");
	  if(V<=0)
	    throw std::logic_error("Cannot have attractive Junquera potential!\n");
	  return junq_confinement(iel, N, V, arma::max(get_bval()), shift_pot);
	} else
	  throw std::logic_error("Case not implemented!\n");
      }


      arma::mat FEMRadialBasis::model_potential(const modelpotential::ModelPotential *model,
                                                size_t iel) const {
        return matrix_element(iel, BasisKind::B0, BasisKind::B0,
                              [model](double r){ return model->V(r); });
      }

      arma::mat FEMRadialBasis::nuclear_offcenter(size_t iel, double Rhalf, int L) const {
        if (fem.element_begin(iel) <= Rhalf)
          return -sqrt(4.0 * M_PI / (2 * L + 1)) * radial_integral(-L - 1, iel) *
            std::pow(Rhalf, L);
        else if (fem.element_end(iel) >= Rhalf)
          return -sqrt(4.0 * M_PI / (2 * L + 1)) * radial_integral(L, iel) *
            std::pow(Rhalf, -L - 1);
        else {
          throw std::logic_error("Nucleus placed within element!\n");
          arma::mat ret;
          return ret;
        }
      }

      arma::mat FEMRadialBasis::twoe_integral(int L, size_t iel) const {
        double Rmin(fem.element_begin(iel));
        double Rmax(fem.element_end(iel));

        // Integral by quadrature
        std::shared_ptr<const polynomial_basis::PolynomialBasis> p(fem.get_basis(iel));
        arma::mat tei(quadrature::twoe_integral(Rmin, Rmax, xq, wq, p, L));
        if(tei.has_nan()) {
          printf("twoe_integral(%i,%i) has NaN!\n",L,(int) iel);
        }
        return tei;
      }

      arma::mat FEMRadialBasis::yukawa_integral(int L, double lambda, size_t iel) const {
        double Rmin(fem.element_begin(iel));
        double Rmax(fem.element_end(iel));

        // Integral by quadrature
        std::shared_ptr<const polynomial_basis::PolynomialBasis> p(fem.get_basis(iel));
        arma::mat tei(quadrature::yukawa_integral(Rmin, Rmax, xq, wq, p, L, lambda));

        return tei;
      }

      arma::mat FEMRadialBasis::erfc_integral(int L, double mu, size_t iel, size_t kel) const {
        // Number of quadrature points
        size_t Nq = xq.n_elem;
        // Number of subintervals
        size_t Nint;

        if (iel == kel) {
          // Intraelement integral is harder to converge due to the
          // electronic cusp, so let's do the same trick as in the
          // separable case and use a tighter grid for the "inner"
          // integral.
          Nint = Nq;

        } else {
          // A single interval suffices since there's no cusp.
          Nint = 1;
        }

        // Get lh quadrature points
        arma::vec xi, wi;
        chebyshev::chebyshev(Nq, xi, wi);
        // and basis function values
        arma::mat ibf(fem.eval_f(xi, iel));
        double Rmini(fem.element_begin(iel));
        double Rmaxi(fem.element_end(iel));

        // Rh quadrature points
        arma::vec xk(Nq * Nint);
        arma::vec wk(Nq * Nint);
        for (size_t ii = 0; ii < Nint; ii++) {
          // Interval starts at
          double istart = ii * 2.0 / Nint - 1.0;
          double iend = (ii + 1) * 2.0 / Nint - 1.0;
          // Midpoint and half-length of interval
          double imid = 0.5 * (iend + istart);
          double ilen = 0.5 * (iend - istart);

          // Place quadrature points at
          xk.subvec(ii * Nq, (ii + 1) * Nq - 1) = arma::ones<arma::vec>(Nq) * imid + xi * ilen;
          // which have the renormalized weights
          wk.subvec(ii * Nq, (ii + 1) * Nq - 1) = wi * ilen;
        }
        // and basis function values
        arma::mat kbf(fem.eval_f(xk, kel));
        double Rmink(fem.element_begin(kel));
        double Rmaxk(fem.element_end(kel));

        // Evaluate integral
        arma::mat tei(quadrature::erfc_integral(Rmini, Rmaxi, ibf, xi, wi, Rmink,
                                                Rmaxk, kbf, xk, wk, L, mu));
        // Symmetrize just to be sure, since quadrature points were
        // different
        if (iel == kel)
          tei = 0.5 * (tei + tei.t());

        return tei;
      }

      arma::mat FEMRadialBasis::spherical_potential(size_t iel) const {
        double Rmin(fem.element_begin(iel));
        double Rmax(fem.element_end(iel));

        // Integral by quadrature
        std::shared_ptr<polynomial_basis::PolynomialBasis> p(fem.get_basis(iel));
        arma::mat pot(quadrature::spherical_potential(Rmin, Rmax, xq, wq, p));

        return pot;
      }

      arma::mat FEMRadialBasis::get_bf(size_t iel) const {
        return get_bf(xq, iel);
      }

      // get_taylor() has been removed in favour of fem.eval_over_r().

      arma::vec FEMRadialBasis::eval_orbs(const arma::mat & C, double r) const {
        if(r > fem.element_end(fem.get_nelem()-1)) {
          // The wave function is zero here
          arma::vec val(C.n_cols, arma::fill::zeros);
          return val;
        } else {
          // Find the element we are in
          size_t iel = fem.find_element(r);
          // Find the value of the primitive coordinate
          arma::vec x(fem.eval_prim(arma::vec({r}), iel));

          // Evaluate the basis functions in the element
          arma::mat val(get_bf(x, iel));
          // Figure out the corresponding indices of the basis functions
          size_t ifirst, ilast;
          get_idx(iel, ifirst, ilast);
          arma::mat Csub(C.rows(ifirst, ilast));
          return (val*Csub).t();
        }
      }

      arma::mat FEMRadialBasis::get_bf(const arma::vec & x, size_t iel) const {
        // For the element starting at r=0 use the analytic eval_over_r path
        // (B(r)/r evaluated by deflating the (x+1) factor that the Dirichlet
        // BC at r=0 guarantees). For other elements r > 0 so direct division
        // is numerically fine.
        if (iel == 0)
          return fem.eval_over_r(x, 0, iel);
        arma::mat val(fem.eval_f(x, iel));
        arma::vec r(fem.eval_coord(x, iel));
        for (size_t ifun = 0; ifun < val.n_cols; ifun++)
          for (size_t ir = 0; ir < x.n_elem; ir++)
            val(ir, ifun) /= r(ir);
        return val;
      }

      arma::mat FEMRadialBasis::get_df(size_t iel) const {
        return get_df(xq ,iel);
      }

      arma::mat FEMRadialBasis::get_df(const arma::vec & x, size_t iel) const {
        if (iel == 0)
          return fem.eval_over_r(x, 1, iel);
        arma::mat fval(fem.eval_f(x, iel));
        arma::mat dval(fem.eval_df(x, iel));
        arma::vec r(fem.eval_coord(x, iel));
        arma::mat der(fval.n_rows, fval.n_cols);
        for (size_t ifun = 0; ifun < der.n_cols; ifun++)
          for (size_t ir = 0; ir < x.n_elem; ir++) {
            const double invr = 1.0 / r(ir);
            der(ir, ifun) = (-fval(ir, ifun) * invr + dval(ir, ifun)) * invr;
          }
        return der;
      }

      arma::mat FEMRadialBasis::get_lf(size_t iel) const {
        return get_lf(xq, iel);
      }

      arma::mat FEMRadialBasis::get_lf(const arma::vec & x, size_t iel) const {
        if (iel == 0)
          return fem.eval_over_r(x, 2, iel);
        arma::mat fval(fem.eval_f(x, iel));
        arma::mat dval(fem.eval_df(x, iel));
        arma::mat lval(fem.eval_d2f(x, iel));
        arma::vec r(fem.eval_coord(x, iel));
        arma::mat lapl(fval.n_rows, fval.n_cols);
        for (size_t ifun = 0; ifun < lapl.n_cols; ifun++)
          for (size_t ir = 0; ir < x.n_elem; ir++) {
            const double invr = 1.0 / r(ir);
            lapl(ir, ifun) = ((2.0 * fval(ir, ifun) * invr - 2.0 * dval(ir, ifun)) * invr
                              + lval(ir, ifun)) * invr;
          }
        return lapl;
      }

      arma::vec FEMRadialBasis::get_wrad(size_t iel) const {
        return get_wrad(wq, iel);
      }

      arma::vec FEMRadialBasis::get_wrad(const arma::vec & w, size_t iel) const {
        // This is just the radial rule, no r^2 factor included here
        return fem.scaling_factor(iel) * w;
      }

      arma::vec FEMRadialBasis::get_r(size_t iel) const {
        return get_r(xq, iel);
      }

      arma::vec FEMRadialBasis::get_r(const arma::vec & x, size_t iel) const {
        return fem.eval_coord(x, iel);
      }

      double FEMRadialBasis::get_r(double x, size_t iel) const {
        return fem.eval_coord(x, iel);
      }

      double FEMRadialBasis::nuclear_density(const arma::mat &Prad) const {
        if (Prad.n_rows != Nbf() || Prad.n_cols != Nbf())
          throw std::logic_error("nuclear_density expects a radial density matrix\n");

        // Nuclear coordinate
        arma::vec x(1);
        // Remember that the primitive basis polynomials belong to [-1,1]
        x(0) = -1.0;

        // Evaluate derivative at nucleus
        arma::mat der(fem.eval_df(x, 0));

        // Radial functions in element
        size_t ifirst, ilast;
        get_idx(0, ifirst, ilast);
        // Density submatrix
        arma::mat Psub(Prad.submat(ifirst, ifirst, ilast, ilast));
        // P_uv B_u'(0) B_v'(0)
        double den(arma::as_scalar(der * Psub * arma::trans(der)));

        return den;
      }

      double FEMRadialBasis::nuclear_density_gradient(const arma::mat &Prad) const {
        if (Prad.n_rows != Nbf() || Prad.n_cols != Nbf())
          throw std::logic_error("nuclear_density_gradient expects a radial density matrix\n");

        // Nuclear coordinate
        arma::vec x(1);
        // Remember that the primitive basis polynomials belong to [-1,1]
        x(0) = -1.0;

        // Evaluate derivative at nucleus
        arma::mat der(fem.eval_df(x, 0));
        arma::mat lapl(fem.eval_d2f(x, 0));

        // Radial functions in element
        size_t ifirst, ilast;
        get_idx(0, ifirst, ilast);
        // Density submatrix
        arma::mat Psub(Prad.submat(ifirst, ifirst, ilast, ilast));
        // P_uv B_u'(0) B_v''(0)
        double den(arma::as_scalar(der * Psub * arma::trans(lapl)));

        return den;
      }

      arma::rowvec FEMRadialBasis::nuclear_orbital(const arma::mat &C) const {
        // Nuclear coordinate
        arma::vec x(1);
        // Remember that the primitive basis polynomials belong to [-1,1]
        x(0) = -1.0;

        // Derivative
        arma::mat der(fem.eval_df(x, 0));
        // Radial functions in element
        size_t ifirst, ilast;
        get_idx(0, ifirst, ilast);
        // Density submatrix
        arma::mat Csub(C.rows(ifirst, ilast));

        // C_ui B_u'(0)
        return der * Csub;
      }
    } // namespace basis
  } // namespace atomic
} // namespace helfem
