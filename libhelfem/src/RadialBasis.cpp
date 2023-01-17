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
#include "helfem/RadialBasis.h"
#include "RadialPotential.h"
#include "chebyshev.h"
#include "quadrature.h"
#include "utils.h"
#include <map>
#include <cfloat>

#ifdef _OPENMP
#include <omp.h>
#endif

// For factorials
extern "C" {
#include <gsl/gsl_sf_gamma.h>
}

inline static double factorial(unsigned int n) {
  if(n==0)
    return 1.0;

  return gsl_sf_fact(n);
}

namespace helfem {
  namespace atomic {
    namespace basis {
      RadialBasis::RadialBasis() {}

      RadialBasis::RadialBasis(const polynomial_basis::FiniteElementBasis & fem_, int n_quad) : fem(fem_) {
        // Get quadrature rule
        chebyshev::chebyshev(n_quad, xq, wq);
        for (size_t i = 0; i < xq.n_elem; i++) {
          if (!std::isfinite(xq[i]))
            printf("xq[%i]=%e\n", (int)i, xq[i]);
          if (!std::isfinite(wq[i]))
            printf("wq[%i]=%e\n", (int)i, wq[i]);
        }

        // Determine small r Taylor cutoff. We use a 5th order Taylor
        // polynomial f(r) = 0 + f'(0) r + 1/2 f''(0) r^2 + ... + 1/5!
        // f^(5)(0) r^5 but we also need up to second derivatives. We
        // should be able to safely use a truncation of machine
        // epsilon^(1/3), since then the last term in the second
        // derivative should be negligible compared to the first one.
        double small_r_cutoff_eps = std::cbrt(DBL_EPSILON);

        // However, if the first element is very small, such as in the
        // case of finite nuclei, we would not be portraying our basis
        // functions very accurately in the element. This is why the
        // cutoff has to be defined in terms of the element size, the
        // number of functions in the element, as well as a safety
        // factor.
        double small_r_cutoff_fun = fem.element_length(0) / fem.get_max_nprim() / 100;

        // We use the minimum of the two for our cutoff
        small_r_taylor_cutoff = std::min(small_r_cutoff_eps, small_r_cutoff_fun);
      }

      RadialBasis::~RadialBasis() {}

      int RadialBasis::get_nquad() const {
        return (int)xq.n_elem;
      }

      arma::vec RadialBasis::get_xq() const {
        return xq;
      }

      size_t RadialBasis::Nbf() const {
        return fem.get_nbf();
      }

      size_t RadialBasis::Nel() const {
        return fem.get_nelem();
      }

      size_t RadialBasis::Nprim(size_t iel) const {
        return fem.get_nprim(iel);
      }

      size_t RadialBasis::max_Nprim() const {
        return fem.get_max_nprim();
      }

      void RadialBasis::get_idx(size_t iel, size_t &ifirst, size_t &ilast) const {
        fem.get_idx(iel, ifirst, ilast);
      }

      arma::vec RadialBasis::get_bval() const {
        return fem.get_bval();
      }

      int RadialBasis::get_poly_id() const {
        return fem.get_poly_id();
      }

      int RadialBasis::get_poly_nnodes() const {
        return fem.get_poly_nnodes();
      }

      double RadialBasis::get_small_r_taylor_cutoff() const {
        return small_r_taylor_cutoff;
      }

      arma::mat RadialBasis::radial_integral(int Rexp, size_t iel) const {
        std::function<double(double)> rpowL = [Rexp](double r) { return std::pow(r, Rexp); };
        return fem.matrix_element(iel, false, false, xq, wq, rpowL);
      }

      arma::mat RadialBasis::bessel_il_integral(int L, double lambda, size_t iel) const {
        std::function<double(double)> besselil = [L, lambda](double r) { return utils::bessel_il(r*lambda, L); };
        return fem.matrix_element(iel, false, false, xq, wq, besselil);
      }

      arma::mat RadialBasis::bessel_kl_integral(int L, double lambda, size_t iel) const {
        std::function<double(double)> besselkl = [L, lambda](double r) { return utils::bessel_kl(r*lambda, L); };
        return fem.matrix_element(iel, false, false, xq, wq, besselkl);
      }

      arma::mat RadialBasis::radial_integral(const RadialBasis &rh, int n, bool lhder,
                                             bool rhder) const {
        modelpotential::RadialPotential rad(n);
        return model_potential(rh, &rad, lhder, rhder);
      }

      arma::mat RadialBasis::model_potential(const RadialBasis &rh,
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

      arma::mat RadialBasis::overlap(const RadialBasis &rh) const {
        return radial_integral(rh, 0);
      }

      arma::mat RadialBasis::kinetic(size_t iel) const {
        std::function<double(double)> dummy;
        return 0.5*fem.matrix_element(iel, true, true, xq, wq, dummy);
      }

      arma::mat RadialBasis::kinetic_l(size_t iel) const {
        return 0.5 * radial_integral(-2, iel);
      }

      arma::mat RadialBasis::nuclear(size_t iel) const { return -radial_integral(-1, iel); }

      arma::mat RadialBasis::model_potential(const modelpotential::ModelPotential *model,
                                             size_t iel) const {
        std::function<double(double)> modelpot = [model](double r) { return model->V(r); };
        return fem.matrix_element(iel, false, false, xq, wq, modelpot);
      }

      arma::mat RadialBasis::nuclear_offcenter(size_t iel, double Rhalf, int L) const {
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

      arma::mat RadialBasis::twoe_integral(int L, size_t iel) const {
        double Rmin(fem.element_begin(iel));
        double Rmax(fem.element_end(iel));

        // Integral by quadrature
        std::shared_ptr<const polynomial_basis::PolynomialBasis> p(fem.get_basis(iel));
        arma::mat tei(quadrature::twoe_integral(Rmin, Rmax, xq, wq, p, L));

        return tei;
      }

      arma::mat RadialBasis::yukawa_integral(int L, double lambda, size_t iel) const {
        double Rmin(fem.element_begin(iel));
        double Rmax(fem.element_end(iel));

        // Integral by quadrature
        std::shared_ptr<const polynomial_basis::PolynomialBasis> p(fem.get_basis(iel));
        arma::mat tei(quadrature::yukawa_integral(Rmin, Rmax, xq, wq, p, L, lambda));

        return tei;
      }

      arma::mat RadialBasis::erfc_integral(int L, double mu, size_t iel, size_t kel) const {
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

      arma::mat RadialBasis::spherical_potential(size_t iel) const {
        double Rmin(fem.element_begin(iel));
        double Rmax(fem.element_end(iel));

        // Integral by quadrature
        std::shared_ptr<polynomial_basis::PolynomialBasis> p(fem.get_basis(iel));
        arma::mat pot(quadrature::spherical_potential(Rmin, Rmax, xq, wq, p));

        return pot;
      }

      arma::mat RadialBasis::get_bf(size_t iel) const {
        return get_bf(xq, iel);
      }

      void RadialBasis::get_taylor(const arma::vec & r, const arma::uvec & taylorind, arma::mat & val, int ider) const {
        if(taylorind[0]!=0 || taylorind[taylorind.n_elem-1] != taylorind.n_elem-1)
          throw std::logic_error("Taylor points not consecutive!\n");

        // The series of B(r)/r is
        //  [B(0) + B'(0)r + 1/2 B''(0) r^2 + ...]/r
        //= B'(0) + 1/2 B''(0) r + 1/6 B'''(0) r^2 + ...

        // Coefficients of the various derivatives
        arma::vec taylorcoeff(5);
        taylorcoeff(0) = 1.0;
        for(size_t i=1; i<taylorcoeff.n_elem; i++)
          taylorcoeff(i) = taylorcoeff(i-1)/(i+1);

        // The related order of the derivatives
        arma::ivec derorder(arma::linspace<arma::ivec>(1,taylorcoeff.n_elem,taylorcoeff.n_elem));
        // and the corresponding r exponents
        arma::ivec rexp(arma::linspace<arma::ivec>(0,taylorcoeff.n_elem-1,taylorcoeff.n_elem));

        // Compute derivatives: c r^n -> cn r^(n-1)
        for(size_t i=0; i<taylorcoeff.n_elem; i++) {
          for(int d=0; d<ider; d++) {
            taylorcoeff(i) *= rexp(i);
            rexp(i)--;
          }
        }

        // Form Taylor series at the origin
        arma::vec origin(1);
        origin(1)=-1;
        std::vector<arma::rowvec> df(taylorcoeff.n_elem);

        size_t iel=0;
        if(ider==0)
          df[0] = fem.eval_df(origin, iel);
        if(ider<=1)
          df[1] = fem.eval_d2f(origin, iel);
        df[2] = fem.eval_d3f(origin, iel);
        df[3] = fem.eval_d4f(origin, iel);
        df[4] = fem.eval_d5f(origin, iel);

        // Exponentiate r
        std::vector<arma::vec> rexpval(taylorcoeff.n_elem);
        for(size_t i=0; i < rexpval.size(); i++)
          rexpval[i] = arma::pow(r, rexp[i]);

        for (size_t ifun = 0; ifun < val.n_cols; ifun++)
          for (size_t ir = 0; ir < taylorind.n_elem; ir++) {
            val(ir, ifun) = 0.0;
            // Terms below this are zero
            for(size_t iterm=ider;iterm < taylorcoeff.n_elem;iterm++) {
              val(ir, ifun) += taylorcoeff[iterm]*df[iterm](ifun)*rexpval[iterm](ir);
            }
          }
      }

      arma::mat RadialBasis::get_bf(const arma::vec & x, size_t iel) const {
        // Element function values at quadrature points are
        arma::mat val(fem.eval_f(x, iel));
        // but we also need to put in the 1/r factor
        arma::vec r(fem.eval_coord(x, iel));

        // Indices where to apply Taylor series
        arma::uvec taylorind;
        if(iel==0)
          taylorind = arma::find(r <= small_r_taylor_cutoff);

        // Special handling for points close to the nucleus
        if(taylorind.n_elem>0) {
          get_taylor(r, taylorind, val, 0);
        }
        // Normal handling elsewhere
        for (size_t ifun = 0; ifun < val.n_cols; ifun++)
          for (size_t ir = taylorind.n_elem; ir < x.n_elem; ir++)
            val(ir, ifun) /= r(ir);

        return val;
      }

      arma::mat RadialBasis::get_df(size_t iel) const {
        return get_df(xq ,iel);
      }

      arma::mat RadialBasis::get_df(const arma::vec & x, size_t iel) const {
        // Element function values at quadrature points are
        arma::mat fval(fem.eval_f(x, iel));
        arma::mat dval(fem.eval_df(x, iel));
        arma::vec r(fem.eval_coord(x, iel));
        arma::mat der(fval);

        // Indices where to apply Taylor series
        arma::uvec taylorind;
        if(iel==0)
          taylorind = arma::find(r <= small_r_taylor_cutoff);

        // Special handling for points close to the nucleus
        if(taylorind.n_elem>0) {
          get_taylor(r, taylorind, der, 1);
        }
        // Normal handling elsewhere
        for (size_t ifun = 0; ifun < der.n_cols; ifun++)
          for (size_t ir = taylorind.n_elem; ir < x.n_elem; ir++)
            der(ir, ifun) = dval(ir, ifun) / r(ir) - fval(ir, ifun) / (r(ir) * r(ir));

        return der;
      }

      arma::mat RadialBasis::get_lf(size_t iel) const {
        return get_lf(xq, iel);
      }

      arma::mat RadialBasis::get_lf(const arma::vec & x, size_t iel) const {
        // Element function values at quadrature points are
        arma::mat fval(fem.eval_f(x, iel));
        arma::mat dval(fem.eval_df(x, iel));
        arma::mat lval(fem.eval_d2f(x, iel));
        arma::vec r(fem.eval_coord(x, iel));
        arma::mat lapl(fval);

        // Indices where to apply Taylor series
        arma::uvec taylorind;
        if(iel==0)
          taylorind = arma::find(r <= small_r_taylor_cutoff);

        // Special handling for points close to the nucleus
        if(taylorind.n_elem>0) {
          get_taylor(r, taylorind, lapl, 2);
        }
        // Normal handling elsewhere
        for (size_t ifun = 0; ifun < lapl.n_cols; ifun++)
          for (size_t ir = taylorind.n_elem; ir < x.n_elem; ir++)
            lapl(ir, ifun) = lval(ir, ifun) / r(ir) -
              2.0 * dval(ir, ifun) / (r(ir) * r(ir)) +
              2.0 * fval(ir, ifun) / (r(ir) * r(ir) * r(ir));

        return lapl;
      }

      arma::vec RadialBasis::get_wrad(size_t iel) const {
        return get_wrad(wq, iel);
      }

      arma::vec RadialBasis::get_wrad(const arma::vec & w, size_t iel) const {
        // This is just the radial rule, no r^2 factor included here
        return fem.scaling_factor(iel) * w;
      }

      arma::vec RadialBasis::get_r(size_t iel) const {
        return get_r(xq, iel);
      }

      arma::vec RadialBasis::get_r(const arma::vec & x, size_t iel) const {
        return fem.eval_coord(x, iel);
      }

      double RadialBasis::nuclear_density(const arma::mat &Prad) const {
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

      double RadialBasis::nuclear_density_gradient(const arma::mat &Prad) const {
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

      arma::rowvec RadialBasis::nuclear_orbital(const arma::mat &C) const {
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

      /// Comparison operator needed for std::map
      bool operator<(const radial_function_t & lh, const radial_function_t & rh) {
        if(lh.rpow < rh.rpow)
          return true;
        if(lh.rpow > rh.rpow)
          return false;
        if(lh.deriv < rh.deriv)
          return true;
        if(lh.deriv > rh.deriv)
          return false;
        return false;
      }

      /// Function to calculate arbitrary derivatives
      std::map<radial_function_t, int> calculate_derivative(int nder) {
        std::map<radial_function_t, int> ret;
        radial_function_t base({-1, 0});
        ret[base] = 1;

        for(int ider=0; ider<nder; ider++) {
          std::map<radial_function_t, int> newret;
          for(auto it=ret.begin(); it!=ret.end(); it++) {
            // Current values
            int cur_rpow = it->first.rpow;
            int cur_deriv = it->first.deriv;
            int cur_coeff = it->second;

            // New terms: attack shape function
            radial_function_t term1({cur_rpow, cur_deriv+1});
            auto it1 = newret.find(term1);
            if(it1 == newret.end()) {
              newret[term1] = cur_coeff;
            } else {
              it1->second += cur_coeff;
            }

            // Attack power of r
            radial_function_t term2({cur_rpow-1, cur_deriv});
            int new_coeff = cur_coeff*cur_rpow;

            auto it2 = newret.find(term2);
            if(it2 == newret.end()) {
              newret[term2] = new_coeff;
            } else {
              it2->second += new_coeff;
            }
          }
          ret = newret;
        }

        return ret;
      }

      /// Comparison operator needed for std::map
      bool operator<(const radial_product_t & lh, const radial_product_t & rh) {
        if(lh.rpow < rh.rpow)
          return true;
        if(lh.rpow > rh.rpow)
          return false;
        if(lh.ider < rh.ider)
          return true;
        if(lh.ider > rh.ider)
          return false;
        if(lh.jder < rh.jder)
          return true;
        if(lh.jder > rh.jder)
          return false;
        return false;
      }

      /// Ensure ider>=jder
      radial_product_t radial_prod(int rpow, int ider, int jder) {
        if(jder<ider)
          return radial_product_t({rpow, ider, jder});
        else
          return radial_product_t({rpow, jder, ider});
      }

      /// Apply l'Hôpital's theorem to make the evaluation accurate
      std::map<radial_product_t, double> apply_hopital(int ider, int jder, int rpow) {
        // First, build the expressions for the individual basis functions
        auto ifunc(calculate_derivative(ider));
        auto jfunc(calculate_derivative(jder));

        // .. and assemble the result
        std::map<radial_product_t, int> result;
        for(auto it=ifunc.begin(); it!=ifunc.end(); it++) {
          for(auto jt=jfunc.begin(); jt!=jfunc.end(); jt++) {
            auto prod(radial_prod(it->first.rpow + jt->first.rpow + rpow, it->first.deriv, jt->first.deriv));
            increment_term(result, prod, it->second*jt->second);
          }
        }

        std::cout << "Initial form of chi^(" << ider << ")" << " * chi^(" << jder << ") * r^" << rpow << " is" << std::endl;
        for(auto it=result.begin(); it!=result.end(); it++) {
          std::cout << it->second << " * r^" << it->first.rpow << " * B^(" << it->first.ider << ") * B^(" << it->first.jder << ")" << std::endl;
        }

        // Now all that remains is to go through the terms and see if any behave badly at the origin.
        std::map<radial_product_t, double> cleanup;
        for(auto iterm=result.begin(); iterm!=result.end(); iterm++) {
          // If the term has a non-negative overall r exponent, it is well-behaved.
          if(iterm->first.rpow >= 0) {
            increment_term(cleanup, iterm->first, static_cast<double>(iterm->second));
            continue;
          }
          // Alternatively, if the term only contains derivatives, it is okay at the origin as well.
          if(iterm->first.ider>0 && iterm->first.jder > 0) {
            increment_term(cleanup, iterm->first, static_cast<double>(iterm->second));
            continue;
          }

          // The only problem case is when we have something that
          // looks like B_u(r) B_v(r)/r^n, in which case we have to use
          // l'Hopital's rule n times.
          int ndenom(-iterm->first.rpow);

          std::map<radial_product_t, int> termcleanup;
          {
            // Initiate recursion: first application of Leibniz' rule gives
            auto term1(radial_prod(0, iterm->first.ider+1, iterm->first.jder));
            auto term2(radial_prod(0, iterm->first.ider, iterm->first.jder+1));
            increment_term(termcleanup, term1, 1);
            increment_term(termcleanup, term2, 1);
          }
          // Use recursion to evaluate derivatives
          for(int ider=1;ider<ndenom;ider++) {
            std::map<radial_product_t, int> reclean;
            for(auto it=termcleanup.begin(); it!=termcleanup.end(); it++) {
              // Initiate recursion: first application of Leibniz' rule gives
              auto term1(radial_prod(0, it->first.ider+1, it->first.jder));
              increment_term(reclean, term1, it->second);
              auto term2(radial_prod(0, it->first.ider, it->first.jder+1));
              increment_term(reclean, term2, it->second);
            }
            termcleanup=reclean;
          }

          // Add into the final result terms
          for(auto it=termcleanup.begin(); it!=termcleanup.end(); it++) {
            increment_term(cleanup, it->first, iterm->second*it->second*1.0/factorial(ndenom));
          }

          std::cout << "Cleaned up form of B^(" <<  iterm->first.ider << ")" << " * B^(" <<  iterm->first.jder << ") * r^" << iterm->first.rpow << " is" << std::endl;
          for(auto it=termcleanup.begin(); it!=termcleanup.end(); it++) {
            std::cout << it->second*iterm->second*1.0/factorial(ndenom) << " * r^" << it->first.rpow << " * B^(" << it->first.ider << ") * B^(" << it->first.jder << ")" << std::endl;
          }
        }

        // Delete all terms that vanish
        std::map<radial_product_t, double> final_result;
        for(auto iterm=cleanup.begin(); iterm!=cleanup.end(); iterm++) {
          // Any term that has an undifferentiated function or a positive r exponent vanishes at the origin
          if(iterm->first.ider == 0 || iterm->first.jder == 0 || iterm->first.rpow>0) {
            continue;
          }
          final_result[iterm->first]=iterm->second;
        }

        return final_result;
      }

    } // namespace basis
  } // namespace atomic
} // namespace helfem
