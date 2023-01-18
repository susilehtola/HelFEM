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
#include <cfloat>

#ifdef _OPENMP
#include <omp.h>
#endif

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
        std::function<double(double)> dummy;
        std::function<arma::mat(const arma::vec &,size_t)> radial_bf;
        radial_bf = [this](const arma::vec & xq, size_t iel) { return this->get_bf(xq, iel); };

        return 0.5 * fem.matrix_element(iel, radial_bf, radial_bf, xq, wq, dummy);
      }

      arma::mat RadialBasis::nuclear(size_t iel) const {
        std::function<double(double)> dummy;
        std::function<arma::mat(const arma::vec &,size_t)> radial_bf, fem_bf;
        radial_bf = [this](const arma::vec & xq, size_t iel) { return this->get_bf(xq, iel); };
        fem_bf = [this](const arma::vec & xq, size_t iel) { return this->fem.eval_f(xq, iel); };
        arma::mat Vnuc=-fem.matrix_element(iel, radial_bf, fem_bf, xq, wq, dummy);
        return 0.5*(Vnuc+Vnuc.t());
      }

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
        origin(0)=-1;
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
    } // namespace basis
  } // namespace atomic
} // namespace helfem
