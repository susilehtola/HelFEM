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

      arma::mat RadialBasis::get_bf(const arma::vec & x, size_t iel) const {
        // Element function values at quadrature points are
        arma::mat val(fem.eval_f(x, iel));
        // but we also need to put in the 1/r factor
        arma::vec r(fem.eval_coord(x, iel));
        for (size_t j = 0; j < val.n_cols; j++)
          for (size_t i = 0; i < val.n_rows; i++)
            val(i, j) /= r(i);

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

        // Derivative is then
        arma::mat der(fval);
        for (size_t j = 0; j < fval.n_cols; j++)
          for (size_t i = 0; i < fval.n_rows; i++)
            der(i, j) = dval(i, j) / r(i) - fval(i, j) / (r(i) * r(i));

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

        // Laplacian is then
        arma::mat lapl(fval);
        for (size_t j = 0; j < fval.n_cols; j++)
          for (size_t i = 0; i < fval.n_rows; i++)
            lapl(i, j) = lval(i, j) / r(i) -
              2.0 * dval(i, j) / (r(i) * r(i)) +
              2.0 * fval(i, j) / (r(i) * r(i) * r(i));

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
