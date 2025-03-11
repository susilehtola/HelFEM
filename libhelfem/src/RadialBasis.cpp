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
      RadialBasis::RadialBasis() {}

      RadialBasis::RadialBasis(const polynomial_basis::FiniteElementBasis & fem_, int n_quad, int taylor_order_) : fem(fem_), taylor_order(taylor_order_) {
        // Get quadrature rule
        chebyshev::chebyshev(n_quad, xq, wq);
        for (size_t i = 0; i < xq.n_elem; i++) {
          if (!std::isfinite(xq[i]))
            printf("xq[%i]=%e\n", (int)i, xq[i]);
          if (!std::isfinite(wq[i]))
            printf("wq[%i]=%e\n", (int)i, wq[i]);
        }

        // Compute Taylor series at the origin
        arma::vec origin(1);
        origin(0)=-1;

        size_t iel=0;
        taylor_df.resize(taylor_order);
        for(int i=0;i<taylor_order;i++) {
          // f(r) = B'(0) + 1/2 B''(0) r + ...: Constant term only
          // survives without derivative, first-order term only up to
          // first derivative etc.
          taylor_df[i] = fem.eval_dnf(origin, i+1, iel);
        }

        // Adjust cutoff
        set_small_r_taylor_cutoff();
      }

      void RadialBasis::set_small_r_taylor_cutoff() {
        // Determine small r Taylor cutoff by minimizing the
        // difference of the analytic and Taylor values of the
        // function and its first two derivatives.

        // Start out by ensuring that rcut values are to the left of
        // any nodes in the basis
        std::shared_ptr<const polynomial_basis::PolynomialBasis> p(fem.get_basis(0));
        arma::vec nodes(arma::sort(p->get_nodes()));
        arma::vec minx(1);
        minx(0)=nodes(1);
        arma::vec maxr(fem.eval_coord(minx, 0));

        // Now divvy out the space with a logarithmic grid
        arma::vec rcut(arma::logspace<arma::vec>(-10, 0, 1000)*maxr(0));

        // Find the primitive coordinates corresponding to the cutoffs
        arma::vec xprim(fem.eval_prim(rcut, 0));

        // Evaluate the basis functions and their derivatives at the cutoff
        small_r_taylor_cutoff = -1.0;
        arma::mat bf0(get_bf(xprim, 0));
        arma::mat df0(get_df(xprim, 0));
        arma::mat lf0(get_lf(xprim, 0));

        // Taylor expansions
        small_r_taylor_cutoff = DBL_MAX;
        arma::mat bft(bf0), dft(df0), lft(lf0);
        arma::uvec taylorind(arma::linspace<arma::uvec>(0, rcut.n_elem-1, rcut.n_elem));
        get_taylor(rcut, taylorind, bft, 0);
        get_taylor(rcut, taylorind, dft, 1);
        get_taylor(rcut, taylorind, lft, 2);

        // Differences
        arma::mat diff_f(bft-bf0);
        arma::mat diff_df(dft-df0);
        arma::mat diff_lf(lft-lf0);

        // Accumulated differences
        arma::mat diffs(diff_f.n_rows,5,arma::fill::zeros);
        diffs.col(0)=rcut/fem.element_length(0);
        for(size_t i=0;i<diffs.n_rows;i++) {
          diffs(i,1) = arma::norm(diff_f.row(i),2)/arma::norm(bf0.row(i),2);
          // Only try to maximize fit of derivatives if they are nonzero
          if(taylor_order>=1)
            diffs(i,2) = arma::norm(diff_df.row(i),2)/arma::norm(df0.row(i),2);
          if(taylor_order>1)
            diffs(i,3) = arma::norm(diff_lf.row(i),2)/arma::norm(lf0.row(i),2);
          diffs(i,4) = diffs(i,1)+diffs(i,2)+diffs(i,3);
        }

        // Use the first local minimum coming in from large r
        size_t icut;
        for(icut=rcut.n_elem-2;icut>0;icut--) {
          if(diffs(icut,4) > diffs(icut+1,4))
            break;
        }
        if(small_r_taylor_cutoff == DBL_MAX) {
          icut=rcut.n_elem-1;
        }
        small_r_taylor_cutoff = rcut(icut);

        // Save error
        taylor_diff=diffs(icut,4);

        /*
        // Print out agreement at cutoff
        printf("\nAnalytic vs Taylor\n");
        printf("%4s %22s %22s %22s %22s %22s %22s\n","ifun","bfanal","bftayl","dfanal","dftayl","lfanal","lftayl");
        for(size_t ifun=0;ifun<bf0.n_cols;ifun++)
	printf("%4i % .15e % .15e % .15e % .15e % .15e % .15e\n",ifun, bf0(icut,ifun), bft(icut,ifun), df0(icut,ifun), dft(icut,ifun), lf0(icut,ifun), lft(icut,ifun));
        diffs.row(icut).print("Relative differences at cutoff");
        */
        //diffs.save("taylor_diff.dat", arma::raw_ascii);
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

      int RadialBasis::get_taylor_order() const {
        return taylor_order;
      }

      double RadialBasis::get_taylor_diff() const {
        return taylor_diff;
      }


      arma::mat RadialBasis::radial_integral(int Rexp, size_t iel, double x_left, double x_right) const {
        std::function<double(double)> rpowL = [Rexp](double r){return std::pow(r,Rexp+2);};
        std::function<arma::mat(const arma::vec &,size_t)> radial_bf;
        radial_bf = [this](const arma::vec & xq_, size_t iel_) { return this->get_bf(xq_, iel_); };
        arma::mat ret(fem.matrix_element(iel, radial_bf, radial_bf, xq, wq, rpowL, x_left, x_right));
        if(ret.has_nan()) {
          printf("radial_integral(%i,%i) has NaN!\n",Rexp,(int) iel);
        }
        return ret;
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

      arma::mat RadialBasis::overlap(size_t iel) const {
        std::function<double(double)> dummy;
        return fem.matrix_element(iel, false, false, xq, wq, dummy);
      }

      arma::mat RadialBasis::overlap() const {
        std::function<double(double)> dummy;
        return fem.matrix_element(false, false, xq, wq, dummy);
      }

      arma::mat RadialBasis::kinetic() const {
        std::function<double(double)> dummy;
        return 0.5*fem.matrix_element(true, true, xq, wq, dummy);
      }

      arma::mat RadialBasis::kinetic(size_t iel) const {
        std::function<double(double)> dummy;
        return 0.5*fem.matrix_element(iel, true, true, xq, wq, dummy);
      }

      arma::mat RadialBasis::kinetic_l() const {
        std::function<double(double)> dummy;
        std::function<arma::mat(const arma::vec &,size_t)> radial_bf;
        radial_bf = [this](const arma::vec & xq_, size_t iel_) { return this->get_bf(xq_, iel_); };

        return 0.5 * fem.matrix_element(radial_bf, radial_bf, xq, wq, dummy);
      }

      arma::mat RadialBasis::kinetic_l(size_t iel) const {
        std::function<double(double)> dummy;
        std::function<arma::mat(const arma::vec &,size_t)> radial_bf;
        radial_bf = [this](const arma::vec & xq_, size_t iel_) { return this->get_bf(xq_, iel_); };

        return 0.5 * fem.matrix_element(iel, radial_bf, radial_bf, xq, wq, dummy);
      }

      arma::mat RadialBasis::nuclear() const {
        std::function<double(double)> r = [](double r){return r;};
        std::function<arma::mat(const arma::vec &,size_t)> radial_bf;
        radial_bf = [this](const arma::vec & xq_, size_t iel_) { return this->get_bf(xq_, iel_); };
        return -fem.matrix_element(radial_bf, radial_bf, xq, wq, r);
      }

      arma::mat RadialBasis::nuclear(size_t iel) const {
        std::function<double(double)> r = [](double r){return r;};
        std::function<arma::mat(const arma::vec &,size_t)> radial_bf;
        radial_bf = [this](const arma::vec & xq_, size_t iel_) { return this->get_bf(xq_, iel_); };
        return -fem.matrix_element(iel, radial_bf, radial_bf, xq, wq, r);
      }

      arma::mat RadialBasis::polynomial_confinement(size_t iel, int N, double shift_pot) const {
	std::function<double(double)> rpow = [N, shift_pot](double r){
	  if(r<shift_pot)
	    return 0.0;
	  return std::pow(r-shift_pot,N+2);
	};
	std::function<arma::mat(const arma::vec &,size_t)> radial_bf = [this](const arma::vec & xq_, size_t iel_) { return this->get_bf(xq_, iel_); };
        return fem.matrix_element(iel, radial_bf, radial_bf, xq, wq, rpow);
      }

      arma::mat RadialBasis::exponential_confinement(size_t iel, int N, double r_0, double shift_pot) const {
	std::function<double(double)> r_exp = [r_0, N, shift_pot](double r) {
	  if(r<shift_pot)
	    return 0.0;
	  const double r_ratio = (r-shift_pot)/r_0;
	  double fact = 1.0;

	  double V=0.0;
	  double r_ratio_pow_k = 1.0;
	  for (int k=0; k<N; k++) {
	    // r^k / k!
	    V -= r_ratio_pow_k / fact;
	    // Prepare values for next iteration
	    fact *= k+1;
	    r_ratio_pow_k *= r_ratio;
	  }
	  V += std::exp(r_ratio);
	  V *= fact;
	  V *= std::pow(r, 2);
	  return V;
	};
	std::function<arma::mat(const arma::vec &, size_t)> radial_bf;
	radial_bf = [this](const arma::vec & xq_, size_t iel_) { return this->get_bf(xq_, iel_); };
	return fem.matrix_element(iel, radial_bf, radial_bf, xq, wq, r_exp);
      }

      arma::mat RadialBasis::barrier_confinement(size_t iel, double V, double shift_pot) const {
	std::function<double(double)> barrier = [V, shift_pot](double r) {
	  if(r<shift_pot)
	    return 0.0;
	  return V * std::pow(r, 2);
	};
	std::function<arma::mat(const arma::vec &, size_t)> radial_bf;
	radial_bf = [this](const arma::vec & xq_, size_t iel_) { return this->get_bf(xq_, iel_); };
	return fem.matrix_element(iel, radial_bf, radial_bf, xq, wq, barrier);
      }

      arma::mat RadialBasis::junq_confinement(size_t iel, int N, double V0, double r_c, double shift_pot) const {
	std::function<double(double)> r_exp = [N, r_c, V0, shift_pot](double r) {
	  if(r<shift_pot)
	    return 0.0;
	  const double denominator = std::pow(r_c-r,N);
	  const double exponential = std::exp(-(r_c - shift_pot) / (r - shift_pot));
	  return V0 * exponential / denominator * std::pow(r,2);
	};
	std::function<arma::mat(const arma::vec &, size_t)> radial_bf;
	radial_bf = [this](const arma::vec & xq_, size_t iel_) { return this->get_bf(xq_, iel_); };
	return fem.matrix_element(iel, radial_bf, radial_bf, xq, wq, r_exp);
      }
      
      arma::mat RadialBasis::confinement_potential(size_t iel, int N, double r_0, int iconf, double V, double shift_pot) const {
	// Attractive potential does not make sense for shift_pot != 0

	// sign of r0 controls if the potential is attractive or repulsive
	int sign = (r_0<0) ? -1 : 1;
	r_0 = std::abs(r_0);

	if(iconf==1) {
	  if(N<0) {
	    if(shift_pot != 0.0)
	      throw std::logic_error("Cannot have a divergent potential with a shift!\n");
	    return sign*std::pow(r_0, N)*polynomial_confinement(iel, N, shift_pot);
	  } else {
	    return sign*std::pow(r_0, -N)*polynomial_confinement(iel, N, shift_pot);
	  }

	} else if(iconf==2) {
	  if(N<0)
	    throw std::logic_error("Exponential confinement potential does not make sense with negative N!\n");
	  if(N==0)
	    throw std::logic_error("Exponential confinement potential requires N >= 1!");

	  return exponential_confinement(iel, N, r_0, shift_pot);
	
	} else if(iconf==3) {
	  if(V<0)
	    throw std::logic_error("Can not have attractive barrier!\n");
	  return barrier_confinement(iel, V, shift_pot);
	} else if(iconf==4) {
	  if(N<=0)
	    throw std::logic_error("Junquera confinement potential requires N >= 1!");
	  if(V<=0)
	    throw std::logic_error("Can not have attractive Junquera potential!\n");
	  return junq_confinement(iel, N, V, r_0, shift_pot);
	} else
	  throw std::logic_error("Case not implemented!\n");
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
        if(tei.has_nan()) {
          printf("twoe_integral(%i,%i) has NaN!\n",L,(int) iel);
        }
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

        /*
	  if(taylor_order<3 || taylor_order>9) {
          std::ostringstream oss;
          oss << "Got taylor_order = " << taylor_order << ". Taylor expansion order needs to be 3 <= taylor_order <= 9.\n";
          throw std::logic_error(oss.str());
	  }
        */

        // Coefficients of the various derivatives in the expansion of
        // the function itself. Note that the zeroth element already
        // corresponds to the first derivative!
        arma::vec taylorcoeff(taylor_order);
        taylorcoeff(0) = 1.0;
        for(int i=1; i<taylor_order; i++)
          taylorcoeff(i) = taylorcoeff(i-1)/(i+1);
        // and the corresponding r exponents
        arma::ivec rexp(arma::linspace<arma::ivec>(0,taylor_order-1,taylor_order));

        // Compute derivatives: c r^n -> c n r^(n-1)
        for(int i=0; i<taylor_order; i++) {
          for(int d=0; d<ider; d++) {
            taylorcoeff(i) *= rexp(i);
            rexp(i)--;
          }
        }

        // Exponentiate r
        std::vector<arma::vec> rexpval(taylor_order);
        for(int i=ider; i < taylor_order; i++) {
          if(rexp[i]<0)
            throw std::logic_error("This should not have happened!\n");
          rexpval[i] = arma::pow(r, rexp[i]);
        }

        // Collect results
        for(size_t ifun = 0; ifun < val.n_cols; ifun++)
          for(size_t ir = 0; ir < taylorind.n_elem; ir++) {
            val(ir, ifun) = 0.0;
            // Terms below this are zero
            for(int iterm=ider;iterm < taylor_order;iterm++) {
              val(ir, ifun) += taylorcoeff[iterm]*taylor_df[iterm](ifun)*rexpval[iterm](ir);
            }
          }
      }

      arma::vec RadialBasis::eval_orbs(const arma::mat & C, double r) const {
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
          for (size_t ir = taylorind.n_elem; ir < x.n_elem; ir++) {
            double invr = 1.0/r(ir);
            der(ir, ifun) = (-fval(ir, ifun) * invr + dval(ir, ifun)) * invr;
          }
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
          for (size_t ir = taylorind.n_elem; ir < x.n_elem; ir++) {
            double invr = 1.0/r(ir);
            lapl(ir, ifun) = ((2.0 * fval(ir, ifun)*invr - 2.0*dval(ir,ifun))*invr + lval(ir,ifun))*invr;
          }

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

      double RadialBasis::get_r(double x, size_t iel) const {
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
