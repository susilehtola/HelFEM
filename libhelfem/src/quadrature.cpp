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
#include "quadrature.h"
#include "erfc_expn.h"
#include "chebyshev.h"
#include "polynomial.h"
#include "utils.h"

namespace helfem {
  namespace quadrature {
    arma::mat radial_integral(double rmin, double rmax, int n, const arma::vec & x, const arma::vec & wx, const arma::mat & bf) {
#ifndef ARMA_NO_DEBUG
      if(x.n_elem != wx.n_elem) {
        std::ostringstream oss;
        oss << "x and wx not compatible: " << x.n_elem << " vs " << wx.n_elem << "!\n";
        throw std::logic_error(oss.str());
      }
      if(x.n_elem != bf.n_rows) {
        std::ostringstream oss;
        oss << "x and bf not compatible: " << x.n_elem << " vs " << bf.n_rows << "!\n";
        throw std::logic_error(oss.str());
      }
#endif

      // Midpoint is at
      double rmid(0.5*(rmax+rmin));
      // and half-length of interval is
      double rlen(0.5*(rmax-rmin));
      // r values are then
      arma::vec r(rmid*arma::ones<arma::vec>(x.n_elem)+rlen*x);

      // Calculate total weight per point
      arma::vec wp(wx*rlen);
      if(n!=0)
        wp%=arma::pow(r,n);

      // Put in weight
      arma::mat wbf(bf);
      for(size_t i=0;i<bf.n_cols;i++)
	wbf.col(i)%=wp;

      // Matrix elements are then
      return arma::trans(wbf)*bf;
    }

    arma::mat derivative_integral(double rmin, double rmax, const arma::vec & x, const arma::vec & wx, const arma::mat & dbf) {
#ifndef ARMA_NO_DEBUG
      if(x.n_elem != wx.n_elem) {
        std::ostringstream oss;
        oss << "x and wx not compatible: " << x.n_elem << " vs " << wx.n_elem << "!\n";
        throw std::logic_error(oss.str());
      }
      if(x.n_elem != dbf.n_rows) {
        std::ostringstream oss;
        oss << "x and dbf not compatible: " << x.n_elem << " vs " << dbf.n_rows << "!\n";
        throw std::logic_error(oss.str());
      }
#endif

      // Half-length of interval is
      double rlen(0.5*(rmax-rmin));

      // Put in weight
      arma::mat wdbf(dbf);
      for(size_t i=0;i<dbf.n_cols;i++)
	// We get +1 rlen from the jacobian, but -2 from the derivatives
	wdbf.col(i)%=wx/rlen;

      // Integral is
      return arma::trans(wdbf)*dbf;
    }

    arma::mat model_potential_integral(double rmin, double rmax, const modelpotential::ModelPotential * nuc, const arma::vec & x, const arma::vec & wx, const arma::mat & bf) {
#ifndef ARMA_NO_DEBUG
      if(x.n_elem != wx.n_elem) {
        std::ostringstream oss;
        oss << "x and wx not compatible: " << x.n_elem << " vs " << wx.n_elem << "!\n";
        throw std::logic_error(oss.str());
      }
      if(x.n_elem != bf.n_rows) {
        std::ostringstream oss;
        oss << "x and bf not compatible: " << x.n_elem << " vs " << bf.n_rows << "!\n";
        throw std::logic_error(oss.str());
      }
#endif

      // Midpoint is at
      double rmid(0.5*(rmax+rmin));
      // and half-length of interval is
      double rlen(0.5*(rmax-rmin));
      // r values are then
      arma::vec r(rmid*arma::ones<arma::vec>(x.n_elem)+rlen*x);

      // Calculate total weight per point
      arma::vec wp(wx*rlen);
      wp%=nuc->V(r);

      // Put in weight
      arma::mat wbf(bf);
      for(size_t i=0;i<bf.n_cols;i++)
	wbf.col(i)%=wp;

      // Matrix elements are then
      return arma::trans(wbf)*bf;
    }

    arma::mat bessel_il_integral(double rmin, double rmax, int L, double lambda, const arma::vec & x, const arma::vec & wx, const arma::mat & bf) {
#ifndef ARMA_NO_DEBUG
      if(x.n_elem != wx.n_elem) {
        std::ostringstream oss;
        oss << "x and wx not compatible: " << x.n_elem << " vs " << wx.n_elem << "!\n";
        throw std::logic_error(oss.str());
      }
      if(x.n_elem != bf.n_rows) {
        std::ostringstream oss;
        oss << "x and bf not compatible: " << x.n_elem << " vs " << bf.n_rows << "!\n";
        throw std::logic_error(oss.str());
      }
#endif

      // Midpoint is at
      double rmid(0.5*(rmax+rmin));
      // and half-length of interval is
      double rlen(0.5*(rmax-rmin));
      // r values are then
      arma::vec r(rmid*arma::ones<arma::vec>(x.n_elem)+rlen*x);

      // Calculate total weight per point
      arma::vec wp(wx*rlen);
      wp%=utils::bessel_il(r*lambda,L);

      // Put in weight
      arma::mat wbf(bf);
      for(size_t i=0;i<bf.n_cols;i++)
	wbf.col(i)%=wp;

      // Matrix elements are then
      return arma::trans(wbf)*bf;
    }

    arma::mat bessel_kl_integral(double rmin, double rmax, int L, double lambda, const arma::vec & x, const arma::vec & wx, const arma::mat & bf) {
#ifndef ARMA_NO_DEBUG
      if(x.n_elem != wx.n_elem) {
        std::ostringstream oss;
        oss << "x and wx not compatible: " << x.n_elem << " vs " << wx.n_elem << "!\n";
        throw std::logic_error(oss.str());
      }
      if(x.n_elem != bf.n_rows) {
        std::ostringstream oss;
        oss << "x and bf not compatible: " << x.n_elem << " vs " << bf.n_rows << "!\n";
        throw std::logic_error(oss.str());
      }
#endif

      // Midpoint is at
      double rmid(0.5*(rmax+rmin));
      // and half-length of interval is
      double rlen(0.5*(rmax-rmin));
      // r values are then
      arma::vec r(rmid*arma::ones<arma::vec>(x.n_elem)+rlen*x);

      // Calculate total weight per point
      arma::vec wp(wx*rlen);
      wp%=utils::bessel_kl(r*lambda,L);

      // Put in weight
      arma::mat wbf(bf);
      for(size_t i=0;i<bf.n_cols;i++)
	wbf.col(i)%=wp;

      // Matrix elements are then
      return arma::trans(wbf)*bf;
    }

    static arma::vec twoe_inner_integral_wrk(double rmin, double rmax, double rmin0, double rmax0, const arma::vec & x, const arma::vec & wx, const polynomial_basis::PolynomialBasis * poly, int L) {
      // Midpoint is at
      double rmid(0.5*(rmax+rmin));
      // and half-length of interval is
      double rlen(0.5*(rmax-rmin));
      // r values are then
      arma::vec r(rmid*arma::ones<arma::vec>(x.n_elem)+rlen*x);

      // Midpoint of original interval is at
      double rmid0(0.5*(rmax0+rmin0));
      // and half-length of original interval is
      double rlen0(0.5*(rmax0-rmin0));

      // Calculate total weight per point
      arma::vec wp((wx%arma::pow(r,L))*rlen);

      // Calculate x values the polynomials should be evaluated at
      arma::vec xpoly((r-rmid0*arma::ones<arma::vec>(x.n_elem))/rlen0);
      // Evaluate the polynomials at these points
      arma::mat bf(poly->eval(xpoly));

      // Put in weight
      arma::mat wbf(bf);
      for(size_t i=0;i<wbf.n_cols;i++)
        wbf.col(i)%=wp;

      // The integrals are then
      arma::vec inner(arma::vectorise(arma::trans(wbf)*bf));

      return inner;
    }

    arma::mat twoe_inner_integral(double rmin, double rmax, const arma::vec & x, const arma::vec & wx, const polynomial_basis::PolynomialBasis * poly, int L) {
      // Midpoint is at
      double rmid(0.5*(rmax+rmin));
      // and half-length of interval is
      double rlen(0.5*(rmax-rmin));
      // r values are then
      arma::vec r(rmid*arma::ones<arma::vec>(x.n_elem)+rlen*x);

      // Compute the "inner" integrals as function of r.
      arma::mat inner(x.n_elem,std::pow(poly->get_nbf(),2));
      inner.row(0)=arma::trans(twoe_inner_integral_wrk(rmin, r(0), rmin, rmax, x, wx, poly, L));
      // Every subinterval uses a fresh nquad points!
      for(size_t ip=1;ip<x.n_elem;ip++)
        inner.row(ip)=inner.row(ip-1)+arma::trans(twoe_inner_integral_wrk(r(ip-1), r(ip), rmin, rmax, x, wx, poly, L));

      // Put in the 1/r^(L+1) factors now that the integrals have been computed
      arma::vec rpopl(arma::pow(r,L+1));
      for(size_t ip=0;ip<x.n_elem;ip++)
        inner.row(ip)/=rpopl(ip);

      return inner;
    }

    arma::mat twoe_integral(double rmin, double rmax, const arma::vec & x, const arma::vec & wx, const polynomial_basis::PolynomialBasis * poly, int L) {
#ifndef ARMA_NO_DEBUG
      if(x.n_elem != wx.n_elem) {
        std::ostringstream oss;
        oss << "x and wx not compatible: " << x.n_elem << " vs " << wx.n_elem << "!\n";
        throw std::logic_error(oss.str());
      }
#endif
      // and half-length of interval is
      double rlen(0.5*(rmax-rmin));

      // Compute the inner integrals
      arma::mat inner(twoe_inner_integral(rmin, rmax, x, wx, poly, L));

      // Evaluate basis functions at quadrature points
      arma::mat bf(poly->eval(x));

      // Product functions
      arma::mat bfprod(bf.n_rows,bf.n_cols*bf.n_cols);
      for(size_t fi=0;fi<bf.n_cols;fi++)
        for(size_t fj=0;fj<bf.n_cols;fj++)
          bfprod.col(fi*bf.n_cols+fj)=bf.col(fi)%bf.col(fj);
      // Put in the weights for the outer integral
      arma::vec wp(wx*rlen);
      for(size_t i=0;i<bfprod.n_cols;i++)
        bfprod.col(i)%=wp;

      // Integrals are then
      arma::mat ints(arma::trans(bfprod)*inner);
      // but we are still missing the second term which can be
      // obtained as simply as
      ints+=arma::trans(ints);

      return ints;
    }

    static arma::vec yukawa_inner_integral_wrk(double rmin, double rmax, double rmin0, double rmax0, const arma::vec & x, const arma::vec & wx, const polynomial_basis::PolynomialBasis * poly, int L, double lambda) {
      // Midpoint is at
      double rmid(0.5*(rmax+rmin));
      // and half-length of interval is
      double rlen(0.5*(rmax-rmin));
      // r values are then
      arma::vec r(rmid*arma::ones<arma::vec>(x.n_elem)+rlen*x);

      // Midpoint of original interval is at
      double rmid0(0.5*(rmax0+rmin0));
      // and half-length of original interval is
      double rlen0(0.5*(rmax0-rmin0));

      // Calculate total weight per point
      arma::vec wp((wx%utils::bessel_il(r*lambda,L))*rlen);

      // Calculate x values the polynomials should be evaluated at
      arma::vec xpoly((r-rmid0*arma::ones<arma::vec>(x.n_elem))/rlen0);
      // Evaluate the polynomials at these points
      arma::mat bf(poly->eval(xpoly));

      // Put in weight
      arma::mat wbf(bf);
      for(size_t i=0;i<wbf.n_cols;i++)
        wbf.col(i)%=wp;

      // The integrals are then
      arma::vec inner(arma::vectorise(arma::trans(wbf)*bf));

      return inner;
    }

    arma::mat yukawa_inner_integral(double rmin, double rmax, const arma::vec & x, const arma::vec & wx, const polynomial_basis::PolynomialBasis * poly, int L, double lambda) {
      // Midpoint is at
      double rmid(0.5*(rmax+rmin));
      // and half-length of interval is
      double rlen(0.5*(rmax-rmin));
      // r values are then
      arma::vec r(rmid*arma::ones<arma::vec>(x.n_elem)+rlen*x);

      // Compute the "inner" integrals as function of r.
      arma::mat inner(x.n_elem,std::pow(poly->get_nbf(),2));
      inner.row(0)=arma::trans(yukawa_inner_integral_wrk(rmin, r(0), rmin, rmax, x, wx, poly, L, lambda));
      // Every subinterval uses a fresh nquad points!
      for(size_t ip=1;ip<x.n_elem;ip++)
        inner.row(ip)=inner.row(ip-1)+arma::trans(yukawa_inner_integral_wrk(r(ip-1), r(ip), rmin, rmax, x, wx, poly, L, lambda));

      // Put in the k_L(r) factors now that the integrals have been computed
      arma::vec rpopl(utils::bessel_kl(r*lambda,L));
      for(size_t ip=0;ip<x.n_elem;ip++)
        inner.row(ip)*=rpopl(ip);

      return inner;
    }

    arma::mat yukawa_integral(double rmin, double rmax, const arma::vec & x, const arma::vec & wx, const polynomial_basis::PolynomialBasis * poly, int L, double lambda) {
#ifndef ARMA_NO_DEBUG
      if(x.n_elem != wx.n_elem) {
        std::ostringstream oss;
        oss << "x and wx not compatible: " << x.n_elem << " vs " << wx.n_elem << "!\n";
        throw std::logic_error(oss.str());
      }
#endif
      // and half-length of interval is
      double rlen(0.5*(rmax-rmin));

      // Compute the inner integrals
      arma::mat inner(yukawa_inner_integral(rmin, rmax, x, wx, poly, L, lambda));

      // Evaluate basis functions at quadrature points
      arma::mat bf(poly->eval(x));

      // Product functions
      arma::mat bfprod(bf.n_rows,bf.n_cols*bf.n_cols);
      for(size_t fi=0;fi<bf.n_cols;fi++)
        for(size_t fj=0;fj<bf.n_cols;fj++)
          bfprod.col(fi*bf.n_cols+fj)=bf.col(fi)%bf.col(fj);
      // Put in the weights for the outer integral
      arma::vec wp(wx*rlen);
      for(size_t i=0;i<bfprod.n_cols;i++)
        bfprod.col(i)%=wp;

      // Integrals are then
      arma::mat ints(arma::trans(bfprod)*inner);
      // but we are still missing the second term which can be
      // obtained as simply as
      ints+=arma::trans(ints);

      return ints;
    }

    arma::mat erfc_integral(double rmini, double rmaxi, const arma::mat & bfi, const arma::vec & xi, const arma::vec & wi, double rmink, double rmaxk, const arma::mat & bfk, const arma::vec & xk, const arma::vec & wk, int L, double mu) {
#ifndef ARMA_NO_DEBUG
      if(xi.n_elem != wi.n_elem) {
        std::ostringstream oss;
        oss << "xi and wi not compatible: " << xi.n_elem << " vs " << wi.n_elem << "!\n";
        throw std::logic_error(oss.str());
      }
      if(xk.n_elem != wk.n_elem) {
        std::ostringstream oss;
        oss << "xk and wk not compatible: " << xk.n_elem << " vs " << wk.n_elem << "!\n";
        throw std::logic_error(oss.str());
      }
#endif
      // and half-lengths of the intervals are
      double rmidi(0.5*(rmaxi+rmini));
      double rmidk(0.5*(rmaxk+rmink));

      double rleni(0.5*(rmaxi-rmini));
      double rlenk(0.5*(rmaxk-rmink));

      // Radii
      arma::vec ri(rmidi*arma::ones<arma::vec>(xi.n_elem)+rleni*xi);
      arma::vec rk(rmidk*arma::ones<arma::vec>(xk.n_elem)+rlenk*xk);

      // Green's function
      arma::mat Fn(ri.n_elem,rk.n_elem);
      for(size_t i=0;i<ri.n_elem;i++)
        for(size_t k=0;k<rk.n_elem;k++)
          Fn(i,k) = atomic::erfc_expn::Phi(L,mu*ri(i),mu*rk(k));

      // Product functions
      arma::mat bfprodij(bfi.n_rows,bfi.n_cols*bfi.n_cols);
      for(size_t fi=0;fi<bfi.n_cols;fi++)
        for(size_t fj=0;fj<bfi.n_cols;fj++)
          bfprodij.col(fi*bfi.n_cols+fj)=bfi.col(fi)%bfi.col(fj);
      arma::mat bfprodkl(bfk.n_rows,bfk.n_cols*bfk.n_cols);
      for(size_t fi=0;fi<bfk.n_cols;fi++)
        for(size_t fj=0;fj<bfk.n_cols;fj++)
          bfprodkl.col(fi*bfk.n_cols+fj)=bfk.col(fi)%bfk.col(fj);
      // Put in the weights
      arma::vec wpi(wi*rleni);
      for(size_t i=0;i<bfprodij.n_cols;i++)
        bfprodij.col(i)%=wpi;
      arma::vec wpk(wk*rlenk);
      for(size_t i=0;i<bfprodkl.n_cols;i++)
        bfprodkl.col(i)%=wpk;

      // Integrals are then
      arma::mat ints(arma::trans(bfprodij)*Fn*bfprodkl);

      return ints;
    }

    arma::mat spherical_potential(double rmin, double rmax, const arma::vec & x, const arma::vec & wx, const polynomial_basis::PolynomialBasis * poly) {
      // Midpoint is at
      double rmid(0.5*(rmax+rmin));
      // and half-length of interval is
      double rlen(0.5*(rmax-rmin));
      // r values are then
      arma::vec r(rmid*arma::ones<arma::vec>(x.n_elem)+rlen*x);

      // Compute the "inner" integrals as function of r.
      arma::mat zero(std::pow(poly->get_nbf(),2),x.n_elem);
      zero.zeros();
      arma::mat minusone(std::pow(poly->get_nbf(),2),x.n_elem);
      minusone.zeros();
      // Every subinterval uses a fresh nquad points!
      for(size_t ip=0;ip<x.n_elem;ip++) {
        double low = ip ? r(ip-1) : rmin;
        double high = r(ip);
        zero.col(ip)=twoe_inner_integral_wrk(low, high, rmin, rmax, x, wx, poly, 0);
      }
      for(size_t ip=0;ip<x.n_elem;ip++) {
        double low = r(ip);
        double high = (ip == x.n_elem-1) ? rmax : r(ip+1);
        minusone.col(ip)=twoe_inner_integral_wrk(low, high, rmin, rmax, x, wx, poly, -1);
      }

      // The potential itself
      arma::mat V(std::pow(poly->get_nbf(),2),x.n_elem);
      V.zeros();
      for(size_t ip=0;ip<x.n_elem;ip++) {
        // int_0^r Bi(r) Bj(r)
        for(size_t jp=0;jp<=ip;jp++)
          V.col(ip)+=zero.col(jp);
        // divided by r
        V.col(ip) /= r(ip);

        // plus the integral to infinity
        for(size_t jp=ip;jp<x.n_elem;jp++)
          V.col(ip)+=minusone.col(jp);
      }

      // Should be returned as transpose
      return arma::trans(V);
    }
  }
}
