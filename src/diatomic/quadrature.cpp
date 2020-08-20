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
#include "chebyshev.h"
#include "../general/polynomial.h"

namespace helfem {
  namespace diatomic {
    namespace quadrature {
      arma::mat radial_integral(double mumin, double mumax, int m, int n, const arma::vec & x, const arma::vec & wx, const arma::mat & bf) {
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
        double mumid(0.5*(mumax+mumin));
        // and half-length of interval is
        double mulen(0.5*(mumax-mumin));
        // mu values are then
        arma::vec mu(mumid*arma::ones<arma::vec>(x.n_elem)+mulen*x);

        // Calculate total weight per point
        arma::vec wp(wx*mulen);
        if(m!=0)
          wp%=arma::pow(arma::sinh(mu),m);
        if(n!=0)
          wp%=arma::pow(arma::cosh(mu),n);

        // Put in weight
        arma::mat wbf(bf);
        for(size_t i=0;i<bf.n_cols;i++)
          wbf.col(i)%=wp;

        // Matrix elements are then
        return arma::trans(wbf)*bf;
      }

      arma::mat Plm_radial_integral(double mumin, double mumax, int k, const arma::vec & x, const arma::vec & wx, const arma::mat & bf, int L, int M, const legendretable::LegendreTable & tab) {
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
        double mumid(0.5*(mumax+mumin));
        // and half-length of interval is
        double mulen(0.5*(mumax-mumin));
        // mu values are then
        arma::vec mu(mumid*arma::ones<arma::vec>(x.n_elem)+mulen*x);
        arma::vec chmu(arma::cosh(mu));

        // Calculate total weight per point
        arma::vec wp(wx*mulen);
        wp%=arma::sinh(mu);
        if(k!=0)
          wp%=arma::pow(chmu,k);
        wp%=tab.get_Plm(L,M,chmu);

        // Put in weight
        arma::mat wbf(bf);
        for(size_t i=0;i<bf.n_cols;i++)
          wbf.col(i)%=wp;

        // Matrix elements are then
        return arma::trans(wbf)*bf;
      }

      arma::mat Qlm_radial_integral(double mumin, double mumax, int l, const arma::vec & x, const arma::vec & wx, const arma::mat & bf, int L, int M, const legendretable::LegendreTable & tab) {
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
        double mumid(0.5*(mumax+mumin));
        // and half-length of interval is
        double mulen(0.5*(mumax-mumin));
        // mu values are then
        arma::vec mu(mumid*arma::ones<arma::vec>(x.n_elem)+mulen*x);
        arma::vec chmu(arma::cosh(mu));

        // Calculate total weight per point
        arma::vec wp(wx*mulen);
        wp%=arma::sinh(mu);
        if(l!=0)
          // cosh term
          wp%=arma::pow(chmu,l);
        // Legendre polynomial
        wp%=tab.get_Qlm(L,M,chmu);

        // Put in weight
        arma::mat wbf(bf);
        for(size_t i=0;i<wbf.n_cols;i++)
          wbf.col(i)%=wp;

        // The integrals are then
        return arma::trans(wbf)*bf;
      }

      static arma::vec twoe_inner_integral_wrk(double mumin, double mumax, double mumin0, double mumax0, int l, const arma::vec & x, const arma::vec & wx, const polynomial_basis::PolynomialBasis * poly, int L, int M, const legendretable::LegendreTable & tab) {
        // Midpoint is at
        double mumid(0.5*(mumax+mumin));
        // and half-length of interval is
        double mulen(0.5*(mumax-mumin));
        // mu values are then
        arma::vec mu(mumid*arma::ones<arma::vec>(x.n_elem)+mulen*x);
        arma::vec chmu(arma::cosh(mu));

        // Midpoint of original interval is at
        double mumid0(0.5*(mumax0+mumin0));
        // and half-length of original interval is
        double mulen0(0.5*(mumax0-mumin0));

        // Calculate total weight per point
        arma::vec wp(wx*mulen);
        wp%=arma::sinh(mu);
        if(l!=0)
          // cosh term
          wp%=arma::pow(chmu,l);
        // Legendre polynomial
        wp%=tab.get_Plm(L,M,chmu);

        // Calculate x values the polynomials should be evaluated at
        arma::vec xpoly((mu-mumid0*arma::ones<arma::vec>(x.n_elem))/mulen0);
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

      arma::mat twoe_inner_integral(double mumin, double mumax, int l, const arma::vec & x, const arma::vec & wx, const polynomial_basis::PolynomialBasis * poly, int L, int M, const legendretable::LegendreTable & tab) {
        // Midpoint is at
        double mumid(0.5*(mumax+mumin));
        // and half-length of interval is
        double mulen(0.5*(mumax-mumin));
        // r values are then
        arma::vec mu(mumid*arma::ones<arma::vec>(x.n_elem)+mulen*x);

        // Compute the "inner" integrals as function of r.
        arma::mat inner(x.n_elem,std::pow(poly->get_nbf(),2));
        inner.row(0)=arma::trans(twoe_inner_integral_wrk(mumin, mu(0), mumin, mumax, l, x, wx, poly, L, M, tab));
        // Every subinterval uses a fresh nquad points!
        for(size_t ip=1;ip<x.n_elem;ip++)
          inner.row(ip)=inner.row(ip-1)+arma::trans(twoe_inner_integral_wrk(mu(ip-1), mu(ip), mumin, mumax, l, x, wx, poly, L, M, tab));

        return inner;
      }

      static arma::mat twoe_integral_wrk(double mumin, double mumax, int k, int l, const arma::vec & x, const arma::vec & wx, const polynomial_basis::PolynomialBasis * poly, int L, int M, const legendretable::LegendreTable & tab) {
#ifndef ARMA_NO_DEBUG
        if(x.n_elem != wx.n_elem) {
          std::ostringstream oss;
          oss << "x and wx not compatible: " << x.n_elem << " vs " << wx.n_elem << "!\n";
          throw std::logic_error(oss.str());
        }
#endif
        // Midpoint is at
        double mumid(0.5*(mumax+mumin));
        // and half-length of interval is
        double mulen(0.5*(mumax-mumin));
        // mu values are then
        arma::vec mu(mumid*arma::ones<arma::vec>(x.n_elem)+mulen*x);
        arma::vec chmu(arma::cosh(mu));

        // Compute the inner integrals
        arma::mat inner(twoe_inner_integral(mumin, mumax, l, x, wx, poly, L, M, tab));

        // Evaluate basis functions at quadrature points
        arma::mat bf(poly->eval(x));

        // Product functions
        arma::mat bfprod(bf.n_rows,bf.n_cols*bf.n_cols);
        for(size_t fi=0;fi<bf.n_cols;fi++)
          for(size_t fj=0;fj<bf.n_cols;fj++)
            bfprod.col(fi*bf.n_cols+fj)=bf.col(fi)%bf.col(fj);
        // Put in the weights for the outer integral
        arma::vec wp(wx*mulen);
        wp%=arma::sinh(mu);
        if(k!=0)
          wp%=arma::pow(chmu,k);
        wp%=tab.get_Qlm(L,M,chmu);

        for(size_t i=0;i<bfprod.n_cols;i++)
          bfprod.col(i)%=wp;

        // Integrals are then
        arma::mat ints(arma::trans(bfprod)*inner);

        return ints;
      }

      arma::mat twoe_integral(double mumin, double mumax, int k, int l, const arma::vec & x, const arma::vec & wx, const polynomial_basis::PolynomialBasis * poly, int L, int M, const legendretable::LegendreTable & tab) {
        return twoe_integral_wrk(mumin,mumax,k,l,x,wx,poly,L,M,tab) + arma::trans(twoe_integral_wrk(mumin,mumax,l,k,x,wx,poly,L,M,tab));
      }
    }
  }
}
