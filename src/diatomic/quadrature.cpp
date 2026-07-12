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
#include "quadrature.h"
#include "chebyshev.h"
#include <ArmaEigen.h>
#include <sstream>

namespace helfem {
  namespace diatomic {
    namespace quadrature {
      // Legendre P/Q tables are still arma-typed (legendre subsystem);
      // evaluate against a bridged arma copy and return Eigen.
      static helfem::Vector Plm(const legendretable::LegendreTable & tab, int L, int M, const helfem::Vector & chmu) {
        return helfem::to_eigen(tab.get_Plm(L, M, helfem::to_arma(chmu)));
      }
      static helfem::Vector Qlm(const legendretable::LegendreTable & tab, int L, int M, const helfem::Vector & chmu) {
        return helfem::to_eigen(tab.get_Qlm(L, M, helfem::to_arma(chmu)));
      }

      static helfem::Vector twoe_inner_integral_wrk(double mumin, double mumax, double mumin0, double mumax0, int l, const helfem::Vector & x, const helfem::Vector & wx, const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly, int L, int M, const legendretable::LegendreTable & tab) {
        // Midpoint is at
        double mumid(0.5*(mumax+mumin));
        // and half-length of interval is
        double mulen(0.5*(mumax-mumin));
        // mu values are then
        helfem::Vector mu = mumid*helfem::Vector::Ones(x.size()) + mulen*x;
        helfem::Vector chmu = mu.array().cosh();

        // Midpoint of original interval is at
        double mumid0(0.5*(mumax0+mumin0));
        // and half-length of original interval is
        double mulen0(0.5*(mumax0-mumin0));

        // Calculate total weight per point
        helfem::Vector wp = mulen*wx;
        wp.array() *= mu.array().sinh();
        if(l!=0)
          // cosh term
          wp.array() *= chmu.array().pow(l);
        // Legendre polynomial
        wp.array() *= Plm(tab, L, M, chmu).array();

        // Calculate x values the polynomials should be evaluated at
        helfem::Vector xpoly = (mu - mumid0*helfem::Vector::Ones(x.size())) / mulen0;
        // Evaluate the polynomials at these points
        helfem::Matrix bf = poly->eval_dnf(xpoly, 0, mulen0);

        // Put in weight
        helfem::Matrix wbf = bf;
        for(Eigen::Index i=0;i<wbf.cols();i++)
          wbf.col(i).array() *= wp.array();

        // The integrals are then (column-major flatten of wbf^T bf)
        const helfem::Matrix prod = wbf.transpose()*bf;
        return Eigen::Map<const helfem::Vector>(prod.data(), prod.size());
      }

      helfem::Matrix twoe_inner_integral(double mumin, double mumax, int l, const helfem::Vector & x, const helfem::Vector & wx, const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly, int L, int M, const legendretable::LegendreTable & tab) {
        // Midpoint is at
        double mumid(0.5*(mumax+mumin));
        // and half-length of interval is
        double mulen(0.5*(mumax-mumin));
        // r values are then
        helfem::Vector mu = mumid*helfem::Vector::Ones(x.size()) + mulen*x;

        // Compute the "inner" integrals as function of r.
        helfem::Matrix inner(x.size(), (Eigen::Index) std::pow(poly->get_nbf(),2));
        inner.row(0)=twoe_inner_integral_wrk(mumin, mu(0), mumin, mumax, l, x, wx, poly, L, M, tab).transpose();
        // Every subinterval uses a fresh nquad points!
        for(Eigen::Index ip=1;ip<x.size();ip++)
          inner.row(ip)=inner.row(ip-1)+twoe_inner_integral_wrk(mu(ip-1), mu(ip), mumin, mumax, l, x, wx, poly, L, M, tab).transpose();

        return inner;
      }

      static helfem::Matrix twoe_integral_wrk(double mumin, double mumax, int k, int l, const helfem::Vector & x, const helfem::Vector & wx, const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly, int L, int M, const legendretable::LegendreTable & tab) {
        if(x.size() != wx.size()) {
          std::ostringstream oss;
          oss << "x and wx not compatible: " << x.size() << " vs " << wx.size() << "!\n";
          throw std::logic_error(oss.str());
        }
        // Midpoint is at
        double mumid(0.5*(mumax+mumin));
        // and half-length of interval is
        double mulen(0.5*(mumax-mumin));
        // mu values are then
        helfem::Vector mu = mumid*helfem::Vector::Ones(x.size()) + mulen*x;
        helfem::Vector chmu = mu.array().cosh();

        // Compute the inner integrals
        const helfem::Matrix inner = twoe_inner_integral(mumin, mumax, l, x, wx, poly, L, M, tab);

        // Evaluate basis functions at quadrature points
        const helfem::Matrix bf = poly->eval_dnf(x, 0, mulen);

        // Product functions
        helfem::Matrix bfprod(bf.rows(), bf.cols()*bf.cols());
        for(Eigen::Index fi=0;fi<bf.cols();fi++)
          for(Eigen::Index fj=0;fj<bf.cols();fj++)
            bfprod.col(fi*bf.cols()+fj)=bf.col(fi).cwiseProduct(bf.col(fj));
        // Put in the weights for the outer integral
        helfem::Vector wp = mulen*wx;
        wp.array() *= mu.array().sinh();
        if(k!=0)
          wp.array() *= chmu.array().pow(k);
        wp.array() *= Qlm(tab, L, M, chmu).array();

        for(Eigen::Index i=0;i<bfprod.cols();i++)
          bfprod.col(i).array() *= wp.array();

        // Integrals are then
        return bfprod.transpose()*inner;
      }

      helfem::Matrix twoe_integral(double mumin, double mumax, int k, int l, const helfem::Vector & x, const helfem::Vector & wx, const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly, int L, int M, const legendretable::LegendreTable & tab) {
        return twoe_integral_wrk(mumin,mumax,k,l,x,wx,poly,L,M,tab) + twoe_integral_wrk(mumin,mumax,l,k,x,wx,poly,L,M,tab).transpose();
      }
    }
  }
}
