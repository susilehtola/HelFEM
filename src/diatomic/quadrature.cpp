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
#include <sstream>

namespace helfem {
  namespace diatomic {
    namespace quadrature {
      // Legendre P/Q tables are Eigen-typed.
      static helfem::Vector Plm(const legendretable::LegendreTable & tab, int L, int M, const helfem::Vector & chmu) {
        return tab.get_Plm(L, M, chmu);
      }
      static helfem::Vector Qlm(const legendretable::LegendreTable & tab, int L, int M, const helfem::Vector & chmu) {
        return tab.get_Qlm(L, M, chmu);
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

      TwoElectronElement twoe_element(double mumin, double mumax, const helfem::Vector & x, const helfem::Vector & wx, const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly) {
        if(x.size() != wx.size()) {
          std::ostringstream oss;
          oss << "x and wx not compatible: " << x.size() << " vs " << wx.size() << "!\n";
          throw std::logic_error(oss.str());
        }

        TwoElectronElement el;
        el.x  = x;
        el.wx = wx;

        // Element midpoint and half-length
        const double mumid(0.5*(mumax+mumin));
        el.mulen = 0.5*(mumax-mumin);

        // Outer quadrature points
        el.mu   = mumid*helfem::Vector::Ones(x.size()) + el.mulen*x;
        el.chmu = el.mu.array().cosh();
        el.shmu = el.mu.array().sinh();

        // Outer basis functions and the product table B_i B_j. Neither depends
        // on k, l, L or M.
        el.bf = poly->eval_dnf(x, 0, el.mulen);
        const Eigen::Index nbf = el.bf.cols();
        el.bfprod.resize(el.bf.rows(), nbf*nbf);
        for(Eigen::Index fi=0;fi<nbf;fi++)
          for(Eigen::Index fj=0;fj<nbf;fj++)
            el.bfprod.col(fi*nbf+fj) = el.bf.col(fi).cwiseProduct(el.bf.col(fj));

        // Inner (cumulative) integral: subinterval ip spans [mu(ip-1), mu(ip)],
        // with [mumin, mu(0)] for ip = 0. Each uses a fresh set of nquad points,
        // and the polynomials there depend only on where those points fall.
        el.sub.resize(x.size());
        for(Eigen::Index ip=0;ip<x.size();ip++) {
          const double submin = (ip==0) ? mumin : el.mu(ip-1);
          const double submax = el.mu(ip);

          const double submid(0.5*(submax+submin));
          const double sublen(0.5*(submax-submin));

          const helfem::Vector submu = submid*helfem::Vector::Ones(x.size()) + sublen*x;

          TwoElectronElement::Subinterval & si = el.sub[ip];
          si.mulen = sublen;
          si.chmu  = submu.array().cosh();
          si.shmu  = submu.array().sinh();
          // The polynomials are those of the PARENT element, evaluated at the
          // subinterval's points -- hence the element midpoint / half-length.
          const helfem::Vector xpoly = (submu - mumid*helfem::Vector::Ones(x.size())) / el.mulen;
          si.bf = poly->eval_dnf(xpoly, 0, el.mulen);
        }

        return el;
      }

      helfem::Matrix twoe_integral(const TwoElectronElement & el, int k, int l, int L, int M, const legendretable::LegendreTable & tab) {
        const Eigen::Index nq  = el.x.size();
        const Eigen::Index nbf = el.bf.cols();

        // Scratch, allocated ONCE per call and reused. Previously every one of
        // the nq subintervals allocated its own wp/wbf/prod, and each outer()
        // call copied the whole (nq x nbf^2) bfprod table just to scale it by
        // the weights -- hundreds of MB of allocation traffic through
        // compute_tei, which the profile saw as malloc/memmove/memset.
        helfem::Matrix inner(nq, nbf*nbf);
        helfem::Matrix wbf(nq, nbf);
        helfem::Matrix prod(nbf, nbf);
        helfem::Vector acc(nbf*nbf);
        helfem::Vector wp(nq);
        helfem::Matrix wbfprod(nq, nbf*nbf);
        helfem::Vector legscratch(nq);

        // Inner integrals as a function of the outer point, accumulated over
        // the subintervals. Depends on l (through cosh^l) and on (L, M)
        // (through P_{L|M|}), but on no polynomial evaluation.
        auto inner_integral = [&](int lval) {
          acc.setZero();
          for(Eigen::Index ip=0;ip<nq;ip++) {
            const TwoElectronElement::Subinterval & si = el.sub[ip];

            wp = si.mulen*el.wx;
            wp.array() *= si.shmu.array();
            if(lval!=0)
              wp.array() *= si.chmu.array().pow(lval);
            // Fill via the scalar accessor: the vector-returning overload
            // bridges Eigen -> arma, allocates, and copies back, three
            // allocations per subinterval.
            for(Eigen::Index i=0;i<nq;i++)
              legscratch(i) = tab.get_Plm(L, M, si.chmu(i));
            wp.array() *= legscratch.array();

            wbf = si.bf;
            for(Eigen::Index i=0;i<wbf.cols();i++)
              wbf.col(i).array() *= wp.array();

            prod.noalias() = wbf.transpose()*si.bf;
            acc += Eigen::Map<const helfem::Vector>(prod.data(), prod.size());
            inner.row(ip) = acc.transpose();
          }
        };

        // Outer integral, for the (k, l) ordering
        auto outer = [&](int kval, int lval, helfem::Matrix & out) {
          inner_integral(lval);

          wp = el.mulen*el.wx;
          wp.array() *= el.shmu.array();
          if(kval!=0)
            wp.array() *= el.chmu.array().pow(kval);
          for(Eigen::Index i=0;i<nq;i++)
            legscratch(i) = tab.get_Qlm(L, M, el.chmu(i));
          wp.array() *= legscratch.array();

          wbfprod = el.bfprod;
          for(Eigen::Index i=0;i<wbfprod.cols();i++)
            wbfprod.col(i).array() *= wp.array();

          out.noalias() = wbfprod.transpose()*inner;
        };

        helfem::Matrix t1(nbf*nbf, nbf*nbf), t2(nbf*nbf, nbf*nbf);
        outer(k, l, t1);
        outer(l, k, t2);
        return helfem::Matrix(t1 + t2.transpose());
      }

    }
  }
}
