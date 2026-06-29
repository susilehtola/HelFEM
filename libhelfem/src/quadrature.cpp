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

// Phase 5.7: quadrature.cpp migrated arma -> Eigen. Public API takes
// and returns Eigen Vec / Mat; internals are native Eigen ops.

#include "quadrature.h"
#include "erfc_expn.h"
#include "chebyshev.h"
#include "utils.h"
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace helfem {
  namespace quadrature {

    using Vec = helfem::Vector;
    using Mat = helfem::Matrix;

    Vec twoe_inner_integral_wrk(double rmin, double rmax, double rmin0, double rmax0,
                                const Vec & x, const Vec & wx,
                                const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly,
                                const std::function<double(double,double)> & fsmallbig,
                                const std::function<double(double)> & fbig) {
      (void) fbig;
      const double rmid  = 0.5 * (rmax + rmin);
      const double rlen  = 0.5 * (rmax - rmin);
      const Vec r = Vec::Constant(x.size(), rmid) + rlen * x;

      const double rmid0 = 0.5 * (rmax0 + rmin0);
      const double rlen0 = 0.5 * (rmax0 - rmin0);

      Vec fsmallbigr(r.size());
      for (Eigen::Index i = 0; i < r.size(); ++i)
        fsmallbigr(i) = fsmallbig(r(i), rmax);

      // wp = wx .* fsmallbigr * rlen (element-wise)
      const Vec wp = (wx.array() * fsmallbigr.array() * rlen).matrix();

      // xpoly = (r - rmid0) / rlen0
      const Vec xpoly = (r - Vec::Constant(r.size(), rmid0)) / rlen0;

      // Evaluate polynomials at xpoly
      Mat bf;
      poly->eval_dnf(xpoly, bf, 0, rlen0);

      // Weighted bf: wbf(:, k) = bf(:, k) .* wp
      Mat wbf = bf;
      for (Eigen::Index k = 0; k < wbf.cols(); ++k)
        wbf.col(k).array() *= wp.array();

      // inner = vec(wbf^T * bf), column-major flatten.
      const Mat M = wbf.transpose() * bf;
      Vec inner(M.size());
      std::memcpy(inner.data(), M.data(), sizeof(double) * (size_t) M.size());
      return inner;
    }

    Mat twoe_inner_integral(double rmin, double rmax,
                            const Vec & x, const Vec & wx,
                            const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly,
                            const std::function<double(double,double)> & fsmallbig,
                            const std::function<double(double)> & fbig) {
      const double rmid = 0.5 * (rmax + rmin);
      const double rlen = 0.5 * (rmax - rmin);
      const Vec r = Vec::Constant(x.size(), rmid) + rlen * x;

      const int nbf = poly->get_nbf();
      Mat inner(x.size(), nbf * nbf);
      // Row 0: integral from rmin to r(0)
      inner.row(0) = twoe_inner_integral_wrk(rmin, r(0), rmin, rmax, x, wx, poly, fsmallbig, fbig).transpose();
      for (Eigen::Index ip = 1; ip < x.size(); ++ip)
        inner.row(ip) = twoe_inner_integral_wrk(r(ip - 1), r(ip), rmin, rmax, x, wx, poly, fsmallbig, fbig).transpose();

      // Rescale: undo the per-segment R^(-L-1) factor we applied for
      // numerical stability.
      for (Eigen::Index ip = 1; ip < x.size(); ++ip)
        inner.row(ip) += inner.row(ip - 1) * (fbig(r(ip)) / fbig(r(ip - 1)));

      return inner;
    }

    Mat twoe_inner_integral(double rmin, double rmax,
                            const Vec & x, const Vec & wx,
                            const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly,
                            int L) {
      std::function<double(double,double)> fsmallbig = [L](double r, double R) { return std::pow(r/R, L) / R; };
      std::function<double(double)> fbig = [L](double r) { return std::pow(r, -L - 1); };
      return twoe_inner_integral(rmin, rmax, x, wx, poly, fsmallbig, fbig);
    }

    namespace {
      // Helper: bf-product columns bfprod(:, fi*Nf+fj) = bf(:, fi) .* bf(:, fj).
      inline Mat make_bfprod(const Mat & bf) {
        const Eigen::Index N = bf.cols();
        Mat bfprod(bf.rows(), N * N);
        for (Eigen::Index fi = 0; fi < N; ++fi)
          for (Eigen::Index fj = 0; fj < N; ++fj)
            bfprod.col(fi * N + fj).array() = bf.col(fi).array() * bf.col(fj).array();
        return bfprod;
      }
    } // namespace

    Mat twoe_integral(double rmin, double rmax,
                      const Vec & x, const Vec & wx,
                      const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly,
                      int L) {
      if (x.size() != wx.size()) {
        std::ostringstream oss;
        oss << "x and wx not compatible: " << x.size() << " vs " << wx.size() << "!\n";
        throw std::logic_error(oss.str());
      }
      const double rlen = 0.5 * (rmax - rmin);

      const Mat inner = twoe_inner_integral(rmin, rmax, x, wx, poly, L);

      Mat bf;
      poly->eval_dnf(x, bf, 0, rlen);

      Mat bfprod = make_bfprod(bf);
      const Vec wp = wx * rlen;
      for (Eigen::Index i = 0; i < bfprod.cols(); ++i)
        bfprod.col(i).array() *= wp.array();

      Mat ints = bfprod.transpose() * inner;
      ints += ints.transpose().eval();
      return ints;
    }

    Mat yukawa_inner_integral(double rmin, double rmax,
                              const Vec & x, const Vec & wx,
                              const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly,
                              int L, double lambda) {
      std::function<double(double,double)> fsmallbig = [L, lambda](double r, double R) {
        return utils::bessel_il(r * lambda, L) * utils::bessel_kl(R * lambda, L);
      };
      std::function<double(double)> fbig = [L, lambda](double r) { return utils::bessel_kl(r * lambda, L); };
      return twoe_inner_integral(rmin, rmax, x, wx, poly, fsmallbig, fbig);
    }

    Mat yukawa_integral(double rmin, double rmax,
                        const Vec & x, const Vec & wx,
                        const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly,
                        int L, double lambda) {
      if (x.size() != wx.size()) {
        std::ostringstream oss;
        oss << "x and wx not compatible: " << x.size() << " vs " << wx.size() << "!\n";
        throw std::logic_error(oss.str());
      }
      const double rlen = 0.5 * (rmax - rmin);

      const Mat inner = yukawa_inner_integral(rmin, rmax, x, wx, poly, L, lambda);

      Mat bf;
      poly->eval_dnf(x, bf, 0, rlen);

      Mat bfprod = make_bfprod(bf);
      const Vec wp = wx * rlen;
      for (Eigen::Index i = 0; i < bfprod.cols(); ++i)
        bfprod.col(i).array() *= wp.array();

      Mat ints = bfprod.transpose() * inner;
      ints += ints.transpose().eval();
      return ints;
    }

    Mat erfc_integral(double rmini, double rmaxi,
                      const Mat & bfi, const Vec & xi, const Vec & wi,
                      double rmink, double rmaxk,
                      const Mat & bfk, const Vec & xk, const Vec & wk,
                      int L, double mu) {
      if (xi.size() != wi.size()) {
        std::ostringstream oss;
        oss << "xi and wi not compatible: " << xi.size() << " vs " << wi.size() << "!\n";
        throw std::logic_error(oss.str());
      }
      if (xk.size() != wk.size()) {
        std::ostringstream oss;
        oss << "xk and wk not compatible: " << xk.size() << " vs " << wk.size() << "!\n";
        throw std::logic_error(oss.str());
      }

      const double rmidi = 0.5 * (rmaxi + rmini);
      const double rmidk = 0.5 * (rmaxk + rmink);
      const double rleni = 0.5 * (rmaxi - rmini);
      const double rlenk = 0.5 * (rmaxk - rmink);

      const Vec ri = Vec::Constant(xi.size(), rmidi) + rleni * xi;
      const Vec rk = Vec::Constant(xk.size(), rmidk) + rlenk * xk;

      // Green's function matrix Fn(i, k) = Phi(L, mu*ri(i), mu*rk(k))
      Mat Fn(ri.size(), rk.size());
      for (Eigen::Index i = 0; i < ri.size(); ++i)
        for (Eigen::Index k = 0; k < rk.size(); ++k)
          Fn(i, k) = atomic::erfc_expn::Phi(L, mu * ri(i), mu * rk(k));

      Mat bfprodij = make_bfprod(bfi);
      Mat bfprodkl = make_bfprod(bfk);

      const Vec wpi = wi * rleni;
      for (Eigen::Index i = 0; i < bfprodij.cols(); ++i)
        bfprodij.col(i).array() *= wpi.array();
      const Vec wpk = wk * rlenk;
      for (Eigen::Index i = 0; i < bfprodkl.cols(); ++i)
        bfprodkl.col(i).array() *= wpk.array();

      return bfprodij.transpose() * Fn * bfprodkl;
    }

    Mat spherical_potential(double rmin, double rmax,
                            const Vec & x, const Vec & wx,
                            const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly) {
      const double rmid = 0.5 * (rmax + rmin);
      const double rlen = 0.5 * (rmax - rmin);
      const Vec r = Vec::Constant(x.size(), rmid) + rlen * x;

      const int nbf = poly->get_nbf();
      Mat zero     = Mat::Zero(nbf * nbf, x.size());
      Mat minusone = Mat::Zero(nbf * nbf, x.size());

      std::function<double(double)> fsmall = [](double) { return 1.0; };
      std::function<double(double)> fbig   = [](double r) { return 1.0 / r; };
      std::function<double(double,double)> fsmallbig = [](double, double R) { return 1.0 / R; };
      std::function<double(double,double)> fbigsmall = [](double r, double) { return 1.0 / r; };

      for (Eigen::Index ip = 0; ip < x.size(); ++ip) {
        const double low  = ip ? r(ip - 1) : rmin;
        const double high = r(ip);
        zero.col(ip) = twoe_inner_integral_wrk(low, high, rmin, rmax, x, wx, poly, fsmallbig, fbig);
      }
      for (Eigen::Index ip = 0; ip < x.size(); ++ip) {
        const double low  = r(ip);
        const double high = (ip == x.size() - 1) ? rmax : r(ip + 1);
        minusone.col(ip) = twoe_inner_integral_wrk(low, high, rmin, rmax, x, wx, poly, fbigsmall, fsmall);
      }

      Mat V = Mat::Zero(nbf * nbf, x.size());
      for (Eigen::Index ip = 0; ip < x.size(); ++ip) {
        // \int_0^r Bi(r) Bj(r) dr
        for (Eigen::Index jp = 0; jp <= ip; ++jp)
          V.col(ip) += zero.col(jp) * r(jp);
        V.col(ip) /= r(ip);

        // + \int_r^infty r^{-1} Bi(r) Bj(r) dr
        for (Eigen::Index jp = ip; jp < x.size(); ++jp)
          V.col(ip) += minusone.col(jp);
      }

      return V.transpose();
    }

  } // namespace quadrature
} // namespace helfem
