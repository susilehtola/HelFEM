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
//
// Templated on the scalar type; explicitly instantiated for double and
// long double at the bottom of the namespace.

#include "quadrature.h"
#include "erfc_expn.h"
#include "utils.h"
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace helfem {
  namespace quadrature {

    template <typename T> using Vec = helfem::Vec<T>;
    template <typename T> using Mat = helfem::Mat<T>;

    template <typename T>
    Vec<T> twoe_inner_integral_wrk(T rmin, T rmax, T rmin0, T rmax0,
                                   const Vec<T> & x, const Vec<T> & wx,
                                   const std::shared_ptr<const PolyBasis<T>> & poly,
                                   const std::function<T(T,T)> & fsmallbig,
                                   const std::function<T(T)> & fbig) {
      (void) fbig;
      const T rmid  = T(0.5) * (rmax + rmin);
      const T rlen  = T(0.5) * (rmax - rmin);
      const Vec<T> r = Vec<T>::Constant(x.size(), rmid) + rlen * x;

      const T rmid0 = T(0.5) * (rmax0 + rmin0);
      const T rlen0 = T(0.5) * (rmax0 - rmin0);

      Vec<T> fsmallbigr(r.size());
      for (Eigen::Index i = 0; i < r.size(); ++i)
        fsmallbigr(i) = fsmallbig(r(i), rmax);

      // wp = wx .* fsmallbigr * rlen (element-wise)
      const Vec<T> wp = (wx.array() * fsmallbigr.array() * rlen).matrix();

      // xpoly = (r - rmid0) / rlen0
      const Vec<T> xpoly = (r - Vec<T>::Constant(r.size(), rmid0)) / rlen0;

      // Evaluate polynomials at xpoly
      Mat<T> bf;
      poly->eval_dnf(xpoly, bf, 0, rlen0);

      // Weighted bf: wbf(:, k) = bf(:, k) .* wp
      Mat<T> wbf = bf;
      for (Eigen::Index k = 0; k < wbf.cols(); ++k)
        wbf.col(k).array() *= wp.array();

      // inner = vec(wbf^T * bf), column-major flatten.
      const Mat<T> M = wbf.transpose() * bf;
      Vec<T> inner(M.size());
      std::memcpy(inner.data(), M.data(), sizeof(T) * (size_t) M.size());
      return inner;
    }

    template <typename T>
    Mat<T> twoe_inner_integral(T rmin, T rmax,
                               const Vec<T> & x, const Vec<T> & wx,
                               const std::shared_ptr<const PolyBasis<T>> & poly,
                               const std::function<T(T,T)> & fsmallbig,
                               const std::function<T(T)> & fbig) {
      const T rmid = T(0.5) * (rmax + rmin);
      const T rlen = T(0.5) * (rmax - rmin);
      const Vec<T> r = Vec<T>::Constant(x.size(), rmid) + rlen * x;

      const int nbf = poly->get_nbf();
      Mat<T> inner(x.size(), nbf * nbf);
      // Row 0: integral from rmin to r(0)
      inner.row(0) = twoe_inner_integral_wrk<T>(rmin, r(0), rmin, rmax, x, wx, poly, fsmallbig, fbig).transpose();
      for (Eigen::Index ip = 1; ip < x.size(); ++ip)
        inner.row(ip) = twoe_inner_integral_wrk<T>(r(ip - 1), r(ip), rmin, rmax, x, wx, poly, fsmallbig, fbig).transpose();

      // Rescale: undo the per-segment R^(-L-1) factor we applied for
      // numerical stability.
      for (Eigen::Index ip = 1; ip < x.size(); ++ip)
        inner.row(ip) += inner.row(ip - 1) * (fbig(r(ip)) / fbig(r(ip - 1)));

      return inner;
    }

    template <typename T>
    Mat<T> twoe_inner_integral(NonDeduced<T> rmin, NonDeduced<T> rmax,
                               const Vec<T> & x, const Vec<T> & wx,
                               const std::shared_ptr<const PolyBasis<NonDeduced<T>>> & poly,
                               int L) {
      std::function<T(T,T)> fsmallbig = [L](T r, T R) { return std::pow(r/R, L) / R; };
      std::function<T(T)> fbig = [L](T r) { return std::pow(r, -L - 1); };
      return twoe_inner_integral<T>(rmin, rmax, x, wx, poly, fsmallbig, fbig);
    }

    namespace {
      // Helper: bf-product columns bfprod(:, fi*Nf+fj) = bf(:, fi) .* bf(:, fj).
      template <typename T>
      inline Mat<T> make_bfprod(const Mat<T> & bf) {
        const Eigen::Index N = bf.cols();
        Mat<T> bfprod(bf.rows(), N * N);
        for (Eigen::Index fi = 0; fi < N; ++fi)
          for (Eigen::Index fj = 0; fj < N; ++fj)
            bfprod.col(fi * N + fj).array() = bf.col(fi).array() * bf.col(fj).array();
        return bfprod;
      }
    } // namespace

    template <typename T>
    Mat<T> twoe_integral(NonDeduced<T> rmin, NonDeduced<T> rmax,
                         const Vec<T> & x, const Vec<T> & wx,
                         const std::shared_ptr<const PolyBasis<NonDeduced<T>>> & poly,
                         int L) {
      if (x.size() != wx.size()) {
        std::ostringstream oss;
        oss << "x and wx not compatible: " << x.size() << " vs " << wx.size() << "!\n";
        throw std::logic_error(oss.str());
      }
      const T rlen = T(0.5) * (rmax - rmin);

      const Mat<T> inner = twoe_inner_integral<T>(rmin, rmax, x, wx, poly, L);

      Mat<T> bf;
      poly->eval_dnf(x, bf, 0, rlen);

      Mat<T> bfprod = make_bfprod<T>(bf);
      const Vec<T> wp = wx * rlen;
      for (Eigen::Index i = 0; i < bfprod.cols(); ++i)
        bfprod.col(i).array() *= wp.array();

      Mat<T> ints = bfprod.transpose() * inner;
      ints += ints.transpose().eval();
      return ints;
    }

    template <typename T>
    Mat<T> yukawa_inner_integral(NonDeduced<T> rmin, NonDeduced<T> rmax,
                                 const Vec<T> & x, const Vec<T> & wx,
                                 const std::shared_ptr<const PolyBasis<NonDeduced<T>>> & poly,
                                 int L, NonDeduced<T> lambda) {
      std::function<T(T,T)> fsmallbig = [L, lambda](T r, T R) {
        return utils::bessel_il<T>(r * lambda, L) * utils::bessel_kl<T>(R * lambda, L);
      };
      std::function<T(T)> fbig = [L, lambda](T r) { return utils::bessel_kl<T>(r * lambda, L); };
      return twoe_inner_integral<T>(rmin, rmax, x, wx, poly, fsmallbig, fbig);
    }

    template <typename T>
    Mat<T> yukawa_integral(NonDeduced<T> rmin, NonDeduced<T> rmax,
                           const Vec<T> & x, const Vec<T> & wx,
                           const std::shared_ptr<const PolyBasis<NonDeduced<T>>> & poly,
                           int L, NonDeduced<T> lambda) {
      if (x.size() != wx.size()) {
        std::ostringstream oss;
        oss << "x and wx not compatible: " << x.size() << " vs " << wx.size() << "!\n";
        throw std::logic_error(oss.str());
      }
      const T rlen = T(0.5) * (rmax - rmin);

      const Mat<T> inner = yukawa_inner_integral<T>(rmin, rmax, x, wx, poly, L, lambda);

      Mat<T> bf;
      poly->eval_dnf(x, bf, 0, rlen);

      Mat<T> bfprod = make_bfprod<T>(bf);
      const Vec<T> wp = wx * rlen;
      for (Eigen::Index i = 0; i < bfprod.cols(); ++i)
        bfprod.col(i).array() *= wp.array();

      Mat<T> ints = bfprod.transpose() * inner;
      ints += ints.transpose().eval();
      return ints;
    }

    template <typename T>
    Mat<T> erfc_integral(NonDeduced<T> rmini, NonDeduced<T> rmaxi,
                         const Mat<T> & bfi, const Vec<T> & xi, const Vec<T> & wi,
                         NonDeduced<T> rmink, NonDeduced<T> rmaxk,
                         const Mat<T> & bfk, const Vec<T> & xk, const Vec<T> & wk,
                         int L, NonDeduced<T> mu) {
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

      const T rmidi = T(0.5) * (rmaxi + rmini);
      const T rmidk = T(0.5) * (rmaxk + rmink);
      const T rleni = T(0.5) * (rmaxi - rmini);
      const T rlenk = T(0.5) * (rmaxk - rmink);

      const Vec<T> ri = Vec<T>::Constant(xi.size(), rmidi) + rleni * xi;
      const Vec<T> rk = Vec<T>::Constant(xk.size(), rmidk) + rlenk * xk;

      // Green's function matrix Fn(i, k) = Phi(L, mu*ri(i), mu*rk(k)).
      //
      // erfc_expn::Phi is the one piece of the radial layer that is still
      // double-only: it is a long series of double-precision special-function
      // evaluations (erfc, Boys-like damping functions) with hard-coded
      // double tolerances. This is the single double boundary in the
      // templated radial code -- the range-separated (erfc) two-electron
      // integrals are therefore computed at double accuracy even at
      // T = long double. Everything else in the chain is at T.
      Mat<T> Fn(ri.size(), rk.size());
      for (Eigen::Index i = 0; i < ri.size(); ++i)
        for (Eigen::Index k = 0; k < rk.size(); ++k)
          Fn(i, k) = static_cast<T>(atomic::erfc_expn::Phi(
              L, static_cast<double>(mu * ri(i)), static_cast<double>(mu * rk(k))));

      Mat<T> bfprodij = make_bfprod<T>(bfi);
      Mat<T> bfprodkl = make_bfprod<T>(bfk);

      const Vec<T> wpi = wi * rleni;
      for (Eigen::Index i = 0; i < bfprodij.cols(); ++i)
        bfprodij.col(i).array() *= wpi.array();
      const Vec<T> wpk = wk * rlenk;
      for (Eigen::Index i = 0; i < bfprodkl.cols(); ++i)
        bfprodkl.col(i).array() *= wpk.array();

      return bfprodij.transpose() * Fn * bfprodkl;
    }

    template <typename T>
    Mat<T> spherical_potential(NonDeduced<T> rmin, NonDeduced<T> rmax,
                               const Vec<T> & x, const Vec<T> & wx,
                               const std::shared_ptr<const PolyBasis<NonDeduced<T>>> & poly) {
      const T rmid = T(0.5) * (rmax + rmin);
      const T rlen = T(0.5) * (rmax - rmin);
      const Vec<T> r = Vec<T>::Constant(x.size(), rmid) + rlen * x;

      const int nbf = poly->get_nbf();
      Mat<T> zero     = Mat<T>::Zero(nbf * nbf, x.size());
      Mat<T> minusone = Mat<T>::Zero(nbf * nbf, x.size());

      std::function<T(T)> fsmall = [](T) { return T(1); };
      std::function<T(T)> fbig   = [](T r_) { return T(1) / r_; };
      std::function<T(T,T)> fsmallbig = [](T, T R) { return T(1) / R; };
      std::function<T(T,T)> fbigsmall = [](T r_, T) { return T(1) / r_; };

      for (Eigen::Index ip = 0; ip < x.size(); ++ip) {
        const T low  = ip ? r(ip - 1) : rmin;
        const T high = r(ip);
        zero.col(ip) = twoe_inner_integral_wrk<T>(low, high, rmin, rmax, x, wx, poly, fsmallbig, fbig);
      }
      for (Eigen::Index ip = 0; ip < x.size(); ++ip) {
        const T low  = r(ip);
        const T high = (ip == x.size() - 1) ? rmax : r(ip + 1);
        minusone.col(ip) = twoe_inner_integral_wrk<T>(low, high, rmin, rmax, x, wx, poly, fbigsmall, fsmall);
      }

      Mat<T> V = Mat<T>::Zero(nbf * nbf, x.size());
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

    // Explicit instantiations.
#define HELFEM_INSTANTIATE_QUADRATURE(T)                                       \
  template Mat<T> twoe_inner_integral<T>(T, T, const Vec<T> &, const Vec<T> &, \
                                         const std::shared_ptr<const PolyBasis<T>> &, int); \
  template Mat<T> twoe_integral<T>(T, T, const Vec<T> &, const Vec<T> &,       \
                                   const std::shared_ptr<const PolyBasis<T>> &, int); \
  template Mat<T> yukawa_inner_integral<T>(T, T, const Vec<T> &, const Vec<T> &, \
                                           const std::shared_ptr<const PolyBasis<T>> &, int, T); \
  template Mat<T> yukawa_integral<T>(T, T, const Vec<T> &, const Vec<T> &,     \
                                     const std::shared_ptr<const PolyBasis<T>> &, int, T); \
  template Mat<T> erfc_integral<T>(T, T, const Mat<T> &, const Vec<T> &, const Vec<T> &, \
                                   T, T, const Mat<T> &, const Vec<T> &, const Vec<T> &, \
                                   int, T);                                    \
  template Mat<T> spherical_potential<T>(T, T, const Vec<T> &, const Vec<T> &, \
                                         const std::shared_ptr<const PolyBasis<T>> &);

    HELFEM_INSTANTIATE_QUADRATURE(double)
    HELFEM_INSTANTIATE_QUADRATURE(long double)

#undef HELFEM_INSTANTIATE_QUADRATURE

  } // namespace quadrature
} // namespace helfem
