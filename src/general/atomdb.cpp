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

#include "atomdb.h"
#include "dftfuncs.h"
#include <lobatto.h>
#include <xc.h>
#include <utility>
#include <sstream>
#include <stdexcept>

namespace helfem {
  namespace atomdb {

    int max_Z() { return data::max_Z; }
    int lmax() { return data::lmax; }
    int Nbf() { return data::Nbf; }

    helfem::Vector element_boundaries() {
      return Eigen::Map<const helfem::Vector>(data::bval, data::nelem + 1);
    }

    /// Bounds check shared by the accessors.
    static void check(int Z, int l) {
      if (Z < 1 || Z > data::max_Z) {
        std::ostringstream oss;
        oss << "atomdb: no record for Z = " << Z << "; the database covers 1.."
            << data::max_Z << ".\n";
        throw std::logic_error(oss.str());
      }
      if (l < 0 || l > data::lmax) {
        std::ostringstream oss;
        oss << "atomdb: no record for l = " << l << "; the database covers 0.."
            << data::lmax << ".\n";
        throw std::logic_error(oss.str());
      }
    }

    int norb(int Z, int l) {
      check(Z, l);
      return data::norb[Z - 1][l];
    }

    helfem::Vector occupations(int Z, int l) {
      check(Z, l);
      return Eigen::Map<const helfem::Vector>(
          data::occupations + data::offset[Z - 1][l], data::norb[Z - 1][l]);
    }

    helfem::Matrix coefficients(int Z, int l) {
      check(Z, l);
      // The table is orbital-major, i.e. the coefficients of one orbital
      // are contiguous; the returned matrix is Nbf x norb, so map it as
      // column-major with Nbf rows.
      return Eigen::Map<const helfem::Matrix>(
          data::coefficients + static_cast<size_t>(data::offset[Z - 1][l]) * data::Nbf,
          data::Nbf, data::norb[Z - 1][l]);
    }

    /// The radial basis the coefficients were solved in. Every record
    /// shares it, so the parameters come from the table header and the
    /// basis is built on first use and then reused.
    static const atomic::basis::FEMRadialBasis & shared_basis() {
      static const atomic::basis::FEMRadialBasis basis = []() {
        std::shared_ptr<const polynomial_basis::PolynomialBasis> poly(
            polynomial_basis::make_basis(data::primbas, data::nnodes));
        // Matches sadatom::basis::TwoDBasis: the function vanishes at the
        // origin and at the practical infinity, its derivative is free.
        polynomial_basis::FiniteElementBasis fem(poly, element_boundaries(), true,
                                                 false, true, false);
        return atomic::basis::FEMRadialBasis(fem, data::nquad);
      }();
      return basis;
    }

    Atom::Atom(int Z) : Z_(Z), radial_(shared_basis()) {
      check(Z, 0);

      // Contract the orbitals into the total density matrix.
      helfem::Matrix P = helfem::Matrix::Zero(radial_.Nbf(), radial_.Nbf());
      for (int l = 0; l <= data::lmax; l++) {
        if (!norb(Z, l))
          continue;
        const helfem::Matrix C = coefficients(Z, l);
        P += C * occupations(Z, l).asDiagonal() * C.transpose();
      }

      // Per-element charge moments. radial_integral(0) integrates the
      // density itself, radial_integral(-1) the density over r, which are
      // precisely the two halves of the multipole expansion of 1/r_>.
      const size_t Nel = radial_.Nel();
      Psub_.resize(Nel);
      helfem::Vector qel(Nel), mel(Nel);
      for (size_t iel = 0; iel < Nel; iel++) {
        size_t ifirst, ilast;
        radial_.idx(iel, ifirst, ilast);
        Psub_[iel] = P.block(ifirst, ifirst, ilast - ifirst + 1, ilast - ifirst + 1);
        qel(iel) = partial_integral(iel, -1.0, 1.0, false);
        mel(iel) = partial_integral(iel, -1.0, 1.0, true);
      }
      // Accumulate inwards and outwards, so an evaluation only has to
      // integrate over the element the point falls in.
      Qbelow_ = helfem::Vector::Zero(Nel);
      Mabove_ = helfem::Vector::Zero(Nel);
      for (size_t iel = 1; iel < Nel; iel++)
        Qbelow_(iel) = Qbelow_(iel - 1) + qel(iel - 1);
      for (size_t iel = Nel - 1; iel-- > 0;)
        Mabove_(iel) = Mabove_(iel + 1) + mel(iel + 1);

      Ntot_ = qel.sum();
      Rmax_ = radial_.fem().element_end(Nel - 1);
    }

    int Atom::charge() const { return Z_; }

    double Atom::nelectrons() const { return Ntot_; }

    const atomic::basis::FEMRadialBasis & Atom::basis() const { return radial_; }

    helfem::Vector Atom::element_boundaries() const {
      return atomdb::element_boundaries();
    }

    /// Locate r and return the element index and the matching coordinate
    /// on the reference element [-1, 1].
    static size_t locate(const polynomial_basis::FiniteElementBasis & fem, double r,
                         double & xprim) {
      const size_t iel = fem.find_element(r);
      xprim = (r - fem.element_midpoint(iel)) / (0.5 * fem.element_length(iel));
      // find_element clamps to the end elements; keep the quadrature
      // limits legal for a point outside the grid.
      xprim = std::min(std::max(xprim, -1.0), 1.0);
      return iel;
    }

    helfem::Vector Atom::orbitals(int l, double r) const {
      const int n = (l >= 0 && l <= data::lmax) ? norb(Z_, l) : 0;
      helfem::Vector out = helfem::Vector::Zero(std::max(n, 0));
      if (n <= 0 || r <= 0.0 || r > Rmax_)
        return out;
      double xprim;
      const size_t iel = locate(radial_.fem(), r, xprim);
      helfem::Vector x(1);
      x(0) = xprim;
      // bf returns B(r)/r, so contracting it with the stored
      // coefficients gives the radial function R(r) itself, not r*R(r).
      const helfem::Matrix bf = radial_.bf(x, iel);
      size_t ifirst, ilast;
      radial_.idx(iel, ifirst, ilast);
      const helfem::Matrix C = coefficients(Z_, l);
      out = (bf * C.block(ifirst, 0, ilast - ifirst + 1, n)).transpose();
      return out;
    }

    double Atom::radial_density(double r) const {
      if (r <= 0.0 || r > Rmax_)
        return 0.0;
      double xprim;
      const size_t iel = locate(radial_.fem(), r, xprim);
      helfem::Vector x(1);
      x(0) = xprim;
      // bf returns B(r)/r, so the contraction is 4 pi rho directly.
      const helfem::Matrix bf = radial_.bf(x, iel);
      return r * r * (bf * Psub_[iel] * bf.transpose())(0, 0);
    }

    double Atom::density(double r) const {
      const double rr = radial_density(r);
      return (r > 0.0) ? rr / (4.0 * M_PI * r * r) : 0.0;
    }

    double Atom::density_gradient(double r) const {
      if (r <= 0.0 || r > Rmax_)
        return 0.0;
      double xprim;
      const size_t iel = locate(radial_.fem(), r, xprim);
      helfem::Vector x(1);
      x(0) = xprim;
      const helfem::Matrix bf = radial_.bf(x, iel);
      const helfem::Matrix df = radial_.df(x, iel);
      // d/dr sum_ij P_ij R_i R_j = 2 sum_ij P_ij R_i R_j', P symmetric.
      return 2.0 * (bf * Psub_[iel] * df.transpose())(0, 0) / (4.0 * M_PI);
    }

    double Atom::density_laplacian(double r) const {
      if (r <= 0.0 || r > Rmax_)
        return 0.0;
      double xprim;
      const size_t iel = locate(radial_.fem(), r, xprim);
      helfem::Vector x(1);
      x(0) = xprim;
      const helfem::Matrix bf = radial_.bf(x, iel);
      const helfem::Matrix df = radial_.df(x, iel);
      const helfem::Matrix lf = radial_.lf(x, iel);
      // rho'' + 2 rho' / r, with rho'' = 2 (R' P R' + R P R'').
      const double d2 = 2.0 * ((df * Psub_[iel] * df.transpose())(0, 0) +
                               (bf * Psub_[iel] * lf.transpose())(0, 0));
      const double d1 = 2.0 * (bf * Psub_[iel] * df.transpose())(0, 0);
      return (d2 + 2.0 * d1 / r) / (4.0 * M_PI);
    }

    /// The fixed rule used for the in-element integrals.
    ///
    /// The enclosed-charge integrand is the density contracted with
    /// itself, a polynomial of degree 2*(nnodes-1) in the reference
    /// coordinate, which an n-point Gauss-Lobatto rule integrates exactly
    /// once 2n-3 >= 2*(nnodes-1), i.e. n >= nnodes+1. The 1/r integrand
    /// is not a polynomial, but it is analytic on the element -- the pole
    /// of 1/r sits outside [-1, 1], closest for the element just outside
    /// the innermost one, where the Bernstein parameter is about 2.8 --
    /// so the surplus points take it geometrically past machine
    /// precision. Measured against quadrature of the pointwise density,
    /// nnodes+4 already sits on the roundoff floor.
    static const std::pair<helfem::Vector, helfem::Vector> & in_element_rule() {
      static const std::pair<helfem::Vector, helfem::Vector> rule = []() {
        helfem::Vector x, w;
        helfem::lobatto::lobatto_compute(data::nnodes + 12, x, w);
        return std::make_pair(x, w);
      }();
      return rule;
    }

    double Atom::partial_integral(size_t iel, double xa, double xb, bool over_r) const {
      const double half_sub = 0.5 * (xb - xa);
      if (half_sub <= 0.0)
        return 0.0;
      const polynomial_basis::FiniteElementBasis & fem = radial_.fem();

      const helfem::Vector & xq = in_element_rule().first;
      const helfem::Vector & wq = in_element_rule().second;
      // Map the rule onto [xa, xb] within the reference element.
      const helfem::Vector xi =
          (0.5 * (xa + xb)) * helfem::Vector::Ones(xq.size()) + half_sub * xq;
      const helfem::Vector r = fem.eval_coord(xi, iel);

      // Contract the density first. Integrating the 20x20 matrix and
      // tracing it against the density matrix afterwards computes 400
      // matrix elements to extract one number; the integrand we actually
      // want is the scalar sum_ij P_ij R_i(r) R_j(r).
      const helfem::Matrix bf = radial_.bf(xi, iel);
      const helfem::Vector quad =
          ((bf * Psub_[iel]).array() * bf.array()).rowwise().sum();

      // bf gives R = B/r, so the quadratic form is 4 pi rho. The
      // weight is r^2 for the charge and r for the 1/r moment -- written
      // as a multiplication rather than a division so that r = 0, which
      // the innermost element reaches, needs no special case.
      helfem::Vector integrand = (quad.array() * r.array()).matrix();
      if (!over_r)
        integrand.array() *= r.array();

      return 0.5 * fem.element_length(iel) * half_sub *
             (wq.array() * integrand.array()).sum();
    }

    double Atom::enclosed_charge(double r) const {
      if (r <= 0.0)
        return 0.0;
      if (r >= Rmax_)
        return nelectrons();
      double xprim;
      const size_t iel = locate(radial_.fem(), r, xprim);
      return Qbelow_(iel) + partial_integral(iel, -1.0, xprim, false);
    }

    double Atom::hartree_screening(double r) const {
      if (r <= 0.0)
        return 0.0;
      if (r >= Rmax_)
        return nelectrons();
      double xprim;
      const size_t iel = locate(radial_.fem(), r, xprim);
      // r * V_H(r) = Q(<r) + r * integral_{r'>r} rho(r') / r' dV.
      const double Qin = Qbelow_(iel) + partial_integral(iel, -1.0, xprim, false);
      const double Mout = Mabove_(iel) + partial_integral(iel, xprim, 1.0, true);
      return Qin + r * Mout;
    }

    /// Initialized libxc handles for one (x_func, c_func) pair.
    struct XCFunctionals {
      int x_func, c_func;
      /// The functionals, with a flag for whether each needs the gradient
      std::vector<std::pair<xc_func_type, bool>> funcs;
      /// Whether any of them does, i.e. whether the density derivatives
      /// have to be evaluated at all
      bool any_gga = false;

      XCFunctionals(int x, int c) : x_func(x), c_func(c) {
        funcs.reserve(2);
        for (int id : {x, c}) {
          if (id <= 0)
            continue;
          bool gga, mggat, mggal;
          ::is_gga_mgga(id, gga, mggat, mggal);
          if (mggat || mggal) {
            std::ostringstream oss;
            oss << "atomdb: functional " << id
                << " is a meta-GGA, which the spherically symmetric "
                   "screening does not implement.\n";
            throw std::logic_error(oss.str());
          }
          xc_func_type f;
          if (xc_func_init(&f, id, XC_POLARIZED) != 0)
            throw std::logic_error("atomdb: could not initialize the functional.\n");
          // libxc's own density threshold is left alone. It cuts the
          // screening off where the atomic density falls below roughly
          // 1e-15, which happens around r = 18 bohr and truncates a
          // contribution of at most ~3e-4 to the effective charge -- the
          // tabulated potential in sap.cpp, which predates the threshold,
          // carries that tail instead. Overriding the threshold is not
          // worth it: it is what keeps functionals out of the regime
          // where their own expressions misbehave.
          //
          // A standalone, libxc-free version of this evaluator can hard
          // code the Slater exchange potential -(3 rho / pi)^(1/3) and
          // drop the threshold altogether, since the cube root is well
          // behaved however small the density gets. That only works
          // because the screening is exchange-only; a general functional
          // needs libxc and therefore its threshold.
          funcs.emplace_back(f, gga);
          any_gga = any_gga || gga;
        }
      }
      ~XCFunctionals() {
        for (auto & f : funcs)
          xc_func_end(&f.first);
      }
      // The handles own libxc-internal allocations; copying would double
      // free them.
      XCFunctionals(const XCFunctionals &) = delete;
      XCFunctionals & operator=(const XCFunctionals &) = delete;
    };

    double Atom::xc_screening(double r, int x_func, int c_func) const {
      if (r <= 0.0 || (x_func <= 0 && c_func <= 0))
        return 0.0;

      // Spin-restricted density, split evenly between the channels: the
      // database is spin-restricted by construction.
      double rho[2];
      rho[0] = rho[1] = 0.5 * density(r);
      if (rho[0] <= 0.0)
        return 0.0;

      if (!xc_ || xc_->x_func != x_func || xc_->c_func != c_func)
        xc_ = std::make_shared<XCFunctionals>(x_func, c_func);

      // The density derivatives are only worth having if something asks
      // for them. The Laplacian in particular costs two orders of
      // magnitude more than the density itself, and an LDA never looks
      // at it -- evaluating it unconditionally made the LDA screening
      // 150x slower than it needed to be.
      double grad[2] = {0.0, 0.0}, lapl[2] = {0.0, 0.0};
      double sigma[3] = {0.0, 0.0, 0.0};
      if (xc_->any_gga) {
        grad[0] = grad[1] = 0.5 * density_gradient(r);
        lapl[0] = lapl[1] = 0.5 * density_laplacian(r);
        // Reduced gradients, in libxc's (aa, ab, bb) order.
        sigma[0] = grad[0] * grad[0];
        sigma[1] = grad[0] * grad[1];
        sigma[2] = grad[1] * grad[1];
      }

      double vxc[2] = {0.0, 0.0};
      double vsigma[3] = {0.0, 0.0, 0.0};
      double v2rhosigma[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      double v2sigma2[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      bool do_gga = false;

      for (const auto & fp : xc_->funcs) {
        const xc_func_type & f = fp.first;
        double v[2] = {0.0, 0.0};
        if (fp.second) {
          double vs[3] = {0.0, 0.0, 0.0}, v2r2[3] = {0.0, 0.0, 0.0};
          double v2rs[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
          double v2s2[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
          xc_gga_vxc(&f, 1, rho, sigma, v, vs);
          xc_gga_fxc(&f, 1, rho, sigma, v2r2, v2rs, v2s2);
          do_gga = true;
          for (int i = 0; i < 3; i++) vsigma[i] += vs[i];
          for (int i = 0; i < 6; i++) { v2rhosigma[i] += v2rs[i]; v2sigma2[i] += v2s2[i]; }
        } else {
          xc_lda_vxc(&f, 1, rho, v);
        }
        vxc[0] += v[0];
        vxc[1] += v[1];
      }

      if (do_gga) {
        // The GGA screening potential is v_rho - div(2 v_sigma grad rho),
        // and for a spherically symmetric density the divergence is
        // d/dr + 2/r. Expanding d/dr of the vsigma terms by the chain
        // rule brings in the second derivatives of the functional and the
        // Laplacian of the density. Mirrors the grid implementation in
        // sadatom::basis::TwoDBasis::xc_screening.
        double corr[2] = {0.0, 0.0};

        // g(t) (d^2 E / d n(t) d sigma(ss')) g(s')
        corr[0] += 2.0 * (grad[0] * v2rhosigma[0] + grad[1] * v2rhosigma[3]) * grad[0];
        corr[0] += (grad[0] * v2rhosigma[1] + grad[1] * v2rhosigma[4]) * grad[1];
        corr[1] += 2.0 * (grad[1] * v2rhosigma[5] + grad[0] * v2rhosigma[2]) * grad[1];
        corr[1] += (grad[0] * v2rhosigma[1] + grad[1] * v2rhosigma[4]) * grad[0];

        // (l(t) g(t') + g(t) l(t')) (d^2 E / d sigma(tt') d sigma(ss')) g(s')
        const double lg = lapl[0] * grad[1] + grad[0] * lapl[1];
        const double d2Edsaa = lapl[0] * grad[0] * v2sigma2[0] + lg * v2sigma2[1] +
                               lapl[1] * grad[1] * v2sigma2[2];
        const double d2Edsab = lapl[0] * grad[0] * v2sigma2[1] + lg * v2sigma2[3] +
                               lapl[1] * grad[1] * v2sigma2[4];
        const double d2Edsbb = lapl[0] * grad[0] * v2sigma2[2] + lg * v2sigma2[4] +
                               lapl[1] * grad[1] * v2sigma2[5];
        corr[0] += 4.0 * d2Edsaa * grad[0] + 2.0 * d2Edsab * grad[1];
        corr[1] += 4.0 * d2Edsbb * grad[1] + 2.0 * d2Edsab * grad[0];

        // dE/dsigma(ss') l(s')
        corr[0] += 2.0 * vsigma[0] * lapl[0] + vsigma[1] * lapl[1];
        corr[1] += vsigma[1] * lapl[0] + 2.0 * vsigma[2] * lapl[1];

        // The 2/r piece of the radial divergence
        corr[0] += 2.0 / r * (2.0 * vsigma[0] * grad[0] + vsigma[1] * grad[1]);
        corr[1] += 2.0 / r * (vsigma[1] * grad[0] + 2.0 * vsigma[2] * grad[1]);

        vxc[0] -= corr[0];
        vxc[1] -= corr[1];
      }

      // Average the two channels; they are equal here, but the average is
      // what the spin-unrestricted reference tabulates as well.
      return r * 0.5 * (vxc[0] + vxc[1]);
    }

    double Atom::effective_charge(double r, int x_func, int c_func) const {
      return Z_ - hartree_screening(r) - xc_screening(r, x_func, c_func);
    }
  }
}
