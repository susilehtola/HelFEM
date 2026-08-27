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
#include "lcao_projection.h"

#include <PolynomialBasis.h>
#include <cmath>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace helfem {
  namespace lcao {

    double gto_rmax(double alpha) { return std::sqrt(50.0 / alpha); }

    double sto_rmax(double zeta) { return 60.0 / zeta; }

    AOBasis make_ao_basis(double rmax, int primbas, int nnodes, int nelem) {
      if (primbas != 4) {
        std::ostringstream oss;
        oss << "make_ao_basis got primbas " << primbas << ", but only 4 (LIP) "
            << "is supported: the AO expansion coefficients are the function "
            << "values at the interpolation nodes, which is a valid "
            << "expansion only for a basis that is cardinal at those nodes. "
            << "A Hermite basis would additionally need the derivative of "
            << "the AO, and a Legendre basis is modal.\n";
        throw std::logic_error(oss.str());
      }
      if (rmax <= 0.0)
        throw std::logic_error("make_ao_basis needs a positive box radius.\n");

      const std::shared_ptr<const polynomial_basis::PolynomialBasis> poly(
          polynomial_basis::make_basis(primbas, nnodes));
      const helfem::Vector bval = helfem::Vector::LinSpaced(nelem + 1, 0.0, rmax);
      // Dirichlet at both ends: u(0) = 0, and rmax is chosen so the AO has
      // decayed to negligible size there.
      polynomial_basis::FiniteElementBasis fem(poly, bval, true, false, true,
                                               false);

      AOBasis ao;
      ao.rad = helfem::atomic::basis::FEMRadialBasis(fem, 2 * nnodes);
      const helfem::Vector x0 = poly->nodes();
      ao.rnode = helfem::Vector::Zero(ao.rad.Nbf());
      for (size_t iel = 0; iel < ao.rad.Nel(); iel++) {
        size_t ifirst, ilast;
        ao.rad.idx(iel, ifirst, ilast);
        const helfem::IVec en = ao.rad.fem().basis(iel)->enabled();
        for (Eigen::Index j = 0; j < en.size(); j++)
          ao.rnode(ifirst + j) = ao.rad.r(x0(en(j)), iel);
      }
      ao.Sao = ao.rad.overlap();
      return ao;
    }

    helfem::Vector ao_coefficients(const AOBasis &ao, int l, double ex,
                                   const RadialAO &eval_ao) {
      helfem::Vector c(ao.rnode.size());
      for (Eigen::Index i = 0; i < c.size(); i++)
        c(i) = ao.rnode(i) * eval_ao(ao.rnode(i), l, ex);
      const double nrm2 = c.dot(ao.Sao * c);
      if (!(nrm2 > 0.0)) {
        std::ostringstream oss;
        oss << "Trial AO with l=" << l << " and exponent " << ex
            << " has vanishing norm on its own basis; the box radius is "
            << "probably wrong for this exponent.\n";
        throw std::logic_error(oss.str());
      }
      c /= std::sqrt(nrm2);
      return c;
    }

    helfem::Vector project_ao(const AOBasis &ao, const helfem::Matrix &Sx,
                              int l, double ex, const RadialAO &eval_ao) {
      return Sx * ao_coefficients(ao, l, ex, eval_ao);
    }

    double completeness(const helfem::Matrix &Sinvh, const helfem::Vector &b) {
      return (Sinvh.transpose() * b).norm();
    }

    double importance(const helfem::Matrix &Cocc, const helfem::Vector &b) {
      return (Cocc.transpose() * b).norm();
    }

    void ao_profiles(const helfem::atomic::basis::FEMRadialBasis &solrad,
                     const helfem::Matrix &Sinvh, const helfem::Cube &C,
                     const Eigen::VectorXi &occs, int lmax,
                     const helfem::Vector &expn, const RadialAO &eval_ao,
                     const std::function<double(double ex)> &ao_rmax,
                     helfem::Matrix &completeness_out,
                     helfem::Matrix &importance_out) {
      completeness_out = helfem::Matrix::Zero(expn.size(), lmax + 2);
      importance_out = helfem::Matrix::Zero(expn.size(), lmax + 2);
      completeness_out.col(0) = expn;
      importance_out.col(0) = expn;

      for (Eigen::Index ix = 0; ix < expn.size(); ix++) {
        const double ex = expn(ix);
        // One AO basis and one cross overlap per exponent, shared by every
        // l channel and both profiles: neither depends on l.
        const AOBasis ao = make_ao_basis(ao_rmax(ex));
        // <u_i | v_j> between the solution basis and the AO basis, via the
        // auto-converging element-pair-intersection quadrature.
        const helfem::Matrix Sx = solrad.overlap(ao.rad);
        for (int l = 0; l <= lmax; l++) {
          const helfem::Vector b = project_ao(ao, Sx, l, ex, eval_ao);
          completeness_out(ix, l + 1) = completeness(Sinvh, b);
          if (l < occs.size() && occs(l) > 0) {
            const int nocc =
                (int)std::ceil(occs(l) / (2.0 * (2.0 * l + 1.0)));
            importance_out(ix, l + 1) = importance(C[l].leftCols(nocc), b);
          }
        }
      }
    }

  } // namespace lcao
} // namespace helfem
