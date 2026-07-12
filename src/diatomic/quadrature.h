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
#ifndef INTEGRALS_H
#define INTEGRALS_H

#include <Matrix.h>
#include <memory>
#include "../general/legendretable.h"
#include "PolynomialBasis.h"
#include <vector>

namespace helfem {
  namespace diatomic {
    namespace quadrature {
      /**
       * Computes the inner in-element two-electron integral:
       * \f$ \phi^{l,LM}(\mu) = \int_{0}^{\mu}d\mu'\cosh^{l}\mu'\sinh\mu'B_{\gamma}(\mu')B_{\delta}(\mu')P_{L,|M|}(\cosh\mu') \f$
       */
      helfem::Matrix twoe_inner_integral(double mumin, double mumax, int l, const helfem::Vector & x, const helfem::Vector & wx, const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly, int L, int M, const legendretable::LegendreTable & tab);

      /**
       * Computes a primitive two-electron in-element integral.
       * Cross-element integrals reduce to products of radial integrals.
       * Note that the routine needs the polynomial representation.
       */
      helfem::Matrix twoe_integral(double rmin, double rmax, int k, int l, const helfem::Vector & x, const helfem::Vector & wx, const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly, int L, int M, const legendretable::LegendreTable & tab);

      /**
       * Everything in the two-electron in-element integrals that depends only
       * on the ELEMENT and the quadrature rule -- not on (k, l, L, M).
       *
       * The polynomials are the expensive part (profiling put the FEM
       * evaluation at ~35% of a run), and they were being re-evaluated inside
       * the (L, M) x (alpha, beta) loops: compute_tei asks for 4 (alpha,beta)
       * combinations per (element, L, M), each of which runs twoe_integral_wrk
       * twice, and each of those re-evaluated the outer basis, the outer
       * product table, and one inner basis per subinterval. None of that
       * depends on k, l, L or M.
       *
       * Build this once per element and hand it to twoe_integral() below.
       */
      struct TwoElectronElement {
        /// Quadrature nodes and weights on [-1, 1]
        helfem::Vector x, wx;
        /// Half-length of the element in mu
        double mulen;
        /// Outer quadrature: mu, cosh(mu), sinh(mu) at the element's points
        helfem::Vector mu, chmu, shmu;
        /// Outer basis functions, (nquad x nbf)
        helfem::Matrix bf;
        /// Outer product table B_i(mu) B_j(mu), (nquad x nbf^2)
        helfem::Matrix bfprod;

        /// Per-subinterval data for the cumulative inner integral. Subinterval
        /// ip runs from mu(ip-1) to mu(ip) (and from mumin to mu(0) for ip=0),
        /// and each uses a fresh set of nquad points.
        struct Subinterval {
          /// Half-length of the subinterval
          double mulen;
          /// cosh(mu), sinh(mu) at the subinterval's points
          helfem::Vector chmu, shmu;
          /// Basis functions there, (nquad x nbf)
          helfem::Matrix bf;
        };
        std::vector<Subinterval> sub;
      };

      /// Build the element-only data above. Call once per element.
      TwoElectronElement twoe_element(double mumin, double mumax, const helfem::Vector & x, const helfem::Vector & wx, const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly);

      /// Primitive two-electron in-element integral, reusing precomputed
      /// element data. Equivalent to the twoe_integral() above, but without
      /// re-evaluating any polynomial.
      helfem::Matrix twoe_integral(const TwoElectronElement & el, int k, int l, int L, int M, const legendretable::LegendreTable & tab);
    }
  }
}

#endif
