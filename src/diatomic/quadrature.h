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
    }
  }
}

#endif
