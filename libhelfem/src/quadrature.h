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
#ifndef QUADRATURE_H
#define QUADRATURE_H

#include <Matrix.h>
#include <memory>
#include <functional>
#include <ModelPotential.h>
#include <PolynomialBasis.h>

// Phase 5.7: quadrature API migrated to Eigen.
namespace helfem {
  namespace quadrature {
    /// Inner in-element two-electron integral
    ///   phi(r) = (1 / r^(L+1)) * integral_0^r dr' r'^L B_k(r') B_l(r')
    helfem::Matrix twoe_inner_integral(double rmin, double rmax,
                                        const helfem::Vector & x, const helfem::Vector & wx,
                                        const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly,
                                        int L);

    /// Primitive in-element two-electron integral. Cross-element pieces
    /// reduce to products of radial integrals (handled by caller).
    helfem::Matrix twoe_integral(double rmin, double rmax,
                                  const helfem::Vector & x, const helfem::Vector & wx,
                                  const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly,
                                  int L);

    /// Inner in-element two-electron Yukawa integral.
    helfem::Matrix yukawa_inner_integral(double rmin, double rmax,
                                          const helfem::Vector & x, const helfem::Vector & wx,
                                          const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly,
                                          int L, double lambda);

    /// Primitive in-element two-electron Yukawa integral.
    helfem::Matrix yukawa_integral(double rmin, double rmax,
                                    const helfem::Vector & x, const helfem::Vector & wx,
                                    const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly,
                                    int L, double lambda);

    /// Primitive two-electron complementary error function integral.
    /// These do not factorise across elements.
    helfem::Matrix erfc_integral(double rmini, double rmaxi,
                                  const helfem::Matrix & bfi,
                                  const helfem::Vector & xi, const helfem::Vector & wi,
                                  double rmink, double rmaxk,
                                  const helfem::Matrix & bfk,
                                  const helfem::Vector & xk, const helfem::Vector & wk,
                                  int L, double mu);

    /// Spherically symmetric potential V(r).
    helfem::Matrix spherical_potential(double rmin, double rmax,
                                        const helfem::Vector & x, const helfem::Vector & wx,
                                        const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly);
  }
}

#endif
