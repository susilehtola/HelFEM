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
//
// Now templated on the scalar type, following FiniteElementBasisT<T>: these
// are the quadratures FEMRadialBasisT<T> runs, so pinning them to double
// would cap the precision of the whole radial layer. Explicitly instantiated
// for double and long double at the bottom of quadrature.cpp.
//
// T is deduced from the Vec<T> / Mat<T> arguments only. The scalar arguments
// (rmin, rmax, lambda, mu) and the polynomial basis sit in a non-deduced
// context: callers pass int / double literals for the former (e.g.
// inttest.cpp's `twoe_integral(0, R, ...)`) and non-const shared_ptrs for the
// latter, neither of which is deducible.
namespace helfem {
  namespace quadrature {
    template <typename T>
    using PolyBasis = helfem::lib1dfem::polynomial_basis::PolynomialBasis<T>;

    /// Inner in-element two-electron integral
    ///   phi(r) = (1 / r^(L+1)) * integral_0^r dr' r'^L B_k(r') B_l(r')
    template <typename T>
    helfem::Mat<T> twoe_inner_integral(NonDeduced<T> rmin, NonDeduced<T> rmax,
                                       const helfem::Vec<T> & x, const helfem::Vec<T> & wx,
                                       const std::shared_ptr<const PolyBasis<NonDeduced<T>>> & poly,
                                       int L);

    /// Primitive in-element two-electron integral. Cross-element pieces
    /// reduce to products of radial integrals (handled by caller).
    template <typename T>
    helfem::Mat<T> twoe_integral(NonDeduced<T> rmin, NonDeduced<T> rmax,
                                 const helfem::Vec<T> & x, const helfem::Vec<T> & wx,
                                 const std::shared_ptr<const PolyBasis<NonDeduced<T>>> & poly,
                                 int L);

    /// Inner in-element two-electron Yukawa integral.
    template <typename T>
    helfem::Mat<T> yukawa_inner_integral(NonDeduced<T> rmin, NonDeduced<T> rmax,
                                         const helfem::Vec<T> & x, const helfem::Vec<T> & wx,
                                         const std::shared_ptr<const PolyBasis<NonDeduced<T>>> & poly,
                                         int L, NonDeduced<T> lambda);

    /// Primitive in-element two-electron Yukawa integral.
    template <typename T>
    helfem::Mat<T> yukawa_integral(NonDeduced<T> rmin, NonDeduced<T> rmax,
                                   const helfem::Vec<T> & x, const helfem::Vec<T> & wx,
                                   const std::shared_ptr<const PolyBasis<NonDeduced<T>>> & poly,
                                   int L, NonDeduced<T> lambda);

    /// Primitive two-electron complementary error function integral.
    /// These do not factorise across elements.
    template <typename T>
    helfem::Mat<T> erfc_integral(NonDeduced<T> rmini, NonDeduced<T> rmaxi,
                                 const helfem::Mat<T> & bfi,
                                 const helfem::Vec<T> & xi, const helfem::Vec<T> & wi,
                                 NonDeduced<T> rmink, NonDeduced<T> rmaxk,
                                 const helfem::Mat<T> & bfk,
                                 const helfem::Vec<T> & xk, const helfem::Vec<T> & wk,
                                 int L, NonDeduced<T> mu);

    /// Spherically symmetric potential V(r).
    template <typename T>
    helfem::Mat<T> spherical_potential(NonDeduced<T> rmin, NonDeduced<T> rmax,
                                       const helfem::Vec<T> & x, const helfem::Vec<T> & wx,
                                       const std::shared_ptr<const PolyBasis<NonDeduced<T>>> & poly);
  }
}

#endif
