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
#ifndef POLYNOMIAL_BASIS_POLYNOMIALBASIS_H
#define POLYNOMIAL_BASIS_POLYNOMIALBASIS_H

// v2 refactor (Phase 1): the abstract PolynomialBasis class is now a
// template living in lib1dfem. libhelfem is compiled at T = double, so
// this header exposes the double-only instantiation under the original
// helfem::polynomial_basis::PolynomialBasis name -- existing concrete
// subclasses (LIPBasis, HIPBasis, GeneralHIPBasis, LegendreBasis) and
// downstream callers continue to compile without source changes.

#include <lib1dfem/PolynomialBasis.h>
#include <lib1dfem/LIPBasis.h>
#include <lib1dfem/HIPBasis.h>
#include <lib1dfem/HIP2Basis.h>
#include <lib1dfem/HIP3Basis.h>
#include <lib1dfem/LegendreBasis.h>
#include <lib1dfem/lobatto.h>
#include <stdexcept>

namespace helfem {
  namespace polynomial_basis {
    /// Alias for the lib1dfem template at T = double.
    using PolynomialBasis = helfem::lib1dfem::polynomial_basis::PolynomialBasis<double>;

    /// Factory: construct a primitive polynomial basis by ID.
    PolynomialBasis * get_basis(int primbas, int Nnodes);

    /// Same factory, templated on the scalar type. The concrete bases
    /// (LegendreBasis<T>, LIPBasis<T>, HIPBasis<T>) and lobatto_compute<T>
    /// were already generic; this just stops the factory from pinning double.
    template<typename T>
    helfem::lib1dfem::polynomial_basis::PolynomialBasis<T> *
    get_basis_T(int primbas, int Nnodes) {
      namespace pb = helfem::lib1dfem::polynomial_basis;
      if(Nnodes < 2)
        throw std::logic_error("Can't have finite element basis with less than two nodes per element.\n");

      switch(primbas) {
      case 3:
        return new pb::LegendreBasis<T>(Nnodes, primbas);
      case 4: {
        helfem::lib1dfem::Vec<T> x, w;
        helfem::lib1dfem::lobatto::lobatto_compute<T>(Nnodes, x, w);
        return new pb::LIPBasis<T>(x, primbas);
      }
      case 5: {
        helfem::lib1dfem::Vec<T> x, w;
        helfem::lib1dfem::lobatto::lobatto_compute<T>(Nnodes, x, w);
        return new pb::HIPBasis<T>(x, primbas);
      }
      case 8: {
        // Analytic HIP2 (closed-form Hermite, 2nd order)
        helfem::lib1dfem::Vec<T> x, w;
        helfem::lib1dfem::lobatto::lobatto_compute<T>(Nnodes, x, w);
        return new pb::HIP2Basis<T>(x, primbas);
      }
      case 9: {
        // Analytic HIP3 (closed-form Hermite, 3rd order)
        helfem::lib1dfem::Vec<T> x, w;
        helfem::lib1dfem::lobatto::lobatto_compute<T>(Nnodes, x, w);
        return new pb::HIP3Basis<T>(x, primbas);
      }
      default:
        throw std::logic_error("Unsupported primitive basis for the templated factory.\n");
      }
    }
  }
}

#endif
