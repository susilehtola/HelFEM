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

// The primitive polynomial bases are templates on the scalar type T
// (PolynomialBasisT<T> and its subclasses LIPBasisT / HIPBasisT /
// HIP2BasisT / HIP3BasisT / LegendreBasisT, each in its own header). Most
// of libhelfem, and every downstream caller, work at T = double; this
// header pulls in the templated definitions and exposes the double
// instantiation of each under the original (un-suffixed) name, so existing
// callers spell `polynomial_basis::PolynomialBasis`, `::LIPBasis`, ... and
// compile unchanged.

#include <PolynomialBasisT.h>
#include <LIPBasis.h>
#include <HIPBasis.h>
#include <HIP2Basis.h>
#include <HIP3Basis.h>
#include <LegendreBasis.h>
#include <lobatto.h>
#include <stdexcept>

namespace helfem {
  namespace polynomial_basis {
    /// Double-precision instantiations of the templated primitive bases,
    /// exposed under their historical (un-suffixed) names.
    using PolynomialBasis = PolynomialBasisT<double>;
    using LIPBasis        = LIPBasisT<double>;
    using HIPBasis        = HIPBasisT<double>;
    using HIP2Basis       = HIP2BasisT<double>;
    using HIP3Basis       = HIP3BasisT<double>;
    using LegendreBasis   = LegendreBasisT<double>;

    /// Factory: construct a primitive polynomial basis by ID.
    PolynomialBasis * get_basis(int primbas, int Nnodes);

    /// Same factory, templated on the scalar type. The concrete bases
    /// (LegendreBasisT<T>, LIPBasisT<T>, HIPBasisT<T>) and lobatto_compute<T>
    /// were already generic; this just stops the factory from pinning double.
    template<typename T>
    helfem::polynomial_basis::PolynomialBasisT<T> *
    get_basis_T(int primbas, int Nnodes) {
      namespace pb = helfem::polynomial_basis;
      if(Nnodes < 2)
        throw std::logic_error("Can't have finite element basis with less than two nodes per element.\n");

      switch(primbas) {
      case 3:
        return new pb::LegendreBasisT<T>(Nnodes, primbas);
      case 4: {
        helfem::Vec<T> x, w;
        helfem::lobatto::lobatto_compute<T>(Nnodes, x, w);
        return new pb::LIPBasisT<T>(x, primbas);
      }
      case 5: {
        helfem::Vec<T> x, w;
        helfem::lobatto::lobatto_compute<T>(Nnodes, x, w);
        return new pb::HIPBasisT<T>(x, primbas);
      }
      case 8: {
        // Analytic HIP2 (closed-form Hermite, 2nd order)
        helfem::Vec<T> x, w;
        helfem::lobatto::lobatto_compute<T>(Nnodes, x, w);
        return new pb::HIP2BasisT<T>(x, primbas);
      }
      case 9: {
        // Analytic HIP3 (closed-form Hermite, 3rd order)
        helfem::Vec<T> x, w;
        helfem::lobatto::lobatto_compute<T>(Nnodes, x, w);
        return new pb::HIP3BasisT<T>(x, primbas);
      }
      default:
        throw std::logic_error("Unsupported primitive basis for the templated factory.\n");
      }
    }
  }
}

#endif
