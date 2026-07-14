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
#ifndef MODELPOTENTIAL_MODELPOTENTIAL_H
#define MODELPOTENTIAL_MODELPOTENTIAL_H

namespace helfem {
  namespace modelpotential {
    /// Model potential.
    ///
    /// Templated on the scalar type, following FiniteElementBasisT<T>: the
    /// potential is evaluated inside the radial quadrature, so pinning it to
    /// double would cap the precision of an otherwise long-double
    /// FEMRadialBasisT<long double>. Instantiated for double and long double.
    template <typename T>
    class ModelPotentialT {
    public:
      /// Constructor
      ModelPotentialT();
      /// Destructor
      virtual ~ModelPotentialT();

      /// Potential at a single radial point.
      virtual T V(T r) const = 0;
    };

    /// The double instantiation, which every existing caller uses.
    using ModelPotential = ModelPotentialT<double>;
  }
}

#endif
