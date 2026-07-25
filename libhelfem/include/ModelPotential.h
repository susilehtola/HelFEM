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

#include <vector>

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

      /// Radii in [a, b] at which V is not smooth -- a hard nuclear
      /// boundary, a tabulation knot, the edge of a charge distribution.
      /// The quadrature splits the element there, because plain
      /// order-refinement stalls across a kink: it converges only
      /// algebraically and grinds to the order cap without ever reaching
      /// eps(T).
      ///
      /// Default: none, i.e. the potential is smooth everywhere. A model
      /// that overrides this must return only the points strictly inside
      /// (a, b); boundaries coinciding with the element ends are already
      /// handled by the element decomposition itself.
      virtual std::vector<T> breakpoints(T a, T b) const {
        (void) a; (void) b;
        return std::vector<T>();
      }
    };

    /// The double instantiation, which every existing caller uses.
    using ModelPotential = ModelPotentialT<double>;
  }
}

#endif
