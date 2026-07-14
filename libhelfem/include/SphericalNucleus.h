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
#ifndef MODELPOTENTIAL_SPHERICALNUCLEUS_H
#define MODELPOTENTIAL_SPHERICALNUCLEUS_H

#include <ModelPotential.h>

namespace helfem {
  namespace modelpotential {
    /// Uniformly charged spherical nucleus
    template <typename T>
    class SphericalNucleusT : public ModelPotentialT<T> {
      /// Charge
      int Z;
      /// Size
      T R0;
    public:
      /// Constructor
      SphericalNucleusT(int Z, T Rrms);
      /// Destructor
      ~SphericalNucleusT();
      /// Potential
      T V(T r) const override;
      /// Get R0
      T get_R0() const;
      /// Set R0
      void set_R0(T R0);
    };

    /// The double instantiation, which every existing caller uses.
    using SphericalNucleus = SphericalNucleusT<double>;
  }
}

#endif
