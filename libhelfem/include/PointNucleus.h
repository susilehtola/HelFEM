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
#ifndef MODELPOTENTIAL_POINTNUCLEUS_H
#define MODELPOTENTIAL_POINTNUCLEUS_H

#include <ModelPotential.h>

namespace helfem {
  namespace modelpotential {
    /// Point nucleus
    template <typename T>
    class PointNucleusT : public ModelPotentialT<T> {
      /// Charge
      int Z;
    public:
      /// Constructor
      PointNucleusT(int Z);
      /// Destructor
      ~PointNucleusT();
      /// Potential
      T V(T r) const override;
    };

    /// The double instantiation, which every existing caller uses.
    using PointNucleus = PointNucleusT<double>;
  }
}

#endif
