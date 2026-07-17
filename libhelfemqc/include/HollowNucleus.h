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
#ifndef MODELPOTENTIAL_HOLLOWNUCLEUS_H
#define MODELPOTENTIAL_HOLLOWNUCLEUS_H

#include <ModelPotential.h>

namespace helfem {
  namespace modelpotential {
    /// Thin hollow nucleus
    template <typename T>
    class HollowNucleusT : public ModelPotentialT<T> {
      /// Charge
      int Z;
      /// Size
      T R;
    public:
      /// Constructor
      HollowNucleusT(int Z, T R);
      /// Destructor
      ~HollowNucleusT();
      /// Potential
      T V(T r) const override;
      /// Get R
      T get_R() const;
      /// Set R
      void set_R(T R);
    };

    /// The double instantiation, which every existing caller uses.
    using HollowNucleus = HollowNucleusT<double>;
  }
}

#endif
