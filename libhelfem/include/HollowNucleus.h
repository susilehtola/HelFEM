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
    class HollowNucleus : public ModelPotential {
      /// Charge
      int Z;
      /// Size
      double R;
    public:
      /// Constructor
      HollowNucleus(int Z, double R);
      /// Destructor
      ~HollowNucleus();
      /// Potential
      double V(double r) const override;
      /// Get R
      double get_R() const;
      /// Set R
      void set_R(double R);
    };
  }
}

#endif
