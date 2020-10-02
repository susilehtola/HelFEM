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
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */
#ifndef MODELPOTENTIAL_HOLLOWNUCLEUS_H
#define MODELPOTENTIAL_HOLLOWNUCLEUS_H

#include <helfem/ModelPotential.h>

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
