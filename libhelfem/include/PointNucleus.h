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
#ifndef MODELPOTENTIAL_POINTNUCLEUS_H
#define MODELPOTENTIAL_POINTNUCLEUS_H

#include <helfem/ModelPotential.h>

namespace helfem {
  namespace modelpotential {
    /// Point nucleus
    class PointNucleus : public ModelPotential {
      /// Charge
      int Z;
    public:
      /// Constructor
      PointNucleus(int Z);
      /// Destructor
      ~PointNucleus();
      /// Potential
      double V(double r) const override;
    };
  }
}

#endif
