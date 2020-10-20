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
#ifndef MODELPOTENTIAL_RADIALPOTENTIAL_H
#define MODELPOTENTIAL_RADIALPOTENTIAL_H

#include <helfem/ModelPotential.h>

namespace helfem {
  namespace modelpotential {
    /// Simple r^n radial potential
    class RadialPotential : public ModelPotential {
      /// Exponent
      int n;
    public:
      /// Constructor
      RadialPotential(int n);
      /// Destructor
      ~RadialPotential();
      /// Potential
      double V(double r) const override;
    };
  }
}

#endif
