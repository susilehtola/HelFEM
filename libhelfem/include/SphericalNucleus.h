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
#ifndef MODELPOTENTIAL_SPHERICALNUCLEUS_H
#define MODELPOTENTIAL_SPHERICALNUCLEUS_H

#include <helfem/ModelPotential.h>

namespace helfem {
  namespace modelpotential {
    /// Uniformly charged spherical nucleus
    class SphericalNucleus : public ModelPotential {
      /// Charge
      int Z;
      /// Size
      double R0;
    public:
      /// Constructor
      SphericalNucleus(int Z, double Rrms);
      /// Destructor
      ~SphericalNucleus();
      /// Potential
      double V(double r) const override;
      /// Get R0
      double get_R0() const;
      /// Set R0
      void set_R0(double R0);
    };
  }
}

#endif
