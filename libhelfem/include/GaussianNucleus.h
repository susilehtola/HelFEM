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
#ifndef MODELPOTENTIAL_GAUSSIANNUCLEUS_H
#define MODELPOTENTIAL_GAUSSIANNUCLEUS_H

#include <ModelPotential.h>

namespace helfem {
  namespace modelpotential {
    /// Gaussian nucleus
    class GaussianNucleus : public ModelPotential {
      /// Charge
      int Z;
      /// Size
      double mu;

      /// Cutoff for Taylor series
      double Rcut;
    public:
      /// Constructor
      GaussianNucleus(int Z, double Rrms);
      /// Destructor
      ~GaussianNucleus();
      /// Potential
      double V(double r) const override;
      /// Get mu
      double get_mu() const;
      /// Set mu
      void set_mu(double mu);
    };
  }
}

#endif
