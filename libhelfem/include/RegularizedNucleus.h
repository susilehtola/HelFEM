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
#ifndef MODELPOTENTIAL_REGULARIZEDNUCLEUS_H
#define MODELPOTENTIAL_REGULARIZEDNUCLEUS_H

#include <ModelPotential.h>

namespace helfem {
  namespace modelpotential {
    /** Regularized nucleus, i.e. Gygi's Analytic Norm-Conserving
        Regularized Potential from doi:10.1021/acs.jctc.2c01191.
     */
    class RegularizedNucleus : public ModelPotential {
      /// Charge
      int Z;
      /// Size parameters
      double a, b;
    public:
      /// Constructor
      RegularizedNucleus(int Z, double a);
      /// Destructor
      ~RegularizedNucleus();
      /// Potential
      double V(double r) const override;
      /// Get a
      double get_a() const;
      /// Get b
      double get_b() const;
      /// Set mu
      void set_a(double a);
    };
  }
}

#endif
