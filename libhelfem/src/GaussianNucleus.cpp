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
#include "GaussianNucleus.h"
#include <cfloat>

namespace helfem {
  namespace modelpotential {
    GaussianNucleus::GaussianNucleus(int Z_, double Rrms) : Z(Z_) {
      // Eqn (11) in Visscher-Dyall 1997
      set_mu(sqrt(3.0/2.0)/Rrms);
    }

    GaussianNucleus::~GaussianNucleus() {
    }

    double GaussianNucleus::V(double R) const {
      // Taylor series for erf(mu*r)/r from Maple
      if(R <= Rcut) {
        double mur2 = std::pow(mu*R,2);
        return -Z*M_2_SQRTPI*mu*( 1.0 + (-1.0/3.0 + (1.0/10.0 - 1.0/42.0*mur2)*mur2)*mur2);
      } else {
        return -Z*erf(mu*R)/R;
      }
    }

    double GaussianNucleus::get_mu() const {
      return mu;
    }

    void GaussianNucleus::set_mu(double mu_) {
      // Set value
      mu=mu_;
      // Update Taylor series cutoff: sixth-order term is epsilon
      Rcut = std::pow(42.0*DBL_EPSILON, 1.0/6.0)/mu;
    }
  }
}
