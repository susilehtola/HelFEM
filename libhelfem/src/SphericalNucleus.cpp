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
#include "helfem/SphericalNucleus.h"

namespace helfem {
  namespace modelpotential {
    SphericalNucleus::SphericalNucleus(int Z_, double Rrms) : Z(Z_) {
      // Eqn (4) in Visscher-Dyall 1997
      R0 = sqrt(5.0/3.0)*Rrms;
    }

    SphericalNucleus::~SphericalNucleus() {
    }

    double SphericalNucleus::V(double r) const {
      if(r>=R0) {
        // See full charge, eqn (7a) in Visscher-Dyall 1997
        return -Z/r;
      } else {
        // See only charge inside, eqn (7b) in Visscher-Dyall 1997
        return -Z/(2.0*R0)*(3.0-std::pow(r/R0,2));
      }
    }

    double SphericalNucleus::get_R0() const {
      return R0;
    }

    void SphericalNucleus::set_R0(double R0_) {
      R0=R0_;
    }
  }
}
