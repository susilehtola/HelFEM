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
#include "helfem/HollowNucleus.h"

namespace helfem {
  namespace modelpotential {
    HollowNucleus::HollowNucleus(int Z_, double R_) : Z(Z_), R(R_) {
    }

    HollowNucleus::~HollowNucleus() {
    }

    double HollowNucleus::V(double r) const {
      if(r>=R) {
        return -Z/r;
      } else {
        return -Z/R;
      }
    }

    double HollowNucleus::get_R() const {
      return R;
    }

    void HollowNucleus::set_R(double R_) {
      R=R_;
    }
  }
}
