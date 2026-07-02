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
#include "HollowNucleus.h"
#include <cmath>

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
