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
#include "RadialPotential.h"

namespace helfem {
  namespace modelpotential {
    RadialPotential::RadialPotential(int n_) : n(n_) {
    }

    RadialPotential::~RadialPotential() {
    }

    double RadialPotential::V(double R) const {
      return std::pow(R,n);
    }
  }
}
