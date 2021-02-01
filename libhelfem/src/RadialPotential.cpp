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
#include "RadialPotential.h"

namespace helfem {
  namespace modelpotential {
    RadialPotential::RadialPotential(int n_) : n(n_) {}

    RadialPotential::~RadialPotential() {}

    double RadialPotential::V(double R) const { return std::pow(R, n); }
  } // namespace modelpotential
} // namespace helfem
