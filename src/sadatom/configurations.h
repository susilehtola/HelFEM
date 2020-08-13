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

#ifndef HELFEM_CONFIGURATION_H
#define HELFEM_CONFIGURATION_H

#include <armadillo>

namespace helfem {
  namespace sadatom {
    /**
     * Get Hartree-Fock ground-state configuration for 1 <= Z <= 118.
     *
     * The configurations are from the paper S. L. Saito,
     * "Hartree-Fock-Roothaan energies and expectation values for the
     * neutral atoms He to Uuo: The B-spline expansion method", Atomic
     * Data and Nuclear Data Tables 95 (2009) 836-870.
     */
    arma::ivec get_configuration(int Z);
  }
}

#endif
