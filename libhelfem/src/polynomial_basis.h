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
#ifndef POLYNOMIAL_BASIS_H
#define POLYNOMIAL_BASIS_H

#include "LegendreBasis.h"
#include "HermiteBasis.h"
#include "LIPBasis.h"
#include <armadillo>
#include "helfem.h"

namespace helfem {
  namespace polynomial_basis {
    /// Get primitive indices for a basis with n nodes and n overlapping functions.
    arma::uvec primitive_indices(int nnodes, int noverlap, bool drop_first, bool drop_last);
  }
}
#endif
