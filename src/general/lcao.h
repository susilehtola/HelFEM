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
#ifndef LCAO_H
#define LCAO_H

// Phase 5.18: lcao migrated arma -> Eigen. The vector overloads have
// no in-tree callers; all callers use the scalar overloads.
#include <Matrix.h>

namespace helfem {
  namespace lcao {
    /// Evaluate radial GTO (scalar)
    double radial_GTO(double r, int l, double alpha);
    /// Evaluate radial GTO (vectorised over r and alpha)
    helfem::Matrix radial_GTO(const helfem::Vector & r, int l, const helfem::Vector & alpha);
    /// Evaluate radial STO (scalar)
    double radial_STO(double r, int l, double zeta);
    /// Evaluate radial STO (vectorised over r and zeta)
    helfem::Matrix radial_STO(const helfem::Vector & r, int l, const helfem::Vector & zeta);
  }
}

#endif
