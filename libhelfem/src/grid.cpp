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

// v2 refactor (Phase 1): the grid-builder implementation now lives in
// libhelfem as a templated header. This translation unit is the double-only
// compatibility shim that keeps the libhelfem public API (declared in
// helfem.source.h) source-compatible during the migration.

#include <helfem.h>
#include <grid.h>
#include <cstring>

helfem::Vector helfem::utils::get_grid(double rmax, int num_el, int igrid,
                                        double zexp) {
  // The FEM grid is Eigen; return directly.
  return helfem::grid::get_grid<double>(rmax, num_el, igrid, zexp,
                                                   helfem::verbose);
}
