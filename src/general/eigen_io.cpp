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

// write_raw_ascii / print_matrix are header-only templates now (eigen_io.h):
// the raw-ASCII precision follows the scalar type, so nothing has to be
// instantiated here. This translation unit is kept so the build's object
// list does not change.
#include "eigen_io.h"
