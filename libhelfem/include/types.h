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
#ifndef HELFEM_FEM_TYPES_H
#define HELFEM_FEM_TYPES_H

// The dynamic-size Eigen shorthand aliases (Vec<T> / Mat<T> / RowVec<T>)
// and the Eigen indexing-placeholder shim both live in <Matrix.h>. This
// header adds the one alias the FEM primitive bases additionally need --
// the signed index vector IVec -- and re-exports the shorthand via Matrix.h
// so the templated basis / quadrature / grid headers keep a single include.

#include <Matrix.h>

namespace helfem {

/// Signed index vector -- typically for the `enabled` column-selection
/// list on PolynomialBasis subclasses.
using IVec = Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>;

} // namespace helfem

#endif // HELFEM_FEM_TYPES_H
