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
#ifndef LIB1DFEM_TYPES_H
#define LIB1DFEM_TYPES_H

#include <Eigen/Core>

// Eigen 5 moved the indexing placeholders `all`, `last`, `lastp1`
// from namespace `Eigen` into `Eigen::placeholders`. Everything in
// HelFEM was written against the 3.x location, so re-expose them at
// the old spelling for any Eigen >= 4.9 (i.e. 5.x prereleases and up).
#if EIGEN_VERSION_AT_LEAST(4, 90, 0)
namespace Eigen {
  using placeholders::all;
  using placeholders::last;
  using placeholders::lastp1;
}
#endif

// Shorthand aliases for the dynamic-size Eigen types used throughout
// lib1dfem (and re-exported by libhelfem). The full
// `Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>` spelling is
// unreadable in template-heavy code, so vectors and matrices spell as
// `Vec<T>` / `Mat<T>` instead. Column-major storage.

namespace helfem {
namespace lib1dfem {

template <typename T>
using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename T>
using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template <typename T>
using RowVec = Eigen::Matrix<T, 1, Eigen::Dynamic>;

/// Signed index vector -- typically for the `enabled` column-selection
/// list on PolynomialBasis subclasses.
using IVec = Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>;

} // namespace lib1dfem
} // namespace helfem

#endif // LIB1DFEM_TYPES_H
