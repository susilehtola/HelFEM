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

// Shorthand aliases for the dynamic-size Eigen types used throughout
// lib1dfem (and re-exported by libhelfem). Phase 5.x of the arma -> Eigen
// migration: the full `Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>`
// type spelling is unreadable in template-heavy code, so the migrated
// headers spell vectors and matrices as `Vec<T>` / `Mat<T>` instead.
//
// Column-major storage to match arma::mat / arma::vec so the libhelfem
// boundary shims stay zero-copy at T = double.

namespace helfem {
namespace lib1dfem {

template <typename T>
using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename T>
using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template <typename T>
using RowVec = Eigen::Matrix<T, 1, Eigen::Dynamic>;

} // namespace lib1dfem
} // namespace helfem

#endif // LIB1DFEM_TYPES_H
