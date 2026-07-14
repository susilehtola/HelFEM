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
#ifndef HELFEM_RADIAL_BLOCK_HELPER_H
#define HELFEM_RADIAL_BLOCK_HELPER_H

#include <Matrix.h>
#include <RadialBasis.h>
#include <type_traits>

namespace helfem {
  // Assemble a block-diagonal radial matrix by summing per-element
  // contributions. Kernel is a callable (size_t iel) -> helfem::Matrix
  // returning the per-element block; the block is placed into
  // rows/cols [ifirst .. ilast] as returned by radial.get_idx(iel).
  //
  // Replaces the copy-pasted pattern:
  //   for (size_t iel = 0; iel < radial.Nel(); ++iel) {
  //     size_t ifirst, ilast;
  //     radial.get_idx(iel, ifirst, ilast);
  //     M.submat(ifirst, ifirst, ilast, ilast) += radial.foo(iel);
  //   }
  // that appears ~13 times across src/atomic/TwoDBasis.cpp and
  // src/sadatom/basis.cpp.
  // The scalar type is deduced from the kernel's return type (every
  // per-element block is an Eigen matrix, which carries ::Scalar), so a
  // FEMRadialBasisT<long double> assembles a long-double matrix and every
  // existing double caller is unchanged.
  template <typename RadialBasis, typename Kernel>
  inline auto assemble_radial_diagonal(const RadialBasis & radial,
                                       Kernel && kernel)
      -> helfem::Mat<typename std::decay_t<decltype(kernel(size_t(0)))>::Scalar> {
    using T = typename std::decay_t<decltype(kernel(size_t(0)))>::Scalar;
    const Eigen::Index Nrad = static_cast<Eigen::Index>(radial.Nbf());
    helfem::Mat<T> M = helfem::Mat<T>::Zero(Nrad, Nrad);
    for (size_t iel = 0; iel < radial.Nel(); ++iel) {
      size_t ifirst, ilast;
      radial.get_idx(iel, ifirst, ilast);
      const Eigen::Index n = static_cast<Eigen::Index>(ilast - ifirst + 1);
      M.block(static_cast<Eigen::Index>(ifirst),
               static_cast<Eigen::Index>(ifirst), n, n) += kernel(iel);
    }
    return M;
  }
} // namespace helfem

#endif
