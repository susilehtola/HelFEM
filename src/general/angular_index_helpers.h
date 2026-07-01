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
#ifndef HELFEM_ANGULAR_INDEX_HELPERS_H
#define HELFEM_ANGULAR_INDEX_HELPERS_H

#include <armadillo>
#include <cstddef>

namespace helfem {
  // Collect global basis-function indices belonging to a subset of
  // angular shells. Shells are laid out linearly in the global basis
  // in the order i = 0 .. nshells-1; shell i occupies shell_size(i)
  // consecutive basis-function slots. This helper walks the shells
  // once to count the matching slots and once more to place them,
  // matching the copy-pasted pattern that appeared in
  // atomic::TwoDBasis::m_indices / lm_indices and
  // diatomic::TwoDBasis::m_indices (both overloads).
  //
  // Usage:
  //   idx = collect_shell_indices(mval.n_elem,
  //             [&](size_t)   { return radial.Nbf(); },   // shell size
  //             [&](size_t i) { return mval(i) == m; });  // predicate
  template <typename ShellSizeFn, typename PredicateFn>
  inline arma::uvec collect_shell_indices(size_t nshells,
                                           ShellSizeFn shell_size,
                                           PredicateFn match) {
    size_t nm = 0;
    for (size_t i = 0; i < nshells; ++i)
      if (match(i)) nm += shell_size(i);

    arma::uvec idx(nm);
    size_t ioff = 0;
    size_t ibf  = 0;
    for (size_t i = 0; i < nshells; ++i) {
      const size_t nsh = shell_size(i);
      if (match(i)) {
        idx.subvec(ioff, ioff + nsh - 1) =
            arma::linspace<arma::uvec>(ibf, ibf + nsh - 1, nsh);
        ioff += nsh;
      }
      ibf += nsh;
    }
    return idx;
  }
} // namespace helfem

#endif
