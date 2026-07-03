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
#ifndef TEI_UTILS_H
#define TEI_UTILS_H

// Arma-typed two-electron-integral (TEI) permutation helpers used by
// the diatomic J/K assembly. These live next to their arma-consuming
// callers in src/general/ so libhelfem itself ships no arma-typed
// TEI helpers. When src/diatomic/basis.cpp's arma-native J/K storage
// is migrated to helfem::Matrix, this header becomes deletable.

#include <armadillo>
#include <cstddef>

namespace helfem {
  namespace utils {
    /// Form two-electron integrals from product of large-r and small-r
    /// radial moment matrices.
    arma::mat product_tei(const arma::mat & big, const arma::mat & small);

    /// Check that the two-electron integral has proper symmetry
    /// i<->j and k<->l.
    void check_tei_symmetry(const arma::mat & tei, size_t Ni, size_t Nj,
                             size_t Nk, size_t Nl);

    /// Permute indices (ij|kl) -> (jk|il).
    arma::mat exchange_tei(const arma::mat & tei, size_t Ni, size_t Nj,
                            size_t Nk, size_t Nl);
  }
}

#endif // TEI_UTILS_H
