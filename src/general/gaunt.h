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
#ifndef GAUNT
#define GAUNT

#include <vector>

namespace helfem {
  namespace gaunt {
    /**
     * Computes Gaunt coefficient \f$ G^{M m m'}_{L l l'} \f$ in the expansion
     * \f$ Y_l^m (\Omega) Y_{l'}^{m'} (\Omega) = \sum_{L,M} G^{M m m'}_{L l l'} Y_L^M (\Omega) \f$
     */
    double gaunt_coefficient(int L, int M, int l, int m, int lp, int mp);

    /// Get "modified" Gaunt coefficient (interim coupling through cos^2)
    double modified_gaunt_coefficient(int L, int M, int l, int m, int lp, int mp);

    /// Table of Gaunt coefficients.
    /// Storage is a flat 5D dense array indexed by (L, M, l, m, lp); the m-sum
    /// selection rule fixes mp = M - m, so an explicit mp axis is omitted.
    /// Callers must pre-enforce the rule (in practice they do this naturally,
    /// since M is computed from outer m-channel indices).
    class Gaunt {
      std::vector<double> table;
      int Lmax = 0, lmax = 0, lpmax = 0;
      std::size_t lm_stride = 0;
      std::size_t Lm_stride = 0;

      std::size_t flat_index(int L, int M, int l, int m, int lp) const {
        const std::size_t LM = static_cast<std::size_t>(L) * (L + 1) + M;
        const std::size_t lm = static_cast<std::size_t>(l) * (l + 1) + m;
        return LM * Lm_stride + lm * lm_stride + static_cast<std::size_t>(lp);
      }

    public:
      Gaunt() = default;
      Gaunt(int Lmax, int lmax, int lpmax);

      /// Get Gaunt coefficient. mp is implicit: mp = M - m. Cells outside the
      /// stored range or violating |M|<=L, |m|<=l return 0.
      double coeff(int L, int M, int l, int m, int lp) const;
      /// Get "modified" Gaunt coefficient (interim coupling through cos^2)
      double mod_coeff(int lj, int mj, int L, int M, int li, int mi) const;

      /// Get cosine type coupling
      double cosine_coupling(int lj, int mj, int li, int mi) const;
      /// Get cosine^2 type coupling
      double cosine2_coupling(int lj, int mj, int li, int mi) const;
      /// Get cosine^3 type coupling
      double cosine3_coupling(int lj, int mj, int li, int mi) const;
      /// Get cosine^4 type coupling
      double cosine4_coupling(int lj, int mj, int li, int mi) const;
      /// Get cosine^5 type coupling
      double cosine5_coupling(int lj, int mj, int li, int mi) const;

      /// Get sine^2 type coupling
      double sine2_coupling(int lj, int mj, int li, int mi) const;
      /// Get cosine^2 sine^2 type coupling
      double cosine2_sine2_coupling(int lj, int mj, int li, int mi) const;
    };
  }
}

#endif
