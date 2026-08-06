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

#include <cstddef>
#include <vector>

namespace helfem {
  namespace gaunt {
    /**
     * Computes Gaunt coefficient \f$ G^{M m m'}_{L l l'} \f$ in the expansion
     * \f$ Y_l^m (\Omega) Y_{l'}^{m'} (\Omega) = \sum_{L,M} G^{M m m'}_{L l l'} Y_L^M (\Omega) \f$
     *
     * Templated on the scalar type. The value comes from libwignernj, which
     * evaluates the coefficient from an EXACT prime-factorised rational and
     * only rounds at the very end -- so it is correctly rounded at whatever
     * precision is asked for, and never caps the calculation. Instantiated
     * for double, long double and (under HELFEM_HAVE_FLOAT128) _Float128.
     */
    template <typename T>
    T gaunt_coefficient_T(int L, int M, int l, int m, int lp, int mp);

    /// Double-precision entry point (unchanged spelling for existing callers).
    double gaunt_coefficient(int L, int M, int l, int m, int lp, int mp);

    /// Get "modified" Gaunt coefficient (interim coupling through cos^2)
    double modified_gaunt_coefficient(int L, int M, int l, int m, int lp, int mp);

    /// Table of Gaunt coefficients.
    /// Storage is a flat 5D dense array indexed by (L, M, l, m, lp); the m-sum
    /// selection rule fixes mp = M - m, so an explicit mp axis is omitted.
    /// Callers must pre-enforce the rule (in practice they do this naturally,
    /// since M is computed from outer m-channel indices).
    ///
    /// Templated on the scalar type, following FiniteElementBasisT<T>: the
    /// Gaunt coefficients multiply the radial integrals in the Coulomb and
    /// exchange assemblies, so a double-only table would cap an otherwise
    /// higher-precision Fock build at double accuracy.
    template <typename T>
    class GauntT {
      std::vector<T> table;
      int Lmax = 0, lmax = 0, lpmax = 0;
      // Caps on |M| and |m|. The triangular packing L*(L+1)+M assumes every
      // |M| <= L occurs, which is true for an atom but wildly false for a
      // diatomic: there L is large while |M| is bounded by the basis, since M
      // is a difference of two basis m values. Cu2 at lmax=46 reaches L=96
      // with |M| <= 4, so the triangular table is ~190x larger than needed --
      // 61 GB against 322 MB. Capping the m axes makes the packing rectangular
      // in those directions and the table proportional to what is used.
      int mcap = 0;
      std::size_t lm_stride = 0;
      std::size_t Lm_stride = 0;

      std::size_t flat_index(int L, int M, int l, int m, int lp) const {
        const std::size_t LM = static_cast<std::size_t>(L) * (2*mcap + 1) + (M + mcap);
        const std::size_t lm = static_cast<std::size_t>(l) * (2*mcap + 1) + (m + mcap);
        return LM * Lm_stride + lm * lm_stride + static_cast<std::size_t>(lp);
      }

    public:
      GauntT() = default;
      /// mcap bounds |M| and |m| alike. Defaulted to the full range, which is
      /// what an atomic basis needs; a diatomic basis should pass its actual,
      /// far smaller, bound. One cap suffices: mod_coeff routes plain basis m
      /// values into the M axis (coeff(lj,mj,li,mi,L)), so the M axis cannot be
      /// bounded more tightly than the m axis anyway.
      GauntT(int Lmax, int lmax, int lpmax, int mcap = -1);

      /// Get Gaunt coefficient. mp is implicit: mp = M - m. Cells outside the
      /// stored range or violating |M|<=L, |m|<=l return 0.
      T coeff(int L, int M, int l, int m, int lp) const;
      /// Get "modified" Gaunt coefficient (interim coupling through cos^2)
      T mod_coeff(int lj, int mj, int L, int M, int li, int mi) const;

      /// Get cosine type coupling
      T cosine_coupling(int lj, int mj, int li, int mi) const;
      /// Get cosine^2 type coupling
      T cosine2_coupling(int lj, int mj, int li, int mi) const;
      /// Get cosine^3 type coupling
      T cosine3_coupling(int lj, int mj, int li, int mi) const;
      /// Get cosine^4 type coupling
      T cosine4_coupling(int lj, int mj, int li, int mi) const;
      /// Get cosine^5 type coupling
      T cosine5_coupling(int lj, int mj, int li, int mi) const;

      /// Get sine^2 type coupling
      T sine2_coupling(int lj, int mj, int li, int mi) const;
      /// Get cosine^2 sine^2 type coupling
      T cosine2_sine2_coupling(int lj, int mj, int li, int mi) const;
    };

    /// The double instantiation, which every existing caller uses.
    using Gaunt = GauntT<double>;
  }
}

#endif
