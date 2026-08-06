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
      // Symmetry-reduced storage. The Gaunt coefficient
      //     Y^{l1 l2 l3}_{m1 m2 m3} = int Y_l1^m1 Y_l2^m2 Y_l3^m3 dOmega
      // is fully symmetric under permutation of its three columns and under
      // negating all three m at once (Rasch & Yu 2004 eqs 4.3-4.5), and it
      // vanishes unless the l form a triangle, l1+l2+l3 is even, and
      // m1+m2+m3 = 0. Storing one representative per orbit removes all of
      // that at once, in place of a dense array over five indices.
      //
      // Layout: one contiguous run per (l1,l2,l3) with l1>=l2>=l3, in which
      // m3 = 0..min(l3,mcap) and, for each m3, m2 spans
      // [max(-l2,-mcap), min(min(l1-m3,l2), mcap)]. m1 is implied by the
      // m-sum rule. triple_base indexes the runs; NPOS marks a triple that
      // fails the triangle or parity test, which is Rasch & Yu's null
      // pointer.
      //
      // NOTE: the m2-interval reductions of Pinchon & Hoggan (2007, eqs
      // 22-24), which exploit the extra symmetries when l1=l2, l2=l3 or
      // m3=0, are NOT applied. They buy a further 10-18% of the coefficients
      // (their Table I) at the cost of several interacting special cases in
      // both build and lookup. Their larger saving -- 61% of the pointer
      // array -- is already obtained here by indexing the runs on (l1,l2,l3)
      // rather than on (l1,l2,l3,m3).
      static constexpr std::size_t NPOS = static_cast<std::size_t>(-1);
      std::vector<std::size_t> triple_base;
      std::vector<T> coeffs;
      int lall = 0;                  ///< max l on any of the three axes
      // Bound on the m actually stored. It is 2*mcap, not mcap: the caller's
      // bound constrains |M| and |m|, but the third index mp = M - m reaches
      // |M|+|m| <= 2*mcap, and in this layout all three m are explicit axes.
      // (In the old dense layout mp was implicit, so it never needed room.)
      int mstore = 0;
      std::size_t triple_index(int l1, int l2, int l3) const {
        return (static_cast<std::size_t>(l1) * (lall + 1) + l2) * (lall + 1) + l3;
      }
      /// Offset of (l1,l2,l3,m3,m2) within its run, or NPOS if not stored.
      std::size_t slot(int l1, int l2, int l3, int m3, int m2) const;
      /// Number of m2 values stored for this (l1,l2,l3,m3).
      int m2_count(int l1, int l2, int l3, int m3) const;

      int Lmax = 0, lmax = 0, lpmax = 0;
      // Caps on |M| and |m|. The triangular packing L*(L+1)+M assumes every
      // |M| <= L occurs, which is true for an atom but wildly false for a
      // diatomic: there L is large while |M| is bounded by the basis, since M
      // is a difference of two basis m values. Cu2 at lmax=46 reaches L=96
      // with |M| <= 4, so the triangular table is ~190x larger than needed --
      // 61 GB against 322 MB. Capping the m axes makes the packing rectangular
      // in those directions and the table proportional to what is used.
      int mcap = 0;
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
