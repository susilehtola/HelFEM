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
#ifndef ATOMIC_BASIS_COULOMB_EXCHANGE_FE_H
#define ATOMIC_BASIS_COULOMB_EXCHANGE_FE_H

#include "RadialBasis.h"
#include <functional>

namespace helfem {
  // Forward declaration of the existing in-tree helper from
  // libhelfem/src/utils.cpp (its header is in libhelfem/src/, not on the
  // public include path; the symbol is resolved at link time -- callers
  // of these helpers link against libhelfem anyway).
  namespace utils {
    /// (ij|kl) -> (jk|il) permutation on the helfem::Mat<T>-typed
    /// in-element TEI. Used internally by the assemble_K_FE_one_multipole
    /// helpers below. Explicitly instantiated in libhelfem/src/utils.cpp
    /// for double, long double and (under HELFEM_HAVE_FLOAT128) _Float128.
    template <typename T>
    helfem::Mat<T> exchange_tei(const helfem::Mat<T> & tei, size_t Ni, size_t Nj,
                                size_t Nk, size_t Nl);
  }
  namespace atomic {
    namespace basis {

      /// Free-function FE-basis assembly of the single-multipole
      /// two-electron radial operators. These are the "BARE" operators in
      /// the sense of TwoDBasis: no 4*pi/(2L+1) prefactor, no Gaunt
      /// coefficient -- the caller (NAORadialBasis, TwoDBasis::coulomb
      /// at one L, ...) applies whatever angular factor its own driver
      /// demands.
      ///
      /// For the in-element 4-index tensor, the assembly uses
      ///   FEMRadialBasis::twoe_integral(L, iel)         (bare Coulomb)
      ///   FEMRadialBasis::yukawa_integral(L, lambda, iel) (Yukawa)
      /// and contracts directly. For cross-element pairs the standard
      /// disjoint factorisation is used:
      ///   R^L(ab, cd) = r_small(iel)_{ab} * r_big(jel)_{cd}    (iel < jel)
      ///              = r_big(iel)_{ab}   * r_small(jel)_{cd}   (iel > jel)
      /// with
      ///   r_small = radial.radial_integral(L, iel)         (bare Coulomb)
      ///           = radial.bessel_il_integral(L, lambda, iel) (Yukawa)
      ///   r_big   = radial.radial_integral(-L-1, iel)
      ///           = radial.bessel_kl_integral(L, lambda, iel)
      ///
      /// Same assembly pattern that sadatom/atomic TwoDBasis::coulomb /
      /// exchange use (single source of truth: the helpers below). The
      /// existing TwoDBasis driver routines can rewrite themselves on
      /// top of these (DRY); doing that migration is a separate PR.
      ///
      /// Sign convention: assemble_J_FE_one_multipole returns the
      /// positive bare integral; assemble_K_FE_one_multipole returns the
      /// positive bare K (no HF minus -- caller flips sign if needed,
      /// matches NAORadialBasis::exchange_radial).
      ///
      /// Implementations recompute the per-element radial_integral /
      /// twoe_integral on each call -- adequate for occasional NAO usage
      /// but expensive for an SCF loop. TwoDBasis (which calls these
      /// every SCF iteration) should cache per-element integrals
      /// externally and... [follow-on PR will add caching-aware
      /// overloads].

      /// Per-element integral accessor: given an element index `iel`,
      /// returns the per-element matrix for the multipole the caller has
      /// in mind. Used by the cached overloads below so that consumers
      /// with their own per-(L, iel) integral caches (e.g. TwoDBasis,
      /// driven from an SCF loop) avoid recomputing on every call.
      // Phase 2c: accessor types now return Eigen Matrix references,
      // matching the new TwoDBasis cache types (std::vector<helfem::Matrix>).
      template <typename T = double>
      using PerElementAccessorT =
          std::function<const helfem::Mat<T> & (size_t iel)>;
      using PerElementAccessor = PerElementAccessorT<double>;

      // Every entry point below is templated on the scalar type T, which is
      // deduced from `radial` (a FEMRadialBasisT<T>) and from the density
      // matrix. The per-element accessors are taken as GENERIC callables
      // rather than as a fixed std::function type: template argument
      // deduction does not see through the user-defined conversion from a
      // lambda to std::function, so a std::function parameter would break
      // every existing call site. As a bonus the lambdas are now called
      // directly, with no type-erasure indirection.

      // -- Uncached entry points --------------------------------------

      template <typename T>
      helfem::Mat<T> assemble_J_FE_one_multipole(
          const FEMRadialBasisT<T> & radial, int L, const helfem::Mat<T> & P_FE);

      template <typename T>
      helfem::Mat<T> assemble_K_FE_one_multipole(
          const FEMRadialBasisT<T> & radial, int L, const helfem::Mat<T> & P_FE);

      template <typename T>
      helfem::Mat<T> assemble_J_FE_one_multipole_yukawa(
          const FEMRadialBasisT<T> & radial, int L, helfem::NonDeduced<T> lambda,
          const helfem::Mat<T> & P_FE);

      template <typename T>
      helfem::Mat<T> assemble_K_FE_one_multipole_yukawa(
          const FEMRadialBasisT<T> & radial, int L, helfem::NonDeduced<T> lambda,
          const helfem::Mat<T> & P_FE);

      // -- Cached entry points ----------------------------------------

      /// Same as assemble_J_FE_one_multipole, but the per-element radial
      /// integrals r_small (= radial_integral(L, iel) for bare Coulomb,
      /// or bessel_il_integral(L, lambda, iel) for Yukawa), r_big
      /// (= radial_integral(-L-1, iel) or bessel_kl_integral), and
      /// twoe_in_element (= twoe_integral(L, iel) or yukawa_integral) are
      /// supplied by the caller via accessors. Same assembly, no
      /// recomputation -- meant for SCF inner-loop callers.
      ///
      /// As above for K: `ktei_in_element` returns the EXCHANGE-PERMUTED
      /// in-element tensor (i.e.
      ///   utils::exchange_tei(twoe_in_element(iel), Ni, Ni, Ni, Ni))
      /// so SCF-driven callers (TwoDBasis, which precomputes prim_ktei)
      /// can pass the cached permuted form directly with zero overhead.

      /// Per-element integral accessor: given an element index `iel`,
      /// returns the per-element matrix for the multipole the caller has
      /// in mind. Kept as a named alias for callers who want to store one;
      /// the assemblers themselves take any callable with this shape.
      template <typename T = double>
      using PerElementPairAccessorT =
          std::function<const helfem::Mat<T> & (size_t iel, size_t jel)>;
      using PerElementPairAccessor = PerElementPairAccessorT<double>;

      // Inline implementations -- header-only to match the rest of the
      // libhelfem/include/ NAO surface.

      namespace detail_fe_2e {

        template <typename T>
        inline helfem::Mat<T> r_small(const FEMRadialBasisT<T> & fem, int L, size_t iel,
                                      bool yukawa, T lambda) {
          return yukawa ? fem.bessel_il_integral(L, lambda, iel)
                        : fem.radial_integral(L, iel);
        }
        template <typename T>
        inline helfem::Mat<T> r_big(const FEMRadialBasisT<T> & fem, int L, size_t iel,
                                    bool yukawa, T lambda) {
          return yukawa ? fem.bessel_kl_integral(L, lambda, iel)
                        : fem.radial_integral(-L - 1, iel);
        }
        template <typename T>
        inline helfem::Mat<T> in_element_tei(const FEMRadialBasisT<T> & fem, int L,
                                             size_t iel,
                                             bool yukawa, T lambda) {
          return yukawa ? fem.yukawa_integral(L, lambda, iel)
                        : fem.twoe_integral(L, iel);
        }

      } // namespace detail_fe_2e

      // -- Per-element cache construction helpers ---------------------
      //
      // Convention: caches are flat vectors with index = L*Nel + iel
      // (or L*Nel*Nel + iel*Nel + jel for cross-element). Matches the
      // existing TwoDBasis cache layouts exactly.

      /// Per-(L, iel) disjoint radial integrals for the multipole
      /// decomposition.
      ///   bare:    disjoint_small[L*Nel+iel] = radial.radial_integral(L, iel)
      ///            disjoint_big  [L*Nel+iel] = radial.radial_integral(-L-1, iel)
      ///   Yukawa:  disjoint_small = radial.bessel_il_integral(L, lambda, iel)
      ///            disjoint_big   = radial.bessel_kl_integral(L, lambda, iel)
      template <typename T>
      inline void compute_disjoint_radial_integrals(
          const FEMRadialBasisT<T> & radial,
          int N_L,
          std::vector<helfem::Mat<T>> & disjoint_small,
          std::vector<helfem::Mat<T>> & disjoint_big,
          bool yukawa = false,
          helfem::NonDeduced<T> lambda = T(0)) {
        const size_t Nel = radial.Nel();
        disjoint_small.resize((size_t) N_L * Nel);
        disjoint_big  .resize((size_t) N_L * Nel);
        for (int L = 0; L < N_L; ++L) {
          for (size_t iel = 0; iel < Nel; ++iel) {
            disjoint_small[(size_t) L * Nel + iel] =
                detail_fe_2e::r_small<T>(radial, L, iel, yukawa, lambda);
            disjoint_big  [(size_t) L * Nel + iel] =
                detail_fe_2e::r_big<T>  (radial, L, iel, yukawa, lambda);
          }
        }
      }

      /// Per-(L, iel) in-element 4-index 2e integrals. Only diagonal
      /// (iel == iel) entries are populated; cross-element pieces are
      /// assembled on the fly from the disjoint factors in the
      /// assemble_*_FE_one_multipole_cached path. Layout:
      ///   prim_tei[L*Nel*Nel + iel*Nel + iel] = in-element 4-index tensor
      template <typename T>
      inline void compute_in_element_tei(
          const FEMRadialBasisT<T> & radial,
          int N_L,
          std::vector<helfem::Mat<T>> & prim_tei,
          bool yukawa = false,
          helfem::NonDeduced<T> lambda = T(0)) {
        const size_t Nel = radial.Nel();
        prim_tei.resize(Nel * Nel * (size_t) N_L);
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
        for (int L = 0; L < N_L; ++L) {
          for (size_t iel = 0; iel < Nel; ++iel) {
            prim_tei[Nel * Nel * (size_t) L + iel * Nel + iel] =
                detail_fe_2e::in_element_tei<T>(radial, L, iel, yukawa, lambda);
          }
        }
      }

      /// Apply utils::exchange_tei to each in-element prim_tei entry to
      /// produce the exchange-permuted form (prim_ktei) used by the
      /// cached K assembly. Only diagonal entries.
      template <typename T>
      inline void compute_in_element_ktei_from_tei(
          const FEMRadialBasisT<T> & radial,
          int N_L,
          const std::vector<helfem::Mat<T>> & prim_tei,
          std::vector<helfem::Mat<T>> & prim_ktei) {
        const size_t Nel = radial.Nel();
        prim_ktei.resize(Nel * Nel * (size_t) N_L);
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
        for (int L = 0; L < N_L; ++L) {
          for (size_t iel = 0; iel < Nel; ++iel) {
            const size_t Ni = radial.Nprim(iel);
            prim_ktei[Nel * Nel * (size_t) L + iel * Nel + iel] =
                helfem::utils::exchange_tei<T>(
                    prim_tei[Nel * Nel * (size_t) L + iel * Nel + iel],
                    Ni, Ni, Ni, Ni);
          }
        }
      }

      /// Erfc-kernel cross-element ktei cache:
      ///   rs_ktei[L*Nel*Nel + iel*Nel + jel]
      ///     = exchange-permuted erfc 4-index tensor for (iel, jel).
      /// The erfc kernel does NOT factorise into disjoint small/big
      /// factors, so all Nel*Nel pairs are stored explicitly.
      template <typename T>
      inline void compute_erfc_ktei(
          const FEMRadialBasisT<T> & radial,
          int N_L,
          helfem::NonDeduced<T> mu,
          std::vector<helfem::Mat<T>> & rs_ktei) {
        const size_t Nel = radial.Nel();
        rs_ktei.resize(Nel * Nel * (size_t) N_L);
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
        for (int L = 0; L < N_L; ++L) {
          for (size_t iel = 0; iel < Nel; ++iel) {
            for (size_t jel = 0; jel < Nel; ++jel) {
              const size_t Ni = radial.Nprim(iel);
              const size_t Nj = radial.Nprim(jel);
              rs_ktei[Nel * Nel * (size_t) L + iel * Nel + jel] =
                  helfem::utils::exchange_tei<T>(
                      radial.erfc_integral(L, mu, iel, jel),
                      Ni, Ni, Nj, Nj);
            }
          }
        }
      }

      // The core J / K assemblers operate on per-element accessors, so
      // the same body backs both the uncached (recompute-on-the-fly,
      // NAO use) and cached (look-up-from-SCF-cache, TwoDBasis use)
      // entry points.

      template <typename T, typename RSmall, typename RBig, typename TwoE>
      inline helfem::Mat<T> assemble_J_FE_one_multipole_cached(
          const FEMRadialBasisT<T> & radial,
          const RSmall & r_small,
          const RBig & r_big,
          const TwoE & twoe_in_element,
          const helfem::Mat<T> & P_FE) {
        const size_t Nel  = radial.Nel();
        const Eigen::Index Nrad = (Eigen::Index) radial.Nbf();
        const helfem::Mat<T> & P_E = P_FE;
        helfem::Mat<T> J_E = helfem::Mat<T>::Zero(Nrad, Nrad);
        for (size_t jel = 0; jel < Nel; ++jel) {
          size_t jfirst, jlast;
          radial.get_idx(jel, jfirst, jlast);
          const Eigen::Index Nj = (Eigen::Index)(jlast - jfirst + 1);
          const helfem::Mat<T> Psub =
              P_E.block((Eigen::Index) jfirst, (Eigen::Index) jfirst, Nj, Nj);
          const T jsmall = (r_small(jel) * Psub).trace();
          const T jbig   = (r_big  (jel) * Psub).trace();
          for (size_t iel = 0; iel < jel; ++iel) {
            size_t ifirst, ilast;
            radial.get_idx(iel, ifirst, ilast);
            const Eigen::Index Ni = (Eigen::Index)(ilast - ifirst + 1);
            J_E.block((Eigen::Index) ifirst, (Eigen::Index) ifirst, Ni, Ni)
                += jbig * r_small(iel);
          }
          for (size_t iel = jel + 1; iel < Nel; ++iel) {
            size_t ifirst, ilast;
            radial.get_idx(iel, ifirst, ilast);
            const Eigen::Index Ni = (Eigen::Index)(ilast - ifirst + 1);
            J_E.block((Eigen::Index) ifirst, (Eigen::Index) ifirst, Ni, Ni)
                += jsmall * r_big(iel);
          }
          // In-element 4-index contribution: J_sub = twoe * vec(Psub),
          // then reshape back to Nj x Nj.
          const Eigen::Map<const helfem::Vec<T>> Psub_v(Psub.data(), Nj * Nj);
          const helfem::Vec<T> Jsub_v = twoe_in_element(jel) * Psub_v;
          J_E.block((Eigen::Index) jfirst, (Eigen::Index) jfirst, Nj, Nj)
              += Eigen::Map<const helfem::Mat<T>>(Jsub_v.data(), Nj, Nj);
        }
        return J_E;
      }

      template <typename T, typename RSmall, typename RBig, typename KTei>
      inline helfem::Mat<T> assemble_K_FE_one_multipole_cached(
          const FEMRadialBasisT<T> & radial,
          const RSmall & r_small,
          const RBig & r_big,
          const KTei & ktei_in_element,
          const helfem::Mat<T> & P_FE) {
        const size_t Nel  = radial.Nel();
        const Eigen::Index Nrad = (Eigen::Index) radial.Nbf();
        const helfem::Mat<T> & P_E = P_FE;
        helfem::Mat<T> K_E = helfem::Mat<T>::Zero(Nrad, Nrad);
        for (size_t iel = 0; iel < Nel; ++iel) {
          size_t ifirst, ilast;
          radial.get_idx(iel, ifirst, ilast);
          const Eigen::Index Ni = (Eigen::Index)(ilast - ifirst + 1);
          for (size_t jel = 0; jel < Nel; ++jel) {
            size_t jfirst, jlast;
            radial.get_idx(jel, jfirst, jlast);
            const Eigen::Index Nj = (Eigen::Index)(jlast - jfirst + 1);
            if (iel == jel) {
              const helfem::Mat<T> Psub =
                  P_E.block((Eigen::Index) ifirst, (Eigen::Index) jfirst, Ni, Nj);
              const Eigen::Map<const helfem::Vec<T>> Psub_v(Psub.data(), Ni * Nj);
              const helfem::Vec<T> Ksub_v = ktei_in_element(iel) * Psub_v;
              K_E.block((Eigen::Index) ifirst, (Eigen::Index) jfirst, Ni, Nj)
                  += Eigen::Map<const helfem::Mat<T>>(Ksub_v.data(), Ni, Nj);
            } else {
              const helfem::Mat<T> & iint = (iel > jel) ? r_big(iel) : r_small(iel);
              const helfem::Mat<T> & jint = (iel > jel) ? r_small(jel) : r_big(jel);
              const helfem::Mat<T> Psub =
                  P_E.block((Eigen::Index) ifirst, (Eigen::Index) jfirst, Ni, Nj);
              K_E.block((Eigen::Index) ifirst, (Eigen::Index) jfirst, Ni, Nj)
                  += iint * (Psub * jint.transpose());
            }
          }
        }
        return K_E;
      }

      /// Per-(iel, jel) accessor variant of the above K helper, for
      /// kernels whose r1/r2 coupling does NOT factorise into a
      /// small/big disjoint product (in particular: the error-function
      /// kernel used by compute_erfc / rs_exchange). Every element pair
      /// gets its own dense ktei from the cache. O(Nel^2) per call.
      template <typename T, typename KTeiPair>
      inline helfem::Mat<T> assemble_K_FE_one_multipole_cached_pairwise(
          const FEMRadialBasisT<T> & radial,
          const KTeiPair & ktei_pairwise,
          const helfem::Mat<T> & P_FE) {
        const size_t Nel  = radial.Nel();
        const Eigen::Index Nrad = (Eigen::Index) radial.Nbf();
        const helfem::Mat<T> & P_E = P_FE;
        helfem::Mat<T> K_E = helfem::Mat<T>::Zero(Nrad, Nrad);
        for (size_t iel = 0; iel < Nel; ++iel) {
          size_t ifirst, ilast;
          radial.get_idx(iel, ifirst, ilast);
          const Eigen::Index Ni = (Eigen::Index)(ilast - ifirst + 1);
          for (size_t jel = 0; jel < Nel; ++jel) {
            size_t jfirst, jlast;
            radial.get_idx(jel, jfirst, jlast);
            const Eigen::Index Nj = (Eigen::Index)(jlast - jfirst + 1);
            const helfem::Mat<T> Psub =
                P_E.block((Eigen::Index) ifirst, (Eigen::Index) jfirst, Ni, Nj);
            const Eigen::Map<const helfem::Vec<T>> Psub_v(Psub.data(), Ni * Nj);
            const helfem::Vec<T> Ksub_v = ktei_pairwise(iel, jel) * Psub_v;
            K_E.block((Eigen::Index) ifirst, (Eigen::Index) jfirst, Ni, Nj)
                += Eigen::Map<const helfem::Mat<T>>(Ksub_v.data(), Ni, Nj);
          }
        }
        return K_E;
      }

      // --- Cholesky-factored cached helpers -----------------------------
      // Same FE structure as the dense cached J / K helpers above, but
      // the in-element 4-index contraction is rewritten as a sum of
      // outer products over the Cholesky rank. Both take the SAME
      // per-element J-ordered Cholesky factor of shape (Ni^2 x r) with
      //     T(ab, cd) = Sum_p factor(ab, p) * factor(cd, p),
      // built via FEMRadialBasisT::twoe_integral_cholesky.

      template <typename T, typename RSmall, typename RBig, typename Chol>
      inline helfem::Mat<T> assemble_J_FE_one_multipole_cached_chol(
          const FEMRadialBasisT<T> & radial,
          const RSmall & r_small,
          const RBig & r_big,
          const Chol & twoe_chol_J,
          const helfem::Mat<T> & P_FE) {
        const size_t Nel  = radial.Nel();
        const Eigen::Index Nrad = (Eigen::Index) radial.Nbf();
        const helfem::Mat<T> & P_E = P_FE;
        helfem::Mat<T> J_E = helfem::Mat<T>::Zero(Nrad, Nrad);
        for (size_t jel = 0; jel < Nel; ++jel) {
          size_t jfirst, jlast;
          radial.get_idx(jel, jfirst, jlast);
          const Eigen::Index Nj = (Eigen::Index)(jlast - jfirst + 1);
          const helfem::Mat<T> Psub =
              P_E.block((Eigen::Index) jfirst, (Eigen::Index) jfirst, Nj, Nj);
          const T jsmall = (r_small(jel) * Psub).trace();
          const T jbig   = (r_big  (jel) * Psub).trace();
          for (size_t iel = 0; iel < jel; ++iel) {
            size_t ifirst, ilast;
            radial.get_idx(iel, ifirst, ilast);
            const Eigen::Index Ni = (Eigen::Index)(ilast - ifirst + 1);
            J_E.block((Eigen::Index) ifirst, (Eigen::Index) ifirst, Ni, Ni)
                += jbig * r_small(iel);
          }
          for (size_t iel = jel + 1; iel < Nel; ++iel) {
            size_t ifirst, ilast;
            radial.get_idx(iel, ifirst, ilast);
            const Eigen::Index Ni = (Eigen::Index)(ilast - ifirst + 1);
            J_E.block((Eigen::Index) ifirst, (Eigen::Index) ifirst, Ni, Ni)
                += jsmall * r_big(iel);
          }
          // Factored in-element contribution.
          //   J_{ab} = Sum_p L(ab, p) . <L_p, P>
          // with the (ab) pair packed column-major (a-fast, b-slow) so
          // vec(P) (Eigen also column-major) matches L's row layout.
          const helfem::Mat<T> & L = twoe_chol_J(jel);
          const Eigen::Map<const helfem::Vec<T>> Psub_v(Psub.data(), Nj * Nj);
          const helfem::Vec<T> scalars = L.transpose() * Psub_v;  // length r
          const helfem::Vec<T> Jsub_v  = L * scalars;             // length Nj^2
          J_E.block((Eigen::Index) jfirst, (Eigen::Index) jfirst, Nj, Nj)
              += Eigen::Map<const helfem::Mat<T>>(Jsub_v.data(), Nj, Nj);
        }
        return J_E;
      }

      template <typename T, typename RSmall, typename RBig, typename Chol>
      inline helfem::Mat<T> assemble_K_FE_one_multipole_cached_chol(
          const FEMRadialBasisT<T> & radial,
          const RSmall & r_small,
          const RBig & r_big,
          const Chol & twoe_chol_J,
          const helfem::Mat<T> & P_FE) {
        const size_t Nel  = radial.Nel();
        const Eigen::Index Nrad = (Eigen::Index) radial.Nbf();
        const helfem::Mat<T> & P_E = P_FE;
        helfem::Mat<T> K_E = helfem::Mat<T>::Zero(Nrad, Nrad);
        for (size_t iel = 0; iel < Nel; ++iel) {
          size_t ifirst, ilast;
          radial.get_idx(iel, ifirst, ilast);
          const Eigen::Index Ni = (Eigen::Index)(ilast - ifirst + 1);
          for (size_t jel = 0; jel < Nel; ++jel) {
            size_t jfirst, jlast;
            radial.get_idx(jel, jfirst, jlast);
            const Eigen::Index Nj = (Eigen::Index)(jlast - jfirst + 1);
            if (iel == jel) {
              // Factored in-element K via the J-ordered Cholesky factor:
              //   K_{a,c} = Sum_p (M_p . P . M_p^T)_{a,c}
              // with M_p = reshape(L.col(p), Ni, Ni) (column-major so
              // M_p(a, b) = L(a + b*Ni, p), matching the (ab) row pair
              // in twoe_integral).
              const helfem::Mat<T> & L = twoe_chol_J(iel);
              const helfem::Mat<T> Psub =
                  P_E.block((Eigen::Index) ifirst, (Eigen::Index) jfirst, Ni, Nj);
              helfem::Mat<T> Ksub = helfem::Mat<T>::Zero(Ni, Nj);
              for (Eigen::Index p = 0; p < L.cols(); ++p) {
                Eigen::Map<const helfem::Mat<T>> Mp(L.col(p).data(), Ni, Ni);
                Ksub += Mp * (Psub * Mp.transpose());
              }
              K_E.block((Eigen::Index) ifirst, (Eigen::Index) jfirst, Ni, Nj)
                  += Ksub;
            } else {
              const helfem::Mat<T> & iint = (iel > jel) ? r_big(iel) : r_small(iel);
              const helfem::Mat<T> & jint = (iel > jel) ? r_small(jel) : r_big(jel);
              const helfem::Mat<T> Psub =
                  P_E.block((Eigen::Index) ifirst, (Eigen::Index) jfirst, Ni, Nj);
              K_E.block((Eigen::Index) ifirst, (Eigen::Index) jfirst, Ni, Nj)
                  += iint * (Psub * jint.transpose());
            }
          }
        }
        return K_E;
      }

      // The uncached entry points just wire on-the-fly accessors that
      // call the FE primitives directly.

      template <typename T>
      inline helfem::Mat<T> assemble_J_FE_one_multipole(
          const FEMRadialBasisT<T> & radial, int L, const helfem::Mat<T> & P_FE) {
        // Per-call temporaries to keep the accessor lambdas returning a
        // stable const reference (storing each computed matrix in a
        // capture-list scratchpad).
        std::vector<helfem::Mat<T>> scratch_small(radial.Nel());
        std::vector<helfem::Mat<T>> scratch_big  (radial.Nel());
        std::vector<helfem::Mat<T>> scratch_twoe (radial.Nel());
        auto rs = [&](size_t iel) -> const helfem::Mat<T> & {
          if (scratch_small[iel].size() == 0)
            scratch_small[iel] = detail_fe_2e::r_small<T>(radial, L, iel, false, T(0));
          return scratch_small[iel];
        };
        auto rb = [&](size_t iel) -> const helfem::Mat<T> & {
          if (scratch_big[iel].size() == 0)
            scratch_big[iel] = detail_fe_2e::r_big<T>(radial, L, iel, false, T(0));
          return scratch_big[iel];
        };
        auto tw = [&](size_t iel) -> const helfem::Mat<T> & {
          if (scratch_twoe[iel].size() == 0)
            scratch_twoe[iel] = detail_fe_2e::in_element_tei<T>(radial, L, iel, false, T(0));
          return scratch_twoe[iel];
        };
        return assemble_J_FE_one_multipole_cached(radial, rs, rb, tw, P_FE);
      }

      template <typename T>
      inline helfem::Mat<T> assemble_K_FE_one_multipole(
          const FEMRadialBasisT<T> & radial, int L, const helfem::Mat<T> & P_FE) {
        std::vector<helfem::Mat<T>> scratch_small(radial.Nel());
        std::vector<helfem::Mat<T>> scratch_big  (radial.Nel());
        std::vector<helfem::Mat<T>> scratch_ktei (radial.Nel());
        auto rs = [&](size_t iel) -> const helfem::Mat<T> & {
          if (scratch_small[iel].size() == 0)
            scratch_small[iel] = detail_fe_2e::r_small<T>(radial, L, iel, false, T(0));
          return scratch_small[iel];
        };
        auto rb = [&](size_t iel) -> const helfem::Mat<T> & {
          if (scratch_big[iel].size() == 0)
            scratch_big[iel] = detail_fe_2e::r_big<T>(radial, L, iel, false, T(0));
          return scratch_big[iel];
        };
        auto kt = [&](size_t iel) -> const helfem::Mat<T> & {
          if (scratch_ktei[iel].size() == 0) {
            size_t ifirst, ilast;
            radial.get_idx(iel, ifirst, ilast);
            const size_t Ni = ilast - ifirst + 1;
            scratch_ktei[iel] = utils::exchange_tei<T>(
                detail_fe_2e::in_element_tei<T>(radial, L, iel, false, T(0)),
                Ni, Ni, Ni, Ni);
          }
          return scratch_ktei[iel];
        };
        return assemble_K_FE_one_multipole_cached(radial, rs, rb, kt, P_FE);
      }

      template <typename T>
      inline helfem::Mat<T> assemble_J_FE_one_multipole_yukawa(
          const FEMRadialBasisT<T> & radial, int L, helfem::NonDeduced<T> lambda,
          const helfem::Mat<T> & P_FE) {
        std::vector<helfem::Mat<T>> scratch_small(radial.Nel());
        std::vector<helfem::Mat<T>> scratch_big  (radial.Nel());
        std::vector<helfem::Mat<T>> scratch_twoe (radial.Nel());
        auto rs = [&](size_t iel) -> const helfem::Mat<T> & {
          if (scratch_small[iel].size() == 0)
            scratch_small[iel] = detail_fe_2e::r_small<T>(radial, L, iel, true, lambda);
          return scratch_small[iel];
        };
        auto rb = [&](size_t iel) -> const helfem::Mat<T> & {
          if (scratch_big[iel].size() == 0)
            scratch_big[iel] = detail_fe_2e::r_big<T>(radial, L, iel, true, lambda);
          return scratch_big[iel];
        };
        auto tw = [&](size_t iel) -> const helfem::Mat<T> & {
          if (scratch_twoe[iel].size() == 0)
            scratch_twoe[iel] = detail_fe_2e::in_element_tei<T>(radial, L, iel, true, lambda);
          return scratch_twoe[iel];
        };
        return assemble_J_FE_one_multipole_cached(radial, rs, rb, tw, P_FE);
      }

      template <typename T>
      inline helfem::Mat<T> assemble_K_FE_one_multipole_yukawa(
          const FEMRadialBasisT<T> & radial, int L, helfem::NonDeduced<T> lambda,
          const helfem::Mat<T> & P_FE) {
        std::vector<helfem::Mat<T>> scratch_small(radial.Nel());
        std::vector<helfem::Mat<T>> scratch_big  (radial.Nel());
        std::vector<helfem::Mat<T>> scratch_ktei (radial.Nel());
        auto rs = [&](size_t iel) -> const helfem::Mat<T> & {
          if (scratch_small[iel].size() == 0)
            scratch_small[iel] = detail_fe_2e::r_small<T>(radial, L, iel, true, lambda);
          return scratch_small[iel];
        };
        auto rb = [&](size_t iel) -> const helfem::Mat<T> & {
          if (scratch_big[iel].size() == 0)
            scratch_big[iel] = detail_fe_2e::r_big<T>(radial, L, iel, true, lambda);
          return scratch_big[iel];
        };
        auto kt = [&](size_t iel) -> const helfem::Mat<T> & {
          if (scratch_ktei[iel].size() == 0) {
            size_t ifirst, ilast;
            radial.get_idx(iel, ifirst, ilast);
            const size_t Ni = ilast - ifirst + 1;
            scratch_ktei[iel] = utils::exchange_tei<T>(
                detail_fe_2e::in_element_tei<T>(radial, L, iel, true, lambda),
                Ni, Ni, Ni, Ni);
          }
          return scratch_ktei[iel];
        };
        return assemble_K_FE_one_multipole_cached(radial, rs, rb, kt, P_FE);
      }
    } // namespace basis
  } // namespace atomic
} // namespace helfem

#endif
