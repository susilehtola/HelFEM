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
#include "ArmaEigen.h"
#include <armadillo>
#include <functional>

namespace helfem {
  // Forward declaration of the existing in-tree helper from
  // libhelfem/src/utils.cpp (its header is in libhelfem/src/, not on the
  // public include path; the symbol is resolved at link time -- callers
  // of these helpers link against libhelfem anyway).
  namespace utils {
    arma::mat exchange_tei(const arma::mat & tei, size_t Ni, size_t Nj,
                            size_t Nk, size_t Nl);
    /// Eigen overload (Phase 2c wrap-up): same (ij|kl) -> (jk|il) permutation
    /// for helfem::Matrix-typed in-element TEIs, avoids the
    /// to_eigen(to_arma(...)) round-trip the helpers used to do.
    helfem::Matrix exchange_tei(const helfem::Matrix & tei, size_t Ni, size_t Nj,
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
      using PerElementAccessor =
          std::function<const helfem::Matrix & (size_t iel)>;

      // -- Uncached entry points --------------------------------------

      helfem::Matrix assemble_J_FE_one_multipole(
          const FEMRadialBasis & radial, int L, const helfem::Matrix & P_FE);

      helfem::Matrix assemble_K_FE_one_multipole(
          const FEMRadialBasis & radial, int L, const helfem::Matrix & P_FE);

      helfem::Matrix assemble_J_FE_one_multipole_yukawa(
          const FEMRadialBasis & radial, int L, double lambda,
          const helfem::Matrix & P_FE);

      helfem::Matrix assemble_K_FE_one_multipole_yukawa(
          const FEMRadialBasis & radial, int L, double lambda,
          const helfem::Matrix & P_FE);

      // -- Cached entry points ----------------------------------------

      /// Same as assemble_J_FE_one_multipole, but the per-element radial
      /// integrals r_small (= radial_integral(L, iel) for bare Coulomb,
      /// or bessel_il_integral(L, lambda, iel) for Yukawa), r_big
      /// (= radial_integral(-L-1, iel) or bessel_kl_integral), and
      /// twoe_in_element (= twoe_integral(L, iel) or yukawa_integral) are
      /// supplied by the caller via accessors. Same assembly, no
      /// recomputation -- meant for SCF inner-loop callers.
      helfem::Matrix assemble_J_FE_one_multipole_cached(
          const FEMRadialBasis & radial,
          const PerElementAccessor & r_small,
          const PerElementAccessor & r_big,
          const PerElementAccessor & twoe_in_element,
          const helfem::Matrix & P_FE);

      /// As above for K. Note `ktei_in_element` here returns the
      /// EXCHANGE-PERMUTED in-element tensor (i.e.
      ///   utils::exchange_tei(twoe_in_element(iel), Ni, Ni, Ni, Ni))
      /// so SCF-driven callers (TwoDBasis, which precomputes prim_ktei)
      /// can pass the cached permuted form directly with zero overhead.
      helfem::Matrix assemble_K_FE_one_multipole_cached(
          const FEMRadialBasis & radial,
          const PerElementAccessor & r_small,
          const PerElementAccessor & r_big,
          const PerElementAccessor & ktei_in_element,
          const helfem::Matrix & P_FE);

      /// Per-(iel, jel) accessor variant of the above K helper, for
      /// kernels whose r1/r2 coupling does NOT factorise into a
      /// small/big disjoint product (in particular: the error-function
      /// kernel used by HelFEM's compute_erfc / rs_exchange). Every
      /// element pair gets its own dense ktei from the cache, then the
      /// standard K contraction is applied. No disjoint optimisation
      /// available -- O(Nel^2) per call.
      using PerElementPairAccessor =
          std::function<const helfem::Matrix & (size_t iel, size_t jel)>;
      helfem::Matrix assemble_K_FE_one_multipole_cached_pairwise(
          const FEMRadialBasis & radial,
          const PerElementPairAccessor & ktei_pairwise,
          const helfem::Matrix & P_FE);

      /// Cholesky-factored variants of the cached J / K helpers. Both
      /// take the SAME per-element J-ordered Cholesky factor of shape
      /// (Ni^2 x r) such that
      ///     T(ab, cd) = twoe_integral(L, iel)(ab, cd)
      ///              = Sum_p factor(ab, p) * factor(cd, p)
      /// where (ab) is the row pair (vec(.) column-major; a-fast b-slow).
      /// Build via FEMRadialBasis::twoe_integral_cholesky.
      ///
      /// Asymmetric cost picture (per element):
      ///   J: O(r * Ni^2) -- inner product per Cholesky vector then
      ///                     scalar * outer; ~4x FASTER than dense
      ///                     matvec at typical FE rank ~ 2*Ni.
      ///   K: O(r * Ni^3) -- two matmuls per Cholesky vector via
      ///                     K = Sum_p M_p . P . M_p^T (M_p = reshape
      ///                     of the p-th Cholesky column to Ni x Ni).
      ///                     SLOWER than dense K (O(Ni^4)) for typical
      ///                     FE rank-vs-Ni ratios; the K-PERMUTED tensor
      ///                     happens to be ~full-rank (the bivariate
      ///                     space u_a(r1) u_c(r2) is Ni^2-dimensional),
      ///                     so K-side Cholesky compression buys nothing.
      ///
      /// Why expose the K factored form anyway: it is the canonical
      /// RI / density-fitting representation expected by external
      /// drivers (PySCF's DF backend forms K from the same (mu nu | P)
      /// 3-index factor that backs J). Memory-wise, callers who store
      /// only the J-ordered Cholesky factor save ~Ni^2/r ~ 7x relative
      /// to caching the full in-element tensor; the slower K contraction
      /// is the price paid for that compression.
      ///
      /// Cross-element pieces are unchanged -- still use the disjoint
      /// r_small / r_big factorisation. Only the in-element 4-index
      /// path is factored.
      helfem::Matrix assemble_J_FE_one_multipole_cached_chol(
          const FEMRadialBasis & radial,
          const PerElementAccessor & r_small,
          const PerElementAccessor & r_big,
          const PerElementAccessor & twoe_chol_J,
          const helfem::Matrix & P_FE);

      helfem::Matrix assemble_K_FE_one_multipole_cached_chol(
          const FEMRadialBasis & radial,
          const PerElementAccessor & r_small,
          const PerElementAccessor & r_big,
          const PerElementAccessor & twoe_chol_J,
          const helfem::Matrix & P_FE);

      // Inline implementations -- header-only to match the rest of the
      // libhelfem/include/ NAO surface. ~150 LOC total.

      namespace detail_fe_2e {

        // Phase 2c: caches are std::vector<helfem::Matrix>; the bridges
        // these used to do are no longer needed -- direct return.
        inline helfem::Matrix r_small(const FEMRadialBasis & fem, int L, size_t iel,
                                       bool yukawa, double lambda) {
          return yukawa ? fem.bessel_il_integral(L, lambda, iel)
                        : fem.radial_integral(L, iel);
        }
        inline helfem::Matrix r_big(const FEMRadialBasis & fem, int L, size_t iel,
                                     bool yukawa, double lambda) {
          return yukawa ? fem.bessel_kl_integral(L, lambda, iel)
                        : fem.radial_integral(-L - 1, iel);
        }
        inline helfem::Matrix in_element_tei(const FEMRadialBasis & fem, int L,
                                              size_t iel,
                                              bool yukawa, double lambda) {
          return yukawa ? fem.yukawa_integral(L, lambda, iel)
                        : fem.twoe_integral(L, iel);
        }

      } // namespace detail_fe_2e

      // -- Per-element cache construction helpers ---------------------
      //
      // The cached assembly helpers below
      // (assemble_*_FE_one_multipole_cached) take per-(L, iel) integrals
      // via accessor lambdas; SCF-driven callers (atomic::TwoDBasis,
      // sadatom::TwoDBasis) populate those caches once per basis change.
      // The fill loops are identical in structure between the two
      // callers, so we share them here.
      //
      // Convention: caches are flat vectors with index = L*Nel + iel
      // (or L*Nel*Nel + iel*Nel + jel for cross-element). Matches the
      // existing TwoDBasis cache layouts exactly so the helpers below
      // can be dropped into existing call sites without changing the
      // surrounding accessor lambdas.

      /// Per-(L, iel) disjoint radial integrals for the multipole
      /// decomposition.
      ///   bare:    disjoint_small[L*Nel+iel] = radial.radial_integral(L, iel)
      ///            disjoint_big  [L*Nel+iel] = radial.radial_integral(-L-1, iel)
      ///   Yukawa:  disjoint_small = radial.bessel_il_integral(L, lambda, iel)
      ///            disjoint_big   = radial.bessel_kl_integral(L, lambda, iel)
      inline void compute_disjoint_radial_integrals(
          const FEMRadialBasis & radial,
          int N_L,
          std::vector<helfem::Matrix> & disjoint_small,
          std::vector<helfem::Matrix> & disjoint_big,
          bool yukawa = false,
          double lambda = 0.0) {
        const size_t Nel = radial.Nel();
        disjoint_small.resize((size_t) N_L * Nel);
        disjoint_big  .resize((size_t) N_L * Nel);
        for (int L = 0; L < N_L; ++L) {
          for (size_t iel = 0; iel < Nel; ++iel) {
            disjoint_small[(size_t) L * Nel + iel] =
                detail_fe_2e::r_small(radial, L, iel, yukawa, lambda);
            disjoint_big  [(size_t) L * Nel + iel] =
                detail_fe_2e::r_big  (radial, L, iel, yukawa, lambda);
          }
        }
      }

      /// Per-(L, iel) in-element 4-index 2e integrals. Only diagonal
      /// (iel == iel) entries are populated; cross-element pieces are
      /// assembled on the fly from the disjoint factors in the
      /// assemble_*_FE_one_multipole_cached path. Layout:
      ///   prim_tei[L*Nel*Nel + iel*Nel + iel] = in-element 4-index tensor
      /// For bare Coulomb (default): radial.twoe_integral(L, iel).
      /// For Yukawa: radial.yukawa_integral(L, lambda, iel).
      inline void compute_in_element_tei(
          const FEMRadialBasis & radial,
          int N_L,
          std::vector<helfem::Matrix> & prim_tei,
          bool yukawa = false,
          double lambda = 0.0) {
        const size_t Nel = radial.Nel();
        prim_tei.resize(Nel * Nel * (size_t) N_L);
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
        for (int L = 0; L < N_L; ++L) {
          for (size_t iel = 0; iel < Nel; ++iel) {
            prim_tei[Nel * Nel * (size_t) L + iel * Nel + iel] =
                detail_fe_2e::in_element_tei(radial, L, iel, yukawa, lambda);
          }
        }
      }

      /// Apply utils::exchange_tei to each in-element prim_tei entry to
      /// produce the exchange-permuted form (prim_ktei) used by the
      /// cached K assembly. Only diagonal entries.
      inline void compute_in_element_ktei_from_tei(
          const FEMRadialBasis & radial,
          int N_L,
          const std::vector<helfem::Matrix> & prim_tei,
          std::vector<helfem::Matrix> & prim_ktei) {
        const size_t Nel = radial.Nel();
        prim_ktei.resize(Nel * Nel * (size_t) N_L);
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
        for (int L = 0; L < N_L; ++L) {
          for (size_t iel = 0; iel < Nel; ++iel) {
            const size_t Ni = radial.Nprim(iel);
            // Phase 2c wrap-up: utils::exchange_tei now has an Eigen
            // overload, so prim_tei (Eigen) -> prim_ktei (Eigen) is
            // direct -- no arma round-trip.
            prim_ktei[Nel * Nel * (size_t) L + iel * Nel + iel] =
                helfem::utils::exchange_tei(
                    prim_tei[Nel * Nel * (size_t) L + iel * Nel + iel],
                    Ni, Ni, Ni, Ni);
          }
        }
      }

      /// Erfc-kernel cross-element ktei cache:
      ///   rs_ktei[L*Nel*Nel + iel*Nel + jel]
      ///     = exchange-permuted erfc 4-index tensor for (iel, jel).
      /// The erfc kernel does NOT factorise into disjoint small/big
      /// factors, so all Nel*Nel pairs are stored explicitly. The
      /// pairwise K assembly path
      /// (assemble_K_FE_one_multipole_cached_pairwise) consumes this
      /// layout.
      inline void compute_erfc_ktei(
          const FEMRadialBasis & radial,
          int N_L,
          double mu,
          std::vector<helfem::Matrix> & rs_ktei) {
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
              // Phase 2c wrap-up: utils::exchange_tei Eigen overload
              // -- direct erfc_integral (Eigen) -> rs_ktei (Eigen).
              rs_ktei[Nel * Nel * (size_t) L + iel * Nel + jel] =
                  helfem::utils::exchange_tei(
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

      // Phase 2c: assemble helpers migrated to Eigen internals so they
      // match the new Eigen-typed accessors and caches. P_FE input and
      // J/K output are still arma (the TwoDBasis SCF assembly is still
      // arma-typed) -- bridged at the entry/exit boundary.
      inline helfem::Matrix assemble_J_FE_one_multipole_cached(
          const FEMRadialBasis & radial,
          const PerElementAccessor & r_small,
          const PerElementAccessor & r_big,
          const PerElementAccessor & twoe_in_element,
          const helfem::Matrix & P_FE) {
        const size_t Nel  = radial.Nel();
        const Eigen::Index Nrad = (Eigen::Index) radial.Nbf();
        const helfem::Matrix & P_E = P_FE;
        helfem::Matrix J_E = helfem::Matrix::Zero(Nrad, Nrad);
        for (size_t jel = 0; jel < Nel; ++jel) {
          size_t jfirst, jlast;
          radial.get_idx(jel, jfirst, jlast);
          const Eigen::Index Nj = (Eigen::Index)(jlast - jfirst + 1);
          const helfem::Matrix Psub =
              P_E.block((Eigen::Index) jfirst, (Eigen::Index) jfirst, Nj, Nj);
          const double jsmall = (r_small(jel) * Psub).trace();
          const double jbig   = (r_big  (jel) * Psub).trace();
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
          const Eigen::Map<const Eigen::VectorXd> Psub_v(Psub.data(), Nj * Nj);
          const Eigen::VectorXd Jsub_v = twoe_in_element(jel) * Psub_v;
          J_E.block((Eigen::Index) jfirst, (Eigen::Index) jfirst, Nj, Nj)
              += Eigen::Map<const Eigen::MatrixXd>(Jsub_v.data(), Nj, Nj);
        }
        return J_E;
      }

      inline helfem::Matrix assemble_K_FE_one_multipole_cached(
          const FEMRadialBasis & radial,
          const PerElementAccessor & r_small,
          const PerElementAccessor & r_big,
          const PerElementAccessor & ktei_in_element,
          const helfem::Matrix & P_FE) {
        const size_t Nel  = radial.Nel();
        const Eigen::Index Nrad = (Eigen::Index) radial.Nbf();
        const helfem::Matrix & P_E = P_FE;
        helfem::Matrix K_E = helfem::Matrix::Zero(Nrad, Nrad);
        for (size_t iel = 0; iel < Nel; ++iel) {
          size_t ifirst, ilast;
          radial.get_idx(iel, ifirst, ilast);
          const Eigen::Index Ni = (Eigen::Index)(ilast - ifirst + 1);
          for (size_t jel = 0; jel < Nel; ++jel) {
            size_t jfirst, jlast;
            radial.get_idx(jel, jfirst, jlast);
            const Eigen::Index Nj = (Eigen::Index)(jlast - jfirst + 1);
            if (iel == jel) {
              const helfem::Matrix Psub =
                  P_E.block((Eigen::Index) ifirst, (Eigen::Index) jfirst, Ni, Nj);
              const Eigen::Map<const Eigen::VectorXd> Psub_v(Psub.data(), Ni * Nj);
              const Eigen::VectorXd Ksub_v = ktei_in_element(iel) * Psub_v;
              K_E.block((Eigen::Index) ifirst, (Eigen::Index) jfirst, Ni, Nj)
                  += Eigen::Map<const Eigen::MatrixXd>(Ksub_v.data(), Ni, Nj);
            } else {
              const helfem::Matrix & iint = (iel > jel) ? r_big(iel) : r_small(iel);
              const helfem::Matrix & jint = (iel > jel) ? r_small(jel) : r_big(jel);
              const helfem::Matrix Psub =
                  P_E.block((Eigen::Index) ifirst, (Eigen::Index) jfirst, Ni, Nj);
              K_E.block((Eigen::Index) ifirst, (Eigen::Index) jfirst, Ni, Nj)
                  += iint * (Psub * jint.transpose());
            }
          }
        }
        return K_E;
      }

      inline helfem::Matrix assemble_K_FE_one_multipole_cached_pairwise(
          const FEMRadialBasis & radial,
          const PerElementPairAccessor & ktei_pairwise,
          const helfem::Matrix & P_FE) {
        const size_t Nel  = radial.Nel();
        const Eigen::Index Nrad = (Eigen::Index) radial.Nbf();
        const helfem::Matrix & P_E = P_FE;
        helfem::Matrix K_E = helfem::Matrix::Zero(Nrad, Nrad);
        for (size_t iel = 0; iel < Nel; ++iel) {
          size_t ifirst, ilast;
          radial.get_idx(iel, ifirst, ilast);
          const Eigen::Index Ni = (Eigen::Index)(ilast - ifirst + 1);
          for (size_t jel = 0; jel < Nel; ++jel) {
            size_t jfirst, jlast;
            radial.get_idx(jel, jfirst, jlast);
            const Eigen::Index Nj = (Eigen::Index)(jlast - jfirst + 1);
            const helfem::Matrix Psub =
                P_E.block((Eigen::Index) ifirst, (Eigen::Index) jfirst, Ni, Nj);
            const Eigen::Map<const Eigen::VectorXd> Psub_v(Psub.data(), Ni * Nj);
            const Eigen::VectorXd Ksub_v = ktei_pairwise(iel, jel) * Psub_v;
            K_E.block((Eigen::Index) ifirst, (Eigen::Index) jfirst, Ni, Nj)
                += Eigen::Map<const Eigen::MatrixXd>(Ksub_v.data(), Ni, Nj);
          }
        }
        return K_E;
      }

      // --- Cholesky-factored cached helpers -----------------------------
      // Same FE structure as the dense cached J / K helpers above, but
      // the in-element 4-index contraction is rewritten as a sum of
      // outer products over the Cholesky rank.

      inline helfem::Matrix assemble_J_FE_one_multipole_cached_chol(
          const FEMRadialBasis & radial,
          const PerElementAccessor & r_small,
          const PerElementAccessor & r_big,
          const PerElementAccessor & twoe_chol_J,
          const helfem::Matrix & P_FE) {
        const size_t Nel  = radial.Nel();
        const Eigen::Index Nrad = (Eigen::Index) radial.Nbf();
        const helfem::Matrix & P_E = P_FE;
        helfem::Matrix J_E = helfem::Matrix::Zero(Nrad, Nrad);
        for (size_t jel = 0; jel < Nel; ++jel) {
          size_t jfirst, jlast;
          radial.get_idx(jel, jfirst, jlast);
          const Eigen::Index Nj = (Eigen::Index)(jlast - jfirst + 1);
          const helfem::Matrix Psub =
              P_E.block((Eigen::Index) jfirst, (Eigen::Index) jfirst, Nj, Nj);
          const double jsmall = (r_small(jel) * Psub).trace();
          const double jbig   = (r_big  (jel) * Psub).trace();
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
          const helfem::Matrix & L = twoe_chol_J(jel);
          const Eigen::Map<const Eigen::VectorXd> Psub_v(Psub.data(), Nj * Nj);
          const Eigen::VectorXd scalars = L.transpose() * Psub_v;  // length r
          const Eigen::VectorXd Jsub_v  = L * scalars;             // length Nj^2
          J_E.block((Eigen::Index) jfirst, (Eigen::Index) jfirst, Nj, Nj)
              += Eigen::Map<const Eigen::MatrixXd>(Jsub_v.data(), Nj, Nj);
        }
        return J_E;
      }

      inline helfem::Matrix assemble_K_FE_one_multipole_cached_chol(
          const FEMRadialBasis & radial,
          const PerElementAccessor & r_small,
          const PerElementAccessor & r_big,
          const PerElementAccessor & twoe_chol_J,
          const helfem::Matrix & P_FE) {
        const size_t Nel  = radial.Nel();
        const Eigen::Index Nrad = (Eigen::Index) radial.Nbf();
        const helfem::Matrix & P_E = P_FE;
        helfem::Matrix K_E = helfem::Matrix::Zero(Nrad, Nrad);
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
              const helfem::Matrix & L = twoe_chol_J(iel);
              const helfem::Matrix Psub =
                  P_E.block((Eigen::Index) ifirst, (Eigen::Index) jfirst, Ni, Nj);
              helfem::Matrix Ksub = helfem::Matrix::Zero(Ni, Nj);
              for (Eigen::Index p = 0; p < L.cols(); ++p) {
                Eigen::Map<const Eigen::MatrixXd> Mp(L.col(p).data(), Ni, Ni);
                Ksub += Mp * (Psub * Mp.transpose());
              }
              K_E.block((Eigen::Index) ifirst, (Eigen::Index) jfirst, Ni, Nj)
                  += Ksub;
            } else {
              const helfem::Matrix & iint = (iel > jel) ? r_big(iel) : r_small(iel);
              const helfem::Matrix & jint = (iel > jel) ? r_small(jel) : r_big(jel);
              const helfem::Matrix Psub =
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

      // Phase 2c: scratchpads are now std::vector<helfem::Matrix>; the
      // is_empty check becomes .size() == 0 (Eigen's empty-by-default).
      inline helfem::Matrix assemble_J_FE_one_multipole(
          const FEMRadialBasis & radial, int L, const helfem::Matrix & P_FE) {
        // Per-call temporaries to keep the accessor lambdas returning a
        // stable const reference (storing each computed matrix in a
        // capture-list scratchpad).
        std::vector<helfem::Matrix> scratch_small(radial.Nel());
        std::vector<helfem::Matrix> scratch_big  (radial.Nel());
        std::vector<helfem::Matrix> scratch_twoe (radial.Nel());
        auto rs = [&](size_t iel) -> const helfem::Matrix & {
          if (scratch_small[iel].size() == 0)
            scratch_small[iel] = detail_fe_2e::r_small(radial, L, iel, false, 0.0);
          return scratch_small[iel];
        };
        auto rb = [&](size_t iel) -> const helfem::Matrix & {
          if (scratch_big[iel].size() == 0)
            scratch_big[iel] = detail_fe_2e::r_big(radial, L, iel, false, 0.0);
          return scratch_big[iel];
        };
        auto tw = [&](size_t iel) -> const helfem::Matrix & {
          if (scratch_twoe[iel].size() == 0)
            scratch_twoe[iel] = detail_fe_2e::in_element_tei(radial, L, iel, false, 0.0);
          return scratch_twoe[iel];
        };
        return assemble_J_FE_one_multipole_cached(radial, rs, rb, tw, P_FE);
      }

      inline helfem::Matrix assemble_K_FE_one_multipole(
          const FEMRadialBasis & radial, int L, const helfem::Matrix & P_FE) {
        std::vector<helfem::Matrix> scratch_small(radial.Nel());
        std::vector<helfem::Matrix> scratch_big  (radial.Nel());
        std::vector<helfem::Matrix> scratch_ktei (radial.Nel());
        auto rs = [&](size_t iel) -> const helfem::Matrix & {
          if (scratch_small[iel].size() == 0)
            scratch_small[iel] = detail_fe_2e::r_small(radial, L, iel, false, 0.0);
          return scratch_small[iel];
        };
        auto rb = [&](size_t iel) -> const helfem::Matrix & {
          if (scratch_big[iel].size() == 0)
            scratch_big[iel] = detail_fe_2e::r_big(radial, L, iel, false, 0.0);
          return scratch_big[iel];
        };
        auto kt = [&](size_t iel) -> const helfem::Matrix & {
          if (scratch_ktei[iel].size() == 0) {
            size_t ifirst, ilast;
            radial.get_idx(iel, ifirst, ilast);
            const size_t Ni = ilast - ifirst + 1;
            scratch_ktei[iel] = utils::exchange_tei(
                detail_fe_2e::in_element_tei(radial, L, iel, false, 0.0),
                Ni, Ni, Ni, Ni);
          }
          return scratch_ktei[iel];
        };
        return assemble_K_FE_one_multipole_cached(radial, rs, rb, kt, P_FE);
      }

      inline helfem::Matrix assemble_J_FE_one_multipole_yukawa(
          const FEMRadialBasis & radial, int L, double lambda,
          const helfem::Matrix & P_FE) {
        std::vector<helfem::Matrix> scratch_small(radial.Nel());
        std::vector<helfem::Matrix> scratch_big  (radial.Nel());
        std::vector<helfem::Matrix> scratch_twoe (radial.Nel());
        auto rs = [&](size_t iel) -> const helfem::Matrix & {
          if (scratch_small[iel].size() == 0)
            scratch_small[iel] = detail_fe_2e::r_small(radial, L, iel, true, lambda);
          return scratch_small[iel];
        };
        auto rb = [&](size_t iel) -> const helfem::Matrix & {
          if (scratch_big[iel].size() == 0)
            scratch_big[iel] = detail_fe_2e::r_big(radial, L, iel, true, lambda);
          return scratch_big[iel];
        };
        auto tw = [&](size_t iel) -> const helfem::Matrix & {
          if (scratch_twoe[iel].size() == 0)
            scratch_twoe[iel] = detail_fe_2e::in_element_tei(radial, L, iel, true, lambda);
          return scratch_twoe[iel];
        };
        return assemble_J_FE_one_multipole_cached(radial, rs, rb, tw, P_FE);
      }

      inline helfem::Matrix assemble_K_FE_one_multipole_yukawa(
          const FEMRadialBasis & radial, int L, double lambda,
          const helfem::Matrix & P_FE) {
        std::vector<helfem::Matrix> scratch_small(radial.Nel());
        std::vector<helfem::Matrix> scratch_big  (radial.Nel());
        std::vector<helfem::Matrix> scratch_ktei (radial.Nel());
        auto rs = [&](size_t iel) -> const helfem::Matrix & {
          if (scratch_small[iel].size() == 0)
            scratch_small[iel] = detail_fe_2e::r_small(radial, L, iel, true, lambda);
          return scratch_small[iel];
        };
        auto rb = [&](size_t iel) -> const helfem::Matrix & {
          if (scratch_big[iel].size() == 0)
            scratch_big[iel] = detail_fe_2e::r_big(radial, L, iel, true, lambda);
          return scratch_big[iel];
        };
        auto kt = [&](size_t iel) -> const helfem::Matrix & {
          if (scratch_ktei[iel].size() == 0) {
            size_t ifirst, ilast;
            radial.get_idx(iel, ifirst, ilast);
            const size_t Ni = ilast - ifirst + 1;
            scratch_ktei[iel] = utils::exchange_tei(
                detail_fe_2e::in_element_tei(radial, L, iel, true, lambda),
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
