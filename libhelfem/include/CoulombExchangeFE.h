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
      using PerElementAccessor =
          std::function<const arma::mat & (size_t iel)>;

      // -- Uncached entry points --------------------------------------

      arma::mat assemble_J_FE_one_multipole(
          const FEMRadialBasis & radial, int L, const arma::mat & P_FE);

      arma::mat assemble_K_FE_one_multipole(
          const FEMRadialBasis & radial, int L, const arma::mat & P_FE);

      arma::mat assemble_J_FE_one_multipole_yukawa(
          const FEMRadialBasis & radial, int L, double lambda,
          const arma::mat & P_FE);

      arma::mat assemble_K_FE_one_multipole_yukawa(
          const FEMRadialBasis & radial, int L, double lambda,
          const arma::mat & P_FE);

      // -- Cached entry points ----------------------------------------

      /// Same as assemble_J_FE_one_multipole, but the per-element radial
      /// integrals r_small (= radial_integral(L, iel) for bare Coulomb,
      /// or bessel_il_integral(L, lambda, iel) for Yukawa), r_big
      /// (= radial_integral(-L-1, iel) or bessel_kl_integral), and
      /// twoe_in_element (= twoe_integral(L, iel) or yukawa_integral) are
      /// supplied by the caller via accessors. Same assembly, no
      /// recomputation -- meant for SCF inner-loop callers.
      arma::mat assemble_J_FE_one_multipole_cached(
          const FEMRadialBasis & radial,
          const PerElementAccessor & r_small,
          const PerElementAccessor & r_big,
          const PerElementAccessor & twoe_in_element,
          const arma::mat & P_FE);

      /// As above for K. Note `ktei_in_element` here returns the
      /// EXCHANGE-PERMUTED in-element tensor (i.e.
      ///   utils::exchange_tei(twoe_in_element(iel), Ni, Ni, Ni, Ni))
      /// so SCF-driven callers (TwoDBasis, which precomputes prim_ktei)
      /// can pass the cached permuted form directly with zero overhead.
      arma::mat assemble_K_FE_one_multipole_cached(
          const FEMRadialBasis & radial,
          const PerElementAccessor & r_small,
          const PerElementAccessor & r_big,
          const PerElementAccessor & ktei_in_element,
          const arma::mat & P_FE);

      /// Per-(iel, jel) accessor variant of the above K helper, for
      /// kernels whose r1/r2 coupling does NOT factorise into a
      /// small/big disjoint product (in particular: the error-function
      /// kernel used by HelFEM's compute_erfc / rs_exchange). Every
      /// element pair gets its own dense ktei from the cache, then the
      /// standard K contraction is applied. No disjoint optimisation
      /// available -- O(Nel^2) per call.
      using PerElementPairAccessor =
          std::function<const arma::mat & (size_t iel, size_t jel)>;
      arma::mat assemble_K_FE_one_multipole_cached_pairwise(
          const FEMRadialBasis & radial,
          const PerElementPairAccessor & ktei_pairwise,
          const arma::mat & P_FE);

      // Inline implementations -- header-only to match the rest of the
      // libhelfem/include/ NAO surface. ~150 LOC total.

      namespace detail_fe_2e {

        inline arma::mat r_small(const FEMRadialBasis & fem, int L, size_t iel,
                                  bool yukawa, double lambda) {
          return yukawa ? fem.bessel_il_integral(L, lambda, iel)
                        : fem.radial_integral(L, iel);
        }
        inline arma::mat r_big(const FEMRadialBasis & fem, int L, size_t iel,
                                bool yukawa, double lambda) {
          return yukawa ? fem.bessel_kl_integral(L, lambda, iel)
                        : fem.radial_integral(-L - 1, iel);
        }
        inline arma::mat in_element_tei(const FEMRadialBasis & fem, int L,
                                         size_t iel,
                                         bool yukawa, double lambda) {
          return yukawa ? fem.yukawa_integral(L, lambda, iel)
                        : fem.twoe_integral(L, iel);
        }

      } // namespace detail_fe_2e

      // The core J / K assemblers operate on per-element accessors, so
      // the same body backs both the uncached (recompute-on-the-fly,
      // NAO use) and cached (look-up-from-SCF-cache, TwoDBasis use)
      // entry points.

      inline arma::mat assemble_J_FE_one_multipole_cached(
          const FEMRadialBasis & radial,
          const PerElementAccessor & r_small,
          const PerElementAccessor & r_big,
          const PerElementAccessor & twoe_in_element,
          const arma::mat & P_FE) {
        const size_t Nel  = radial.Nel();
        const size_t Nrad = radial.Nbf();
        arma::mat J_FE(Nrad, Nrad, arma::fill::zeros);
        for (size_t jel = 0; jel < Nel; ++jel) {
          size_t jfirst, jlast;
          radial.get_idx(jel, jfirst, jlast);
          const size_t Nj = jlast - jfirst + 1;
          arma::mat Psub = P_FE.submat(jfirst, jfirst, jlast, jlast);
          const double jsmall = arma::trace(r_small(jel) * Psub);
          const double jbig   = arma::trace(r_big  (jel) * Psub);
          for (size_t iel = 0; iel < jel; ++iel) {
            size_t ifirst, ilast;
            radial.get_idx(iel, ifirst, ilast);
            J_FE.submat(ifirst, ifirst, ilast, ilast) +=
                jbig * r_small(iel);
          }
          for (size_t iel = jel + 1; iel < Nel; ++iel) {
            size_t ifirst, ilast;
            radial.get_idx(iel, ifirst, ilast);
            J_FE.submat(ifirst, ifirst, ilast, ilast) +=
                jsmall * r_big(iel);
          }
          // In-element 4-index contribution.
          arma::vec Psub_v = arma::vectorise(Psub);
          arma::vec Jsub_v = twoe_in_element(jel) * Psub_v;
          J_FE.submat(jfirst, jfirst, jlast, jlast) +=
              arma::reshape(Jsub_v, Nj, Nj);
        }
        return J_FE;
      }

      inline arma::mat assemble_K_FE_one_multipole_cached(
          const FEMRadialBasis & radial,
          const PerElementAccessor & r_small,
          const PerElementAccessor & r_big,
          const PerElementAccessor & ktei_in_element,
          const arma::mat & P_FE) {
        const size_t Nel  = radial.Nel();
        const size_t Nrad = radial.Nbf();
        arma::mat K_FE(Nrad, Nrad, arma::fill::zeros);
        for (size_t iel = 0; iel < Nel; ++iel) {
          size_t ifirst, ilast;
          radial.get_idx(iel, ifirst, ilast);
          const size_t Ni = ilast - ifirst + 1;
          for (size_t jel = 0; jel < Nel; ++jel) {
            size_t jfirst, jlast;
            radial.get_idx(jel, jfirst, jlast);
            const size_t Nj = jlast - jfirst + 1;
            if (iel == jel) {
              arma::vec Psub_v = arma::vectorise(
                  P_FE.submat(ifirst, jfirst, ilast, jlast));
              arma::vec Ksub_v = ktei_in_element(iel) * Psub_v;
              K_FE.submat(ifirst, jfirst, ilast, jlast) +=
                  arma::reshape(Ksub_v, Ni, Nj);
            } else {
              const arma::mat & iint = (iel > jel) ? r_big(iel) : r_small(iel);
              const arma::mat & jint = (iel > jel) ? r_small(jel) : r_big(jel);
              const arma::mat Psub = P_FE.submat(ifirst, jfirst, ilast, jlast);
              K_FE.submat(ifirst, jfirst, ilast, jlast) +=
                  iint * (Psub * jint.t());
            }
          }
        }
        return K_FE;
      }

      inline arma::mat assemble_K_FE_one_multipole_cached_pairwise(
          const FEMRadialBasis & radial,
          const PerElementPairAccessor & ktei_pairwise,
          const arma::mat & P_FE) {
        const size_t Nel  = radial.Nel();
        const size_t Nrad = radial.Nbf();
        arma::mat K_FE(Nrad, Nrad, arma::fill::zeros);
        for (size_t iel = 0; iel < Nel; ++iel) {
          size_t ifirst, ilast;
          radial.get_idx(iel, ifirst, ilast);
          const size_t Ni = ilast - ifirst + 1;
          for (size_t jel = 0; jel < Nel; ++jel) {
            size_t jfirst, jlast;
            radial.get_idx(jel, jfirst, jlast);
            const size_t Nj = jlast - jfirst + 1;
            arma::vec Psub_v = arma::vectorise(
                P_FE.submat(ifirst, jfirst, ilast, jlast));
            arma::vec Ksub_v = ktei_pairwise(iel, jel) * Psub_v;
            K_FE.submat(ifirst, jfirst, ilast, jlast) +=
                arma::reshape(Ksub_v, Ni, Nj);
          }
        }
        return K_FE;
      }

      // The uncached entry points just wire on-the-fly accessors that
      // call the FE primitives directly.

      inline arma::mat assemble_J_FE_one_multipole(
          const FEMRadialBasis & radial, int L, const arma::mat & P_FE) {
        // Per-call temporaries to keep the accessor lambdas returning a
        // stable const reference (storing each computed matrix in a
        // capture-list scratchpad).
        std::vector<arma::mat> scratch_small(radial.Nel());
        std::vector<arma::mat> scratch_big  (radial.Nel());
        std::vector<arma::mat> scratch_twoe (radial.Nel());
        auto rs = [&](size_t iel) -> const arma::mat & {
          if (scratch_small[iel].is_empty())
            scratch_small[iel] = detail_fe_2e::r_small(radial, L, iel, false, 0.0);
          return scratch_small[iel];
        };
        auto rb = [&](size_t iel) -> const arma::mat & {
          if (scratch_big[iel].is_empty())
            scratch_big[iel] = detail_fe_2e::r_big(radial, L, iel, false, 0.0);
          return scratch_big[iel];
        };
        auto tw = [&](size_t iel) -> const arma::mat & {
          if (scratch_twoe[iel].is_empty())
            scratch_twoe[iel] = detail_fe_2e::in_element_tei(radial, L, iel, false, 0.0);
          return scratch_twoe[iel];
        };
        return assemble_J_FE_one_multipole_cached(radial, rs, rb, tw, P_FE);
      }

      inline arma::mat assemble_K_FE_one_multipole(
          const FEMRadialBasis & radial, int L, const arma::mat & P_FE) {
        std::vector<arma::mat> scratch_small(radial.Nel());
        std::vector<arma::mat> scratch_big  (radial.Nel());
        std::vector<arma::mat> scratch_ktei (radial.Nel());
        auto rs = [&](size_t iel) -> const arma::mat & {
          if (scratch_small[iel].is_empty())
            scratch_small[iel] = detail_fe_2e::r_small(radial, L, iel, false, 0.0);
          return scratch_small[iel];
        };
        auto rb = [&](size_t iel) -> const arma::mat & {
          if (scratch_big[iel].is_empty())
            scratch_big[iel] = detail_fe_2e::r_big(radial, L, iel, false, 0.0);
          return scratch_big[iel];
        };
        auto kt = [&](size_t iel) -> const arma::mat & {
          if (scratch_ktei[iel].is_empty()) {
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

      inline arma::mat assemble_J_FE_one_multipole_yukawa(
          const FEMRadialBasis & radial, int L, double lambda,
          const arma::mat & P_FE) {
        std::vector<arma::mat> scratch_small(radial.Nel());
        std::vector<arma::mat> scratch_big  (radial.Nel());
        std::vector<arma::mat> scratch_twoe (radial.Nel());
        auto rs = [&](size_t iel) -> const arma::mat & {
          if (scratch_small[iel].is_empty())
            scratch_small[iel] = detail_fe_2e::r_small(radial, L, iel, true, lambda);
          return scratch_small[iel];
        };
        auto rb = [&](size_t iel) -> const arma::mat & {
          if (scratch_big[iel].is_empty())
            scratch_big[iel] = detail_fe_2e::r_big(radial, L, iel, true, lambda);
          return scratch_big[iel];
        };
        auto tw = [&](size_t iel) -> const arma::mat & {
          if (scratch_twoe[iel].is_empty())
            scratch_twoe[iel] = detail_fe_2e::in_element_tei(radial, L, iel, true, lambda);
          return scratch_twoe[iel];
        };
        return assemble_J_FE_one_multipole_cached(radial, rs, rb, tw, P_FE);
      }

      inline arma::mat assemble_K_FE_one_multipole_yukawa(
          const FEMRadialBasis & radial, int L, double lambda,
          const arma::mat & P_FE) {
        std::vector<arma::mat> scratch_small(radial.Nel());
        std::vector<arma::mat> scratch_big  (radial.Nel());
        std::vector<arma::mat> scratch_ktei (radial.Nel());
        auto rs = [&](size_t iel) -> const arma::mat & {
          if (scratch_small[iel].is_empty())
            scratch_small[iel] = detail_fe_2e::r_small(radial, L, iel, true, lambda);
          return scratch_small[iel];
        };
        auto rb = [&](size_t iel) -> const arma::mat & {
          if (scratch_big[iel].is_empty())
            scratch_big[iel] = detail_fe_2e::r_big(radial, L, iel, true, lambda);
          return scratch_big[iel];
        };
        auto kt = [&](size_t iel) -> const arma::mat & {
          if (scratch_ktei[iel].is_empty()) {
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
