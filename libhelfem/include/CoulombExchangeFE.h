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

        inline arma::mat assemble_J(const FEMRadialBasis & radial, int L,
                                     const arma::mat & P_FE,
                                     bool yukawa, double lambda) {
          const size_t Nel  = radial.Nel();
          const size_t Nrad = radial.Nbf();
          arma::mat J_FE(Nrad, Nrad, arma::fill::zeros);
          for (size_t jel = 0; jel < Nel; ++jel) {
            size_t jfirst, jlast;
            radial.get_idx(jel, jfirst, jlast);
            const size_t Nj = jlast - jfirst + 1;
            arma::mat Psub = P_FE.submat(jfirst, jfirst, jlast, jlast);
            const arma::mat rs = r_small(radial, L, jel, yukawa, lambda);
            const arma::mat rb = r_big  (radial, L, jel, yukawa, lambda);
            const double jsmall = arma::trace(rs * Psub);
            const double jbig   = arma::trace(rb * Psub);
            for (size_t iel = 0; iel < jel; ++iel) {
              size_t ifirst, ilast;
              radial.get_idx(iel, ifirst, ilast);
              J_FE.submat(ifirst, ifirst, ilast, ilast) +=
                  jbig * r_small(radial, L, iel, yukawa, lambda);
            }
            for (size_t iel = jel + 1; iel < Nel; ++iel) {
              size_t ifirst, ilast;
              radial.get_idx(iel, ifirst, ilast);
              J_FE.submat(ifirst, ifirst, ilast, ilast) +=
                  jsmall * r_big(radial, L, iel, yukawa, lambda);
            }
            // In-element 4-index contribution.
            arma::vec Psub_v = arma::vectorise(Psub);
            arma::vec Jsub_v = in_element_tei(radial, L, jel, yukawa, lambda) * Psub_v;
            J_FE.submat(jfirst, jfirst, jlast, jlast) +=
                arma::reshape(Jsub_v, Nj, Nj);
          }
          return J_FE;
        }

        inline arma::mat assemble_K(const FEMRadialBasis & radial, int L,
                                     const arma::mat & P_FE,
                                     bool yukawa, double lambda) {
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
                const arma::mat tei  = in_element_tei(radial, L, iel, yukawa, lambda);
                const arma::mat ktei = utils::exchange_tei(tei, Ni, Ni, Ni, Ni);
                arma::vec Psub_v = arma::vectorise(
                    P_FE.submat(ifirst, jfirst, ilast, jlast));
                arma::vec Ksub_v = ktei * Psub_v;
                K_FE.submat(ifirst, jfirst, ilast, jlast) +=
                    arma::reshape(Ksub_v, Ni, Nj);
              } else {
                const arma::mat iint = (iel > jel)
                    ? r_big  (radial, L, iel, yukawa, lambda)
                    : r_small(radial, L, iel, yukawa, lambda);
                const arma::mat jint = (iel > jel)
                    ? r_small(radial, L, jel, yukawa, lambda)
                    : r_big  (radial, L, jel, yukawa, lambda);
                const arma::mat Psub = P_FE.submat(ifirst, jfirst, ilast, jlast);
                K_FE.submat(ifirst, jfirst, ilast, jlast) +=
                    iint * (Psub * jint.t());
              }
            }
          }
          return K_FE;
        }

      } // namespace detail_fe_2e

      inline arma::mat assemble_J_FE_one_multipole(
          const FEMRadialBasis & radial, int L, const arma::mat & P_FE) {
        return detail_fe_2e::assemble_J(radial, L, P_FE, /*yukawa=*/false, 0.0);
      }
      inline arma::mat assemble_K_FE_one_multipole(
          const FEMRadialBasis & radial, int L, const arma::mat & P_FE) {
        return detail_fe_2e::assemble_K(radial, L, P_FE, /*yukawa=*/false, 0.0);
      }
      inline arma::mat assemble_J_FE_one_multipole_yukawa(
          const FEMRadialBasis & radial, int L, double lambda,
          const arma::mat & P_FE) {
        return detail_fe_2e::assemble_J(radial, L, P_FE, /*yukawa=*/true, lambda);
      }
      inline arma::mat assemble_K_FE_one_multipole_yukawa(
          const FEMRadialBasis & radial, int L, double lambda,
          const arma::mat & P_FE) {
        return detail_fe_2e::assemble_K(radial, L, P_FE, /*yukawa=*/true, lambda);
      }

    } // namespace basis
  } // namespace atomic
} // namespace helfem

#endif
