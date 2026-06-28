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
#ifndef ATOMIC_BASIS_STORADIALBASIS_H
#define ATOMIC_BASIS_STORADIALBASIS_H

#include "RadialBasis.h"
#include <armadillo>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace helfem {
  namespace atomic {
    namespace basis {

      /// Primitive Slater-type-orbital (STO) radial basis. Each function
      /// is characterised by an integer power n_i >= 1 and a positive
      /// exponent zeta_i:
      ///
      ///     R_i(r) = N_i * r^{n_i - 1} * exp(-zeta_i * r)
      ///     u_i(r) = r * R_i(r) = N_i * r^{n_i} * exp(-zeta_i * r)
      ///
      /// (the u convention matches FEMRadialBasis and NAORadialBasis,
      /// so u_i(0) = 0 and all matrix elements are in dr measure).
      ///
      /// Functions are normalised individually:
      ///     N_i = sqrt( (2*zeta_i)^{2 n_i + 1} / (2 n_i)! )
      /// so the diagonal of the overlap matrix is 1 (off-diagonals are
      /// nonzero -- STOs at distinct (n, zeta) are NOT orthogonal).
      ///
      /// All one-electron matrix elements have closed forms via
      ///     I(p, alpha) := integral_0^inf r^p exp(-alpha r) dr
      ///                  = p! / alpha^{p+1}    (integer p >= 0).
      ///
      /// Contracted STOs (fixed linear combinations of primitives) are
      /// handled by composing this class with NAORadialBasis: build the
      /// primitive STORadialBasis once, then wrap with NAORadialBasis
      /// carrying the contraction coefficients.
      ///
      /// Currently non-templated (operates on arma::mat = double) to
      /// match the non-templated RadialBasis base.
      class STORadialBasis : public RadialBasis {
       protected:
        arma::ivec n_;       ///< principal powers (n_i >= 1)
        arma::vec  zeta_;    ///< exponents (zeta_i > 0)
        arma::vec  norm_;    ///< per-function normalisation constants

        /// Integer factorial; STOs typically need p <= 2*max(n_i), small.
        static double fact(int p) {
          if (p < 0)
            throw std::logic_error("STORadialBasis: factorial of negative argument.\n");
          double f = 1.0;
          for (int k = 2; k <= p; ++k) f *= k;
          return f;
        }

        /// Convenience: I(p, alpha) = p! / alpha^{p+1}.
        static double I(int p, double alpha) {
          return fact(p) / std::pow(alpha, p + 1);
        }

       public:
        /// Construct from per-function (n, zeta). Both vectors must have
        /// the same nonzero length; all n_i >= 1; all zeta_i > 0.
        STORadialBasis(arma::ivec n, arma::vec zeta)
            : n_(std::move(n)), zeta_(std::move(zeta)) {
          if (n_.n_elem != zeta_.n_elem)
            throw std::logic_error("STORadialBasis: n and zeta sizes differ.\n");
          if (n_.n_elem == 0)
            throw std::logic_error("STORadialBasis: empty basis.\n");
          if (arma::any(n_ < 1))
            throw std::logic_error("STORadialBasis: every n_i must be >= 1 "
                                    "(to enforce u(0)=0).\n");
          if (arma::any(zeta_ <= 0.0))
            throw std::logic_error("STORadialBasis: every zeta_i must be positive.\n");
          norm_.set_size(n_.n_elem);
          for (arma::uword i = 0; i < n_.n_elem; ++i) {
            const int    ni = n_(i);
            const double zi = zeta_(i);
            // N_i^2 = (2 zeta_i)^{2 n_i + 1} / (2 n_i)!
            norm_(i) = std::sqrt(std::pow(2.0 * zi, 2 * ni + 1) / fact(2 * ni));
          }
        }

        ~STORadialBasis() override = default;

        /// Read-only accessors.
        const arma::ivec & n_powers() const { return n_; }
        const arma::vec  & zetas()   const { return zeta_; }
        const arma::vec  & norms()   const { return norm_; }

        size_t Nbf() const override { return n_.n_elem; }

        /// S_ij = N_i N_j (n_i + n_j)! / (zeta_i + zeta_j)^{n_i + n_j + 1}
        arma::mat overlap() const override {
          const arma::uword N = Nbf();
          arma::mat S(N, N);
          for (arma::uword j = 0; j < N; ++j)
            for (arma::uword i = 0; i < N; ++i)
              S(i, j) = norm_(i) * norm_(j) *
                        I(n_(i) + n_(j), zeta_(i) + zeta_(j));
          return S;
        }

        /// T_ij = (1/2) integral u'_i u'_j dr.
        /// With u'_i = N_i r^{n_i - 1} (n_i - zeta_i r) exp(-zeta_i r),
        ///   u'_i u'_j = N_i N_j r^{p-2} [n_i n_j
        ///                                - (n_i zeta_j + n_j zeta_i) r
        ///                                + zeta_i zeta_j r^2] exp(-alpha r)
        /// where p = n_i + n_j, alpha = zeta_i + zeta_j.
        arma::mat kinetic() const override {
          const arma::uword N = Nbf();
          arma::mat T(N, N);
          for (arma::uword j = 0; j < N; ++j) {
            for (arma::uword i = 0; i < N; ++i) {
              const int    ni = n_(i);
              const int    nj = n_(j);
              const double zi = zeta_(i);
              const double zj = zeta_(j);
              const int    p  = ni + nj;
              const double a  = zi + zj;
              const double term = double(ni * nj) * I(p - 2, a)
                                - (ni * zj + nj * zi) * I(p - 1, a)
                                + zi * zj * I(p, a);
              T(i, j) = 0.5 * norm_(i) * norm_(j) * term;
            }
          }
          return T;
        }

        /// (Centrifugal)_ij = (1/2) integral u_i u_j / r^2 dr
        ///                  = (1/2) N_i N_j (p-2)! / alpha^{p-1}
        ///                  (well-defined since p = n_i + n_j >= 2).
        arma::mat kinetic_l() const override {
          const arma::uword N = Nbf();
          arma::mat L(N, N);
          for (arma::uword j = 0; j < N; ++j)
            for (arma::uword i = 0; i < N; ++i)
              L(i, j) = 0.5 * norm_(i) * norm_(j) *
                        I(n_(i) + n_(j) - 2, zeta_(i) + zeta_(j));
          return L;
        }

        /// V_ij = -integral u_i u_j / r dr
        ///      = -N_i N_j (p-1)! / alpha^p     (Z=1; caller multiplies by Z).
        arma::mat nuclear() const override {
          const arma::uword N = Nbf();
          arma::mat V(N, N);
          for (arma::uword j = 0; j < N; ++j)
            for (arma::uword i = 0; i < N; ++i)
              V(i, j) = -norm_(i) * norm_(j) *
                         I(n_(i) + n_(j) - 1, zeta_(i) + zeta_(j));
          return V;
        }

        /// psi_alpha(r) = sum_i C_{i,alpha} R_i(r)
        ///              = sum_i C_{i,alpha} N_i r^{n_i - 1} exp(-zeta_i r).
        /// Finite at r=0 (n_i >= 1 guarantees r^{n_i - 1} bounded).
        arma::vec eval_orbs(const arma::mat & C, double r) const override {
          if (C.n_rows != Nbf()) {
            std::ostringstream oss;
            oss << "STORadialBasis::eval_orbs: C has " << C.n_rows
                << " rows but Nbf() = " << Nbf() << "\n";
            throw std::logic_error(oss.str());
          }
          arma::rowvec R_at_r(Nbf());
          for (arma::uword i = 0; i < Nbf(); ++i)
            R_at_r(i) = norm_(i) * std::pow(r, n_(i) - 1) *
                        std::exp(-zeta_(i) * r);
          return (R_at_r * C).t();
        }
      };

    } // namespace basis
  } // namespace atomic
} // namespace helfem

#endif
