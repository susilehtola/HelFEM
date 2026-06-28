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
#ifndef ATOMIC_BASIS_GTORADIALBASIS_H
#define ATOMIC_BASIS_GTORADIALBASIS_H

#include "RadialBasis.h"
#include <armadillo>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace helfem {
  namespace atomic {
    namespace basis {

      /// Primitive Gaussian-type-orbital (GTO) radial basis. Each
      /// function is characterised by an integer power n_i >= 0 (usually
      /// n_i = l for a primitive of angular momentum l) and a positive
      /// exponent alpha_i:
      ///
      ///     R_i(r) = N_i * r^{n_i} * exp(-alpha_i * r^2)
      ///     u_i(r) = r * R_i(r) = N_i * r^{n_i + 1} * exp(-alpha_i * r^2)
      ///
      /// (u convention matches FEMRadialBasis / NAORadialBasis;
      /// u_i(0) = 0 for any n_i >= 0).
      ///
      /// Functions are normalised individually:
      ///     N_i = 1 / sqrt( I(2 n_i + 2, 2 alpha_i) )
      /// so the diagonal of the overlap matrix is 1 (off-diagonals are
      /// nonzero -- GTOs at distinct (n, alpha) are NOT orthogonal).
      ///
      /// All one-electron matrix elements have closed forms via
      ///     I(p, alpha) := integral_0^inf r^p exp(-alpha r^2) dr
      ///                  = (1/2) Gamma((p+1)/2) / alpha^{(p+1)/2}
      /// implemented with std::tgamma so any non-negative integer p works.
      ///
      /// Contracted GTOs (fixed linear combinations of primitives) are
      /// handled by composing with NAORadialBasis (just like for
      /// STORadialBasis).
      ///
      /// Currently non-templated (operates on arma::mat = double) to
      /// match the non-templated RadialBasis base.
      class GTORadialBasis : public RadialBasis {
       protected:
        arma::ivec n_;       ///< powers (n_i >= 0; typically n_i = l)
        arma::vec  alpha_;   ///< exponents (alpha_i > 0)
        arma::vec  norm_;    ///< per-function normalisation constants

        /// I(p, a) = (1/2) Gamma((p+1)/2) / a^{(p+1)/2}; p >= 0.
        static double I(int p, double a) {
          if (p < 0)
            throw std::logic_error("GTORadialBasis: I called with negative p.\n");
          const double half = 0.5 * (p + 1);
          return 0.5 * std::tgamma(half) / std::pow(a, half);
        }

       public:
        /// Construct from per-function (n, alpha). Both vectors must have
        /// the same nonzero length; all n_i >= 0; all alpha_i > 0.
        GTORadialBasis(arma::ivec n, arma::vec alpha)
            : n_(std::move(n)), alpha_(std::move(alpha)) {
          if (n_.n_elem != alpha_.n_elem)
            throw std::logic_error("GTORadialBasis: n and alpha sizes differ.\n");
          if (n_.n_elem == 0)
            throw std::logic_error("GTORadialBasis: empty basis.\n");
          if (arma::any(n_ < 0))
            throw std::logic_error("GTORadialBasis: every n_i must be >= 0.\n");
          if (arma::any(alpha_ <= 0.0))
            throw std::logic_error("GTORadialBasis: every alpha_i must be positive.\n");
          norm_.set_size(n_.n_elem);
          for (arma::uword i = 0; i < n_.n_elem; ++i)
            norm_(i) = 1.0 / std::sqrt(I(2 * n_(i) + 2, 2.0 * alpha_(i)));
        }

        ~GTORadialBasis() override = default;

        const arma::ivec & n_powers() const { return n_; }
        const arma::vec  & alphas()   const { return alpha_; }
        const arma::vec  & norms()    const { return norm_; }

        size_t Nbf() const override { return n_.n_elem; }

        /// S_ij = N_i N_j I(n_i + n_j + 2, alpha_i + alpha_j).
        arma::mat overlap() const override {
          const arma::uword N = Nbf();
          arma::mat S(N, N);
          for (arma::uword j = 0; j < N; ++j)
            for (arma::uword i = 0; i < N; ++i)
              S(i, j) = norm_(i) * norm_(j) *
                        I(n_(i) + n_(j) + 2, alpha_(i) + alpha_(j));
          return S;
        }

        /// T_ij = (1/2) integral u'_i u'_j dr.
        /// u'_i = N_i [(n_i + 1) r^{n_i} - 2 alpha_i r^{n_i + 2}] exp(-alpha_i r^2)
        /// u'_i u'_j = N_i N_j r^{p} [A - 2 B r^2 + 4 alpha_i alpha_j r^4] exp(-beta r^2)
        ///   with p = n_i + n_j, A = (n_i + 1)(n_j + 1),
        ///        B = (n_i + 1) alpha_j + (n_j + 1) alpha_i,
        ///        beta = alpha_i + alpha_j.
        arma::mat kinetic() const override {
          const arma::uword N = Nbf();
          arma::mat T(N, N);
          for (arma::uword j = 0; j < N; ++j) {
            for (arma::uword i = 0; i < N; ++i) {
              const int    ni = n_(i);
              const int    nj = n_(j);
              const double ai = alpha_(i);
              const double aj = alpha_(j);
              const int    p  = ni + nj;
              const double b  = ai + aj;
              const double A  = double(ni + 1) * (nj + 1);
              const double B  = (ni + 1) * aj + (nj + 1) * ai;
              const double term = A * I(p, b)
                                - 2.0 * B * I(p + 2, b)
                                + 4.0 * ai * aj * I(p + 4, b);
              T(i, j) = 0.5 * norm_(i) * norm_(j) * term;
            }
          }
          return T;
        }

        /// (Centrifugal)_ij = (1/2) integral u_i u_j / r^2 dr
        ///                  = (1/2) N_i N_j I(n_i + n_j, alpha_i + alpha_j).
        arma::mat kinetic_l() const override {
          const arma::uword N = Nbf();
          arma::mat L(N, N);
          for (arma::uword j = 0; j < N; ++j)
            for (arma::uword i = 0; i < N; ++i)
              L(i, j) = 0.5 * norm_(i) * norm_(j) *
                        I(n_(i) + n_(j), alpha_(i) + alpha_(j));
          return L;
        }

        /// V_ij = -integral u_i u_j / r dr
        ///      = -N_i N_j I(n_i + n_j + 1, alpha_i + alpha_j)  (Z=1).
        arma::mat nuclear() const override {
          const arma::uword N = Nbf();
          arma::mat V(N, N);
          for (arma::uword j = 0; j < N; ++j)
            for (arma::uword i = 0; i < N; ++i)
              V(i, j) = -norm_(i) * norm_(j) *
                         I(n_(i) + n_(j) + 1, alpha_(i) + alpha_(j));
          return V;
        }

        /// psi_alpha(r) = sum_i C_{i,alpha} N_i r^{n_i} exp(-alpha_i r^2).
        arma::vec eval_orbs(const arma::mat & C, double r) const override {
          if (C.n_rows != Nbf()) {
            std::ostringstream oss;
            oss << "GTORadialBasis::eval_orbs: C has " << C.n_rows
                << " rows but Nbf() = " << Nbf() << "\n";
            throw std::logic_error(oss.str());
          }
          arma::rowvec R_at_r(Nbf());
          for (arma::uword i = 0; i < Nbf(); ++i)
            R_at_r(i) = norm_(i) * std::pow(r, n_(i)) *
                        std::exp(-alpha_(i) * r * r);
          return (R_at_r * C).t();
        }
      };

    } // namespace basis
  } // namespace atomic
} // namespace helfem

#endif
