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
#ifndef ATOMIC_BASIS_NAORADIALBASIS_H
#define ATOMIC_BASIS_NAORADIALBASIS_H

#include "RadialBasis.h"
#include <armadillo>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace helfem {
  namespace atomic {
    namespace basis {

      /// Numerical-Atomic-Orbital (NAO) radial basis. Each NAO is a linear
      /// combination of the shape functions of an underlying RadialBasis
      /// (typically a high-resolution FEMRadialBasis using LIP/HIP/HIP2/
      /// HIP3/Legendre primitives). The class is just a thin view: it
      /// holds a shared_ptr to the underlying basis and a coefficient
      /// matrix C (shape Nbf_underlying x Nbf_NAO) defining each NAO as
      ///
      ///     |NAO_alpha> = sum_i C(i, alpha) |underlying_i>
      ///
      /// Every one-electron integral reduces to a basis transform of the
      /// underlying matrix:
      ///
      ///     M_NAO  =  C^T  M_underlying  C
      ///
      /// so the NAO basis automatically inherits the underlying basis's
      /// quadrature accuracy. No tabulation, no interpolation, no log-grid
      /// reweighting; the underlying basis already integrates everything
      /// to its own (typically near-machine-precision) standard.
      ///
      /// Typical usage:
      ///   1. Run an atomic SCF in a high-resolution FEMRadialBasis.
      ///   2. Pick the C columns to keep (e.g. the occupied orbitals plus
      ///      a handful of virtuals, per (n, l) channel).
      ///   3. Hand the (underlying, C) pair to NAORadialBasis; use the
      ///      result wherever a RadialBasis* is expected (e.g. as the
      ///      atomic-centre basis in a downstream diatomic calculation).
      ///
      /// The underlying basis is held by shared_ptr so the NAO basis can
      /// outlive the constructing scope, and so the same underlying FE
      /// basis can back multiple NAO projections (e.g. one per (n, l)).
      ///
      /// Currently non-templated (operates on arma::mat = double) to
      /// match the non-templated RadialBasis base; the broader
      /// templated-precision migration is a follow-up.
      class NAORadialBasis : public RadialBasis {
       protected:
        /// Underlying radial basis (typically a FEMRadialBasis).
        std::shared_ptr<const RadialBasis> underlying_;
        /// Orbital coefficient matrix, shape (Nbf_underlying x Nbf_NAO).
        arma::mat C_;

       public:
        /// Constructor. `C` must have `underlying->Nbf()` rows and at
        /// least one column (one column = one NAO).
        NAORadialBasis(std::shared_ptr<const RadialBasis> underlying,
                       arma::mat C)
            : underlying_(std::move(underlying)), C_(std::move(C)) {
          if (!underlying_)
            throw std::logic_error("NAORadialBasis: null underlying basis.\n");
          if (C_.n_rows != underlying_->Nbf()) {
            std::ostringstream oss;
            oss << "NAORadialBasis: C has " << C_.n_rows
                << " rows but underlying basis has " << underlying_->Nbf()
                << " basis functions.\n";
            throw std::logic_error(oss.str());
          }
          if (C_.n_cols == 0)
            throw std::logic_error("NAORadialBasis: C has zero columns (no NAOs).\n");
        }

        ~NAORadialBasis() override = default;

        /// Access the underlying basis.
        const RadialBasis & underlying() const { return *underlying_; }
        /// Access the orbital coefficient matrix.
        const arma::mat & coeffs() const { return C_; }

        size_t Nbf() const override { return C_.n_cols; }

        /// S_NAO = C^T S_underlying C.
        arma::mat overlap() const override {
          return C_.t() * underlying_->overlap() * C_;
        }

        /// T_NAO = C^T T_underlying C.
        arma::mat kinetic() const override {
          return C_.t() * underlying_->kinetic() * C_;
        }

        /// (Centrifugal-per-l(l+1))_NAO = C^T (...)_underlying C.
        arma::mat kinetic_l() const override {
          return C_.t() * underlying_->kinetic_l() * C_;
        }

        /// V_NAO = C^T V_underlying C (Z=1; caller multiplies by +Z).
        arma::mat nuclear() const override {
          return C_.t() * underlying_->nuclear() * C_;
        }

        /// Evaluate the NAO orbitals (columns of C_orb in the NAO basis)
        /// at radius r. Internally promotes to the underlying basis via
        /// C_promoted = C * C_orb and delegates to underlying.eval_orbs.
        arma::vec eval_orbs(const arma::mat & C_orb, double r) const override {
          if (C_orb.n_rows != Nbf()) {
            std::ostringstream oss;
            oss << "NAORadialBasis::eval_orbs: C_orb has " << C_orb.n_rows
                << " rows but NAO Nbf() = " << Nbf() << "\n";
            throw std::logic_error(oss.str());
          }
          return underlying_->eval_orbs(C_ * C_orb, r);
        }
      };

    } // namespace basis
  } // namespace atomic
} // namespace helfem

#endif
