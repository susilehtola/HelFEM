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
#include "ArmaEigen.h"
#include "CoulombExchangeFE.h"
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

        /// Convenience factory: wrap an existing concrete RadialBasis
        /// (typically a FEMRadialBasis fresh out of a TwoDBasis) in a
        /// shared_ptr by copy, then construct. Useful when the caller
        /// holds the underlying basis by value and does not want to
        /// manage the shared_ptr themselves.
        template <typename RB>
        static NAORadialBasis from_owned_radial(RB underlying, arma::mat C) {
          return NAORadialBasis(
              std::make_shared<RB>(std::move(underlying)), std::move(C));
        }

        /// Access the underlying basis.
        const RadialBasis & underlying() const { return *underlying_; }
        /// Access the orbital coefficient matrix.
        const arma::mat & coeffs() const { return C_; }

        size_t Nbf() const override { return C_.n_cols; }

        /// S_NAO = C^T S_underlying C.
        helfem::Matrix overlap() const override {
          // Bridge: underlying_->overlap() is Eigen (Phase 2a); C_ is
          // still arma. Convert at the boundary; subsequent Phase 2
          // PR migrates C_ to Eigen and drops the round-trip. Force
          // evaluation of the arma expression template into a concrete
          // mat before to_eigen to avoid ADL ambiguity.
          arma::mat S_arma = C_.t() * helfem::to_arma(underlying_->overlap()) * C_;
          return helfem::to_eigen(S_arma);
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

        // ---------------------------------------------------------------
        // Per-multipole two-electron radial operators (for libatomscf-style
        // consumers that drive the SCF/FCI from outside HelFEM).
        //
        // Notation. The BARE radial Slater integral for a single multipole
        // k is
        //
        //   R^k(ab, cd) = integral_0..inf integral_0..inf
        //                     R_a(r1) R_b(r1) [r_<^k / r_>^{k+1}]
        //                     R_c(r2) R_d(r2) dr1 dr2
        //
        // with NO 4*pi/(2k+1) prefactor and NO Gaunt coefficient -- the
        // caller multiplies those in afterwards based on the angular
        // structure they're driving. *this is the l_a channel; `other` is
        // the l_b channel; both NAO bases MUST wrap the SAME underlying
        // FEMRadialBasis (asserted at runtime).
        //
        // coulomb_radial(k, other, D)_{ij} = sum_{mn} D_{mn} R^k(a_i a_j, b_m b_n)
        // exchange_radial(k, other, D)_{ij} = sum_{mn} D_{mn} R^k(a_i b_m, a_j b_n)
        //
        // Implementation. Both reduce to a C-transform of an FE-basis
        // single-multipole operator:
        //
        //   M_NAO  =  C_a^T  M_FE(P_FE)  C_a,    P_FE = C_b D C_b^T.
        //
        // The FE-side single-multipole J^L_FE and K^L_FE are assembled by
        // re-using FEMRadialBasis::twoe_integral(L, iel) for in-element
        // pieces and the disjoint factorisation
        //     R^L_FE(ab, cd) = r_small(iel)_{ab} . r_big(jel)_{cd}     (iel < jel)
        //                    = r_big(iel)_{ab}   . r_small(jel)_{cd}   (iel > jel)
        // for cross-element pieces, where r_small = radial_integral(L, .)
        // and r_big = radial_integral(-L-1, .). Same pattern as TwoDBasis
        // (sadatom/basis.cpp + atomic/TwoDBasis.cpp), but at a SINGLE L,
        // with Lfac == 1.
        //
        // Range-separated variants: coulomb_radial_yukawa /
        // exchange_radial_yukawa replace the bare kernel with the
        // Yukawa-screened r_<^L / r_>^{L+1} -> i_L(lambda r_<) k_L(lambda r_>)
        // form. Bessel-l/bessel-k disjoint integrals are used in place of
        // the r^L / r^{-L-1} pieces. erfc is NOT exposed here -- its
        // cross-element structure does not factorise and the assembly is
        // expensive.

        /// J^k_NAO(D) := sum_{mn} D_{mn} R^k(a_i a_j, b_m b_n) (bare radial,
        /// no 4 pi / (2k+1)). `other` is the l_b channel; both NAO bases
        /// must share the same underlying FEMRadialBasis.
        arma::mat coulomb_radial(int k, const NAORadialBasis & other,
                                 const arma::mat & D) const {
          require_shared_fe_(other, "coulomb_radial");
          require_density_shape_(other, D, "coulomb_radial");
          const FEMRadialBasis & fem = underlying_fem_();
          const arma::mat P_FE = other.C_ * D * other.C_.t();
          return C_.t() * assemble_J_FE_one_multipole(fem, k, P_FE) * C_;
        }

        /// K^k_NAO(D) := sum_{mn} D_{mn} R^k(a_i b_m, a_j b_n) (bare radial).
        /// Sign convention: returns the POSITIVE bare integral; callers
        /// driving Hartree-Fock should flip sign on their end.
        arma::mat exchange_radial(int k, const NAORadialBasis & other,
                                  const arma::mat & D) const {
          require_shared_fe_(other, "exchange_radial");
          require_density_shape_(other, D, "exchange_radial");
          const FEMRadialBasis & fem = underlying_fem_();
          const arma::mat P_FE = other.C_ * D * other.C_.t();
          return C_.t() * assemble_K_FE_one_multipole(fem, k, P_FE) * C_;
        }

        /// Yukawa-screened Coulomb at multipole k, screening lambda > 0:
        ///   kernel = (2k+1) i_k(lambda r_<) k_k(lambda r_>)
        /// (the same kernel used in HelFEM's TwoDBasis::compute_yukawa /
        /// FEMRadialBasis::yukawa_integral).
        arma::mat coulomb_radial_yukawa(int k, double lambda,
                                        const NAORadialBasis & other,
                                        const arma::mat & D) const {
          require_shared_fe_(other, "coulomb_radial_yukawa");
          require_density_shape_(other, D, "coulomb_radial_yukawa");
          if (lambda <= 0.0)
            throw std::logic_error("coulomb_radial_yukawa: lambda must be positive.\n");
          const FEMRadialBasis & fem = underlying_fem_();
          const arma::mat P_FE = other.C_ * D * other.C_.t();
          return C_.t() *
                 assemble_J_FE_one_multipole_yukawa(fem, k, lambda, P_FE) * C_;
        }

        /// Yukawa-screened exchange at multipole k.
        arma::mat exchange_radial_yukawa(int k, double lambda,
                                         const NAORadialBasis & other,
                                         const arma::mat & D) const {
          require_shared_fe_(other, "exchange_radial_yukawa");
          require_density_shape_(other, D, "exchange_radial_yukawa");
          if (lambda <= 0.0)
            throw std::logic_error("exchange_radial_yukawa: lambda must be positive.\n");
          const FEMRadialBasis & fem = underlying_fem_();
          const arma::mat P_FE = other.C_ * D * other.C_.t();
          return C_.t() *
                 assemble_K_FE_one_multipole_yukawa(fem, k, lambda, P_FE) * C_;
        }

       protected:
        // --- helpers --------------------------------------------------

        /// Throw unless `other` wraps the exact same FEMRadialBasis as *this.
        void require_shared_fe_(const NAORadialBasis & other,
                                const char * site) const {
          if (underlying_.get() != other.underlying_.get()) {
            std::ostringstream oss;
            oss << "NAORadialBasis::" << site
                << ": this and other must share the same underlying basis.\n";
            throw std::logic_error(oss.str());
          }
          (void)underlying_fem_();  // also asserts both are FEM
        }

        void require_density_shape_(const NAORadialBasis & other,
                                    const arma::mat & D,
                                    const char * site) const {
          if (D.n_rows != other.Nbf() || D.n_cols != other.Nbf()) {
            std::ostringstream oss;
            oss << "NAORadialBasis::" << site << ": density matrix is "
                << D.n_rows << "x" << D.n_cols << " but other.Nbf() = "
                << other.Nbf() << ".\n";
            throw std::logic_error(oss.str());
          }
        }

        /// Downcast the underlying basis to FEMRadialBasis; throws if it
        /// isn't one (NAO two-electron currently only supports FE-backed
        /// underlying bases -- STO/GTO have closed-form 4-index tensors
        /// and would need a different code path).
        const FEMRadialBasis & underlying_fem_() const {
          const auto * fem = dynamic_cast<const FEMRadialBasis *>(underlying_.get());
          if (!fem)
            throw std::logic_error(
                "NAORadialBasis::coulomb_radial / exchange_radial: underlying "
                "basis is not a FEMRadialBasis; per-multipole two-electron "
                "operators are only implemented for FE-backed NAOs.\n");
          return *fem;
        }
      };

    } // namespace basis
  } // namespace atomic
} // namespace helfem

#endif
