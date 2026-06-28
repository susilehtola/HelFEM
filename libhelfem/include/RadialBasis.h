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
#ifndef ATOMIC_BASIS_RADIALBASIS_H
#define ATOMIC_BASIS_RADIALBASIS_H

#include "ModelPotential.h"
#include "FiniteElementBasis.h"
#include <armadillo>

namespace helfem {
  namespace atomic {
    namespace basis {
      /// Abstract radial basis interface. Implementations provide the
      /// global one-electron matrices in the u = r * R(r) representation
      /// (integration measure dr; the r^2 volume element is absorbed in u^2).
      /// Concrete subclasses: FEMRadialBasis (this file); future
      /// NAORadialBasis, STORadialBasis, GTORadialBasis.
      class RadialBasis {
       public:
        virtual ~RadialBasis() = default;

        /// Number of basis functions
        virtual size_t Nbf() const = 0;

        /// Overlap matrix S_{ij} = integral u_i(r) u_j(r) dr
        virtual arma::mat overlap() const = 0;
        /// Radial kinetic matrix (1/2) integral u'_i(r) u'_j(r) dr.
        /// EXCLUDES the centrifugal term -- caller adds l*(l+1) * kinetic_l().
        virtual arma::mat kinetic() const = 0;
        /// Half the centrifugal-per-l(l+1) matrix:
        /// (1/2) integral u_i(r) u_j(r) / r^2 dr.
        /// Full centrifugal contribution is l*(l+1) * kinetic_l().
        virtual arma::mat kinetic_l() const = 0;
        /// Nuclear attraction at Z=1 (sign included):
        /// - integral u_i(r) u_j(r) / r dr.
        /// For arbitrary Z, the caller multiplies by +Z.
        virtual arma::mat nuclear() const = 0;

        /// Evaluate orbitals (columns of C) at a given r.
        virtual arma::vec eval_orbs(const arma::mat & C, double r) const = 0;
      };

      /// Finite-element radial basis set.
      ///
      /// Implements RadialBasis using a Lagrange/Hermite finite-element basis
      /// on a user-supplied radial grid; this is HelFEM's original concrete
      /// implementation (formerly named RadialBasis prior to the v2 refactor).
      class FEMRadialBasis : public RadialBasis {
        /// Quadrature points
        arma::vec xq;
        /// Quadrature weights
        arma::vec wq;
        /// Finite element basis
        polynomial_basis::FiniteElementBasis fem;

        // The Taylor pipeline used in earlier HelFEM versions to handle
        // B(r)/r near r=0 has been replaced by an analytic deflation in
        // PolynomialBasis::eval_over_r (called via FiniteElementBasis).
        // The small_r_taylor_cutoff / taylor_order / taylor_diff state and
        // the set_small_r_taylor_cutoff() helper are no longer needed.

      public:
        /// Dummy constructor
        FEMRadialBasis();
        /// Construct radial basis
        FEMRadialBasis(const polynomial_basis::FiniteElementBasis & fem, int n_quad, int taylor_order);
        /// Explicit destructor
        ~FEMRadialBasis() override;

        /// Add an element boundary
        void add_boundary(double r);

        /// Get polynomial basis
        std::shared_ptr<polynomial_basis::PolynomialBasis> get_poly() const;
        /// Get the underlying FE basis (read-only).
        const polynomial_basis::FiniteElementBasis & get_fem() const { return fem; }

        /// Get number of quadrature points
        int get_nquad() const;
        /// Get quadrature points
        arma::vec get_xq() const;
        /// Get boundary values
        arma::vec get_bval() const;
        /// Get polynomial basis identifier
        int get_poly_id() const;
        /// Get number of nodes in polynomial basis
        int get_poly_nnodes() const;
        /// Get small r Taylor cutoff
        double get_small_r_taylor_cutoff() const;
        /// Get the order of the Taylor series
        int get_taylor_order() const;
        /// Get the error in the Taylor series
        double get_taylor_diff() const;

        /// Get number of overlapping functions
        size_t get_noverlap() const;
        /// Number of basis functions
        size_t Nbf() const override;
        /// Number of primitive functions in element
        size_t Nprim(size_t iel) const;
        /// Number of primitive functions in element
        size_t max_Nprim() const;

        /// Number of elements
        size_t Nel() const;
        /// Get function indices
        void get_idx(size_t iel, size_t &ifirst, size_t &ilast) const;

        /// Compute radial matrix elements <r^n> in element (overlap is n=0,
        /// nuclear is n=-1)
        arma::mat radial_integral(const arma::mat &bf_c, int n,
                                  size_t iel) const;
        /// Compute radial matrix elements <r^n> in element (overlap is n=0,
        /// nuclear is n=-1)
        arma::mat radial_integral(int n, size_t iel, double x_left = -1.0, double x_right = 1.0) const;

        /// Compute Bessel i_L integral
        arma::mat bessel_il_integral(int L, double lambda, size_t iel) const;
        /// Compute Bessel k_L integral
        arma::mat bessel_kl_integral(int L, double lambda, size_t iel) const;

        /// Identifies which representation of the radial basis to use when
        /// evaluating bra/ket factors in a matrix element. Bn is the raw FE
        /// polynomial B(r) = r*R(r) (or its n-th r-derivative); Rn is the
        /// physical radial wavefunction R(r) = B(r)/r (or its n-th derivative),
        /// computed via the analytic eval_over_r deflation on the first
        /// element and direct division elsewhere. Both are first-class:
        /// pick whichever makes the integrand smooth for the weight function
        /// in question.
        enum class BasisKind { B0, B1, B2, R0, R1, R2 };

        /// Generic single-element matrix element
        ///     M_ij = integral  bra_i(r) * weight(r) * ket_j(r)  dr
        /// over element `iel`, with bra/ket chosen via BasisKind. Mirrors
        /// FiniteElementBasis::matrix_element and is the engine that the
        /// named matrix-element methods below (overlap, kinetic, nuclear, ...)
        /// can be composed from. Default weight is identity (1).
        arma::mat matrix_element(
            size_t iel, BasisKind bra, BasisKind ket,
            const std::function<double(double)> & weight =
                std::function<double(double)>()) const;

        /// Same as above, summed over every element of the basis.
        arma::mat matrix_element(
            BasisKind bra, BasisKind ket,
            const std::function<double(double)> & weight =
                std::function<double(double)>()) const;

        /// Compute overlap matrix
        arma::mat overlap() const override;
        /// Compute overlap matrix in element
        arma::mat overlap(size_t iel) const;

        /// Compute primitive kinetic energy matrix (excluding l part)
        arma::mat kinetic() const override;
        /// Compute primitive kinetic energy matrix in element (excluding l
        /// part)
        arma::mat kinetic(size_t iel) const;
        /// Compute l part of kinetic energy matrix
        arma::mat kinetic_l() const override;
        /// Compute l part of kinetic energy matrix in element
        arma::mat kinetic_l(size_t iel) const;
        /// Compute nuclear attraction matrix
        arma::mat nuclear() const override;
        /// Compute nuclear attraction matrix in element
        arma::mat nuclear(size_t iel) const;
	/// Compute polynomial confinement potential matrix in element
	arma::mat polynomial_confinement(size_t iel, int N, double shift_pot) const;
	/// Compute exponential confinement potential matrix in element
	arma::mat exponential_confinement(size_t iel, int N, double r_0, double shift_pot) const;
	/// Compute barrier confinement potential matrix in element
	arma::mat barrier_confinement(size_t iel, double V, double r_c) const;
	/// Compute Junquera et al. confinement potential matrix in element
	arma::mat junq_confinement(size_t iel, int N, double V0, double r_c, double shift_pot) const;
	/// Driver for computing confinement potential
	arma::mat confinement_potential(size_t iel, int N, double r_0, int iconf, double V, double shift_pot) const;

        /// Compute model potential matrix in element
        arma::mat model_potential(const modelpotential::ModelPotential *nuc,
                                  size_t iel) const;
        /// Compute off-center nuclear attraction matrix in element
        arma::mat nuclear_offcenter(size_t iel, double Rhalf, int L) const;

        /// Compute primitive two-electron integral
        arma::mat twoe_integral(int L, size_t iel) const;
        /// Compute primitive Yukawa-screened two-electron integral
        arma::mat yukawa_integral(int L, double lambda, size_t iel) const;
        /// Compute primitive complementary error function two-electron integral
        arma::mat erfc_integral(int L, double lambda, size_t iel,
                                size_t jel) const;
        /// Compute a spherically symmetric potential
        arma::mat spherical_potential(size_t iel) const;

        /// Compute cross-basis integral
        arma::mat radial_integral(const FEMRadialBasis &rh, int n,
                                  bool lhder = false, bool rhder = false) const;
        /// Compute cross-basis model potential integral
        arma::mat model_potential(const FEMRadialBasis &rh,
                                  const modelpotential::ModelPotential *model,
                                  bool lhder = false, bool rhder = false) const;
        /// Compute projection
        arma::mat overlap(const FEMRadialBasis &rh) const;

        /// Evaluate basis functions at quadrature points
        arma::mat get_bf(size_t iel) const;
        /// Evaluate basis functions at given points
        arma::mat get_bf(const arma::vec & x, size_t iel) const;
        /// Evaluate derivatives of basis functions at quadrature points
        arma::mat get_df(size_t iel) const;
        /// Evaluate basis functions at given points
        arma::mat get_df(const arma::vec & x, size_t iel) const;
        /// Evaluate second derivatives of basis functions at quadrature points
        arma::mat get_lf(size_t iel) const;
        /// Evaluate basis functions at given points
        arma::mat get_lf(const arma::vec & x, size_t iel) const;
        /// Evaluate orbitals at a given point
        arma::vec eval_orbs(const arma::mat & C, double r) const override;

        /// Get quadrature weights
        arma::vec get_wrad(size_t iel) const;
        /// Get quadrature weights
        arma::vec get_wrad(const arma::vec & w, size_t iel) const;
        /// Get r values
        arma::vec get_r(size_t iel) const;
        /// Get r values
        arma::vec get_r(const arma::vec & x, size_t iel) const;
	/// Get r value
        double get_r(double x, size_t iel) const;

        /// Evaluate nuclear density
        double nuclear_density(const arma::mat &P) const;
        /// Evaluate nuclear density gradient
        double nuclear_density_gradient(const arma::mat &P) const;
        /// Evaluate orbitals at nucleus
        arma::rowvec nuclear_orbital(const arma::mat &C) const;
      };
    } // namespace basis
  }   // namespace atomic
} // namespace helfem

#endif
