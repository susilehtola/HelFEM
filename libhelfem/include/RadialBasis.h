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
#include "Matrix.h"
#include <vector>

namespace helfem {
  namespace atomic {
    namespace basis {
      /// Abstract radial basis interface. Implementations provide the
      /// global one-electron matrices in the u = r * R(r) representation
      /// (integration measure dr; the r^2 volume element is absorbed in u^2).
      /// Concrete subclasses: FEMRadialBasisT (this file); future
      /// NAORadialBasis, STORadialBasis, GTORadialBasis.
      ///
      /// Templated on the scalar type, following FiniteElementBasisT<T>.
      template <typename T>
      class RadialBasisT {
       public:
        virtual ~RadialBasisT() = default;

        /// Number of basis functions
        virtual size_t Nbf() const = 0;

        /// Overlap matrix S_{ij} = integral u_i(r) u_j(r) dr.
        virtual helfem::Mat<T> overlap() const = 0;
        /// Radial kinetic matrix (1/2) integral u'_i(r) u'_j(r) dr.
        /// EXCLUDES the centrifugal term -- caller adds l*(l+1) * kinetic_l().
        /// Eigen-typed (Phase 2a migration; follows overlap() in PR #103).
        virtual helfem::Mat<T> kinetic() const = 0;
        /// Half the centrifugal-per-l(l+1) matrix:
        /// (1/2) integral u_i(r) u_j(r) / r^2 dr.
        /// Full centrifugal contribution is l*(l+1) * kinetic_l().
        /// Eigen-typed (Phase 2a migration).
        virtual helfem::Mat<T> kinetic_l() const = 0;
        /// Nuclear attraction at Z=1 (sign included):
        /// - integral u_i(r) u_j(r) / r dr.
        /// For arbitrary Z, the caller multiplies by +Z.
        /// Eigen-typed (Phase 2a migration).
        virtual helfem::Mat<T> nuclear() const = 0;

        /// Evaluate orbitals (columns of C) at a given r.
        /// Phase 5.23: Eigen-typed argument + return.
        virtual helfem::Vec<T> eval_orbs(const helfem::Mat<T> & C, T r) const = 0;
      };

      /// The double instantiation, which every existing caller uses.
      using RadialBasis = RadialBasisT<double>;

      /// Finite-element radial basis set.
      ///
      /// Implements RadialBasisT using a Lagrange/Hermite finite-element basis
      /// on a user-supplied radial grid; this is HelFEM's original concrete
      /// implementation (formerly named RadialBasis prior to the v2 refactor).
      ///
      /// Templated on the scalar type, exactly as FiniteElementBasisT<T>:
      /// everything below it (the FE basis, the Gauss-Chebyshev quadrature,
      /// the model potentials, the Bessel functions) now follows T, so a
      /// FEMRadialBasisT<long double> is genuinely a long-double calculation.
      /// Explicitly instantiated for double and long double.
      template <typename T>
      class FEMRadialBasisT : public RadialBasisT<T> {
        /// Quadrature points
        // Phase 5.6: quadrature node/weight members migrated to Eigen.
        helfem::Vec<T> xq;
        /// Quadrature weights
        helfem::Vec<T> wq;
        /// Finite element basis
        polynomial_basis::FiniteElementBasisT<T> fem;

      public:
        /// Dummy constructor
        FEMRadialBasisT();
        /// Construct radial basis
        FEMRadialBasisT(const polynomial_basis::FiniteElementBasisT<T> & fem, int n_quad);
        /// Explicit destructor
        ~FEMRadialBasisT() override;

        /// Add an element boundary
        void add_boundary(T r);

        /// Get polynomial basis
        std::shared_ptr<helfem::lib1dfem::polynomial_basis::PolynomialBasis<T>> get_poly() const;
        /// Get the underlying FE basis (read-only).
        const polynomial_basis::FiniteElementBasisT<T> & get_fem() const { return fem; }

        /// Get number of quadrature points
        int get_nquad() const;
        /// Get quadrature points (Phase 5.20: Eigen at the public boundary).
        helfem::Vec<T> get_xq() const;
        /// Get boundary values (Phase 5.20: Eigen at the public boundary).
        helfem::Vec<T> get_bval() const;
        /// Get polynomial basis identifier
        int get_poly_id() const;
        /// Get number of nodes in polynomial basis
        int get_poly_nnodes() const;

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
        /// nuclear is n=-1). Eigen-typed (Phase 2a migration).
        helfem::Mat<T> radial_integral(int n, size_t iel, T x_left = T(-1), T x_right = T(1)) const;

        /// Compute Bessel i_L integral (Eigen; Phase 2a).
        helfem::Mat<T> bessel_il_integral(int L, T lambda, size_t iel) const;
        /// Compute Bessel k_L integral (Eigen; Phase 2a).
        helfem::Mat<T> bessel_kl_integral(int L, T lambda, size_t iel) const;

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
        /// FiniteElementBasisT::matrix_element and is the engine that the
        /// named matrix-element methods below (overlap, kinetic, nuclear, ...)
        /// can be composed from. Default weight is identity (1).
        /// Eigen-typed (Phase 2a migration). Phase 2: routes through the
        /// finite-element basis's auto-converging quadrature. `breakpoints`
        /// are real-space points where the weight is non-smooth (the element
        /// is split there); `poly_degree_f` is the polynomial degree of the
        /// weight if known (>=0, else -1), used only to seed the order.
        helfem::Mat<T> matrix_element(
            size_t iel, BasisKind bra, BasisKind ket,
            const std::function<T(T)> & weight = std::function<T(T)>(),
            const std::vector<T> & breakpoints = std::vector<T>(),
            int poly_degree_f = -1) const;

        /// Same as above, summed over every element of the basis.
        helfem::Mat<T> matrix_element(
            BasisKind bra, BasisKind ket,
            const std::function<T(T)> & weight = std::function<T(T)>(),
            const std::vector<T> & breakpoints = std::vector<T>(),
            int poly_degree_f = -1) const;

        /// Per-element matrix element restricted to a sub-range of the
        /// reference element [x_left, x_right] (defaults to the full range
        /// [-1, +1]). Useful for two-electron inner integrals and other
        /// piecewise integrations.
        helfem::Mat<T> matrix_element(
            size_t iel, BasisKind bra, BasisKind ket,
            const std::function<T(T)> & weight,
            T x_left, T x_right) const;

        /// Cross-basis matrix element
        ///     M_ij = integral  bra_i(r) * weight(r) * ket_j(r) dr,
        /// where bra is taken from this basis and ket from `rh`. The two
        /// bases may have different element layouts; this routine finds
        /// every overlapping (iel, jel) element pair and runs a separate
        /// Chebyshev quadrature on each pair's r-overlap interval, with
        /// n_quad = max(this->n_quad, rh.n_quad). Only B0, B1, B2 are
        /// meaningful here -- R-kinds (B(r)/r) are tied to a single basis's
        /// element-length and aren't well-defined cross-basis.
        helfem::Mat<T> matrix_element(
            const FEMRadialBasisT & rh,
            BasisKind bra, BasisKind ket,
            const std::function<T(T)> & weight = std::function<T(T)>()) const;

        /// Compute overlap matrix (Eigen-typed; Phase 2a migration).
        helfem::Mat<T> overlap() const override;
        /// Compute overlap matrix in element (Eigen-typed; Phase 2a).
        helfem::Mat<T> overlap(size_t iel) const;

        /// Compute primitive kinetic energy matrix (excluding l part).
        /// Eigen-typed; Phase 2a migration.
        helfem::Mat<T> kinetic() const override;
        /// Compute primitive kinetic energy matrix in element (excluding l
        /// part). Eigen-typed (Phase 2a).
        helfem::Mat<T> kinetic(size_t iel) const;
        /// Compute l part of kinetic energy matrix (Eigen; Phase 2a).
        helfem::Mat<T> kinetic_l() const override;
        /// Compute l part of kinetic energy matrix in element (Eigen; Phase 2a).
        helfem::Mat<T> kinetic_l(size_t iel) const;
        /// Compute nuclear attraction matrix (Eigen; Phase 2a).
        helfem::Mat<T> nuclear() const override;
        /// Compute nuclear attraction matrix in element (Eigen; Phase 2a).
        helfem::Mat<T> nuclear(size_t iel) const;
	// Phase 2a wrap-up: confinement helpers Eigen-typed; chain together
	// via confinement_potential which dispatches by iconf.
	/// Compute polynomial confinement potential matrix in element
	helfem::Mat<T> polynomial_confinement(size_t iel, int N, T shift_pot) const;
	/// Compute exponential confinement potential matrix in element
	helfem::Mat<T> exponential_confinement(size_t iel, int N, T r_0, T shift_pot) const;
	/// Compute barrier confinement potential matrix in element
	helfem::Mat<T> barrier_confinement(size_t iel, T V, T r_c) const;
	/// Compute Junquera et al. confinement potential matrix in element
	helfem::Mat<T> junq_confinement(size_t iel, int N, T V0, T r_c, T shift_pot) const;
	/// Driver for computing confinement potential
	helfem::Mat<T> confinement_potential(size_t iel, int N, T r_0, int iconf, T V, T shift_pot) const;

        /// Compute model potential matrix in element (Eigen; Phase 2a).
        helfem::Mat<T> model_potential(const modelpotential::ModelPotentialT<T> *nuc,
                                       size_t iel) const;
        /// Compute off-center nuclear attraction matrix in element
        /// (Eigen; Phase 2a wrap-up).
        helfem::Mat<T> nuclear_offcenter(size_t iel, T Rhalf, int L) const;

        /// Compute primitive two-electron integral (Eigen; Phase 2a).
        helfem::Mat<T> twoe_integral(int L, size_t iel) const;
        /// Compute primitive Yukawa-screened two-electron integral
        /// (Eigen; Phase 2a).
        helfem::Mat<T> yukawa_integral(int L, T lambda, size_t iel) const;

        /// Pivoted-Cholesky decomposition of the in-element 4-index
        /// two-electron tensor twoe_integral(L, iel). Returns L of shape
        /// (Ni^2 x r) such that L * L^T == twoe_integral(L, iel) to the
        /// requested absolute tolerance on the residual diagonal (default
        /// 1e-12). The rank r is determined by truncation; for smooth FE
        /// kernels in a single element r is typically much smaller than
        /// Ni^2 -- often O(Ni).
        ///
        /// Same factored representation as the cross-element disjoint
        /// pieces (radial_integral(L)/(-L-1)), so a single contraction
        /// routine over Sum_p L_p(ab) * L_p(cd) handles every (iel, jel)
        /// case uniformly (in-element: r columns; disjoint: rank-1).
        ///
        /// Assumes the in-element tensor is positive semi-definite; this
        /// holds for the bare Coulomb and Yukawa kernels (both PSD as
        /// integral kernels).
        /// Eigen-typed (Phase 2a migration).
        helfem::Mat<T> twoe_integral_cholesky(int L, size_t iel,
                                              T tol = T(1e-12)) const;
        /// Same as above for the Yukawa-screened tensor yukawa_integral.
        /// Eigen-typed (Phase 2a).
        helfem::Mat<T> yukawa_integral_cholesky(int L, T lambda, size_t iel,
                                                T tol = T(1e-12)) const;

        // NOTE: the K-permuted in-element tensor (exchange_tei) is
        // essentially full rank as a PSD matrix (~Ni^2) -- its Cholesky
        // does not compress. Both J and K therefore use the J-ordered
        // factor twoe_integral_cholesky above: J via the inner-product
        // contraction (one matvec * 2 per p), K via the matrix-matrix
        // form M_p . P . M_p^T (two matmuls per p, slower than dense K
        // for typical FE rank-vs-Ni ratios but the canonical
        // RI/density-fitting form expected by external drivers like
        // PySCF's DF backend).
        /// Compute primitive complementary error function two-electron
        /// integral (Eigen; Phase 2a).
        helfem::Mat<T> erfc_integral(int L, T lambda, size_t iel,
                                     size_t jel) const;
        /// Compute a spherically symmetric potential (Eigen; Phase 2a).
        helfem::Mat<T> spherical_potential(size_t iel) const;

        /// Compute cross-basis integral (Eigen; Phase 2a).
        helfem::Mat<T> radial_integral(const FEMRadialBasisT &rh, int n,
                                       bool lhder = false, bool rhder = false) const;
        /// Compute cross-basis model potential integral (Eigen; Phase 2a).
        helfem::Mat<T> model_potential(const FEMRadialBasisT &rh,
                                       const modelpotential::ModelPotentialT<T> *model,
                                       bool lhder = false, bool rhder = false) const;
        /// Compute projection (Eigen; Phase 2a).
        helfem::Mat<T> overlap(const FEMRadialBasisT &rh) const;

        /// Evaluate basis functions at quadrature points (Eigen return).
        helfem::Mat<T> get_bf(size_t iel) const;
        /// Evaluate basis functions at given points (Phase 5.24: Eigen input + return).
        helfem::Mat<T> get_bf(const helfem::Vec<T> & x, size_t iel) const;
        /// Evaluate derivatives of basis functions at quadrature points (Eigen return).
        helfem::Mat<T> get_df(size_t iel) const;
        /// Evaluate derivatives of basis functions at given points
        /// (Phase 5.24: Eigen input + return).
        helfem::Mat<T> get_df(const helfem::Vec<T> & x, size_t iel) const;
        /// Evaluate second derivatives of basis functions at quadrature points (Eigen return).
        helfem::Mat<T> get_lf(size_t iel) const;
        /// Evaluate second derivatives of basis functions at given points
        /// (Phase 5.24: Eigen input + return).
        helfem::Mat<T> get_lf(const helfem::Vec<T> & x, size_t iel) const;
        /// Evaluate orbitals at a given point (Phase 5.23: Eigen-typed).
        helfem::Vec<T> eval_orbs(const helfem::Mat<T> & C, T r) const override;

        /// Get quadrature weights in element.
        helfem::Vec<T> get_wrad(size_t iel) const;
        /// Get quadrature weights in element from user-supplied weight
        /// vector (Phase 5.25: Eigen input + return).
        helfem::Vec<T> get_wrad(const helfem::Vec<T> & w, size_t iel) const;
        /// Get r values at quadrature points in element.
        helfem::Vec<T> get_r(size_t iel) const;
        /// Get r values at user-supplied x points in element
        /// (Phase 5.25: Eigen input + return).
        helfem::Vec<T> get_r(const helfem::Vec<T> & x, size_t iel) const;
	/// Get r value
        T get_r(T x, size_t iel) const;

        /// Evaluate nuclear density (Phase 5.22: Eigen-typed argument).
        T nuclear_density(const helfem::Mat<T> &P) const;
        /// Evaluate nuclear density gradient (Phase 5.22: Eigen-typed argument).
        T nuclear_density_gradient(const helfem::Mat<T> &P) const;
        /// Evaluate orbitals at nucleus (Phase 5.22: Eigen-typed argument
        /// and Eigen row-vector return).
        helfem::RowVec<T> nuclear_orbital(const helfem::Mat<T> &C) const;
      };

      /// The double instantiation, which every existing caller uses.
      using FEMRadialBasis = FEMRadialBasisT<double>;
    } // namespace basis
  }   // namespace atomic
} // namespace helfem

#endif
