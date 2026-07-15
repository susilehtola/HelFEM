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
#ifndef ATOMIC_BASIS_TWODBASIS_H
#define ATOMIC_BASIS_TWODBASIS_H

#include <armadillo>
#include <limits>
#include "../general/model_potential.h"
#include "../general/sap.h"
#include <Matrix.h>
#include <RadialBasis.h>

namespace helfem {
  namespace atomic {
    namespace basis {
      /// Two-dimensional basis set.
      ///
      /// Templated on the scalar type, following FiniteElementBasisT<T> and
      /// FEMRadialBasisT<T>. This is the top of the arbitrary-precision stack:
      /// with it, the whole HARTREE-FOCK path -- overlap, kinetic, nuclear,
      /// coulomb, exchange, Sinvh, compute_tei -- runs at any T, and the
      /// accuracy of a converged calculation is set by the basis rather than
      /// by the arithmetic. Instantiated for double, long double and (under
      /// HELFEM_HAVE_FLOAT128) _Float128.
      ///
      /// NOT templated, and deliberately so:
      ///   * the DFT path. libxc is a double-only C library, so a
      ///     higher-precision XC energy is not merely unimplemented but
      ///     meaningless. The DFT grid lives in atomic/dftgrid.* and consumes
      ///     TwoDBasis (= TwoDBasisT<double>) as before.
      ///   * the basis-function EVALUATION routines used by that grid and by
      ///     the analysis binaries -- eval_bf / eval_df / eval_lf (which return
      ///     arma::cx_mat and go through the double-only ::spherical_harmonics).
      ///     These are compiled for every T but THROW unless T = double; they
      ///     are not on the Fock path. (The quadrature-point accessors get_bval
      ///     / get_wrad / get_r are precision-generic helfem::Vec<T>.)
      /// The angular-index bookkeeping (get_lval / m_indices / get_sym_idx,
      /// Eigen::VectorXi / std::vector<Eigen::Index>) is integer-valued, hence
      /// precision-free, and is shared by all instantiations unchanged.
      template <typename T>
      class TwoDBasisT {
        /// Nuclear charge
        int Z;
        /// Nuclear model
        modelpotential::nuclear_model_t model;
        /// Rms radius
        T Rrms;

        /// Left-hand nuclear charge
        int Zl;
        /// Right-hand nuclear charge
        int Zr;
        /// Bond length
        T Rhalf;

        /// Yukawa exchange?
        bool yukawa;
        /// Range separation parameter
        T lambda;

        /// Radial basis set
        FEMRadialBasisT<T> radial;
        /// Angular basis set: function l values
        Eigen::VectorXi lval;
        /// Angular basis set: function m values
        Eigen::VectorXi mval;
        /// Zero out derivative at practical infinity?
        bool zeroder;

        /// Auxiliary integrals
        std::vector<helfem::Mat<T>> disjoint_L, disjoint_m1L;
        /// Auxiliary integrals for Yukawa separation
        std::vector<helfem::Mat<T>> disjoint_iL, disjoint_kL;
        /// Pivoted-Cholesky factors of the IN-ELEMENT two-electron integrals,
        /// indexed [L*Nel + iel]: L_p of shape (Ni^2 x r) with T = L L'.
        ///
        /// The in-element block is the only 4-index object left -- the
        /// cross-element contributions to J and K are already contracted in
        /// factorized (disjoint r^L / r^-L-1) form. The kernel here is the
        /// single-channel r_<^L / r_>^(L+1), which is a positive kernel, so a
        /// genuine pivoted Cholesky applies (unlike the diatomic 2-channel
        /// kernel, which is indefinite for odd |M| and needs signs).
        ///
        /// It compresses hard: rank 27-29 out of Ni^2 = 196-225 for 15-node
        /// LIPs, reproducing the exact tensor to ~1e-16. Storage and the J
        /// contraction go from O(Ni^4) to O(Ni^2 r).
        ///
        /// The exchange PAIRING of the same integrals is full rank (196/196),
        /// so it cannot be compressed directly; K instead comes from the
        /// standard RI contraction on these J-ordered factors, which is why no
        /// exchange-ordered tensor is stored.
        std::vector<helfem::Mat<T>> prim_chol;
        /// Tolerance for the pivoted Cholesky above.
        ///
        /// This one has to FOLLOW THE ARITHMETIC, or it silently becomes the
        /// accuracy limit of every higher-precision calculation. Truncating the
        /// in-element two-electron tensor at a fixed 1e-12 is invisible at
        /// double -- it is orders of magnitude below double's own roundoff --
        /// but at _Float128 it would be the single largest error in the
        /// calculation, and quad would buy nothing at all.
        ///
        /// So it is held at the same multiple of eps(T) at every precision.
        /// At T = double the ratio is exactly 1 and the value is exactly the
        /// 1e-12 it has always been, so double results are bit-for-bit
        /// unchanged; long double gets ~4.9e-16 and _Float128 ~8.7e-31.
        T chol_tol = T(1e-12) * (std::numeric_limits<T>::epsilon() /
                                 T(std::numeric_limits<double>::epsilon()));
        /// Same factorization for the Yukawa-screened in-element integrals.
        /// The rank bound 2*Nprim-1 is a property of the orbital PRODUCT basis,
        /// not of the kernel, so it applies to the Yukawa kernel unchanged.
        /// (The erfc path keeps its explicit tensors for now: there the
        /// cross-element blocks are not multipole-separable and are stored in
        /// full, so it needs its own treatment.)
        std::vector<helfem::Mat<T>> rs_chol;
        /// Primitive range-separated two-electron integrals: <Nel^2 * (2L+1)> sorted for exchange
        std::vector<helfem::Mat<T>> rs_ktei;

        /// Add a radial block onto the (iang, jang) angular submatrix
        void add_sub(helfem::Mat<T> & M, size_t iang, size_t jang, const helfem::Mat<T> & Msub) const;
        /// Set the (iang, jang) angular submatrix to a radial block
        void set_sub(helfem::Mat<T> & M, size_t iang, size_t jang, const helfem::Mat<T> & Msub) const;

      public:
        TwoDBasisT();
        /// Constructor
        // Phase 5.19: bval/lval/mval accepted as Eigen at the public
        // boundary so a consumer does not need to include <armadillo>
        // to instantiate the basis.
        TwoDBasisT(int Z, modelpotential::nuclear_model_t model, T Rrms,
                   const std::shared_ptr<const helfem::lib1dfem::polynomial_basis::PolynomialBasis<T>> &poly,
                   bool zeroder, int n_quad,
                   const helfem::Vec<T> & bval,
                   const Eigen::VectorXi & lval,
                   const Eigen::VectorXi & mval,
                   int Zl, int Zr, T Rhalf);
        /// Destructor
        ~TwoDBasisT();

        /// Get Z
        int get_Z() const;
        /// Get Zl
        int get_Zl() const;
        /// Get Zr
        int get_Zr() const;
        /// Get Rhalf
        T get_Rhalf() const;

        /// Get nuclear model
        int get_nuclear_model() const;
        /// Get nuclear size
        T get_nuclear_size() const;

        /// Get l values
        Eigen::VectorXi get_lval() const;
        /// Get m values
        Eigen::VectorXi get_mval() const;

        /// Get number of quadrature points
        int get_nquad() const;
        /// Get boundary values
        helfem::Vec<T> get_bval() const;
        /// Get polynomial basis identifier
        int get_poly_id() const;
        /// Get number of nodes in polynomial
        int get_poly_nnodes() const;
        /// Is derivative zeroed at infinity?
        int get_zeroder() const;


        /// Compute two-electron integrals
        void compute_tei(bool exchange);
        /// Compute range-separated two-electron integrals
        void compute_yukawa(T lambda);
        /// Compute range-separated two-electron integrals
        void compute_erfc(T mu);

        /// Number of basis functions
        size_t Nbf() const;
        /// Number of dummy basis functions
        size_t Ndummy() const;

        /// Number of radial functions
        size_t Nrad() const;
        /// Number of angular shells
        size_t Nang() const;

        /// Form half-inverse overlap matrix
        helfem::Mat<T> Sinvh(bool chol, int sym) const;
        /// Form overlap matrix
        helfem::Mat<T> overlap() const;
        /// Form kinetic energy matrix
        helfem::Mat<T> kinetic() const;
        /// Form nuclear attraction matrix
        helfem::Mat<T> nuclear() const;
	/// Form confinement potential matrix
	helfem::Mat<T> confinement(const int N, const T r_0, const int iconf, const T V, const T shift_pot) const;
	/// Form model potential matrix
	helfem::Mat<T> model_potential(const modelpotential::ModelPotentialT<T> * model) const;
        /// Form dipole coupling matrix
        helfem::Mat<T> dipole_z() const;
        /// Form quadrupole coupling matrix
        helfem::Mat<T> quadrupole_zz() const;

        /// Cross-basis overlap matrix S12 = <this | rh>. Used by the
        /// SCF driver's --load path to project a saved density from an
        /// old basis into the current basis via C1 = S_new^-1 * S12 * C_old.
        helfem::Mat<T> overlap(const TwoDBasisT & rh) const;

        /// Coupling to magnetic field in z direction
        helfem::Mat<T> Bz_field(T B) const;

        /// Form Coulomb matrix
        helfem::Mat<T> coulomb(const helfem::Mat<T> & P) const;
        /// Form exchange matrix
        helfem::Mat<T> exchange(const helfem::Mat<T> & P) const;
        /// Form range-separated exchange matrix
        helfem::Mat<T> rs_exchange(const helfem::Mat<T> & P) const;

        /// Density-fitted (Cholesky-factored) BARE radial Slater
        /// integrals R^k(i, j, m, n) for k = 0..2*max(lval).
        ///
        /// Returns vec[k][Q] = helfem::Mat<T> of shape (Nrad, Nrad),
        /// the Q-th Cholesky factor for multipole k, so that:
        ///     R^k(i, j, m, n)  =  sum_Q  B[k][Q](i, j) * B[k][Q](m, n)
        /// where R^k is the BARE radial Slater integral (no 4*pi/(2k+1),
        /// no Gaunt -- libatomscf applies those at angular assembly).
        ///
        /// Computed via pivoted Cholesky on R^k as a Nrad^2 x Nrad^2
        /// matrix, with columns generated on-the-fly via
        /// assemble_J_FE_one_multipole_cached_chol using the SCF-cached
        /// per-element integrals. R^k is naturally sparse in the FE
        /// basis (cross-element pairs have zero density), so naux_k
        /// is bounded by sum_iel Ni^2 (typically ~Nel * Ni << Nrad^2).
        ///
        /// `tol` is the residual diagonal threshold for stopping the
        /// Cholesky iteration; pivots below `tol` are dropped.
        std::vector<std::vector<helfem::Mat<T>>>
        radial_df_factors(T tol = T(1e-10)) const;


        /// Get indices of basis functions with wanted m quantum number
        std::vector<Eigen::Index> m_indices(int m) const;
        /// Get indices of basis functions with wanted l and m quantum numbers
        std::vector<Eigen::Index> lm_indices(int l, int m) const;
        /// Get indices for wanted symmetry (one index list per block)
        std::vector<std::vector<Eigen::Index>> get_sym_idx(int isym) const;

        /// Evaluate basis functions (T = double only)
        arma::cx_mat eval_bf(size_t iel, double cth, double phi) const;
        /// Evaluate basis functions derivatives (T = double only)
        void eval_df(size_t iel, double cth, double phi, arma::cx_mat & dr, arma::cx_mat & dth, arma::cx_mat & dphi) const;
        /// Evaluate Laplacian of basis functions (T = double only)
        arma::cx_mat eval_lf(size_t iel, double cth, double phi) const;
        /// Get list of basis function indices in element
        std::vector<Eigen::Index> bf_list(size_t iel) const;

        /// Get number of radial elements
        size_t get_rad_Nel() const;
        /// Get (ifirst, ilast) inclusive radial-function index range
        /// for element iel. Adjacent elements OVERLAP at boundary
        /// indices (one shared index for C^0 LIP, two for C^1 HIP);
        /// callers must account for this when partitioning per-element
        /// quantities.
        std::pair<size_t, size_t> radial_element_range(size_t iel) const;
        /// Get radial quadrature weights
        helfem::Vec<T> get_wrad(size_t iel) const;
        /// Get r values
        helfem::Vec<T> get_r(size_t iel) const;
      };

      /// The double instantiation, which every existing caller uses.
      using TwoDBasis = TwoDBasisT<double>;
    }
  }
}


#endif
