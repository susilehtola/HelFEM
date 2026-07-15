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
#ifndef DIATOMIC_BASIS_H
#define DIATOMIC_BASIS_H

#include <armadillo>
#include "FiniteElementBasis.h"
#include "../general/gaunt.h"
#include "../general/legendretable.h"
#include "quadrature.h"

namespace helfem {
  namespace diatomic {
    namespace basis {
      /// Radial basis set
      class RadialBasis {
        /// Quadrature points
        helfem::Vector xq;
        /// Quadrature weights
        helfem::Vector wq;
        /// Finite element basis
        polynomial_basis::FiniteElementBasis fem;

      public:
        /// Dummy constructor
        RadialBasis();
        /// Construct radial basis
        RadialBasis(const polynomial_basis::FiniteElementBasis & fem, int n_quad);
        /// Destructor
        ~RadialBasis();

        /// Get number of quadrature points
        int get_nquad() const;
        /// Get boundary values
        helfem::Vector get_bval() const;
        /// Get polynomial basis identifier
        int get_poly_id() const;
        /// Get number of nodes in polynomial
        int get_poly_nnodes() const;

        /// Get number of overlapping functions
        size_t get_noverlap() const;
        /// Number of basis functions
        size_t Nbf() const;
        /// Number of primitive functions in element
        size_t Nprim(size_t iel) const;
        /// Number of primitive functions in element
        size_t max_Nprim() const;

        /// Number of elements
        size_t Nel() const;
        /// Get function indices
        void get_idx(size_t iel, size_t & ifirst, size_t & ilast) const;

        /// Compute radial matrix elements
        helfem::Matrix radial_integral(int m, int n) const;
        /// Compute primitive kinetic energy matrix in element (excluding l and m parts)
        helfem::Matrix kinetic(size_t iel) const;
        /// Compute primitive kinetic energy matrix
        helfem::Matrix kinetic() const;

        /// Cross-basis radial overlap with (typically different) rh, with
        /// the sinh(mu) * cosh^n(mu) weight. Used by TwoDBasis::overlap
        /// below when building the cross-basis S12 for the SCF driver's
        /// --load path.
        helfem::Matrix overlap(const RadialBasis & rh, int n) const;

        /// Compute Plm integral
        helfem::Matrix Plm_integral(int beta, size_t iel, int L, int M, const legendretable::LegendreTable & legtab) const;
        /// Compute Qlm integral
        helfem::Matrix Qlm_integral(int alpha, size_t iel, int L, int M, const legendretable::LegendreTable & legtab) const;
        /// Compute primitive two-electron integral
        helfem::Matrix twoe_integral(int alpha, int beta, size_t iel, int L, int M, const legendretable::LegendreTable & legtab) const;

        /// Build the element-only two-electron data (basis functions, product
        /// table, subinterval geometry). Independent of (alpha, beta, L, M), so
        /// compute_tei builds it once per element and reuses it across them.
        quadrature::TwoElectronElement twoe_element(size_t iel) const;
        /// Primitive two-electron integral from precomputed element data
        helfem::Matrix twoe_integral(int alpha, int beta, const quadrature::TwoElectronElement & el, int L, int M, const legendretable::LegendreTable & legtab) const;

        /// Get quadrature points
        helfem::Vector get_chmu_quad() const;
        /// Evaluate basis functions at quadrature points
        helfem::Matrix get_bf(size_t iel) const;
        /// Evaluate basis functions at wanted point in [-1,1]
        helfem::Matrix get_bf(size_t iel, const helfem::Vector & x) const;
        /// Evaluate derivatives of basis functions at quadrature points
        helfem::Matrix get_df(size_t iel) const;
        /// Evaluate second derivatives of basis functions at quadrature points
        helfem::Matrix get_d2f(size_t iel) const;
        /// Get quadrature weights
        helfem::Vector get_wrad(size_t iel) const;
        /// Get r values
        helfem::Vector get_r(size_t iel) const;
      };

      /// L, |M| index type
      typedef std::pair<int, int> lmidx_t;
      /// Sort operator
      bool operator<(const lmidx_t & lh, const lmidx_t & rh);
      /// Equivalence operator
      bool operator==(const lmidx_t & lh, const lmidx_t & rh);

      /// l(m) array to l, m arrays
      void lm_to_l_m(const Eigen::VectorXi & lmmax, Eigen::VectorXi & lval, Eigen::VectorXi & mval);

      /// Two-dimensional basis set
      class TwoDBasis {
        /// Nuclear charges
        int Z1, Z2;
        /// Half-bond distance
        double Rhalf;

        /// Radial basis set
        RadialBasis radial;
        /// Angular basis set: function l values
        arma::ivec lval;
        /// Angular basis set: function m values
        arma::ivec mval;

        /// Gaunt coefficient table
        gaunt::Gaunt gaunt;
        /// Legendre function table
        legendretable::LegendreTable legtab;

        /// Cached pure_indices(): dummy index of each real basis function.
        /// Built once; pure_indices() itself rebuilt it on every call, and it
        /// is called on every boundary expansion / removal.
        std::vector<Eigen::Index> pure_idx;

        /// L, |M| map
        std::vector<lmidx_t> lm_map;
        /// L, M map
        std::vector<lmidx_t> LM_map;
        /// Auxiliary integrals, Plm
        std::vector<helfem::Matrix> disjoint_P0, disjoint_P2;
        /// Auxiliary integrals, Qlm
        std::vector<helfem::Matrix> disjoint_Q0, disjoint_Q2;
        /// Low-rank (Cholesky-type) factorization of the IN-ELEMENT two-electron
        /// integrals, indexed [ilm*Nel + iel].
        ///
        /// The in-element block is the only 4-index object in the code -- the
        /// cross-element contributions to J and K are already contracted in
        /// factorized (disjoint P/Q) form. Written as the symmetric 2-channel
        /// kernel
        ///
        ///     W = [  T00  -T02 ]        (T20 = T02', by construction)
        ///         [ -T02'  T22 ]
        ///
        /// it is 2*Nprim^2 square but has a numerical rank of only about
        /// 2*Nprim -- ~30 of 450 for 15-node LIPs, flat from 1e-6 down to
        /// 1e-12, since the P_L(cosh mu_<) Q_L(cosh mu_>) Green's function is
        /// semi-separable. Storing
        ///
        ///     W = B diag(sigma) B'      B: (2*Nprim^2 x r),  sigma = +-1
        ///
        /// replaces O(Nprim^4) storage and contraction by O(Nprim^2 * r), and
        /// puts the integrals in DF/RI form, so exchange follows from the
        /// standard RI-K contraction and prim_ktei is not needed at all. (The
        /// exchange PAIRING is full rank -- 225 of 225 -- so compressing that
        /// directly is not an option; one has to go through RI.)
        ///
        /// W is indefinite for odd |M|, hence the signs: this is an
        /// eigenvalue-thresholded factorization, not a plain Cholesky.
        std::vector<helfem::Matrix> cd_B;
        std::vector<helfem::Vector> cd_sigma;
        /// Relative eigenvalue threshold for the factorization above
        double cd_thresh;

        /// Add to radial submatrix
        void add_sub(helfem::Matrix & M, size_t iang, size_t jang, const helfem::Matrix & Msub) const;
        /// Set radial submatrix
        void set_sub(helfem::Matrix & M, size_t iang, size_t jang, const helfem::Matrix & Msub) const;

        /// Find index in (L,|M|) table
        size_t lmind(int L, int M, bool check=true) const;
        /// Find index in (L,M) table
        size_t LMind(int L, int M, bool check=true) const;

        /// Build the unsigned-M angular prefactor table
        ///   4 * pi * Rhalf^5 / ((L + |M|)! / (L - |M|)!)
        /// indexed in lockstep with lm_map. The (-1)^M sign is applied at
        /// each lookup site so the same table serves both Coulomb (which
        /// iterates LM_map with signed M) and exchange (which derives M
        /// from mj - mi at use-time).
        std::vector<double> build_LMfac_abs() const;

      public:
        // Dummy constructor
        TwoDBasis();
        /// Constructor
        TwoDBasis(int Z1, int Z2, double Rhalf, const std::shared_ptr<const polynomial_basis::PolynomialBasis> &poly, int n_quad, const helfem::Vector & bval, const Eigen::VectorXi & lval, const Eigen::VectorXi & mval, bool legendre=true);
        /// Destructor
        ~TwoDBasis();

        /// Get Z1
        int get_Z1() const;
        /// Get Z2
        int get_Z2() const;
        /// Get Rhalf
        double get_Rhalf() const;

        /// Get l values
        arma::ivec get_lval() const;
        /// Get m values
        arma::ivec get_mval() const;

        /// Get number of quadrature points
        int get_nquad() const;
        /// Get boundary values
        arma::vec get_bval() const;
	/// Get maximum mu value
	double get_mumax() const;
        /// Get polynomial basis identifier
        int get_poly_id() const;
        /// Get polynomial basis order
        int get_poly_nnodes() const;

        /// Get indices of real basis functions
        arma::uvec pure_indices() const;
        /// Expand boundary conditions
        arma::mat expand_boundaries(const arma::mat & H) const;
        /// Remove boundary conditions
        arma::mat remove_boundaries(const arma::mat & H) const;

        /// Eigen-native boundary expansion / removal.
        ///
        /// These run on every Fock build -- coulomb(), exchange() and the XC
        /// grid all bridge a density in and a matrix out -- and the arma
        /// versions above went through arma's generic index-pair submatrix
        /// (subview_elem2), which showed up as ~5% of the run time. They also
        /// rebuilt the index list on every call. These use the cached index
        /// list and a plain gather/scatter loop.
        helfem::Matrix expand_boundaries(const helfem::Matrix & Ppure) const;
        helfem::Matrix remove_boundaries(const helfem::Matrix & Fnob) const;


        /// Compute two-electron integrals
        void compute_tei(bool exchange);

        /// Number of basis functions
        size_t Nbf() const;
        /// Number of dummy basis functions
        size_t Ndummy() const;

        /// Number of radial functions
        size_t Nrad() const;
        /// Number of angular shells
        size_t Nang() const;

        /// Form half-inverse overlap matrix
        helfem::Matrix Sinvh(bool chol, int sym) const;
        /// Form overlap matrix
        helfem::Matrix overlap() const;
        /// Form kinetic energy matrix
        helfem::Matrix kinetic() const;
        /// Form nuclear attraction matrix
        helfem::Matrix nuclear() const;
        /// Form dipole coupling matrix
        helfem::Matrix dipole_z() const;
        /// Form dipole coupling matrix
        helfem::Matrix quadrupole_zz() const;

        /// Cross-basis overlap matrix S12 = <this | rh>. Used by the SCF
        /// driver's --load path to project a saved density from an old
        /// basis into the current one via P_new = P_proj . P_old . P_proj^T
        /// with P_proj = S_new^-1 . S12.
        helfem::Matrix overlap(const TwoDBasis & rh) const;

        /// Coupling to magnetic field in z direction
        helfem::Matrix Bz_field(double B) const;

        /// Form Coulomb matrix (Eigen-typed public boundary; internal
        /// implementation is arma-native, one to_arma bridge at entry
        /// and one to_eigen bridge at exit).
        helfem::Matrix coulomb(const helfem::Matrix & P) const;
        /// Form exchange matrix (Eigen-typed public boundary; same
        /// bridging convention as coulomb() above).
        helfem::Matrix exchange(const helfem::Matrix & P) const;


        /// Assemble the symmetric 2-channel in-element two-electron kernel for
        /// (element, L, |M|):
        ///     W = [  T00  -T02 ]
        ///         [ -T02'  T22 ]
        /// This is the only 4-index object left in the code -- the
        /// cross-element contributions to J and K are already contracted in
        /// factorized (disjoint P/Q) form. Exposed so its compressibility can
        /// be measured.
        helfem::Matrix in_element_kernel(size_t iel, int L, int M) const;

        /// The same in-element integrals, but re-paired the way exchange uses
        /// them: ktei[(j,k),(i,l)] = (ij|kl). Low rank in the Coulomb pairing
        /// says nothing about the exchange pairing, so it has to be measured
        /// separately. Returns the four blocks concatenated as
        /// [ktei00 | ktei02 | ktei20 | ktei22], which is what the K contraction
        /// applies to [R00; R02; R20; R22].
        helfem::Matrix in_element_kernel_exchange(size_t iel, int L, int M) const;

        /// Self-check of the Cholesky/RI machinery for one (element, L, |M|):
        ///  - how well B diag(sigma) B' reproduces the exact kernel W, and
        ///  - whether the RI-K contraction reproduces what the exact
        ///    exchange-ordered tensor gives, for a supplied set of R blocks.
        /// Returns {relative kernel error, relative RI-K error}.
        std::pair<double,double> check_cd(size_t iel, int L, int M) const;

        /// Get indices of basis functions with wanted m quantum number
        std::vector<Eigen::Index> m_indices(int m) const;
        /// Get indices of basis functions with wanted m quantum number and parity
        std::vector<Eigen::Index> m_indices(int m, bool odd) const;
        /// Get indices for wanted symmetry (one index list per block)
        std::vector<std::vector<Eigen::Index>> get_sym_idx(int isym) const;

        /// Evaluate basis functions at quadrature points
        arma::cx_mat eval_bf(size_t iel, size_t irad, double cth, double phi) const;
        /// Evaluate basis functions at wanted x value
        /// Evaluate basis functions with m=m at quadrature point
        arma::mat eval_bf(size_t iel, size_t irad, double cth, int m) const;
        /// Same, but with the element's radial functions already evaluated
        /// (rad_all = get_rad_bf(iel), rows = radial points). The FEM
        /// polynomials depend only on the element, so callers that loop over
        /// angular points should hoist that evaluation out of the loop rather
        /// than redo it -- and throw away all but one row -- per angular point.
        arma::mat eval_bf(size_t iel, size_t irad, double cth, int m, const arma::mat & rad_all) const;

	/// Evaluate basis functions at wanted point
	arma::cx_vec eval_bf(double mu, double cth, double phi) const;

        /// Evaluate basis functions derivatives at quadrature points
        void eval_df(size_t iel, size_t irad, double cth, double phi, arma::cx_mat & dr, arma::cx_mat & dth, arma::cx_mat & dphi) const;

        /// Evaluate the REAL derivatives of the m-block basis functions at a
        /// (mu, nu) quadrature point. Companion to eval_bf(iel,irad,cth,m):
        /// used by the pure-m (analytic-phi) DFT grid.
        ///
        /// For a pure-m function psi = R(mu) Y_l^m(nu) e^{i m phi}, evaluating
        /// at phi = 0 makes both derivatives real:
        ///   d/dmu : Y_l^m(nu,0) * R'(mu)
        ///   d/dnu : [ m cot(th) Y_l^m + sqrt((l-m)(l+m+1)) Y_l^{m+1} ] * R(mu)
        /// The phi derivative is NOT returned: d psi/d phi = i m psi, so it
        /// contributes only the analytic term m^2 |psi|^2 / h_phi^2 (needed for
        /// tau); the density gradient has no phi component at all since
        /// |e^{i m phi}| = 1 makes rho phi-independent.
        void eval_df(size_t iel, size_t irad, double cth, int m, arma::mat & dr, arma::mat & dth) const;

        /// Evaluate the REAL Laplacian of the m-block basis functions at a
        /// (mu, nu) quadrature point. Used by the pure-m DFT grid for
        /// Laplacian-dependent meta-GGAs.
        ///
        /// In prolate spheroidal coordinates, with h = h_mu = h_nu and
        /// h_phi = Rhalf sinh(mu) sin(nu),
        ///
        ///   grad^2 f = (1/h^2) [ f_mumu + coth(mu) f_mu
        ///                        + f_nunu + cot(nu) f_nu ] - (m^2/h_phi^2) f.
        ///
        /// For f = R(mu) Y_l^m(nu) e^{i m phi} the angular combination is
        /// given in closed form by the associated Legendre equation,
        ///
        ///   Y_nunu + cot(nu) Y_nu = [ m^2/sin^2(nu) - l(l+1) ] Y,
        ///
        /// and its 1/sin^2(nu) cancels exactly against the -m^2/h_phi^2 term.
        /// No second angular derivative is needed and nothing diverges at the
        /// poles; what is left is purely radial:
        ///
        ///   grad^2 f = (1/h^2) [ R'' + coth(mu) R'
        ///                        - ( l(l+1) + m^2/sinh^2(mu) ) R ] Y_l^m.
        void eval_lf(size_t iel, size_t irad, double cth, int m, arma::mat & lf) const;
        /// Translate dummy indices to real indices
        arma::uvec dummy_idx_to_real_idx(const arma::uvec & idx) const;
        /// Get list of basis function dummy indices in element
        arma::uvec bf_list_dummy(size_t iel) const;
        /// Get list of basis function dummy indices in element with m=m
        arma::uvec bf_list_dummy(size_t iel, int m) const;
        /// Get list of basis function indices in element
        arma::uvec bf_list(size_t iel) const;

        /// Get number of radial elements
        size_t get_rad_Nel() const;
        /// Get radial quadrature weights
        arma::vec get_wrad(size_t iel) const;
        /// Get r values
        arma::vec get_r(size_t iel) const;
        /// Get radial basis functions at the quadrature points of element iel
        /// (rows = radial points, cols = element primitives)
        arma::mat get_rad_bf(size_t iel) const;
        /// Get their first mu derivatives, same layout
        arma::mat get_rad_df(size_t iel) const;
        /// Get their second mu derivatives, same layout
        arma::mat get_rad_d2f(size_t iel) const;

        /// Electron density at nuclei
      };
    }
  }
}

#endif
