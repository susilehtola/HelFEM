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
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */
#ifndef DIATOMIC_BASIS_H
#define DIATOMIC_BASIS_H

#include <armadillo>
#include "helfem/FiniteElementBasis.h"
#include "../general/gaunt.h"
#include "../general/legendretable.h"

namespace helfem {
  namespace diatomic {
    namespace basis {
      /// Radial basis set
      class RadialBasis {
        /// Quadrature points
        arma::vec xq;
        /// Quadrature weights
        arma::vec wq;
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
        arma::vec get_bval() const;
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

        /// Form density matrix
        arma::mat form_density(const arma::mat & Cl, const arma::mat & Cr, size_t nocc) const;

        /// Compute radial matrix elements
        arma::mat radial_integral(int m, int n) const;
        /// Compute primitive kinetic energy matrix in element (excluding l and m parts)
        arma::mat kinetic(size_t iel) const;
        /// Compute primitive kinetic energy matrix
        arma::mat kinetic() const;

        /// Form overlap matrix
        arma::mat overlap(const RadialBasis & rh, int n) const;

        /// Compute Plm integral
        arma::mat Plm_integral(int beta, size_t iel, int L, int M, const legendretable::LegendreTable & legtab) const;
        /// Compute Qlm integral
        arma::mat Qlm_integral(int alpha, size_t iel, int L, int M, const legendretable::LegendreTable & legtab) const;
        /// Compute primitive two-electron integral
        arma::mat twoe_integral(int alpha, int beta, size_t iel, int L, int M, const legendretable::LegendreTable & legtab) const;

        /// Get quadrature points
        arma::vec get_chmu_quad() const;
        /// Evaluate basis functions at quadrature points
        arma::mat get_bf(size_t iel) const;
        /// Evaluate basis functions at wanted point in [-1,1]
        arma::mat get_bf(size_t iel, const arma::vec & x) const;
        /// Evaluate all basis functions at given value of mu
        arma::mat get_bf(double mu) const;
        /// Evaluate derivatives of basis functions at quadrature points
        arma::mat get_df(size_t iel) const;
        /// Get quadrature weights
        arma::vec get_wrad(size_t iel) const;
        /// Get r values
        arma::vec get_r(size_t iel) const;
      };

      /// L, |M| index type
      typedef std::pair<int, int> lmidx_t;
      /// Sort operator
      bool operator<(const lmidx_t & lh, const lmidx_t & rh);
      /// Equivalence operator
      bool operator==(const lmidx_t & lh, const lmidx_t & rh);

      /// l(m) array to l, m arrays
      void lm_to_l_m(const arma::ivec & lmmax, arma::ivec & lval, arma::ivec & mval);

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

        /// L, |M| map
        std::vector<lmidx_t> lm_map;
        /// L, M map
        std::vector<lmidx_t> LM_map;
        /// Auxiliary integrals, Plm
        std::vector<arma::mat> disjoint_P0, disjoint_P2;
        /// Auxiliary integrals, Qlm
        std::vector<arma::mat> disjoint_Q0, disjoint_Q2;
        /// Primitive two-electron integrals: <Nel^2 * N_L>
        std::vector<arma::mat> prim_tei00, prim_tei02, prim_tei20, prim_tei22;
        /// Primitive two-electron integrals: <Nel^2 * N_L> sorted for exchange
        std::vector<arma::mat> prim_ktei00, prim_ktei02, prim_ktei20, prim_ktei22;

        /// Add to radial submatrix
        void add_sub(arma::mat & M, size_t iang, size_t jang, const arma::mat & Msub) const;
        /// Set radial submatrix
        void set_sub(arma::mat & M, size_t iang, size_t jang, const arma::mat & Msub) const;
        /// Get radial submatrix
        arma::mat get_sub(const arma::mat & M, size_t iang, size_t jang) const;

        /// Find index in (L,|M|) table
        size_t lmind(int L, int M, bool check=true) const;
        /// Find index in (L,M) table
        size_t LMind(int L, int M, bool check=true) const;

      public:
        // Dummy constructor
        TwoDBasis();
        /// Constructor
        TwoDBasis(int Z1, int Z2, double Rbond, const std::shared_ptr<const polynomial_basis::PolynomialBasis> &poly, int n_quad, const arma::vec & bval, const arma::ivec & lval, const arma::ivec & mval, int lpad=0, bool legendre=true);
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
        int get_poly_order() const;
        /// Get polynomial basis order
        int get_poly_nnodes() const;

        /// Get indices of real basis functions
        arma::uvec pure_indices() const;
        /// Expand boundary conditions
        arma::mat expand_boundaries(const arma::mat & H) const;
        /// Remove boundary conditions
        arma::mat remove_boundaries(const arma::mat & H) const;

        /// Memory for one-electron integral matrix
        size_t mem_1el() const;
        /// Memory for auxiliary one-electron integrals (off-center nuclei)
        size_t mem_1el_aux() const;
        /// Memory for auxiliary two-electron integrals
        size_t mem_2el_aux() const;

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

        /// Form half-overlap matrix
        arma::mat Shalf(bool chol, int sym) const;
        /// Form half-inverse overlap matrix
        arma::mat Sinvh(bool chol, int sym) const;
        /// Form radial integral
        arma::mat radial_integral(int n) const;
        /// Form overlap matrix
        arma::mat overlap() const;
        /// Form kinetic energy matrix
        arma::mat kinetic() const;
        /// Form nuclear attraction matrix
        arma::mat nuclear() const;
        /// Form dipole coupling matrix
        arma::mat dipole_z() const;
        /// Form dipole coupling matrix
        arma::mat quadrupole_zz() const;

        /// Form overlap matrix
        arma::mat overlap(const TwoDBasis & rh) const;

        /// Coupling to magnetic field in z direction
        arma::mat Bz_field(double B) const;

        /// <r^2> matrix
        arma::mat radial_moments(const arma::mat & P) const;

        /// Form density matrix
        arma::mat form_density(const arma::mat & C, size_t nocc) const;

        /// Form Coulomb matrix
        arma::mat coulomb(const arma::mat & P) const;
        /// Form exchange matrix
        arma::mat exchange(const arma::mat & P) const;

        /// Get primitive integrals
        std::vector<arma::mat> get_prim_tei() const;

        /// Set elements to zero
        void set_zero(int lmax, arma::mat & M) const;

        /// Get l values
        arma::ivec get_l() const;
        /// Get m values
        arma::ivec get_m() const;
        /// Get indices of basis functions with wanted m quantum number
        arma::uvec m_indices(int m) const;
        /// Get indices of basis functions with wanted m quantum number and parity
        arma::uvec m_indices(int m, bool odd) const;
        /// Get indices for wanted symmetry
        std::vector<arma::uvec> get_sym_idx(int isym) const;

        /// Evaluate basis functions at quadrature points
        arma::cx_mat eval_bf(size_t iel, size_t irad, double cth, double phi) const;
        /// Evaluate basis functions at wanted x value
        arma::cx_mat eval_bf(size_t iel, const arma::vec & x, double cth, double phi) const;
        /// Evaluate basis functions with m=m at quadrature point
        arma::mat eval_bf(size_t iel, size_t irad, double cth, int m) const;

	/// Evaluate basis functions at wanted point
	arma::cx_vec eval_bf(double mu, double cth, double phi) const;

        /// Evaluate basis functions derivatives at quadrature points
        void eval_df(size_t iel, size_t irad, double cth, double phi, arma::cx_mat & dr, arma::cx_mat & dth, arma::cx_mat & dphi) const;
        /// Get list of basis function indices in element
        arma::uvec bf_list(size_t iel) const;
        /// Get list of basis function indices in element with m=m
        arma::uvec bf_list(size_t iel, int m) const;

        /// Get number of radial elements
        size_t get_rad_Nel() const;
        /// Get radial quadrature weights
        arma::vec get_wrad(size_t iel) const;
        /// Get r values
        arma::vec get_r(size_t iel) const;

        /// Electron density at nuclei
        arma::vec nuclear_density(const arma::mat & P) const;
      };
    }
  }
}

#endif
