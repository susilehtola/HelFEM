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
#ifndef ATOMIC_BASIS_TWODBASIS_H
#define ATOMIC_BASIS_TWODBASIS_H

#include <armadillo>
#include "../general/polynomial_basis.h"
#include "../general/model_potential.h"
#include "../general/sap.h"
#include "basis.h"

namespace helfem {
  namespace atomic {
    namespace basis {
      /// Two-dimensional basis set
      class TwoDBasis {
        /// Nuclear charge
        int Z;
        /// Nuclear model
        modelpotential::nuclear_model_t model;
        /// Rms radius
        double Rrms;

        /// Left-hand nuclear charge
        int Zl;
        /// Right-hand nuclear charge
        int Zr;
        /// Bond length
        double Rhalf;

        /// Yukawa exchange?
        bool yukawa;
        /// Range separation parameter
        double lambda;

        /// Radial basis set
        RadialBasis radial;
        /// Angular basis set: function l values
        arma::ivec lval;
        /// Angular basis set: function m values
        arma::ivec mval;

        /// Auxiliary integrals
        std::vector<arma::mat> disjoint_L, disjoint_m1L;
        /// Auxiliary integrals for Yukawa separation
        std::vector<arma::mat> disjoint_iL, disjoint_kL;
        /// Primitive two-electron integrals: <Nel^2 * (2L+1)>
        std::vector<arma::mat> prim_tei;
        /// Primitive two-electron integrals: <Nel^2 * (2L+1)> sorted for exchange
        std::vector<arma::mat> prim_ktei;
        /// Primitive range-separated two-electron integrals: <Nel^2 * (2L+1)> sorted for exchange
        std::vector<arma::mat> rs_ktei;

        /// Add to radial submatrix
        void add_sub(arma::mat & M, size_t iang, size_t jang, const arma::mat & Msub) const;
        /// Set radial submatrix
        void set_sub(arma::mat & M, size_t iang, size_t jang, const arma::mat & Msub) const;
        /// Get radial submatrix
        arma::mat get_sub(const arma::mat & M, size_t iang, size_t jang) const;

      public:
        TwoDBasis();
        /// Constructor
        TwoDBasis(int Z, modelpotential::nuclear_model_t model, double Rrms, const polynomial_basis::PolynomialBasis * poly, int n_quad, const arma::vec & bval, const arma::ivec & lval, const arma::ivec & mval, int Zl, int Zr, double Rhalf);
        /// Destructor
        ~TwoDBasis();

        /// Get Z
        int get_Z() const;
        /// Get Zl
        int get_Zl() const;
        /// Get Zr
        int get_Zr() const;
        /// Get Rhalf
        double get_Rhalf() const;

        /// Get nuclear model
        int get_nuclear_model() const;
        /// Get nuclear size
        double get_nuclear_size() const;

        /// Get l values
        arma::ivec get_lval() const;
        /// Get m values
        arma::ivec get_mval() const;

        /// Get number of quadrature points
        int get_nquad() const;
        /// Get boundary values
        arma::vec get_bval() const;
        /// Get polynomial basis identifier
        int get_poly_id() const;
        /// Get polynomial basis order
        int get_poly_order() const;

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
        /// Compute range-separated two-electron integrals
        void compute_yukawa(double lambda);
        /// Compute range-separated two-electron integrals
        void compute_erfc(double mu);

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
	/// Form model potential matrix
	arma::mat model_potential(const modelpotential::ModelPotential * model) const;
        /// Form dipole coupling matrix
        arma::mat dipole_z() const;
        /// Form quadrupole coupling matrix
        arma::mat quadrupole_zz() const;

        /// Compute overlap matrix
        arma::mat overlap(const TwoDBasis & rh) const;

        /// Coupling to magnetic field in z direction
        arma::mat Bz_field(double B) const;

        /// Form density matrix
        arma::mat form_density(const arma::mat & C, size_t nocc) const;

        /// Form Coulomb matrix
        arma::mat coulomb(const arma::mat & P) const;
        /// Form exchange matrix
        arma::mat exchange(const arma::mat & P) const;
        /// Form range-separated exchange matrix
        arma::mat rs_exchange(const arma::mat & P) const;

        /// Get primitive integrals
        std::vector<arma::mat> get_prim_tei() const;

        /// Get l values
        arma::ivec get_l() const;
        /// Get m values
        arma::ivec get_m() const;
        /// Get indices of basis functions with wanted m quantum number
        arma::uvec m_indices(int m) const;
        /// Get indices of basis functions with wanted l and m quantum numbers
        arma::uvec lm_indices(int l, int m) const;
        /// Get indices for wanted symmetry
        std::vector<arma::uvec> get_sym_idx(int isym) const;

        /// Evaluate basis functions
        arma::cx_mat eval_bf(size_t iel, double cth, double phi) const;
        /// Evaluate basis functions derivatives
        void eval_df(size_t iel, double cth, double phi, arma::cx_mat & dr, arma::cx_mat & dth, arma::cx_mat & dphi) const;
        /// Get list of basis function indices in element
        arma::uvec bf_list(size_t iel) const;

        /// Get number of radial elements
        size_t get_rad_Nel() const;
        /// Get radial quadrature weights
        arma::vec get_wrad(size_t iel) const;
        /// Get r values
        arma::vec get_r(size_t iel) const;

        /// Electron density at nuclei
        arma::vec nuclear_density(const arma::mat & P) const;
        /// Electron density gradient at nuclei
        arma::vec nuclear_density_gradient(const arma::mat & P) const;
      };
    }
  }
}


#endif
