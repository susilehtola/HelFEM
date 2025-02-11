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
#ifndef ATOMIC_BASIS_RADIALBASIS_H
#define ATOMIC_BASIS_RADIALBASIS_H

#include "ModelPotential.h"
#include "FiniteElementBasis.h"
#include <armadillo>

namespace helfem {
  namespace atomic {
    namespace basis {
      /// Radial basis set
      class RadialBasis {
        /// Quadrature points
        arma::vec xq;
        /// Quadrature weights
        arma::vec wq;
        /// Finite element basis
        polynomial_basis::FiniteElementBasis fem;

        /// Value of r for which to switch over to evaluating basis
        /// functions by Taylor series
        double small_r_taylor_cutoff;
        /// Order of Taylor series
        int taylor_order;
        /// Difference at cutoff
        double taylor_diff;

        /// Derivatives of basis functions at origin
        std::vector<arma::rowvec> taylor_df;
        /// Set the cutoff
        void set_small_r_taylor_cutoff();

      public:
        /// Dummy constructor
        RadialBasis();
        /// Construct radial basis
        RadialBasis(const polynomial_basis::FiniteElementBasis & fem, int n_quad, int taylor_order);
        /// Explicit destructor
        ~RadialBasis();

        /// Add an element boundary
        void add_boundary(double r);

        /// Get polynomial basis
        std::shared_ptr<polynomial_basis::PolynomialBasis> get_poly() const;

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
        size_t Nbf() const;
        /// Number of primitive functions in element
        size_t Nprim(size_t iel) const;
        /// Number of primitive functions in element
        size_t max_Nprim() const;

        /// Number of elements
        size_t Nel() const;
        /// Get function indices
        void get_idx(size_t iel, size_t &ifirst, size_t &ilast) const;

        /// Form density matrix
        arma::mat form_density(const arma::mat &Cl, const arma::mat &Cr,
                               size_t nocc) const;

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

        /// Compute overlap matrix
        arma::mat overlap() const;
        /// Compute overlap matrix in element
        arma::mat overlap(size_t iel) const;

        /// Compute primitive kinetic energy matrix (excluding l part)
        arma::mat kinetic() const;
        /// Compute primitive kinetic energy matrix in element (excluding l
        /// part)
        arma::mat kinetic(size_t iel) const;
        /// Compute l part of kinetic energy matrix
        arma::mat kinetic_l() const;
        /// Compute l part of kinetic energy matrix in element
        arma::mat kinetic_l(size_t iel) const;
        /// Compute nuclear attraction matrix
        arma::mat nuclear() const;
        /// Compute nuclear attraction matrix in element
        arma::mat nuclear(size_t iel) const;
	/// Compute polynomial confinement potential matrix in element
	arma::mat polynomial_confinement(size_t iel, int N, double shift_pot) const;
	/// Compute exponential confinement potential matrix in element
	arma::mat exponential_confinement(size_t iel, int N, double r_0, double shift_pot) const;
	/// Compute barrier confinement potential matrix in element
	arma::mat barrier_confinement(size_t iel, double V, double r_c) const;
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
        arma::mat radial_integral(const RadialBasis &rh, int n,
                                  bool lhder = false, bool rhder = false) const;
        /// Compute cross-basis model potential integral
        arma::mat model_potential(const RadialBasis &rh,
                                  const modelpotential::ModelPotential *model,
                                  bool lhder = false, bool rhder = false) const;
        /// Compute projection
        arma::mat overlap(const RadialBasis &rh) const;

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
        /// Evaluate small-r Taylor series
        void get_taylor(const arma::vec & r, const arma::uvec & taylorind, arma::mat & val, int ider) const;

        /// Evaluate orbitals at a given point
        arma::vec eval_orbs(const arma::mat & C, double r) const;

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
