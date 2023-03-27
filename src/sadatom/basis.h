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
#ifndef SADATOM_BASIS_H
#define SADATOM_BASIS_H

#include <armadillo>
#include "../atomic/basis.h"

namespace helfem {
  namespace sadatom {
    namespace basis {
      /// Two-dimensional basis set
      class TwoDBasis {
        /// Nuclear charge
        int Z;
        /// Nuclear model
        modelpotential::nuclear_model_t model;
        /// Rms radius
        double Rrms;

        /// Yukawa exchange?
        bool yukawa;
        /// Range separation parameter
        double lambda;

        /// Radial basis set
        atomic::basis::RadialBasis radial;
        /// Angular basis set: function l values
        arma::ivec lval;

        /// Auxiliary integrals
        std::vector<arma::mat> disjoint_L, disjoint_m1L;
        /// Auxiliary integrals, Yukawa
        std::vector<arma::mat> disjoint_iL, disjoint_kL;
        /// Primitive two-electron integrals: <Nel^2 * (2L+1)>
        std::vector<arma::mat> prim_tei;
        /// Primitive two-electron exchange integrals
        std::vector<arma::mat> prim_ktei;
        /// Primitive two-electron exchange integrals, range separation
        std::vector<arma::mat> rs_ktei;

      public:
        TwoDBasis();
        /// Constructor
        TwoDBasis(int Z, modelpotential::nuclear_model_t model, double Rrms, const std::shared_ptr<const polynomial_basis::PolynomialBasis> &poly, bool zeroder, int n_quad, const arma::vec & bval, int taylor_order, int lmax);
        /// Destructor
        ~TwoDBasis();

        /// Compute two-electron integrals
        void compute_tei();
        /// Compute two-electron integrals
        void compute_yukawa(double lambda);
        /// Compute two-electron integrals
        void compute_erfc(double mu);

        /// Number of basis functions
        size_t Nbf() const;
        /// Get charge
        int charge() const;

        /// Form half-inverse overlap matrix
        arma::mat Sinvh() const;
        /// Form radial integral
        arma::mat radial_integral(int n) const;
        /// Form overlap matrix
        arma::mat overlap() const;
        /// Form basic part kinetic energy matrix
        arma::mat kinetic() const;
        /// Form l part of kinetic energy matrix
        arma::mat kinetic_l() const;
        /// Form nuclear attraction matrix
        arma::mat nuclear() const;
        /// Form model potential matrix
	arma::mat model_potential(const modelpotential::ModelPotential * model) const;
        /// Form Coulomb matrix
        arma::mat coulomb(const arma::mat & P) const;
        /// Form exchange matrix
        arma::cube exchange(const arma::cube & P) const;
        /// Form exchange matrix
        arma::cube rs_exchange(const arma::cube & P) const;

        /// Evaluate basis functions
        arma::mat eval_bf(size_t iel) const;
        /// Evaluate basis functions derivatives
        arma::mat eval_df(size_t iel) const;
        /// Evaluate basis functions second derivatives
        arma::mat eval_lf(size_t iel) const;
        /// Get list of basis function indices in element
        arma::uvec bf_list(size_t iel) const;

        /// Get number of radial elements
        size_t get_rad_Nel() const;
        /// Get radial quadrature weights
        arma::vec get_wrad(size_t iel) const;
        /// Get r values
        arma::vec get_r(size_t iel) const;
        /// Get small r Taylor cutoff
        double get_small_r_taylor_cutoff() const;
        /// Get small r Taylor cutoff
        double get_taylor_diff() const;

        /// Get primitive integrals
        std::vector<arma::mat> get_prim_tei() const;

        /// Electron density at nucleus
        double nuclear_density(const arma::mat & P) const;
        /// Electron density at nucleus
        double nuclear_density_gradient(const arma::mat & P) const;

        /// Get quadrature weights
        arma::vec quadrature_weights() const;
        /// Compute the Coulomb screening of the nucleus
        arma::vec coulomb_screening(const arma::mat & Prad) const;

        /// Get the radial matrices
        std::vector< std::pair<int, arma::mat> > Rmatrices() const;

        /// Radii
        arma::vec radii() const;
        /// Compute radial orbitals
        arma::mat orbitals(const arma::mat & C) const;

        /// Compute the electron density in given element at wanted points
        arma::vec electron_density(const arma::vec & x, size_t iel, const arma::mat & Prad, bool rsqweight = false) const;
        /// Compute the electron density in given element at default quadrature points
        arma::vec electron_density(size_t iel, const arma::mat & Prad, bool rsqweight = false) const;
        /// Compute the electron density in given element at default quadrature points
        double electron_density_maximum(const arma::mat & Prad, double eps=1e-10) const;
        /// Compute the van der Waals radius, see doi:10.1002/chem.201602949
        double vdw_radius(const arma::mat & Prad, double thr=0.001, double eps=1e-10) const;

        /// Compute the electron density
        arma::vec electron_density(const arma::mat & Prad) const;
        /// Compute the electron density gradient
        arma::vec electron_density_gradient(const arma::mat & Prad) const;
        /// Compute the electron density laplacian
        arma::vec electron_density_laplacian(const arma::mat & Prad) const;
        /// Compute the kinetic energy density
        arma::vec kinetic_energy_density(const arma::cube & Pl0) const;

        /// Compute the exchange-correlation screening
        arma::vec xc_screening(const arma::mat & Prad, int x_func, int c_func) const;
        /// Compute the exchange-correlation screening
        arma::mat xc_screening(const arma::mat & Parad, const arma::mat & Pbrad, int x_func, int c_func) const;
      };
    }
  }
}


#endif
