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
#ifndef SADATOM_BASIS_H
#define SADATOM_BASIS_H

#include <armadillo>
#include "../atomic/basis.h"
#include <NAORadialBasis.h>

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
        atomic::basis::FEMRadialBasis radial;
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
        TwoDBasis(int Z, modelpotential::nuclear_model_t model, double Rrms, const std::shared_ptr<const polynomial_basis::PolynomialBasis> &poly, bool zeroder, int n_quad, const arma::vec & bval, int lmax);
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
	/// Form confinement potential matrix
	arma::mat confinement(int N, double r_0, int iconf, double V, double shift_pot) const;
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

        /// Evaluate orbitals
        arma::vec eval_orbs(const arma::mat & C, double r) const;

        /// Get number of radial elements
        size_t get_rad_Nel() const;
        /// Get radial quadrature weights
        arma::vec get_wrad(size_t iel) const;
        /// Get r values
        arma::vec get_r(size_t iel) const;

        /// Read-only access to the underlying radial basis. Useful for
        /// callers that want to build an NAORadialBasis on top of a
        /// converged sadatom SCF (see extract_naos_per_l below).
        const atomic::basis::FEMRadialBasis & get_radial() const { return radial; }

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
        /// Compute radial orbitals' first derivative
        arma::mat orbitals_derivative(const arma::mat & C) const;
        /// Compute radial orbitals' second derivative
        arma::mat orbitals_second_derivative(const arma::mat & C) const;

        /// Compute the electron density in given element at wanted points
        arma::vec electron_density(const arma::vec & x, size_t iel, const arma::mat & Prad, bool rsqweight = false) const;
        /// Compute the electron density in given element at default quadrature points
        arma::vec electron_density(size_t iel, const arma::mat & Prad, bool rsqweight = false) const;
        /// Compute the position of the electron density maximum
        double electron_density_maximum_radius(const arma::mat & Prad, bool rsqweight = true, double conv_thr=1e-10) const;
        /// Compute the van der Waals radius, see doi:10.1002/chem.201602949
        double vdw_radius(const arma::mat & Prad, double eps=0.001, double conv_thr=1e-10) const;
	/// Compute the atomic radius by electron density inclusion
	double electron_count_radius(const arma::mat & Prad, const double eps=0.001, const double conv_thr=1e-10) const;

        /// Compute the electron density
        arma::vec electron_density(const arma::mat & Prad, bool rsqweight = false) const;
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

      /// Extract per-l NAOs from a converged sadatom SCF result.
      ///
      /// `Ccube` is the OrbitalChannel coefficient cube returned by
      /// `sadatom::solver::OrbitalChannel::Coeffs()`; `Ccube.slice(l)`
      /// has shape (Nrad x N_orb_l) and holds the l-channel MOs in
      /// ascending eigenvalue order. `keep_per_l[l]` is how many of the
      /// lowest-energy columns to retain (use -1 to keep all of slice l;
      /// use 0 to skip that l-channel).
      ///
      /// Returns one (l, NAORadialBasis) pair per kept l-channel, in
      /// order of l. Each NAORadialBasis owns its own copy of the
      /// underlying FEMRadialBasis (shared internally via shared_ptr);
      /// the returned basis objects can outlive `sad_basis`.
      std::vector<std::pair<int, atomic::basis::NAORadialBasis>>
      extract_naos_per_l(const TwoDBasis & sad_basis,
                         const arma::cube & Ccube,
                         const std::vector<int> & keep_per_l);
    }
  }
}


#endif
