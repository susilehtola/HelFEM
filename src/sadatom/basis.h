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

#include <Matrix.h>
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
        Eigen::VectorXi lval;

        // Phase 2c: per-element 2e caches migrated to Eigen.
        /// Auxiliary integrals
        std::vector<helfem::Matrix> disjoint_L, disjoint_m1L;
        /// Auxiliary integrals, Yukawa
        std::vector<helfem::Matrix> disjoint_iL, disjoint_kL;
        /// Primitive two-electron integrals: <Nel^2 * (2L+1)>
        /// Pivoted-Cholesky factors of the in-element two-electron integrals,
        /// indexed [L*Nel + iel]: L_p of shape (Ni^2 x r) with T = L L'. Same
        /// low-rank / RI treatment atomic::TwoDBasis uses -- the only 4-index
        /// object left, and the exchange PAIRING of it is full rank, so K goes
        /// through RI rather than storing an exchange-ordered tensor.
        std::vector<helfem::Matrix> prim_chol;
        /// Tolerance for the pivoted Cholesky above
        double chol_tol = 1e-12;
        /// Primitive two-electron exchange integrals
        /// Same factorization for the Yukawa-screened in-element integrals.
        std::vector<helfem::Matrix> rs_chol;
        /// Primitive two-electron exchange integrals, range separation
        std::vector<helfem::Matrix> rs_ktei;

      public:
        TwoDBasis();
        /// Constructor
        TwoDBasis(int Z, modelpotential::nuclear_model_t model, double Rrms, const std::shared_ptr<const polynomial_basis::PolynomialBasis> &poly, bool zeroder, int n_quad, const helfem::Vector & bval, int lmax);
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
        helfem::Matrix Sinvh() const;
        /// Cross-basis overlap S12 = <this | rh>. Sadatom is spherically
        /// symmetric so the cross-basis overlap is a plain radial overlap
        /// between two FE radial bases; the same matrix applies to every
        /// l-block downstream. Used by the SCF driver's --load path.
        helfem::Matrix overlap(const TwoDBasis & rh) const;
        // Phase 3: SCF surface migrated to Eigen.
        /// Form overlap matrix
        helfem::Matrix overlap() const;
        /// Form basic part kinetic energy matrix
        helfem::Matrix kinetic() const;
        /// Form l part of kinetic energy matrix
        helfem::Matrix kinetic_l() const;
        /// Form nuclear attraction matrix
        helfem::Matrix nuclear() const;
	/// Form confinement potential matrix
	helfem::Matrix confinement(int N, double r_0, int iconf, double V, double shift_pot) const;
        /// Form model potential matrix
	helfem::Matrix model_potential(const modelpotential::ModelPotential * model) const;
        /// Form Coulomb matrix
        helfem::Matrix coulomb(const helfem::Matrix & P) const;
        /// Form exchange matrix
        helfem::Cube exchange(const helfem::Cube & P) const;
        /// Form exchange matrix
        helfem::Cube rs_exchange(const helfem::Cube & P) const;

        /// Evaluate basis functions
        helfem::Matrix eval_bf(size_t iel) const;
        /// Evaluate basis functions derivatives
        helfem::Matrix eval_df(size_t iel) const;
        /// Evaluate basis functions second derivatives
        helfem::Matrix eval_lf(size_t iel) const;
        /// Get list of basis function indices in element
        std::vector<Eigen::Index> bf_list(size_t iel) const;

        /// Evaluate orbitals
        helfem::Vector eval_orbs(const helfem::Matrix & C, double r) const;

        /// Get number of radial elements
        size_t get_rad_Nel() const;
        /// Get radial quadrature weights
        helfem::Vector get_wrad(size_t iel) const;
        /// Get r values
        helfem::Vector get_r(size_t iel) const;

        /// Electron density at nucleus
        double nuclear_density(const helfem::Matrix & P) const;
        /// Electron density gradient at nucleus
        double nuclear_density_gradient(const helfem::Matrix & P) const;

        /// Get quadrature weights
        helfem::Vector quadrature_weights() const;
        /// Compute the Coulomb screening of the nucleus
        helfem::Vector coulomb_screening(const helfem::Matrix & Prad) const;

        /// Radii
        helfem::Vector radii() const;
        /// Compute the radial Slater-Condon integral F^k for an orbital
        /// specified by its coefficient vector c (length Nrad) in the
        /// u = r * R basis.
        double slater_F(int k, const helfem::Vector & c) const;
        /// Compute radial orbitals
        helfem::Matrix orbitals(const helfem::Matrix & C) const;
        /// Compute radial orbitals' first derivative
        helfem::Matrix orbitals_derivative(const helfem::Matrix & C) const;
        /// Compute radial orbitals' second derivative
        helfem::Matrix orbitals_second_derivative(const helfem::Matrix & C) const;

        /// Compute the electron density in given element at wanted points
        helfem::Vector electron_density(const helfem::Vector & x, size_t iel, const helfem::Matrix & Prad, bool rsqweight = false) const;
        /// Compute the electron density in given element at default quadrature points
        helfem::Vector electron_density(size_t iel, const helfem::Matrix & Prad, bool rsqweight = false) const;
        /// Compute the position of the electron density maximum
        double electron_density_maximum_radius(const helfem::Matrix & Prad, bool rsqweight = true, double conv_thr=1e-10) const;
        /// Compute the van der Waals radius, see doi:10.1002/chem.201602949
        double vdw_radius(const helfem::Matrix & Prad, double eps=0.001, double conv_thr=1e-10) const;
        /// Compute the atomic radius by electron density inclusion
        double electron_count_radius(const helfem::Matrix & Prad, const double eps=0.001, const double conv_thr=1e-10) const;

        /// Compute the electron density
        helfem::Vector electron_density(const helfem::Matrix & Prad, bool rsqweight = false) const;
        /// Compute the electron density gradient
        helfem::Vector electron_density_gradient(const helfem::Matrix & Prad) const;
        /// Compute the electron density laplacian
        helfem::Vector electron_density_laplacian(const helfem::Matrix & Prad) const;
        /// Compute the kinetic energy density
        helfem::Vector kinetic_energy_density(const helfem::Cube & Pl0) const;

        /// Compute the exchange-correlation screening
        helfem::Vector xc_screening(const helfem::Matrix & Prad, int x_func, int c_func) const;
        /// Compute the exchange-correlation screening
        helfem::Matrix xc_screening(const helfem::Matrix & Parad, const helfem::Matrix & Pbrad, int x_func, int c_func) const;

        /// Read-only access to the underlying radial basis. Useful for
        /// callers that want to build an NAORadialBasis on top of a
        /// converged sadatom SCF (see extract_naos_per_l below).
        const atomic::basis::FEMRadialBasis & get_radial() const { return radial; }


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
                         const helfem::Cube & Ccube,
                         const std::vector<int> & keep_per_l);
    }
  }
}


#endif
