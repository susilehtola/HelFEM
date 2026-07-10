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
        arma::ivec lval;

        // Phase 2c: per-element 2e caches migrated to Eigen.
        /// Auxiliary integrals
        std::vector<helfem::Matrix> disjoint_L, disjoint_m1L;
        /// Auxiliary integrals, Yukawa
        std::vector<helfem::Matrix> disjoint_iL, disjoint_kL;
        /// Primitive two-electron integrals: <Nel^2 * (2L+1)>
        std::vector<helfem::Matrix> prim_tei;
        /// Primitive two-electron exchange integrals
        std::vector<helfem::Matrix> prim_ktei;
        /// Primitive two-electron exchange integrals, range separation
        std::vector<helfem::Matrix> rs_ktei;

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
        helfem::Matrix Sinvh() const;
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
