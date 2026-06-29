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
#ifndef SCF_HELPERS_H
#define SCF_HELPERS_H
#include <armadillo>
#include <helfem.h>
#include <Matrix.h>

namespace helfem {
  namespace scf {
    /// Form density matrix (Phase 5.10: Eigen-typed).
    helfem::Matrix form_density(const helfem::Matrix & C, size_t nocc);
    /// Enforce occupation of wanted symmetries
    void enforce_occupations(arma::mat & C, arma::vec & E, const arma::mat & S, const arma::ivec & nocc, const std::vector<arma::uvec> & m_idx);

    /// Enforce wanted symmetry in the Fock matrix (zero out off-diagonal blocks)
    arma::mat enforce_fock_symmetry(const arma::mat & Fin, const std::vector<arma::uvec> & m_idx);
    /// Average out the Fock matrix
    arma::mat fock_symmetry_average(const arma::mat & Fin, const std::vector< std::vector<arma::uvec> > & sym_idx);

    /// Solve generalized eigenvalue problem (Phase 5.11: Eigen-typed).
    void eig_gsym(helfem::Vector & E, helfem::Matrix & C, const helfem::Matrix & F, const helfem::Matrix & Sinvh);
    /// Solve generalized eigenvalue problem in subspaces (Phase 5.12: Eigen matrices; m_idx kept arma::uvec).
    void eig_gsym_sub(helfem::Vector & E, helfem::Matrix & C, const helfem::Matrix & F, const helfem::Matrix & Sinvh, const std::vector<arma::uvec> & m_idx, bool verbose=true);
    /// Solve eigenvalue problem in subspaces (Phase 5.12: Eigen matrices; m_idx kept arma::uvec).
    void eig_sym_sub(helfem::Vector & E, helfem::Matrix & C, const helfem::Matrix & F, const std::vector<arma::uvec> & m_idx);

    /// Solve eigenvalue problem in subspace
    void eig_sub_wrk(arma::vec & E, arma::mat & Cocc, arma::mat & Cvirt, const arma::mat & F, size_t Nact);
    /// Sort eigenvectors
    void sort_eig(arma::vec & Eorb, arma::mat & Cocc, arma::mat & Cvirt, const arma::mat & Fao, size_t Nact, int maxit, double convthr);
    /// Solve subspace eigenproblem
    void eig_sub(arma::vec & E, arma::mat & Cocc, arma::mat & Cvirt, const arma::mat & F, size_t nsub, int maxit, double convthr);

    /// Iterative eigenvalue solver
    void eig_iter(arma::vec & E, arma::mat & Cocc, arma::mat & Cvirt, const arma::mat & F, const arma::mat & Sinvh, size_t nocc, size_t neig, size_t nsub, int maxit, double convthr);

    /// Random perturbation
    arma::mat perturbation_matrix(size_t N, double ampl);

    /// Form natural orbitals
    void form_NOs(const arma::mat & P, const arma::mat & Sh, const arma::mat & Sinvh, arma::mat & AO_to_NO, arma::mat & NO_to_AO, arma::vec & occs);
    /// Form half-inverse overlap (Phase 5.10: Eigen-typed).
    inline helfem::Matrix form_Sinvh(helfem::Matrix S, bool chol=false) {
      return utils::invh(S, chol);
    }

    /// ROHF update to Fock matrices
    void ROHF_update(arma::mat & Fa_AO, arma::mat & Fb_AO, const arma::mat & P_AO, const arma::mat & Sh, const arma::mat & Sinvh, int nocca, int noccb);

    /// Human readable memory size
    std::string memory_size(size_t size);
    /// Parse number of alpha and beta electrons
    void parse_nela_nelb(int & nela, int & nelb, int & Q, int & M, int Z);

    /// Parse xc parameters
    arma::vec parse_xc_params(const std::string & input);
  }
}

#endif
