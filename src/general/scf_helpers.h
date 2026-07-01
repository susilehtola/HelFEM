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
    /// Enforce occupation of wanted symmetries (Phase 5.13: Eigen matrices; m_idx kept arma::uvec).
    void enforce_occupations(helfem::Matrix & C, helfem::Vector & E, const helfem::Matrix & S, const arma::ivec & nocc, const std::vector<arma::uvec> & m_idx);

    /// Enforce wanted symmetry in the Fock matrix (zero out off-diagonal blocks). Phase 5.13: Eigen.
    helfem::Matrix enforce_fock_symmetry(const helfem::Matrix & Fin, const std::vector<arma::uvec> & m_idx);
    /// Average out the Fock matrix (Phase 5.13: Eigen).
    helfem::Matrix fock_symmetry_average(const helfem::Matrix & Fin, const std::vector< std::vector<arma::uvec> > & sym_idx);

    /// Solve generalized eigenvalue problem (Phase 5.11: Eigen-typed).
    void eig_gsym(helfem::Vector & E, helfem::Matrix & C, const helfem::Matrix & F, const helfem::Matrix & Sinvh);
    /// Solve generalized eigenvalue problem in subspaces (Phase 5.12: Eigen matrices; m_idx kept arma::uvec).
    void eig_gsym_sub(helfem::Vector & E, helfem::Matrix & C, const helfem::Matrix & F, const helfem::Matrix & Sinvh, const std::vector<arma::uvec> & m_idx, bool verbose=true);
    /// Solve eigenvalue problem in subspaces (Phase 5.12: Eigen matrices; m_idx kept arma::uvec).
    void eig_sym_sub(helfem::Vector & E, helfem::Matrix & C, const helfem::Matrix & F, const std::vector<arma::uvec> & m_idx);

    /// Solve eigenvalue problem in subspace (Phase 5.14: Eigen-typed).
    void eig_sub_wrk(helfem::Vector & E, helfem::Matrix & Cocc, helfem::Matrix & Cvirt, const helfem::Matrix & F, size_t Nact);
    /// Sort eigenvectors (Phase 5.14: Eigen-typed).
    void sort_eig(helfem::Vector & Eorb, helfem::Matrix & Cocc, helfem::Matrix & Cvirt, const helfem::Matrix & Fao, size_t Nact, int maxit, double convthr);
    /// Solve subspace eigenproblem (Phase 5.14: Eigen-typed).
    void eig_sub(helfem::Vector & E, helfem::Matrix & Cocc, helfem::Matrix & Cvirt, const helfem::Matrix & F, size_t nsub, int maxit, double convthr);

    /// Iterative eigenvalue solver (Phase 5.14: now a full-spectrum
    /// SelfAdjointEigenSolver in the orthonormal basis; the arma::newarp
    /// Arnoldi path is gone -- it had no callers in-tree and the
    /// matrix dimensions we hit make a full solve cheaper than the
    /// previous iterative path).
    void eig_iter(helfem::Vector & E, helfem::Matrix & Cocc, helfem::Matrix & Cvirt, const helfem::Matrix & F, const helfem::Matrix & Sinvh, size_t nocc, size_t neig, size_t nsub, int maxit, double convthr);

    /// Random perturbation (Phase 5.15: Eigen-typed).
    helfem::Matrix perturbation_matrix(size_t N, double ampl);

    /// Form natural orbitals (Phase 5.16: Eigen-typed).
    void form_NOs(const helfem::Matrix & P, const helfem::Matrix & Sh, const helfem::Matrix & Sinvh, helfem::Matrix & AO_to_NO, helfem::Matrix & NO_to_AO, helfem::Vector & occs);
    /// Form half-inverse overlap (Phase 5.10: Eigen-typed).
    inline helfem::Matrix form_Sinvh(helfem::Matrix S, bool chol=false) {
      return utils::invh(S, chol);
    }

    /// ROHF update to Fock matrices (Phase 5.16: Eigen-typed).
    void ROHF_update(helfem::Matrix & Fa_AO, helfem::Matrix & Fb_AO, const helfem::Matrix & P_AO, const helfem::Matrix & Sh, const helfem::Matrix & Sinvh, int nocca, int noccb);

    /// Human readable memory size
    std::string memory_size(size_t size);
    /// Parse number of alpha and beta electrons
    void parse_nela_nelb(int & nela, int & nelb, int & Q, int & M, int Z);

    /// Parse xc parameters (Phase 5.15: Eigen-typed).
    helfem::Vector parse_xc_params(const std::string & input);
  }
}

#endif
