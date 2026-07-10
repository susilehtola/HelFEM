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
    /// Average out the Fock matrix (Phase 5.13: Eigen).
    helfem::Matrix fock_symmetry_average(const helfem::Matrix & Fin, const std::vector< std::vector<arma::uvec> > & sym_idx);

    /// Solve generalized eigenvalue problem (Phase 5.11: Eigen-typed).
    void eig_gsym(helfem::Vector & E, helfem::Matrix & C, const helfem::Matrix & F, const helfem::Matrix & Sinvh);
    /// Solve generalized eigenvalue problem in subspaces (Phase 5.12: Eigen matrices; m_idx kept arma::uvec).
    void eig_gsym_sub(helfem::Vector & E, helfem::Matrix & C, const helfem::Matrix & F, const helfem::Matrix & Sinvh, const std::vector<arma::uvec> & m_idx, bool verbose=true);

    /// Form half-inverse overlap (Phase 5.10: Eigen-typed).
    inline helfem::Matrix form_Sinvh(helfem::Matrix S, bool chol=false) {
      return utils::invh(S, chol);
    }

    /// Parse number of alpha and beta electrons
    void parse_nela_nelb(int & nela, int & nelb, int & Q, int & M, int Z);

    /// Parse xc parameters (Phase 5.15: Eigen-typed).
    helfem::Vector parse_xc_params(const std::string & input);
  }
}

#endif
