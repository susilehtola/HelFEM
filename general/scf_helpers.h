#ifndef SCF_HELPERS_H
#define SCF_HELPERS_H
#include <armadillo>

namespace helfem {
  namespace scf {
    /// Form density matrix
    arma::mat form_density(const arma::mat & C, size_t nocc);
    /// Enforce occupation of wanted symmetries
    void enforce_occupations(arma::mat & C, arma::vec & E, const arma::ivec & nocc, const std::vector<arma::uvec> & m_idx);

    /// Solve generalized eigenvalue problem
    void eig_gsym(arma::vec & E, arma::mat & C, const arma::mat & F, const arma::mat & Sinvh);
    /// Solve generalized eigenvalue problem in subspaces
    void eig_gsym_sub(arma::vec & E, arma::mat & C, const arma::mat & F, const arma::mat & Sinvh, const std::vector<arma::uvec> & m_idx);
    /// Solve eigenvalue problem in subspaces
    void eig_sym_sub(arma::vec & E, arma::mat & C, const arma::mat & F, const std::vector<arma::uvec> & m_idx);

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

    /// ROHF update to Fock matrices
    void ROHF_update(arma::mat & Fa_AO, arma::mat & Fb_AO, const arma::mat & P_AO, const arma::mat & Sh, const arma::mat & Sinvh, int nocca, int noccb);

    /// Human readable memory size
    std::string memory_size(size_t size);
    /// Parse number of alpha and beta electrons
    void parse_nela_nelb(int & nela, int & nelb, int & Q, int & M, int Z);
  }
}

#endif
