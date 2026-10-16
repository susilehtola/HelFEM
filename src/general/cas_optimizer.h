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
#ifndef HELFEM_CAS_OPTIMIZER_H
#define HELFEM_CAS_OPTIMIZER_H

#include "cas_ci.h"
#include "cas_integrals.h"
#include "otr_solver.h"

#include <string>
#include <utility>
#include <vector>

namespace helfem {
  namespace cas {

    /// A rotation of orbital q into orbital p > q.
    struct RotationPair {
      Eigen::Index p, q;
    };

    /// One rotation parameter: the same angle applied to each listed pair.
    /// The first pair lies in a representative symmetry block, the rest are
    /// its images in the equivalent blocks (+m and -m in a diatomic, every
    /// m of one l in an atom), which keeps equivalent orbitals identical --
    /// the CI alone decides how they are occupied.
    using Rotation = std::vector<RotationPair>;

    /// The non-redundant rotations of a CAS: inactive-active,
    /// inactive-virtual and active-virtual. Active-active rotations are
    /// left out because the CI absorbs them exactly, and rotations between
    /// orbitals of different `label` because they cannot lower a symmetric
    /// state. Orbitals with the same `shell` are images of one another in
    /// equivalent blocks, and their rotations are tied. Empty `label` puts
    /// every orbital in one block; empty `shell` ties nothing.
    std::vector<Rotation> rotation_pairs(Eigen::Index norb, Eigen::Index ninact,
                                         Eigen::Index nact,
                                         const std::vector<int> & label,
                                         const std::vector<int> & shell = {});

    /// Starting orbitals for a CAS from an SCF density P (alpha + beta):
    /// the canonical orbitals of the closed-shell Hartree-Fock operator
    /// h + J[P] - K[P]/2, solved in each symmetry block over its own basis
    /// functions dsym[k] with orthonormalizer Sinvh[k] separately within
    /// the occupied and the empty natural orbitals of P, then sorted by
    /// energy. For a converged RHF these ARE the SCF orbitals; after DFT or
    /// an unrestricted SCF they span the same occupied space as the SCF.
    ///
    /// Blocks whose overlap and core Hamiltonian agree element by element
    /// are equivalent, and get identical orbitals: the operator is averaged
    /// over them, solved once, and copied. label[i] is the block of orbital
    /// i and shell[i] numbers the set of copies it belongs to; the copies of
    /// one shell are contiguous.
    void canonical_orbitals(const JKProvider & jk, const helfem::Matrix & S,
                            const helfem::Matrix & P,
                            const std::vector<std::vector<Eigen::Index>> & dsym,
                            const std::vector<helfem::Matrix> & Sinvh,
                            helfem::Matrix & C, helfem::Vector & energy,
                            std::vector<int> & label, std::vector<int> & shell);

    /// Replace the empty orbitals from canonical_orbitals by correlating
    /// functions: the energy-weighted maximal-exchange orbitals of
    /// Aldossary and Head-Gordon (2022). In each block the virtuals a,b span
    ///
    ///     A_ab = sum_i n_i (ai|ib) / (2 e_i - e_a - e_b)
    ///
    /// over the ACTIVE occupied orbitals i -- the occupied ones above the
    /// `ninact` lowest, with occupation n_i in P -- and its
    /// eigenvectors, most negative eigenvalue first, replace them. A
    /// canonical virtual is set by the one-electron problem alone, and in a
    /// near-complete basis it is a diffuse, nearly free function that
    /// couples to nothing; the correlating function is set by the occupied
    /// density -- the leading one of a bonding orbital is its antibonding
    /// partner -- so it is what an active space wants.
    ///
    /// The occupied orbitals are untouched and come first, by energy. The
    /// active space of `ninact + nact` orbitals is then filled with the
    /// most strongly correlating virtual shells, by |eigenvalue| across all
    /// blocks, whatever their block. The virtuals left outside it are made
    /// canonical again within the span they keep, and follow by energy: the
    /// CASSCF takes its Hessian diagonal from the Fock matrix. Equivalent
    /// blocks share one averaged A and F, which keeps their copies
    /// identical. energy becomes the Fock expectation value of a chosen
    /// orbital and the eigenvalue of a canonical one, and weight is the
    /// eigenvalue of A for a chosen orbital and zero otherwise.
    void correlating_virtuals(const JKProvider & jk, const helfem::Matrix & S,
                              const helfem::Matrix & P, Eigen::Index ninact,
                              Eigen::Index nact, helfem::Matrix & C,
                              helfem::Vector & energy, std::vector<int> & label,
                              std::vector<int> & shell, std::vector<double> & weight);

    struct CASSCFOptions {
      otr::Settings otr;
      int verbosity = 1;
    };

    struct CASSCFResult {
      double energy = 0.0;
      double grad_rms = 0.0;
      size_t n_update = 0, n_objective = 0, n_hessian = 0;
    };

    /// CASSCF on OpenTrustRegion, in one of two parametrizations.
    ///
    /// TwoStep: the parameters are the orbital rotations alone, and the CI
    /// is solved to convergence at every point the optimizer visits. Since
    /// the CI is variational, the orbital gradient at fixed CI is the exact
    /// gradient of that energy; the Hessian-vector product is only the
    /// orbital-orbital block, which leaves out the CI response. The model is
    /// too stiff in the directions that couple to the CI, which costs
    /// iterations, not correctness.
    ///
    /// Coupled: the CI coefficients join the parameters, as a CSF-basis
    /// vector t projected against the current state |0>, with
    ///     |c(t)> = cos|t| |0> + sin|t| t/|t|.
    /// The projection leaves one redundant direction, along |0>, with zero
    /// gradient and zero curvature. Every block of the Hessian is exact:
    ///     kappa-kappa  the orbital Hessian at the RDMs of |0>;
    ///     kappa-c      the orbital gradient at the symmetrised TRANSITION
    ///                  RDMs between t and |0>;
    ///     c-kappa      2 P H' |0>, H' the one-index-transformed Hamiltonian;
    ///     c-c          2 P (H - E) P.
    enum class Coupling { TwoStep, Coupled };

    class CASSCF {
     public:
      /// C holds S-orthonormal orbitals in its columns, inactive first,
      /// then active, then virtual. The CI starts from the CASCI ground
      /// state at C.
      CASSCF(const JKProvider & jk, const helfem::Matrix & S, CASCI & ci,
             helfem::Matrix C, Eigen::Index ninact, Eigen::Index nact,
             const std::vector<int> & label = {}, const std::vector<int> & shell = {},
             Coupling coupling = Coupling::TwoStep);

      CASSCFResult run(const CASSCFOptions & opt);

      /// Number of parameters: rotations, then (Coupled) CSF coefficients.
      size_t nparam() const { return pairs_.size() + nci(); }
      size_t nrot() const { return pairs_.size(); }

      /// The CAS energy at the point x from the current one. TwoStep solves
      /// the CI there; Coupled evaluates the CI vector the parameters name.
      double energy(const std::vector<double> & x);
      /// Gradient at the current point, in parameter order.
      std::vector<double> gradient() const;
      /// Hessian times x at the current point (TwoStep: orbital block only).
      std::vector<double> hessian_vector(const std::vector<double> & x);

      const helfem::Matrix & orbitals() const { return C_; }
      /// The CI state in the determinant basis.
      const std::vector<double> & ci_vector() const { return c_; }

     private:
      const JKProvider & jk_;
      helfem::Matrix S_;
      CASCI & ci_;
      helfem::Matrix C_;
      Eigen::Index ninact_, nact_;
      std::vector<Rotation> pairs_;
      Coupling coupling_;

      /// State at the current point: energy, CI vector (determinants and
      /// CSFs), the active Hamiltonian, sigma of the state in CSFs, RDMs
      /// and the square orbital gradient.
      double E_ = 0.0;
      std::vector<double> c_, ccsf_, sigma_;
      ActiveHamiltonian ah_;
      RDMs rdm_;
      helfem::Matrix g_;

      size_t nci() const { return coupling_ == Coupling::Coupled ? ccsf_.size() : 0; }
      helfem::Matrix kappa(const double * x) const;
      helfem::Matrix rotated(const double * x) const;
      /// The normalized CSF vector the CI parameters at x name.
      std::vector<double> ci_at(const double * x) const;
      /// <c|H|c> for a CSF vector, with H installed in the CI first.
      double expectation(const ActiveHamiltonian & ah, const std::vector<double> & ccsf);
      /// Adopt C (and, Coupled, the CSF vector ccsf) as the current point
      /// and refresh everything above. TwoStep re-solves the CI.
      void adopt(const helfem::Matrix & C, const std::vector<double> & ccsf);
      /// Approximate Hessian diagonal: Fock-matrix estimates for the
      /// rotations, 2(H_II - E) for the CSFs.
      std::vector<double> hessian_diagonal() const;
    };

    /// What a driver asks for with --casact and friends.
    struct CASRequest {
      int ninact = 0, nact = 0;
      /// 2S of the target state
      int twoS = 0;
      Coupling coupling = Coupling::Coupled;
      CASSCFOptions options;
    };

    /// CASSCF from an SCF density: validates the request against Ntot,
    /// builds the starting orbitals with canonical_orbitals and
    /// correlating_virtuals, reports them with `block_name`, and runs the
    /// optimizer. Returns the electronic
    /// energy. The caller must hold a KokkosScope.
    CASSCFResult casscf_from_scf(const JKProvider & jk, const helfem::Matrix & S,
                                 const helfem::Matrix & P,
                                 const std::vector<std::vector<Eigen::Index>> & dsym,
                                 const std::vector<helfem::Matrix> & Sinvh,
                                 const std::vector<std::string> & block_name,
                                 int Ntot, const CASRequest & req);

  } // namespace cas
} // namespace helfem

#endif // HELFEM_CAS_OPTIMIZER_H
