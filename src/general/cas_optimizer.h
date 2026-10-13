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

#include <utility>
#include <vector>

namespace helfem {
  namespace cas {

    /// One rotation parameter: orbital p > q mixed into q.
    struct RotationPair {
      Eigen::Index p, q;
    };

    /// The non-redundant rotations of a CAS: inactive-active,
    /// inactive-virtual and active-virtual. Active-active rotations are
    /// left out because the CI absorbs them exactly, and rotations between
    /// orbitals of different `label` because they cannot lower a symmetric
    /// state. An empty `label` puts every orbital in one symmetry.
    std::vector<RotationPair> rotation_pairs(Eigen::Index norb, Eigen::Index ninact,
                                             Eigen::Index nact,
                                             const std::vector<int> & label);

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
             std::vector<int> label = {}, Coupling coupling = Coupling::TwoStep);

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
      std::vector<RotationPair> pairs_;
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

  } // namespace cas
} // namespace helfem

#endif // HELFEM_CAS_OPTIMIZER_H
