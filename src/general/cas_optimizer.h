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

    /// Two-step CASSCF on OpenTrustRegion: the parameters are the orbital
    /// rotations alone, and the CI is solved to convergence at every point
    /// the optimizer visits.
    ///
    /// Because the CI is variational, the gradient of that energy is the
    /// orbital gradient at fixed CI, exactly. Its Hessian is not: the
    /// Hessian-vector product here is the orbital-orbital block, which
    /// leaves out the CI response. That makes the model too stiff in the
    /// directions that couple to the CI, which costs iterations, not
    /// correctness -- every step is validated against the true energy.
    class CASSCF {
     public:
      /// C holds S-orthonormal orbitals in its columns, inactive first,
      /// then active, then virtual.
      CASSCF(const JKProvider & jk, const helfem::Matrix & S, CASCI & ci,
             helfem::Matrix C, Eigen::Index ninact, Eigen::Index nact,
             std::vector<int> label = {});

      CASSCFResult run(const CASSCFOptions & opt);

      /// Number of rotation parameters.
      size_t nparam() const { return pairs_.size(); }

      /// The CAS energy with the orbitals rotated by x from the current
      /// ones and the CI solved there.
      double energy(const std::vector<double> & x);
      /// Orbital gradient at the current orbitals, in parameter order.
      std::vector<double> gradient();
      /// Orbital-orbital Hessian times x at the current orbitals, CI fixed.
      std::vector<double> hessian_vector(const std::vector<double> & x);

      const helfem::Matrix & orbitals() const { return C_; }
      const std::vector<double> & ci_vector() const { return c_; }

     private:
      const JKProvider & jk_;
      helfem::Matrix S_;
      CASCI & ci_;
      helfem::Matrix C_;
      Eigen::Index ninact_, nact_;
      std::vector<RotationPair> pairs_;

      /// State at the current orbitals: energy, CI vector, RDMs and the
      /// square orbital gradient.
      double E_ = 0.0;
      std::vector<double> c_;
      RDMs rdm_;
      helfem::Matrix g_;

      helfem::Matrix kappa(const double * x) const;
      helfem::Matrix rotated(const double * x) const;
      double solve_ci(const helfem::Matrix & C, std::vector<double> & c);
      /// Adopt C as the current orbitals and refresh everything above.
      void adopt(const helfem::Matrix & C);
      /// Approximate diagonal of the orbital Hessian, from the inactive,
      /// active and generalized Fock matrices.
      std::vector<double> hessian_diagonal() const;
    };

  } // namespace cas
} // namespace helfem

#endif // HELFEM_CAS_OPTIMIZER_H
