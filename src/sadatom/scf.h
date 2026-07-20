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
#ifndef SADATOM_SCF_H
#define SADATOM_SCF_H

#include "basis.h"
#include "../general/model_potential.h"
#include <memory>

namespace helfem {
  namespace sadatom {
    namespace scf {

      /// SCF inputs. All parameters are explicit so this helper can be
      /// invoked both from src/sadatom/main.cpp (CLI-parsed values) and
      /// from src/diatomic/twodquadrature.cpp (hardcoded PBE guess run).
      struct AtomicSCFOptions {
        int    Z          = 0;
        int    lmax       = 0;
        std::shared_ptr<const polynomial_basis::PolynomialBasis> poly;
        int    Nquad      = 0;
        helfem::Vector bval;
        int    nela       = 0;
        int    nelb       = 0;
        bool   restricted = true;
        int    x_func     = 0;
        int    c_func     = 0;
        helfem::Vector x_pars;
        helfem::Vector c_pars;
        double dftthr     = 1e-12;
        /// Initial-guess electron-nuclear potential: 0 core Hamiltonian
        /// (bare Vnuc), 1 GSZ, 2 SAP, 3 Thomas-Fermi. Affects only the
        /// starting Fock matrix; the SCF Fock build always uses the true
        /// nuclear attraction. Default 0 so the twodquadrature sub-SCF
        /// keeps its core-Hamiltonian guess; the gensap CLI defaults to 2.
        int    iguess     = 0;
        modelpotential::nuclear_model_t finitenuc = modelpotential::POINT_NUCLEUS;
        double Rrms       = 0.0;
        bool   zeroder    = false;
        // Confinement (matches main.cpp CLI). iconf == 0 disables.
        int    iconf         = 0;
        int    conf_N        = 0;
        double conf_R        = 0.0;
        double conf_barrier  = 0.0;
        double shift_conf    = 0.0;
        // Frozen per-l per-spin occupation. If either vector has size
        // lmax+1, the corresponding channel's per-l electron count is
        // pinned via OOO's fixed_number_of_particles_per_block API and
        // Aufbau is bypassed for that channel.
        // Restricted: pass fixed_per_l_a with the per-l total (up to
        // 2*(2l+1)) and leave fixed_per_l_b empty.
        // Unrestricted: pass both; each entry is the alpha or beta
        // count in that l (up to 2l+1).
        // Counts may be fractional: OOO's frozen-occupation API takes
        // reals, so a partially filled l shell can be pinned at any
        // occupation in [0, 2l+1] (or [0, 2*(2l+1)] when restricted).
        Eigen::VectorXd fixed_per_l_a;
        Eigen::VectorXd fixed_per_l_b;
        /// OOO verbosity; 0 for silent, higher for per-iteration prints.
        int verbosity = 5;
        /// SCF convergence algorithms handed to OOO's state machine: a
        /// '+' separated subset of DIIS, ODA, CG and LBFGS.
        std::string scf_methods = "DIIS + ODA + CG";
        /// Load orbital guess from checkpoint. Empty = start from core-H
        /// guess as before. When non-empty, the old basis + per-l AO
        /// densities are read from the checkpoint, projected into the
        /// current basis via cross-basis overlap, and used to seed OOO.
        std::string load_file;
        /// Save final basis + per-l AO densities to checkpoint.
        std::string save_file;
      };

      /// SCF outputs.
      struct AtomicSCFResult {
        /// The FE atomic basis used for the SCF (Z, radial grid, lmax).
        sadatom::basis::TwoDBasis basis;
        /// AO->MO coefficient cube: [l] is the (Nbf, Nbf) matrix
        /// of MO coefficients for orbital angular momentum l.
        /// For unrestricted, this is the alpha channel.
        helfem::Cube orbs_a;
        /// Per-l occupation numbers (alpha channel for unrestricted).
        /// Length lmax+1. For restricted, this is the FULL per-l count
        /// (up to 2*(2l+1)).
        Eigen::VectorXi occs_a;
        /// Beta channel (empty for restricted).
        helfem::Cube orbs_b;
        Eigen::VectorXi occs_b;
        /// Converged total radial density matrix (alpha+beta),
        /// Nrad x Nrad. Consumed by the gensap effective-potential /
        /// SAP-table output path (basis::coulomb_screening /
        /// xc_screening / electron_density).
        helfem::Matrix Prad;
        /// Per-l radial density cubes. Pl_a[l] is the l-channel
        /// density; for restricted it holds the full per-l density
        /// (alpha+beta), for unrestricted it is the alpha channel and
        /// Pl_b the beta channel (empty for restricted). Used for the
        /// kinetic-energy-density (tau) column of the SAP table.
        helfem::Cube Pl_a;
        helfem::Cube Pl_b;
      };

      /// Run an OOO-based sadatom SCF. Replaces the bespoke SCFSolver
      /// class that used to live in solver.cpp / solver.h; those files
      /// pull in DIIS and L-BFGS which we retire once this helper
      /// covers both the gensap driver and the twodquadrature
      /// atomic-guess sub-SCF.
      AtomicSCFResult run_atomic_scf(const AtomicSCFOptions & opts);

    } // namespace scf
  } // namespace sadatom
} // namespace helfem

#endif // SADATOM_SCF_H
