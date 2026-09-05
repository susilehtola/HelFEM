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
        /// Outer SCF iteration cap. Matches OOO's own default; raise it
        /// for systems that converge steadily but slowly, such as the
        /// partially filled 3d atoms, where the 4s/3d near-degeneracy
        /// leaves the error decreasing by a fraction of a percent per
        /// iteration.
        int maxiter = 128;
        /// SCF convergence threshold on the DIIS error. Matches OOO's
        /// own default. Some systems have an arithmetic noise floor
        /// above it -- their energy is stable to 1e-11 while the error
        /// sits frozen just above the target -- and need it relaxed.
        double convthr = 1e-7;
        /// Follow the first-order SCF with second-order trust-region
        /// optimization of the orbitals AND the fractional occupations.
        ///
        /// The spherically averaged atom optimizes its occupations as
        /// well as its orbitals, and that is what makes the open-shell
        /// transition metals converge badly: two degenerate orbitals in
        /// different l blocks are connected by no orbital rotation at
        /// all, so moving density between them costs nothing at first
        /// order while its real cost -- the Coulomb and XC coupling
        /// <k_b k_b|W|k_c k_c> -- is entirely second order. See
        /// helfem::trscf for the parametrization.
        bool secondorder = false;
        /// First-order iterations to run before handing over. The
        /// second-order phase optimizes WITHIN an occupation pattern and
        /// cannot discover one, so this is not merely a speed knob:
        /// handing over too early converges tightly to the wrong answer.
        int preiter = 100;
        /// RMS gradient the second-order phase converges to
        double soconvthr = 1e-8;
        /// Trust-region macro- and microiteration caps
        int somacro = 150;
        int somicro = 50;
        /// Ceiling on Hessian-vector products per macroiteration; 0 uses
        /// the optimizer's own default
        int somaxhess = 0;
        /// OpenTrustRegion's residual-reduction factor. Its own default
        /// (1e-3) discards good steps; see helfem::trscf.
        double soredfac = 3e-1;
        /// Reduced-space solver: "davidson" or "tcg"
        std::string sosolver = "davidson";
        /// Use the exact occupation-block preconditioner
        bool soprecond = true;
        /// Instead of optimizing, check the analytic gradient and
        /// Hessian against finite differences with this step size
        double sotest = 0.0;
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
        /// (up to 2*(2l+1)). Rounded to integers for the checkpoint and
        /// the Aufbau-style consumers; see occs_orb_a for the exact
        /// values.
        Eigen::VectorXi occs_a;
        /// Beta channel (empty for restricted).
        helfem::Cube orbs_b;
        Eigen::VectorXi occs_b;
        /// Per-l, per-orbital occupation numbers exactly as converged by
        /// OpenOrbitalOptimizer -- occs_orb_a[l](i) is the occupation of
        /// the i:th orbital of the l channel. These are NOT rounded, so
        /// they retain fractional occupations (OOO optimizes them, and
        /// the spherically averaged open-shell atoms that seed the SAP /
        /// SAD databases genuinely need them). occs_a above is their
        /// rounded per-l sum.
        std::vector<helfem::Vector> occs_orb_a, occs_orb_b;
        /// Per-l, per-orbital orbital energies of the converged
        /// solution, aligned column-by-column with orbs_a / occs_orb_a
        /// (and orbs_b / occs_orb_b). Obtained as the diagonal of the
        /// converged Fock matrix in the converged orbital basis, i.e.
        /// eps_i = <i|F|i>, which is the Janak derivative dE/dn_i and
        /// therefore the quantity the occupations are optimized
        /// against. For a solution converged by the ordinary SCF the
        /// orbitals diagonalize the Fock matrix and this is the usual
        /// eigenvalue; the diagonal form is what stays aligned with the
        /// occupations when the second-order phase hands back orbitals
        /// that are only stationary, not canonical.
        std::vector<helfem::Vector> orb_E_a, orb_E_b;
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
