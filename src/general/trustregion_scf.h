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
#ifndef HELFEM_TRUSTREGION_SCF_H
#define HELFEM_TRUSTREGION_SCF_H

#include <Matrix.h>
#include "otr_solver.h"

#include <functional>
#include <vector>

namespace helfem {
  /// Second-order optimization of orbitals AND fractional occupations for
  /// a set of symmetry blocks that share one basis.
  ///
  /// The problem this solves is the spherically averaged atom (and the
  /// atom in jellium), where the occupations are free parameters rather
  /// than integers. First-order methods handle that badly for a specific
  /// reason: when two orbitals are degenerate, moving density between
  /// them costs nothing at first order in the orbital energies, so the
  /// gradient is flat along exactly the direction that matters and the
  /// true cost -- a dense coupling through the Coulomb and XC kernels --
  /// is entirely second order. Fe is the standard example: 4s and 3d come
  /// out at the same orbital energy, and since they sit in different l
  /// blocks no orbital rotation connects them at all. Their whole
  /// interaction is the occupation coupling <4s 4s|W|3d 3d>.
  ///
  /// PARAMETRIZATION. Orbital energies increase monotonically within an
  /// angular momentum, so each block can hold at most one fractionally
  /// occupied orbital on top of its filled ones. Rather than tracking
  /// which orbital that is, each block carries a single continuous
  /// variable q_b, the number of electrons in the block, and the
  /// occupations follow from filling it up:
  ///
  ///     f_{b,i}(q_b) = clamp(q_b - i*w_b, 0, w_b)
  ///
  /// with w_b the occupation of a filled orbital. That is the same "one
  /// fractional orbital per block" statement with the bookkeeping removed:
  /// when the fractional occupation runs past w_b the next orbital simply
  /// starts filling, continuously, and no re-indexing happens mid-solve.
  /// E(q) does have a kink where an orbital fills exactly -- the gradient
  /// jumps from eps_k to eps_{k+1} -- but that is a measure-zero set and
  /// the solutions of interest sit strictly inside a shell.
  ///
  /// The electron count of each particle type is conserved, so the free
  /// occupation parameters span the sum-zero subspace of its blocks:
  /// n_blocks - 1 of them, in an orthonormal (Helmert) basis so that the
  /// parameter metric stays isotropic rather than singling out one block.
  /// q >= 0 is imposed by Euclidean projection onto the simplex, which
  /// preserves the electron count exactly.
  ///
  /// PRECONDITIONING. The uncoupled Hessian diagonal, 2(f_i-f_j)(F_jj-F_ii),
  /// is *identically zero* along every occupation direction -- the formal
  /// statement that first order sees nothing there. Something has to fill
  /// that in, so the occupation block is built exactly, one Fock response
  /// per column, and inverted exactly in the preconditioner.
  ///
  /// It takes TWO occupation coordinates to see why that matters, which is
  /// most of a system: with one the block is 1x1, there is no coupling to
  /// capture, and turning the whole thing off in favour of the library's
  /// floored diagonal changes neither the iteration count nor the answer.
  /// Almost every open-shell atom has exactly one. Curium has two -- 5f,
  /// 6d and 7s all fractionally occupied at once -- and there the
  /// difference is not subtle:
  ///
  ///                        exact block        floored diagonal
  ///   RMS gradient           9.7e-10                 5.8e-3
  ///   energy            -28376.6749963109      -28376.6711800819
  ///
  /// The diagonal strands the solve 3.8e-3 above the minimum, because the
  /// off-diagonal <k_b k_b|W|k_c k_c> it drops is the same size as the
  /// diagonal it keeps. Hand it a better starting point and it does
  /// converge, but still only to 6.0e-8 against 4.2e-10, and it takes more
  /// objective evaluations to do it.
  ///
  /// Extending the same treatment to the low-lying ORBITAL rotations was
  /// tried and removed. Each parameter in the subspace costs a
  /// Hessian-vector product every macroiteration, and on Curium -- the one
  /// case where preconditioning demonstrably decides the outcome -- it
  /// bought nothing for that: 50 extra rotations cost eight times the
  /// products for no better answer, and 200 or 600 stopped it converging
  /// at all while costing twenty to sixty times. The low-lying ORBITAL
  /// directions are not what these cases founder on; the occupation ones
  /// are, and those are already exact.
  namespace trscf {

    /// Total energy and per-block Fock matrices from per-block densities.
    /// Everything is in the non-orthonormal FEM basis.
    using FockBuilder =
        std::function<double(const helfem::Cube &P, helfem::Cube &F)>;

    /// Linear response of those Fock matrices to a batch of density
    /// perturbations, at the reference density P. Batched because the
    /// expensive parts -- the basis values on the quadrature grid, the
    /// reference density, the libxc kernel -- are shared across the
    /// perturbations, so an exact Hessian in a d-dimensional subspace
    /// costs far less than d separate response builds.
    using ResponseBuilder =
        std::function<void(const helfem::Cube &P,
                           const std::vector<helfem::Cube> &dP,
                           std::vector<helfem::Cube> &dF)>;

    struct Options {
      /// Settings passed through to OpenTrustRegion
      otr::Settings otr;
      /// Use the exact-subspace preconditioner. Turning it off falls back
      /// to OpenTrustRegion's own diagonal one, which is identically zero
      /// along every occupation direction -- useful for measuring what
      /// the exact block actually buys, and not much else.
      bool exact_precond = true;
      /// Ceiling on Hessian-vector products within ONE macroiteration.
      ///
      /// A healthy macroiteration spends at most n_micro of them and then
      /// accepts a step. OpenTrustRegion, though, re-enters its
      /// microiteration loop until a step IS accepted, and never resets
      /// its reduced space in between -- so a macroiteration whose steps
      /// keep failing the ratio test grows that space without bound, and
      /// since every microiteration diagonalizes the augmented Hessian
      /// densely, the cost of each one grows as the cube of the work
      /// already wasted. Left alone a bad starting point does not fail, it
      /// grinds: measured, one such macroiteration ran for 900 seconds
      /// without returning.
      ///
      /// Counting per macroiteration rather than per solve is what makes
      /// this scale-free: it fires on the ratio of work done to progress
      /// made, not on an absolute budget that a legitimately hard case
      /// might need. Zero disables it; the default is a few times n_micro.
      int max_hessian = 0;
      /// How many times the solve may be restarted after the occupation
      /// pattern changes under it (a shell closing, say). Each restart
      /// rebuilds the parametrization around the new pattern.
      int max_restarts = 5;
      /// Output detail
      int verbosity = 5;
    };

    /// Report from a run
    struct Result {
      /// Did the RMS gradient actually reach the threshold? Worth
      /// checking: OpenTrustRegion reports a stalled line search
      /// ("maximum precision reached") as a successful return, so a
      /// zero error code alone does not mean the solve converged.
      bool converged = false;
      /// Number of times the occupation pattern changed under the solve
      /// and the parametrization had to be rebuilt
      size_t n_restart = 0;
      /// Converged energy
      double energy = 0.0;
      /// RMS gradient at the end
      double grad_rms = 0.0;
      /// Number of reference updates (macroiterations)
      size_t n_update = 0;
      /// Number of bare objective evaluations
      size_t n_objective = 0;
      /// Number of Hessian-vector products
      size_t n_hessian = 0;
      /// Number of Fock-response builds (batched, so <= n_hessian)
      size_t n_response = 0;
    };

    class Optimizer {
      /// Half-inverse overlap, Nfem x Nrad
      helfem::Matrix Sinvh_;
      /// Occupation of a filled orbital in each block
      std::vector<double> maxocc_;
      /// Particle type each block belongs to
      std::vector<size_t> particle_;
      /// Blocks of each particle type
      std::vector<std::vector<size_t>> pblocks_;
      /// Blocks of each particle type whose filling is a free
      /// coordinate, i.e. whose topmost occupied orbital really is
      /// fractionally occupied. A block at an exact shell filling or an
      /// empty one sits on a kink or a bound and is held fixed.
      std::vector<std::vector<size_t>> free_;
      /// Orthonormal basis of the sum-zero subspace of each particle
      /// type's FREE block fillings, n_free x (n_free - 1)
      std::vector<helfem::Matrix> sumzero_;

      /// Fock and response builders
      FockBuilder fock_;
      ResponseBuilder response_;

      /// Reference orbitals in the orthonormal basis, Nrad x Nrad
      std::vector<helfem::Matrix> U_;
      /// Reference orbitals in the FEM basis, Sinvh_ * U_
      std::vector<helfem::Matrix> C_;
      /// Electrons in each block
      std::vector<double> q_;
      /// Reference occupations
      std::vector<helfem::Vector> f_;
      /// Index of the fractionally occupied orbital in each block
      std::vector<Eigen::Index> active_;
      /// Reference Fock matrices in the MO basis
      std::vector<helfem::Matrix> F_;
      /// Reference energy
      double energy_ = 0.0;

      /// Parameter layout: rotation pairs of each block, and the offset
      /// of that block's parameters in the flat vector
      std::vector<std::vector<Eigen::Index>> pair_i_, pair_j_;
      std::vector<size_t> koff_;
      /// Offset of the occupation parameters
      size_t qoff_ = 0;
      size_t nparam_ = 0;

      /// Uncoupled Hessian diagonal at the reference
      helfem::Vector hdiag_;
      /// Verbosity for the one-off conditioning report
      int report_verbosity_ = 0;
      /// Ceiling on Hessian-vector products WITHIN one macroiteration,
      /// and whether it was hit
      size_t hess_budget_ = 0;
      size_t hess_this_macro_ = 0;
      bool budget_spent_ = false;
      /// Best point any objective evaluation has reached since the last
      /// accepted step, and its energy. When the solve stalls, this is the
      /// step it kept declining.
      helfem::Vector best_step_;
      double best_energy_ = 0.0;
      /// The occupation pattern the current parametrization was built
      /// around, and whether the reference has since left it
      std::vector<Eigen::Index> entry_active_;
      bool pattern_changed_ = false;
      /// Indices spanning the exactly built subspace, and the eigen-
      /// decomposition of the Hessian restricted to it
      std::vector<size_t> exact_idx_;
      helfem::Matrix exact_vec_;
      helfem::Vector exact_val_;

      /// Counters
      mutable Result stats_;

      /// Number of radial functions
      Eigen::Index Nrad() const { return Sinvh_.cols(); }
      /// Number of blocks
      size_t nblock() const { return maxocc_.size(); }

      /// Build the parameter layout from the current active indices
      void build_layout();
      /// Occupations and active index of one block from its filling
      void fill_occupations(size_t b, double q, helfem::Vector &f,
                            Eigen::Index &k) const;
      /// Project the block fillings of every particle type back onto
      /// {q >= 0, sum q = N}
      void project_fillings(std::vector<double> &q) const;
      /// Decode a parameter vector into per-block rotations and the
      /// change in the block fillings
      void decode(const double *x, std::vector<helfem::Matrix> &kappa,
                  std::vector<double> &dq) const;
      /// Apply rotations and filling changes to the reference, without
      /// adopting the result
      void displace(const std::vector<helfem::Matrix> &kappa,
                    const std::vector<double> &dq,
                    std::vector<helfem::Matrix> &U,
                    std::vector<double> &q) const;
      /// Per-block FEM densities of a given orbital set and filling
      void densities(const std::vector<helfem::Matrix> &U,
                     const std::vector<double> &q, helfem::Cube &P) const;
      /// Adopt a point: recompute C_, F_, the energy, and the gradient
      /// and Hessian data that go with them
      void adopt(const std::vector<helfem::Matrix> &U,
                 const std::vector<double> &q, double *grad, double *h_diag);
      /// Gradient at the reference
      void gradient(double *g) const;
      /// Hessian-vector products for a batch of trial vectors
      void hessian(const std::vector<helfem::Vector> &X,
                   std::vector<helfem::Vector> &HX) const;
      /// The exact occupation block, n_blocks x n_blocks
      helfem::Matrix occupation_hessian() const;
      /// Build the exactly treated subspace and factor it
      void build_exact_subspace();
      /// Apply the preconditioner: exact inverse inside the subspace,
      /// level-shifted diagonal outside it
      void precondition(const double *r, double mu, double *out) const;
      /// Has the reference left the occupation pattern the parametrization
      /// was built around?
      bool pattern_stale() const;
      /// Report how well conditioned the problem is: the spread of the
      /// uncoupled diagonal, how much of it is near-singular, and the
      /// spectrum of the exactly built block. This is what says whether a
      /// case is hard because of the occupations, because of
      /// near-degenerate rotations, or not at all.
      void report_conditioning(int verbosity) const;
      /// Warn if a frozen block's next orbital lies below the Fermi level
      /// of the fractionally occupied ones, i.e. if the occupation pattern
      /// handed over is not aufbau-stable
      void check_aufbau(int verbosity) const;
      /// Look for a block frozen in violation of the KKT conditions of the
      /// constrained problem, and if there is one, move a little charge
      /// across so it becomes a free coordinate again. Returns true if the
      /// occupations were changed.
      bool release_kkt_violation(int verbosity);

    public:
      /// blocks_per_particle gives the number of blocks belonging to each
      /// particle type, in order; maxocc the occupation of a filled
      /// orbital in each block.
      Optimizer(const helfem::Matrix &Sinvh, const std::vector<double> &maxocc,
                const std::vector<size_t> &blocks_per_particle, FockBuilder fock,
                ResponseBuilder response);

      /// Seed the reference from orbitals in the orthonormal basis and
      /// their occupations. The occupations are only read as block
      /// electron counts: the aufbau ordering within a block is taken
      /// from the orbital order, which is what the first-order solver
      /// hands over.
      void set_reference(const std::vector<helfem::Matrix> &orbs,
                         const std::vector<helfem::Vector> &occs);

      /// Optimize. Throws if OpenTrustRegion is unavailable or fails.
      Result run(const Options &opts);

      /// Compare the analytic gradient and Hessian-vector products
      /// against finite differences of the energy surface.
      ///
      /// exact_hessian says whether the response kernel behind the
      /// Hessian is the true one -- it is for an LDA, and deliberately is
      /// not for a GGA or meta-GGA, where the gradient and tau blocks of
      /// the kernel are dropped. With it false the Hessian comparison
      /// stops being a pass/fail test and becomes a measurement of how
      /// good the model Hessian is, since only the convergence rate
      /// depends on it; the gradient must be exact either way, and that
      /// is what the return value then reflects.
      bool verify(double step, double tol, int verbosity,
                  bool exact_hessian = true);

      /// Reference orbitals in the orthonormal basis
      const std::vector<helfem::Matrix> &orbitals() const { return U_; }
      /// Reference occupations
      const std::vector<helfem::Vector> &occupations() const { return f_; }
      /// Reference Fock matrices in the MO basis; the diagonal holds the
      /// orbital energies of the converged solution
      const std::vector<helfem::Matrix> &fock() const { return F_; }
      /// Number of parameters
      size_t nparam() const { return nparam_; }
    };

  } // namespace trscf
} // namespace helfem

#endif
