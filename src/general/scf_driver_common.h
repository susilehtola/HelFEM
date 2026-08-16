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
#ifndef HELFEM_SCF_DRIVER_COMMON_H
#define HELFEM_SCF_DRIVER_COMMON_H

// Shared building blocks for the atomic and diatomic OOO SCF
// drivers. Both driver bodies used to keep byte-identical copies of
// these routines; extracted here so bug fixes only need to happen in
// one place.
//
// The linear algebra is Eigen-native (helfem::Matrix / helfem::Vector);
// the per-symmetry-block gather/scatter uses the Eigen index lists
// (std::vector<Eigen::Index>) the basis sym_idx returns directly.
// Keeping the working matrices Eigen (rather than a LAPACK back end) is
// what lets the SCF driver be instantiated at extended precision --
// Eigen's SelfAdjointEigenSolver is scalar-generic where a LAPACK
// symmetric eigensolver is double-only.

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <sstream>
#include <string>
#include <cstdio>
#include <stdexcept>
#include <utility>
#include <vector>
#include "scf_helpers.h"
#include "timer.h"
#include "openorbitaloptimizer/scfsolver.hpp"

namespace helfem {
  namespace scf_driver {

    /// Wall-clock accounting for the SCF loop, split by Fock component.
    ///
    /// The diatomic Fock build is dominated by exchange and, for DFT, by
    /// the XC quadrature, but which one dominates depends strongly on the
    /// basis -- so the split is worth printing rather than guessing.
    ///
    /// OOO has no timers of its own, so everything it does -- the
    /// diagonalization, the DIIS/ADIIS extrapolation, the occupation
    /// search -- is measured indirectly, as the gap between one Fock
    /// build returning and the next one starting. That bucket ("SCF
    /// solver" below) therefore also absorbs anything the driver itself
    /// does between builds; it is the time NOT spent building Fock
    /// matrices, which is exactly the quantity the old per-component
    /// timings could not show.
    ///
    /// A builder invocation may evaluate SEVERAL densities, and gensap
    /// evaluates them concurrently (the ODA polytope vertices). Component
    /// times are therefore collected per build into a local Components --
    /// never into shared state -- and folded in afterwards with
    /// add_build(), the same discipline the batched builder already uses
    /// for its log lines. When a batch runs n>1 entries at once the
    /// component sums exceed the batch's wall clock, so wall clock is
    /// tracked separately rather than derived from them.
    struct FockTimer {
      /// Component times of ONE Fock build. Local to that build.
      struct Components {
        double density = 0.0, xc = 0.0, coulomb = 0.0, exchange = 0.0;
        /// Wall clock of the whole build, including the leftovers.
        double total = 0.0;
        /// One-electron traces, Fock assembly, block scatter.
        double other() const {
          return total - (density + xc + coulomb + exchange);
        }
      };

      /// Cumulative over the whole SCF. Summed over builds, so under a
      /// concurrent batch these exceed the elapsed wall clock.
      double tot_density = 0.0, tot_xc = 0.0, tot_coulomb = 0.0,
             tot_exchange = 0.0, tot_other = 0.0;
      /// Wall clock actually spent inside the builder, and outside it.
      double tot_fock_wall = 0.0, tot_outside = 0.0;
      /// Individual Fock builds, and builder invocations.
      size_t nbuild = 0, nbatch = 0;

      /// Call at the top of the (possibly batched) Fock builder.
      void enter() {
        outside_ = started_ ? between_.get() : 0.0;
        tot_outside += outside_;
        ++nbatch;
        cur_ = Components();
        nbatch_builds_ = 0;
        batch_.set();
      }

      /// Fold in one completed build. Safe to call repeatedly; call it
      /// from serial code, after any concurrent region has joined.
      void add_build(const Components & c) {
        ++nbuild;
        ++nbatch_builds_;
        tot_density  += c.density;   cur_.density  += c.density;
        tot_xc       += c.xc;        cur_.xc       += c.xc;
        tot_coulomb  += c.coulomb;   cur_.coulomb  += c.coulomb;
        tot_exchange += c.exchange;  cur_.exchange += c.exchange;
        tot_other    += c.other();   cur_.total    += c.total;
      }

      /// Call at the bottom of the builder, after printing.
      void leave() {
        tot_fock_wall += batch_.get();
        between_.set();
        started_ = true;
      }

      /// Seconds spent outside the builder since the previous invocation.
      double outside() const { return outside_; }
      /// Seconds elapsed in this builder invocation so far. A serial
      /// builder uses this as its build's total; a batched one times
      /// each entry itself, since they overlap.
      double build_elapsed() const { return batch_.get(); }

      /// One line per builder invocation, after the energy decomposition.
      void print_build(bool have_xc, bool have_exx) const {
        const double wall = batch_.get();
        printf("  time: Coulomb %.3f s", cur_.coulomb);
        if (have_exx) printf(", exchange %.3f s", cur_.exchange);
        if (have_xc)  printf(", XC %.3f s", cur_.xc);
        printf(", density %.3f s, other %.3f s", cur_.density, cur_.other());
        if (nbatch_builds_ > 1)
          printf(" (summed over %zu concurrent builds)", nbatch_builds_);
        printf("; Fock %.3f s", wall);
        if (started_) printf("; SCF solver %.3f s", outside_);
        printf("\n");
        fflush(stdout);
      }

      /// Cumulative summary, printed once the SCF has finished.
      void print_summary(bool have_xc, bool have_exx) const {
        const double comp = tot_density + tot_xc + tot_coulomb
                          + tot_exchange + tot_other;
        printf("\nSCF wall-clock breakdown over %zu Fock builds", nbuild);
        if (nbatch != nbuild) printf(" in %zu batches", nbatch);
        printf(":\n");
        printf("  Coulomb      %12.3f s\n", tot_coulomb);
        if (have_exx) printf("  exchange     %12.3f s\n", tot_exchange);
        if (have_xc)  printf("  XC           %12.3f s\n", tot_xc);
        printf("  density      %12.3f s\n", tot_density);
        printf("  other (Fock) %12.3f s\n", tot_other);
        printf("  Fock total   %12.3f s", comp);
        // Only when builds ran concurrently do the two differ.
        if (comp > 1.05 * tot_fock_wall)
          printf("  (%.3f s elapsed; builds ran concurrently)", tot_fock_wall);
        printf("\n");
        printf("  SCF solver   %12.3f s  (diagonalization, DIIS/ADIIS,"
               " occupations)\n", tot_outside);
        printf("  SCF total    %12.3f s\n", tot_fock_wall + tot_outside);
        fflush(stdout);
      }

    private:
      Timer batch_, between_;
      Components cur_;
      size_t nbatch_builds_ = 0;
      double outside_ = 0.0;
      bool started_ = false;
    };

    /// Per-block symmetric orthonormalisation of the AO overlap S
    /// restricted to each symmetry index set. Both drivers build this
    /// once and reuse it in the CoreH construction, the --load block
    /// projection, and the --save density reconstruction.
    inline std::vector<helfem::Matrix> build_per_block_Sinvh(
        const helfem::Matrix & S, const std::vector<std::vector<Eigen::Index>> & dsym) {
      const size_t nsym = dsym.size();
      std::vector<helfem::Matrix> out(nsym);
      for (size_t k = 0; k < nsym; ++k) {
        if (dsym[k].empty()) continue;
        const std::vector<Eigen::Index> & idx = dsym[k];
        const helfem::Matrix Sk = S(idx, idx);
        out[k] = scf::form_Sinvh(Sk, /*chol=*/false);
      }
      return out;
    }

    /// Build OOO's per-(spin, block) initial Fock matrix in the
    /// orthonormal basis from a global AO Hamiltonian H0. For
    /// unrestricted magnetic-field runs each spin channel gets its
    /// own +/- 0.5 * Bz * S Zeeman split, matching the split the
    /// steady-state Fock builder applies.
    template <typename Real>
    inline OpenOrbitalOptimizer::FockMatrix<Real> build_coreH_from_H0(
        const helfem::Matrix & H0, const helfem::Matrix & S,
        const std::vector<std::vector<Eigen::Index>> & dsym,
        const std::vector<helfem::Matrix> & Sinvh,
        size_t nparttype, bool have_bfield, double Bz) {
      const size_t nsym = dsym.size();
      OpenOrbitalOptimizer::FockMatrix<Real> CoreH(nsym * nparttype);
      for (size_t t = 0; t < nparttype; ++t) {
        for (size_t k = 0; k < nsym; ++k) {
          if (dsym[k].empty()) {
            CoreH[t * nsym + k] = helfem::Matrix::Zero(0, 0);
            continue;
          }
          const std::vector<Eigen::Index> & idx = dsym[k];
          helfem::Matrix H_sub = H0(idx, idx);
          if (have_bfield && nparttype == 2)
            H_sub += (t == 0 ? -0.5 : 0.5) * Bz * helfem::Matrix(S(idx, idx));
          CoreH[t * nsym + k] = Sinvh[k].transpose() * H_sub * Sinvh[k];
        }
      }
      return CoreH;
    }

    /// Load-path helper: take a saved AO density Pspin projected into
    /// the current basis, diagonalise it inside symmetry block k in
    /// the block's orthonormal basis, and hand OOO the resulting
    /// orbitals + occupations (largest occupation first). Empty
    /// blocks become 0x0 placeholders. Called per spin channel and
    /// per block from the driver's --load path.
    ///
    ///   P_orth = (Sinvh_k^T S_k) . Pspin(dsym[k], dsym[k]) . (S_k Sinvh_k)
    ///          -> V, w  (descending); w clamped to [0, max_occ]
    ///
    /// The S factors are what makes this a DENSITY transformation.
    /// Orbital coefficients transform as C = Sinvh C~, so the density
    /// P = C n C^T pulls back as P~ = Sinvh^-1 P Sinvh^-T, and
    /// Sinvh^T S Sinvh = I gives Sinvh^-1 = Sinvh^T S on the retained
    /// subspace. Dropping the S factors would apply the Fock (covariant)
    /// rule instead: that returns S^-1 P S^-1, whose trace is not the
    /// electron count and whose dominant eigenvectors are the directions
    /// where S^-1 blows up -- the most nearly linearly dependent,
    /// highest-kinetic-energy ones. Restarting from an exact same-basis
    /// density then began at +3.8e5 Eh instead of the converged energy.
    template <typename Real>
    inline void fill_block_from_density(
        size_t out_index,
        OpenOrbitalOptimizer::Orbitals<Real> & orbs,
        OpenOrbitalOptimizer::OrbitalOccupations<Real> & occs,
        const helfem::Matrix & Pspin, const helfem::Matrix & S,
        const std::vector<Eigen::Index> & idx,
        const helfem::Matrix & Sinvh_block, double max_occ) {
      if (idx.empty()) {
        orbs[out_index] = helfem::Matrix::Zero(0, 0);
        occs[out_index] = helfem::Vector::Zero(0);
        return;
      }
      const helfem::Matrix Pblk  = Pspin(idx, idx);
      const helfem::Matrix Sblk  = S(idx, idx);
      const helfem::Matrix SC    = Sblk * Sinvh_block;
      const helfem::Matrix Porth = SC.transpose() * Pblk * SC;
      // SelfAdjointEigenSolver returns eigenvalues in ascending order;
      // reverse for descending (largest occupation first), matching the
      // old symmetric-eigensolver + manual reversal.
      Eigen::SelfAdjointEigenSolver<helfem::Matrix> es(Porth);
      if (es.info() != Eigen::Success)
        throw std::logic_error("--load: eigendecomposition of projected block density failed");
      const Eigen::Index n = es.eigenvalues().size();
      helfem::Matrix V(es.eigenvectors().rows(), n);
      helfem::Vector w(n);
      for (Eigen::Index i = 0; i < n; ++i) {
        V.col(i) = es.eigenvectors().col(n - 1 - i);
        w(i)     = std::min(std::max(es.eigenvalues()(n - 1 - i), 0.0), max_occ);
      }
      orbs[out_index] = V;
      occs[out_index] = w;
    }

    /// Save-path helper: reconstruct the full AO alpha / beta density
    /// matrices from OOO's converged per-block orbitals + occupations.
    /// Restricted case: orbs[k] carries the closed-shell density
    /// (max occ 2); alpha and beta both get half of it. Unrestricted:
    /// alpha in indices [0, nsym), beta in [nsym, 2*nsym).
    template <typename Real>
    inline std::pair<helfem::Matrix, helfem::Matrix> assemble_final_density(
        size_t Nbf, bool restricted,
        const std::vector<std::vector<Eigen::Index>> & dsym,
        const std::vector<helfem::Matrix> & Sinvh,
        const OpenOrbitalOptimizer::Orbitals<Real> & final_orbs,
        const OpenOrbitalOptimizer::OrbitalOccupations<Real> & final_occs) {
      const size_t nsym = dsym.size();
      const Eigen::Index N = static_cast<Eigen::Index>(Nbf);
      helfem::Matrix Pa_final = helfem::Matrix::Zero(N, N);
      helfem::Matrix Pb_final = helfem::Matrix::Zero(N, N);
      for (size_t k = 0; k < nsym; ++k) {
        if (dsym[k].empty()) continue;
        const std::vector<Eigen::Index> & idx = dsym[k];
        const helfem::Matrix orb_a_ao = Sinvh[k] * final_orbs[k];
        const helfem::Vector occ_a    = final_occs[k];
        if (restricted) {
          const helfem::Matrix P_block = 0.5 * (orb_a_ao * occ_a.asDiagonal() * orb_a_ao.transpose());
          Pa_final(idx, idx) += P_block;
          Pb_final(idx, idx) += P_block;
        } else {
          const helfem::Matrix orb_b_ao = Sinvh[k] * final_orbs[nsym + k];
          const helfem::Vector occ_b    = final_occs[nsym + k];
          Pa_final(idx, idx) += orb_a_ao * occ_a.asDiagonal() * orb_a_ao.transpose();
          Pb_final(idx, idx) += orb_b_ao * occ_b.asDiagonal() * orb_b_ao.transpose();
        }
      }
      return {Pa_final, Pb_final};
    }

    /// Assemble global AO-basis orbital matrices from the converged
    /// per-block orthonormal-basis orbitals, for --save. Columns are
    /// sorted by occupation (descending), so leftCols(nocc) spans the
    /// occupied space -- the layout the analysis tools (diatomic_cpl,
    /// ...) expect of the checkpoint's Ca/Cb. The occupation vectors
    /// come back in the same order, as per-spin occupations (the
    /// restricted channel's total is split evenly).
    template <typename Real>
    inline void assemble_final_orbitals(
        size_t Nbf, bool restricted,
        const std::vector<std::vector<Eigen::Index>> & dsym,
        const std::vector<helfem::Matrix> & Sinvh,
        const OpenOrbitalOptimizer::Orbitals<Real> & final_orbs,
        const OpenOrbitalOptimizer::OrbitalOccupations<Real> & final_occs,
        helfem::Matrix & Ca, helfem::Vector & occa,
        helfem::Matrix & Cb, helfem::Vector & occb) {
      const size_t nsym = dsym.size();
      const Eigen::Index N = static_cast<Eigen::Index>(Nbf);

      auto build = [&](size_t block_offset, helfem::Matrix & C, helfem::Vector & occ,
                       double occ_scale) {
        // (occupation, block, column-in-block), sorted descending by
        // occupation; ties keep block-then-column order (stable_sort).
        std::vector<std::tuple<double, size_t, Eigen::Index>> order;
        for (size_t k = 0; k < nsym; ++k) {
          if (dsym[k].empty()) continue;
          const helfem::Vector & o = final_occs[block_offset + k];
          for (Eigen::Index c = 0; c < o.size(); ++c)
            order.emplace_back(o(c), k, c);
        }
        std::stable_sort(order.begin(), order.end(),
                         [](const auto & a, const auto & b) {
                           return std::get<0>(a) > std::get<0>(b);
                         });

        C = helfem::Matrix::Zero(N, static_cast<Eigen::Index>(order.size()));
        occ = helfem::Vector::Zero(static_cast<Eigen::Index>(order.size()));
        for (size_t i = 0; i < order.size(); ++i) {
          const size_t k = std::get<1>(order[i]);
          const Eigen::Index c = std::get<2>(order[i]);
          const helfem::Vector col_ao = Sinvh[k] * final_orbs[block_offset + k].col(c);
          const std::vector<Eigen::Index> & idx = dsym[k];
          for (size_t r = 0; r < idx.size(); ++r)
            C(idx[r], static_cast<Eigen::Index>(i)) = col_ao(static_cast<Eigen::Index>(r));
          occ(static_cast<Eigen::Index>(i)) = occ_scale * std::get<0>(order[i]);
        }
      };

      if (restricted) {
        build(0, Ca, occa, 0.5);
        Cb = Ca;
        occb = occa;
      } else {
        build(0, Ca, occa, 1.0);
        build(nsym, Cb, occb, 1.0);
      }
    }

    /// CLI-input normalisation shared by both drivers:
    /// * scf::parse_nela_nelb fills in nela/nelb from --Q and --M
    ///   when both are zero on entry;
    /// * M = 0 with no explicit nela/nelb runs spin-restricted
    ///   straight from the total electron count Ztotal - Q: the
    ///   restricted SCF sees a single density channel where only
    ///   Ntot matters (and may be odd), so no multiplicity is
    ///   needed. An explicit --restricted=0 with M=0 is an error
    ///   since the alpha/beta split is then undefined.
    /// * restr = -1 means "auto": closed shell -> 1 restricted,
    ///   otherwise 0 unrestricted. Restricted mode with an open
    ///   shell (nela != nelb) is allowed when requested explicitly:
    ///   only Ntot enters the restricted SCF, which distributes any
    ///   unpaired electrons over the frontier orbitals.
    /// Returns the derived (restricted, Ntot = nela + nelb) pair via
    /// out-refs so the drivers can use them directly below.
    inline void derive_nela_nelb_restricted(
        int & nela, int & nelb, int & restr, int Q, int M, int Ztotal,
        bool & restricted, int & Ntot) {
      if (M == 0 && nela == 0 && nelb == 0) {
        if (restr == 0)
          throw std::logic_error("Unrestricted mode needs a multiplicity (--M) or "
                                 "explicit nela/nelb.\n");
        Ntot = Ztotal - Q;
        if (Ntot <= 0)
          throw std::logic_error("No electrons: Z - Q <= 0.\n");
        nela = (Ntot + 1) / 2;
        nelb = Ntot / 2;
        restr = 1;
        restricted = true;
        return;
      }
      scf::parse_nela_nelb(nela, nelb, Q, M, Ztotal);
      if (restr == -1) restr = (nela == nelb) ? 1 : 0;
      restricted = (restr != 0);
      Ntot = nela + nelb;
    }

    /// OOO block wiring. Fills the four IndexVector / Eigen /
    /// std::vector holders that the OOO SCFSolver constructor takes:
    ///   number_of_blocks_per_particle_type (size = nparttype)
    ///   maximum_occupation                 (size = nsym * nparttype)
    ///   number_of_particles                (size = nparttype)
    ///   block_descriptions                 (size = nsym * nparttype)
    ///
    /// Restricted mode packs everything into a single closed-shell
    /// particle type with max_occ = 2 per block. Unrestricted splits
    /// alpha (t=0) and beta (t=1) into two particle types with
    /// max_occ = 1 per block; block descriptions get an "a:" / "b:"
    /// prefix per channel.
    ///
    /// sym_labels names the blocks physically ("m=+1", "l=2 m=-1",
    /// "m=-2 u") and comes from the basis's sym_labels, which is the
    /// same place the block ordering itself is defined. The occupations
    /// OOO prints are then readable as quantum numbers rather than as
    /// bare block indices.
    ///
    /// block_degeneracy is how many degenerate spatial orbitals each
    /// block stands for. It is 1 everywhere except under the diatomic
    /// |m| symmetry, where a block with |m|>0 represents BOTH the +|m|
    /// and -|m| orbitals, so it holds twice the electrons. Folding the
    /// degeneracy into max_occ -- rather than carrying the partner as
    /// its own block -- is what makes the occupations, and not only the
    /// Fock matrix, respect the symmetry: an odd electron in a Pi shell
    /// comes out as 0.5 in each of +-1 rather than landing arbitrarily
    /// in one of two exactly degenerate blocks. (sadatom does the same
    /// thing for l, with max_occ = 2*(2l+1).)
    template <typename Real>
    inline void build_ooo_block_metadata(
        size_t nsym, size_t nparttype, bool restricted,
        int Ntot, int nela, int nelb,
        const std::vector<std::string> & sym_labels,
        const std::vector<int> & block_degeneracy,
        OpenOrbitalOptimizer::IndexVector & number_of_blocks_per_particle_type,
        Eigen::Matrix<Real, Eigen::Dynamic, 1> & maximum_occupation,
        Eigen::Matrix<Real, Eigen::Dynamic, 1> & number_of_particles,
        std::vector<std::string> & block_descriptions) {
      if (block_degeneracy.size() != nsym) {
        std::ostringstream oss;
        oss << "Got " << block_degeneracy.size() << " block degeneracies for "
            << nsym << " blocks.\n";
        throw std::logic_error(oss.str());
      }
      if (sym_labels.size() != nsym) {
        std::ostringstream oss;
        oss << "Got " << sym_labels.size() << " symmetry labels for " << nsym
            << " blocks: sym_labels and sym_idx are out of sync.\n";
        throw std::logic_error(oss.str());
      }
      number_of_blocks_per_particle_type.resize(nparttype);
      maximum_occupation.resize(nsym * nparttype);
      number_of_particles.resize(nparttype);
      block_descriptions.clear();
      block_descriptions.reserve(nsym * nparttype);
      for (size_t t = 0; t < nparttype; ++t) {
        number_of_blocks_per_particle_type(t) = static_cast<int>(nsym);
        number_of_particles(t) = static_cast<Real>(restricted ? Ntot : (t == 0 ? nela : nelb));
        for (size_t k = 0; k < nsym; ++k) {
          maximum_occupation(t * nsym + k) =
              (restricted ? 2.0 : 1.0) * static_cast<Real>(block_degeneracy[k]);
          block_descriptions.push_back(
              (nparttype == 1 ? "" : (t == 0 ? "a:" : "b:")) + sym_labels[k]);
        }
      }
    }

    /// Fock-builder helper: accumulate one block of the AO density
    /// matrix P_full from OOO's per-block (orbitals, occupations)
    /// pair. Called per symmetry block per spin channel from both
    /// drivers' fock_builder lambdas. Empty blocks are no-ops.
    ///
    ///   C_k     = Sinvh_k . orb_e
    ///   P_full(dsym[k], dsym[k]) += C_k . diag(occ_e) . C_k^T
    template <typename Real>
    inline void accumulate_density_block(
        helfem::Matrix & P_full, const std::vector<std::vector<Eigen::Index>> & dsym, size_t k,
        const std::vector<helfem::Matrix> & Sinvh,
        const helfem::Matrix & orb_e, const helfem::Vector & occ_e,
        const std::vector<Eigen::Index> * mirror = nullptr) {
      if (dsym[k].empty()) return;
      const std::vector<Eigen::Index> & idx = dsym[k];
      const helfem::Matrix C_k = Sinvh[k] * orb_e;
      // A mirrored block stands for two degenerate spatial orbitals, so
      // its occupation is shared equally between them. The -|m| partner
      // has the same radial coefficients and the same Sinvh, so the
      // same C_k scatters into both index sets at half weight -- which
      // is exactly the cylindrically averaged density.
      const bool paired = (mirror != nullptr && !mirror->empty());
      const helfem::Vector occ = paired ? helfem::Vector(0.5 * occ_e) : occ_e;
      const helfem::Matrix Pblock = C_k * occ.asDiagonal() * C_k.transpose();
      P_full(idx, idx) += Pblock;
      if (paired) P_full(*mirror, *mirror) += Pblock;
    }

    /// Fock-builder helper: extract block k of a full AO Fock matrix,
    /// transform to that block's orthonormal basis via
    /// Sinvh_k^T . F_k . Sinvh_k, and stash it into the OOO
    /// FockMatrix at index b (as helfem::Matrix). Empty blocks
    /// become 0x0 placeholders.
    template <typename Real>
    inline void orthonormalize_fock_block(
        OpenOrbitalOptimizer::FockMatrix<Real> & fock, size_t b,
        const std::vector<std::vector<Eigen::Index>> & dsym, size_t k,
        const std::vector<helfem::Matrix> & Sinvh,
        const helfem::Matrix & F_full) {
      if (dsym[k].empty()) {
        fock[b] = helfem::Matrix::Zero(0, 0);
        return;
      }
      const std::vector<Eigen::Index> & idx = dsym[k];
      const helfem::Matrix Fk_sub = F_full(idx, idx);
      fock[b] = Sinvh[k].transpose() * Fk_sub * Sinvh[k];
    }

    /// Fock-builder helper: assemble the alpha/beta HF exchange
    /// matrices and their energy contribution from a per-driver
    /// exchange_fn callable. exchange_fn(P) returns the AO K matrix
    /// for a spin-density P, folding in whichever coefficient set
    /// the driver supports (kfrac * K + kshort * K_rs for atomic,
    /// kfrac * K for diatomic). Restricted mode skips the K(Pb)
    /// build; alpha == beta by construction so the beta contribution
    /// is just 2 * the alpha one.
    ///
    /// Ka and Kb are the AO exchange buffers the caller then folds
    /// into the Fock matrix downstream. Exx is the energy
    /// contribution.
    template <typename ExchangeFn>
    inline void assemble_hf_exchange(
        helfem::Matrix & Ka, helfem::Matrix & Kb, double & Exx,
        const helfem::Matrix & Pa, const helfem::Matrix & Pb,
        bool restricted, bool have_exx, ExchangeFn exchange_fn) {
      Exx = 0.0;
      if (!have_exx) return;
      Ka = exchange_fn(Pa);
      Exx = 0.5 * (Pa * Ka).trace();
      if (!restricted) {
        Kb = exchange_fn(Pb);
        Exx += 0.5 * (Pb * Kb).trace();
      } else {
        Exx *= 2.0;
      }
    }

    /// Fock-builder helper: assemble the per-block orthonormal Fock
    /// matrices from the AO ingredients. Both drivers' fock_builder
    /// lambdas end with a byte-identical restricted / unrestricted
    /// branch that
    ///   * adds up H1 + J (+ XC + K) per spin channel,
    ///   * applies the spin-Zeeman +/- Bz/2 * S split (unrestricted
    ///     only),
    ///   * runs the driver-supplied orthonormalize_block callable to
    ///     orthonormalise per block.
    /// XCa / XCb, Ka / Kb are assumed pre-zeroed (their addends only
    /// fire under the corresponding have_* flag), matching the
    /// convention the driver bodies keep for the XC and HF-exchange
    /// pieces.
    template <typename Real, typename OrthoBlock>
    inline void assemble_fock_blocks(
        OpenOrbitalOptimizer::FockMatrix<Real> & fock,
        const helfem::Matrix & H1, const helfem::Matrix & J,
        const helfem::Matrix & XCa, const helfem::Matrix & XCb,
        const helfem::Matrix & Ka,  const helfem::Matrix & Kb,
        const helfem::Matrix & S,
        size_t nsym, bool restricted,
        bool have_xc, bool have_exx, bool have_bfield, double Bz,
        OrthoBlock orthonormalize_block) {
      if (restricted) {
        helfem::Matrix F_ao = H1 + J;
        if (have_xc)  F_ao += XCa;
        if (have_exx) F_ao += Ka;
        for (size_t k = 0; k < nsym; ++k)
          orthonormalize_block(fock, k, F_ao, k);
      } else {
        helfem::Matrix Fa_ao = H1 + J;
        helfem::Matrix Fb_ao = H1 + J;
        if (have_xc)  { Fa_ao += XCa; Fb_ao += XCb; }
        if (have_exx) { Fa_ao += Ka;  Fb_ao += Kb;  }
        // Spin-Zeeman: alpha <- -Bz/2 * S, beta <- +Bz/2 * S.
        if (have_bfield) {
          Fa_ao -= 0.5 * Bz * S;
          Fb_ao += 0.5 * Bz * S;
        }
        for (size_t k = 0; k < nsym; ++k) {
          orthonormalize_block(fock, k,        Fa_ao, k);
          orthonormalize_block(fock, nsym + k, Fb_ao, k);
        }
      }
    }

  } // namespace scf_driver
} // namespace helfem

#endif
