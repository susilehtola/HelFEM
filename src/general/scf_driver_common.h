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

#include <armadillo>
#include <cstdio>
#include <stdexcept>
#include <utility>
#include <vector>
#include "scf_helpers.h"
#include <ArmaEigen.h>
#include "openorbitaloptimizer/scfsolver.hpp"

namespace helfem {
  namespace scf_driver {

    /// Per-block symmetric orthonormalisation of the AO overlap S
    /// restricted to each symmetry index set. Both drivers build this
    /// once and reuse it in the CoreH construction, the --load block
    /// projection, and the --save density reconstruction.
    inline std::vector<arma::mat> build_per_block_Sinvh(
        const arma::mat & S, const std::vector<arma::uvec> & dsym) {
      const size_t nsym = dsym.size();
      std::vector<arma::mat> out(nsym);
      for (size_t k = 0; k < nsym; ++k) {
        if (!dsym[k].n_elem) continue;
        const arma::mat Sk = S(dsym[k], dsym[k]);
        out[k] = helfem::to_arma(scf::form_Sinvh(helfem::to_eigen(Sk), /*chol=*/false));
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
        const arma::mat & H0, const arma::mat & S,
        const std::vector<arma::uvec> & dsym,
        const std::vector<arma::mat> & Sinvh_arma,
        size_t nparttype, bool have_bfield, double Bz) {
      const size_t nsym = dsym.size();
      OpenOrbitalOptimizer::FockMatrix<Real> CoreH(nsym * nparttype);
      for (size_t t = 0; t < nparttype; ++t) {
        for (size_t k = 0; k < nsym; ++k) {
          if (!dsym[k].n_elem) {
            CoreH[t * nsym + k] = helfem::Matrix::Zero(0, 0);
            continue;
          }
          arma::mat H_sub = H0(dsym[k], dsym[k]);
          if (have_bfield && nparttype == 2)
            H_sub += (t == 0 ? -0.5 : 0.5) * Bz * arma::mat(S(dsym[k], dsym[k]));
          const arma::mat H_orth = Sinvh_arma[k].t() * H_sub * Sinvh_arma[k];
          CoreH[t * nsym + k] = helfem::to_eigen(H_orth);
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
    ///   P_orth = Sinvh_k^T . Pspin(dsym[k], dsym[k]) . Sinvh_k
    ///          -> V, w  (descending); w clamped to [0, max_occ]
    template <typename Real>
    inline void fill_block_from_density(
        size_t out_index,
        OpenOrbitalOptimizer::Orbitals<Real> & orbs,
        OpenOrbitalOptimizer::OrbitalOccupations<Real> & occs,
        const arma::mat & Pspin, const arma::uvec & idx,
        const arma::mat & Sinvh_block, double max_occ) {
      if (!idx.n_elem) {
        orbs[out_index] = helfem::Matrix::Zero(0, 0);
        occs[out_index] = helfem::Vector::Zero(0);
        return;
      }
      const arma::mat Pblk  = Pspin(idx, idx);
      const arma::mat Porth = arma::trans(Sinvh_block) * Pblk * Sinvh_block;
      arma::vec occ_eigs;
      arma::mat vec_eigs;
      if (!arma::eig_sym(occ_eigs, vec_eigs, Porth))
        throw std::logic_error("--load: eigendecomposition of projected block density failed");
      const arma::uword n = vec_eigs.n_cols;
      arma::mat V(vec_eigs.n_rows, n);
      arma::vec w(n);
      for (arma::uword i = 0; i < n; ++i) {
        V.col(i) = vec_eigs.col(n - 1 - i);
        w(i)     = std::min(std::max(occ_eigs(n - 1 - i), 0.0), max_occ);
      }
      orbs[out_index] = helfem::to_eigen(V);
      occs[out_index] = helfem::to_eigen(w);
    }

    /// Save-path helper: reconstruct the full AO alpha / beta density
    /// matrices from OOO's converged per-block orbitals + occupations.
    /// Restricted case: orbs[k] carries the closed-shell density
    /// (max occ 2); alpha and beta both get half of it. Unrestricted:
    /// alpha in indices [0, nsym), beta in [nsym, 2*nsym).
    template <typename Real>
    inline std::pair<arma::mat, arma::mat> assemble_final_density(
        size_t Nbf, bool restricted,
        const std::vector<arma::uvec> & dsym,
        const std::vector<arma::mat> & Sinvh_arma,
        const OpenOrbitalOptimizer::Orbitals<Real> & final_orbs,
        const OpenOrbitalOptimizer::OrbitalOccupations<Real> & final_occs) {
      const size_t nsym = dsym.size();
      arma::mat Pa_final(Nbf, Nbf, arma::fill::zeros);
      arma::mat Pb_final(Nbf, Nbf, arma::fill::zeros);
      for (size_t k = 0; k < nsym; ++k) {
        if (!dsym[k].n_elem) continue;
        const arma::mat orb_a_ao = Sinvh_arma[k] * helfem::to_arma(final_orbs[k]);
        const arma::vec occ_a    = helfem::to_arma(final_occs[k]);
        if (restricted) {
          const arma::mat P_block = 0.5 * (orb_a_ao * arma::diagmat(occ_a) * arma::trans(orb_a_ao));
          arma::mat Pa_tmp = Pa_final; Pa_tmp(dsym[k], dsym[k]) += P_block; Pa_final = Pa_tmp;
          arma::mat Pb_tmp = Pb_final; Pb_tmp(dsym[k], dsym[k]) += P_block; Pb_final = Pb_tmp;
        } else {
          const arma::mat orb_b_ao = Sinvh_arma[k] * helfem::to_arma(final_orbs[nsym + k]);
          const arma::vec occ_b    = helfem::to_arma(final_occs[nsym + k]);
          arma::mat Pa_tmp = Pa_final;
          Pa_tmp(dsym[k], dsym[k]) += orb_a_ao * arma::diagmat(occ_a) * arma::trans(orb_a_ao);
          Pa_final = Pa_tmp;
          arma::mat Pb_tmp = Pb_final;
          Pb_tmp(dsym[k], dsym[k]) += orb_b_ao * arma::diagmat(occ_b) * arma::trans(orb_b_ao);
          Pb_final = Pb_tmp;
        }
      }
      return {Pa_final, Pb_final};
    }

    /// CLI-input normalisation shared by both drivers:
    /// * scf::parse_nela_nelb fills in nela/nelb from --Q and --M
    ///   when both are zero on entry;
    /// * restr = -1 means "auto": closed shell -> 1 restricted,
    ///   otherwise 0 unrestricted;
    /// * restricted mode requires nela == nelb, else throws.
    /// Returns the derived (restricted, Ntot = nela + nelb) pair via
    /// out-refs so both drivers can use them directly below.
    inline void derive_nela_nelb_restricted(
        int & nela, int & nelb, int & restr, int Q, int M, int Ztotal,
        bool & restricted, int & Ntot) {
      scf::parse_nela_nelb(nela, nelb, Q, M, Ztotal);
      if (restr == -1) restr = (nela == nelb) ? 1 : 0;
      restricted = (restr != 0);
      if (restricted && nela != nelb)
        throw std::logic_error("Restricted mode requires nela == nelb (closed shell). "
                                "Use --restricted=0 (or leave -1 for auto) for open-shell.");
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
    template <typename Real>
    inline void build_ooo_block_metadata(
        size_t nsym, size_t nparttype, bool restricted,
        int Ntot, int nela, int nelb,
        OpenOrbitalOptimizer::IndexVector & number_of_blocks_per_particle_type,
        Eigen::Matrix<Real, Eigen::Dynamic, 1> & maximum_occupation,
        Eigen::Matrix<Real, Eigen::Dynamic, 1> & number_of_particles,
        std::vector<std::string> & block_descriptions) {
      number_of_blocks_per_particle_type.resize(nparttype);
      maximum_occupation.resize(nsym * nparttype);
      number_of_particles.resize(nparttype);
      block_descriptions.clear();
      block_descriptions.reserve(nsym * nparttype);
      for (size_t t = 0; t < nparttype; ++t) {
        number_of_blocks_per_particle_type(t) = static_cast<int>(nsym);
        number_of_particles(t) = static_cast<Real>(restricted ? Ntot : (t == 0 ? nela : nelb));
        for (size_t k = 0; k < nsym; ++k) {
          maximum_occupation(t * nsym + k) = restricted ? 2.0 : 1.0;
          block_descriptions.push_back(
              (nparttype == 1 ? "" : (t == 0 ? "a:" : "b:"))
              + std::string("sym") + std::to_string(k));
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
        arma::mat & P_full, const std::vector<arma::uvec> & dsym, size_t k,
        const std::vector<arma::mat> & Sinvh_arma,
        const helfem::Matrix & orb_e, const helfem::Vector & occ_e) {
      if (!dsym[k].n_elem) return;
      const arma::mat orb  = helfem::to_arma(orb_e);
      const arma::vec occ  = helfem::to_arma(occ_e);
      const arma::mat C_k  = Sinvh_arma[k] * orb;
      const arma::mat P_k  = C_k * arma::diagmat(occ) * C_k.t();
      P_full(dsym[k], dsym[k]) += P_k;
    }

    /// Fock-builder helper: extract block k of a full AO Fock matrix,
    /// transform to that block's orthonormal basis via
    /// Sinvh_k^T . F_k . Sinvh_k, and stash it into the OOO
    /// FockMatrix at index b (as helfem::Matrix). Empty blocks
    /// become 0x0 placeholders.
    template <typename Real>
    inline void orthonormalize_fock_block(
        OpenOrbitalOptimizer::FockMatrix<Real> & fock, size_t b,
        const std::vector<arma::uvec> & dsym, size_t k,
        const std::vector<arma::mat> & Sinvh_arma,
        const arma::mat & F_full) {
      if (!dsym[k].n_elem) {
        fock[b] = helfem::Matrix::Zero(0, 0);
        return;
      }
      const arma::mat Fk_sub = F_full(dsym[k], dsym[k]);
      const arma::mat F_orth = Sinvh_arma[k].t() * Fk_sub * Sinvh_arma[k];
      fock[b] = helfem::to_eigen(F_orth);
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
        arma::mat & Ka, arma::mat & Kb, double & Exx,
        const arma::mat & Pa, const arma::mat & Pb,
        bool restricted, bool have_exx, ExchangeFn exchange_fn) {
      Exx = 0.0;
      if (!have_exx) return;
      Ka = exchange_fn(Pa);
      Exx = 0.5 * arma::trace(Pa * Ka);
      if (!restricted) {
        Kb = exchange_fn(Pb);
        Exx += 0.5 * arma::trace(Pb * Kb);
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
    ///   * runs the driver-supplied apply_mavg / orthonormalize_block
    ///     callables to symmetrise and orthonormalise per block.
    /// XCa / XCb, Ka / Kb are assumed pre-zeroed (their addends only
    /// fire under the corresponding have_* flag), matching the
    /// convention the driver bodies keep for the XC and HF-exchange
    /// pieces.
    template <typename Real, typename ApplyMAvg, typename OrthoBlock>
    inline void assemble_fock_blocks(
        OpenOrbitalOptimizer::FockMatrix<Real> & fock,
        const arma::mat & H1, const arma::mat & J,
        const arma::mat & XCa, const arma::mat & XCb,
        const arma::mat & Ka,  const arma::mat & Kb,
        const arma::mat & S,
        size_t nsym, bool restricted,
        bool have_xc, bool have_exx, bool have_bfield, double Bz,
        ApplyMAvg apply_mavg, OrthoBlock orthonormalize_block) {
      if (restricted) {
        arma::mat F_ao = H1 + J;
        if (have_xc)  F_ao += XCa;
        if (have_exx) F_ao += Ka;
        apply_mavg(F_ao);
        for (size_t k = 0; k < nsym; ++k)
          orthonormalize_block(fock, k, F_ao, k);
      } else {
        arma::mat Fa_ao = H1 + J;
        arma::mat Fb_ao = H1 + J;
        if (have_xc)  { Fa_ao += XCa; Fb_ao += XCb; }
        if (have_exx) { Fa_ao += Ka;  Fb_ao += Kb;  }
        // Spin-Zeeman: alpha <- -Bz/2 * S, beta <- +Bz/2 * S.
        if (have_bfield) {
          Fa_ao -= 0.5 * Bz * S;
          Fb_ao += 0.5 * Bz * S;
        }
        apply_mavg(Fa_ao);
        apply_mavg(Fb_ao);
        for (size_t k = 0; k < nsym; ++k) {
          orthonormalize_block(fock, k,        Fa_ao, k);
          orthonormalize_block(fock, nsym + k, Fb_ao, k);
        }
      }
    }

  } // namespace scf_driver
} // namespace helfem

#endif
