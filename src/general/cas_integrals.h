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
#ifndef HELFEM_CAS_INTEGRALS_H
#define HELFEM_CAS_INTEGRALS_H

#include <helfem/Matrix.h>
#include <vector>

namespace helfem {
  namespace cas {

    /// Everything a geometry has to supply for a CAS wave function.
    ///
    /// HelFEM never materializes the four-index AO tensor, and none is needed:
    /// `coulomb` contracts the SECOND index pair,
    ///
    ///     coulomb(P)_{mu nu} = sum_{la si} (mu nu|la si) P_{la si}
    ///
    /// so feeding it a pair density and contracting the free pair against MO
    /// coefficients yields MO integrals directly. Every quantity below --
    /// active-space integrals, the generalized Fock matrix, its response --
    /// reduces to that one call, which is why no AO->MO transform appears
    /// anywhere and why the same code serves the atomic and diatomic bases.
    ///
    /// The two geometries differ only in how they implement this interface.
    class JKProvider {
     public:
      virtual ~JKProvider() = default;

      /// J[P]_{mu nu} = sum_{la si} (mu nu|la si) P_{la si}.
      virtual helfem::Matrix coulomb(const helfem::Matrix & P) const = 0;

      /// The SIGNED exchange contribution to the Fock matrix of a SPIN
      /// density -- HelFEM's own convention, stated at src/atomic/main.cpp:383
      /// ("returns the signed contribution that gets ADDED to the Fock
      /// matrix") and used there as Exx = 0.5 tr(P_spin K_spin) per channel.
      /// It is NOT the PySCF convention, where K is positive and subtracted;
      /// the two differ by a sign AND by the spin-vs-total density factor of
      /// two. `closed_shell_veff` below is the single place that reconciles
      /// them, so nothing else has to think about it.
      virtual helfem::Matrix exchange(const helfem::Matrix & P) const = 0;

      /// One-electron Hamiltonian in the same AO basis.
      virtual const helfem::Matrix & hcore() const = 0;
    };

    /// Closed-shell effective potential of a TOTAL density P: J[P] + K[P/2].
    ///
    /// The halving is the spin-vs-total density conversion -- `exchange` wants
    /// a spin density, and for a closed shell that is P/2. Keeping it here
    /// means the convention is stated once.
    helfem::Matrix closed_shell_veff(const JKProvider & jk, const helfem::Matrix & P);

    /// Inactive (doubly occupied) Fock matrix and core energy for the lowest
    /// `ninact` columns of C. F = h + veff[P], E = tr(P h) + tr(P veff)/2 with
    /// P = 2 sum_i C_i C_i^T -- the closed-shell RHF expressions.
    helfem::Matrix inactive_fock(const JKProvider & jk, const helfem::Matrix & C,
                                 Eigen::Index ninact, double & E_inactive);

    /// Chemist-notation (tu|vw) over the columns of Ca, stored row-major as
    /// eri[((t*n + u)*n + v)*n + w] -- the layout the RDMs here are
    /// index-aligned with. NOT reci's: reci takes physicist <pq|rs> = (pr|qs),
    /// and cas_ci.cpp converts at that boundary and nowhere else. Costs n(n+1)/2 `coulomb`
    /// calls; no AO->MO transform.
    ///
    /// The pair density is passed to coulomb() UNSYMMETRISED. That matters for
    /// complex basis functions (any m != 0), where (tu|vw) != (ut|vw) and the
    /// antisymmetric part of the pair density -- a purely imaginary density --
    /// carries genuine information. Contracting the result:
    ///     J_tu = sum_vw (tu|vw) P_vw        K_tu = sum_vw (tw|vu) P_vw
    /// note the K ordering, which reduces to the familiar (tv|uw) only for real
    /// orbitals. Both are exact to 1e-15 at every lmax; the real-orbital K
    /// ordering is wrong by ~2e-1 as soon as lmax > 0.
    std::vector<double> active_eri(const JKProvider & jk, const helfem::Matrix & Ca);

    /// The CAS Hamiltonian handed to a CI solver.
    struct ActiveHamiltonian {
      /// Closed-shell core energy the CI adds to its eigenvalue.
      double E_inactive = 0.0;
      /// (nact x nact) inactive Fock restricted to the active space.
      helfem::Matrix h_eff;
      /// (nact^4) active-space integrals, layout as in active_eri.
      std::vector<double> eri;
    };

    ActiveHamiltonian active_hamiltonian(const JKProvider & jk,
                                         const helfem::Matrix & C,
                                         Eigen::Index ninact, Eigen::Index nact);

    /// The integrals of one set of orbitals C, built once and shared by
    /// everything a CASSCF step needs there: the active Hamiltonian, the
    /// generalized Fock matrix at the state or at any transition RDMs, and
    /// its orbital response.
    ///
    /// The construction costs one closed-shell Fock build for the inactive
    /// density and one Coulomb build per unordered active pair (v,w); the
    /// pair matrices are kept as C^T J[C_v C_w^T] C_a, all orbitals by the
    /// active ones, which is every (mu|vw) with u active that the
    /// generalized Fock matrix contracts. With them the 2-RDM term of any
    /// generalized Fock matrix is a contraction, and the active Hamiltonian
    /// is their active rows. No product here runs over all orbitals twice:
    /// in a basis of thousands of orbitals an N^3 product per call costs
    /// more than the integrals.
    class CASPoint {
     public:
      CASPoint(const JKProvider & jk, const helfem::Matrix & C,
               Eigen::Index ninact, Eigen::Index nact);

      const helfem::Matrix & orbitals() const { return C_; }
      const ActiveHamiltonian & hamiltonian() const { return ah_; }

      /// generalized_fock below, (norb x nocc). Costs one closed-shell Fock
      /// build of the active density, cached for the last D asked for.
      helfem::Matrix generalized_fock(const helfem::Matrix & D, const std::vector<double> & d,
                                      double core_occ = 2.0) const;

      /// fock_response_kappa below, (norb x nocc), and if dH is given
      /// active_hamiltonian_response along the same dkap: the two share
      /// their Coulomb builds, two per unordered active pair, plus two
      /// closed-shell Fock builds.
      helfem::Matrix fock_response(const helfem::Matrix & D, const std::vector<double> & d,
                                   const helfem::Matrix & dkap,
                                   ActiveHamiltonian * dH = nullptr) const;

      /// Diagonal of C^T (F^I + F^A) C over all orbitals, F^A that of the
      /// active density Ca D Ca^T: what the usual approximation to the
      /// orbital Hessian diagonal needs.
      helfem::Vector fock_diagonal(const helfem::Matrix & D) const;

     private:
      const JKProvider & jk_;
      helfem::Matrix C_;
      Eigen::Index ninact_, nact_;
      /// h + veff of the inactive density, AO; C^T Vi C over occupied columns.
      helfem::Matrix Vi_, FIo_;
      /// L_[v*nact + w] = C^T J[C_v C_w^T] C_a.
      std::vector<helfem::Matrix> L_;
      ActiveHamiltonian ah_;
      /// veff of the active density for Va_D_.
      mutable helfem::Matrix Va_, Va_D_;

      const helfem::Matrix & active_veff(const helfem::Matrix & D) const;
      /// T(u,t) = d_tuvw.
      helfem::Matrix pair_block(const std::vector<double> & d, Eigen::Index v,
                                Eigen::Index w) const;
      /// Q[m][t] = sum_uvw d_tuvw (mu|vw), all orbitals by active.
      helfem::Matrix Q(const std::vector<double> & d) const;
      void check_rdms(const helfem::Matrix & D, const std::vector<double> & d) const;
    };

    /// F - F^T restricted to the occupied columns, for F (norb x nocc) whose
    /// columns beyond nocc vanish: the occupied columns of the square
    /// completion's antisymmetric part.
    helfem::Matrix antisymmetrize_occupied(const helfem::Matrix & F);
    /// The same, as a square (norb x norb) matrix.
    helfem::Matrix square_antisymmetric(const helfem::Matrix & F);

    /// d/dt active_hamiltonian(C exp(t dkap)) at t = 0: the one-index
    /// transformed Hamiltonian. Its sigma vector on the CI state is the
    /// CI-orbital block of the coupled Hessian. Same coulomb() primitive as
    /// everything else here: for each active pair, J of the pair density and
    /// of its derivative.
    ActiveHamiltonian active_hamiltonian_response(const JKProvider & jk,
                                                  const helfem::Matrix & C,
                                                  Eigen::Index ninact,
                                                  Eigen::Index nact,
                                                  const helfem::Matrix & dkap);

    /// CASSCF generalized Fock matrix F[m][p]: m over all orbitals, p over
    /// inactive+active.
    ///
    ///   F[m][i] = core_occ F^I[m][i] + 2 F^A[m][i]        i inactive
    ///   F[m][t] = sum_u D_tu F^I[m][u] + Q[m][t]          t active
    ///   Q[m][t] = sum_uvw d_tuvw (mu|vw)
    ///
    /// `core_occ` is the inactive occupation: 2 for a state density, and 0 for
    /// a TRANSITION density. That is not a scaling choice -- for two states of
    /// the same CAS, <I|E_ij|0> = delta_ij <I|0> = 0 over the inactive block,
    /// so the inactive-Fock term drops out of the inactive columns entirely,
    /// while the active columns keep it because their 2-RDM blocks go like
    /// d_tuij ~ D_tu delta_ij. The transition case is not the state case with
    /// different numbers substituted.
    ///
    /// The orbital gradient is 2(F - F^T) over the square completion.
    helfem::Matrix generalized_fock(const JKProvider & jk, const helfem::Matrix & C,
                                    Eigen::Index ninact, Eigen::Index nact,
                                    const helfem::Matrix & D,
                                    const std::vector<double> & d,
                                    double core_occ = 2.0);

    /// g = 2(F - F^T), square over all orbitals: the CAS energy gradient with
    /// respect to an orbital rotation at fixed CI coefficients.
    helfem::Matrix orbital_gradient(const JKProvider & jk, const helfem::Matrix & C,
                                    Eigen::Index ninact, Eigen::Index nact,
                                    const helfem::Matrix & D,
                                    const std::vector<double> & d);

    /// d/dt generalized_fock(C exp(t dkap)) at t = 0, RDMs frozen.
    ///
    /// Every piece of the generalized Fock is a coulomb/exchange call on a
    /// density built from C, so its derivative is the same calls on the
    /// one-index-transformed densities (dC = C dkap) plus the contraction
    /// terms. No new integral machinery, exactly as for the gradient.
    helfem::Matrix fock_response_kappa(const JKProvider & jk,
                                       const helfem::Matrix & C,
                                       Eigen::Index ninact, Eigen::Index nact,
                                       const helfem::Matrix & D,
                                       const std::vector<double> & d,
                                       const helfem::Matrix & dkap);

    /// d/dt of the orbital gradient along dkap, square over all orbitals.
    ///
    /// This is the orbital-orbital Hessian block PLUS (1/2) g . [dkap, .], and
    /// the extra term is not negligible bookkeeping. Differentiating the
    /// gradient walks a PRODUCT of exponentials, exp(t K') exp(s K), whereas
    /// the Hessian is defined on exp(sK + tK'); Baker-Campbell-Hausdorff
    /// separates the two by that commutator. They agree only where the
    /// gradient vanishes. Callers wanting the Hessian must therefore
    /// SYMMETRISE over the two directions, which removes it exactly.
    ///
    /// Measured: the antisymmetric part of this reproduces (1/2) g . [K', K]
    /// to five significant figures at a non-stationary point.
    helfem::Matrix hess_kappa_kappa_raw(const JKProvider & jk,
                                        const helfem::Matrix & C,
                                        Eigen::Index ninact, Eigen::Index nact,
                                        const helfem::Matrix & D,
                                        const std::vector<double> & d,
                                        const helfem::Matrix & dkap);

    /// The symmetric K1 . H_kk . K2, with the BCH ordering term removed.
    double hess_kappa_kappa(const JKProvider & jk, const helfem::Matrix & C,
                            Eigen::Index ninact, Eigen::Index nact,
                            const helfem::Matrix & D,
                            const std::vector<double> & d,
                            const helfem::Matrix & K1, const helfem::Matrix & K2);

    /// The CAS energy at these orbitals with the RDMs held fixed:
    ///     E = E_inactive + sum_pq h_eff_pq D_pq + sum_pqrs eri_pqrs d_pqrs / 2
    /// Rebuilt through active_hamiltonian, so it shares no algebra with the
    /// gradient or the Hessian -- which is what makes finite differences of it
    /// a real check on either, rather than a re-expansion of them.
    double frozen_ci_energy(const JKProvider & jk, const helfem::Matrix & C,
                            Eigen::Index ninact, Eigen::Index nact,
                            const helfem::Matrix & D,
                            const std::vector<double> & d);

  } // namespace cas
} // namespace helfem

#endif // HELFEM_CAS_INTEGRALS_H
