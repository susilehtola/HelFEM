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
#ifndef HELFEM_CAS_CI_H
#define HELFEM_CAS_CI_H

#include "cas_integrals.h"

#include <memory>
#include <stdexcept>
#include <vector>

namespace helfem {
  namespace cas {

    /// The CI half of a CAS calculation, solved by reci's determinant/CSF
    /// engine (reci::core). HelFEM supplies the active-space Hamiltonian
    /// (cas_integrals.h) and gets back energies, CI vectors and RDMs; reci
    /// never sees the basis.
    ///
    /// The two sides index their four-index quantities differently, and this
    /// class is the ONE place that reconciles them:
    ///
    ///   HelFEM (cas_integrals.h)       chemist  eri[t,u,v,w] = (tu|vw)
    ///                                           d[t,u,v,w]   with
    ///                                           E = 1/2 sum d_tuvw (tu|vw)
    ///   reci (CISigmaBuilder)          physicist V[p,q,r,s] = <pq|rs> = (pr|qs)
    ///                                           G[p,q,r,s] = <a+p a+q a_s a_r>
    ///
    /// Both directions are the same swap of the two middle indices. Nothing
    /// outside cas_ci.cpp has to know reci's convention.
    ///
    /// reci's CI is real, so the active orbitals must be real. In HelFEM's
    /// atomic basis an orbital with m != 0 is complex, and then
    /// (tu|vw) != (ut|vw); such a Hamiltonian is refused with
    /// ComplexOrbitalError rather than silently symmetrised -- the same guard,
    /// and for the same reason, as python/helfem/pyscf_driver.py.
    ///
    /// reci_core runs on Kokkos, which must be initialized before a CASCI is
    /// constructed; hold a KokkosScope in main() for the duration.

    /// RAII: initializes Kokkos if nothing else has, and finalizes it only if
    /// it did the initializing. Kokkos may be initialized once per process,
    /// so hold one of these in main(), outliving every CASCI.
    class KokkosScope {
     public:
      KokkosScope();
      ~KokkosScope();
      KokkosScope(const KokkosScope &) = delete;
      KokkosScope & operator=(const KokkosScope &) = delete;

     private:
      bool owner_ = false;
    };

    /// The active-space Hamiltonian is not real-symmetric: some active
    /// orbital is complex (m != 0 in the atomic basis).
    struct ComplexOrbitalError : public std::runtime_error {
      using std::runtime_error::runtime_error;
    };

    /// The spin-free RDMs of a pair of CI vectors, in HelFEM's chemist
    /// convention (see above). For bra == ket these are the state RDMs; for
    /// two different vectors, the transition RDMs the coupled orbital-CI
    /// Hessian needs.
    struct RDMs {
      helfem::Matrix D;          ///< D[t,u] = <L| E_tu |R>
      std::vector<double> d;     ///< chemist order, layout as cas::active_eri
    };

    class CASCI {
     public:
      /// A CAS(nalpha + nbeta, nact) in C1, spin-adapted to S = (nalpha -
      /// nbeta)/2 with M_S = S.
      CASCI(int nact, int nalpha, int nbeta);
      ~CASCI();
      CASCI(const CASCI &) = delete;
      CASCI & operator=(const CASCI &) = delete;

      /// Install the active-space Hamiltonian; strings are kept, so this is
      /// the cheap per-macroiteration call. Throws ComplexOrbitalError for a
      /// Hamiltonian reci cannot represent.
      void set_hamiltonian(const ActiveHamiltonian & ah);

      /// Number of determinants, the length of every CI vector here.
      size_t ndet() const;
      /// Number of CSFs of the target spin.
      size_t ncsf() const;

      /// Lowest root in the target spin: energy (including E_inactive) and
      /// its determinant-basis CI vector.
      ///
      /// Solved by dense diagonalization of H in the CSF basis, built from
      /// reci sigma vectors. Exact and simple, and limited to `max_csf`
      /// CSFs; an iterative solver replaces it when larger active spaces are
      /// needed.
      double solve_ground(std::vector<double> & civec, size_t max_csf = 5000) const;

      /// sigma = H c in the determinant basis, with E_inactive on the
      /// diagonal: the same H whose eigenvalue solve_ground returns.
      std::vector<double> sigma(const std::vector<double> & c) const;

      /// Spin-free RDMs <L| ... |R>, chemist order. L == R gives the state
      /// RDMs.
      RDMs rdms(const std::vector<double> & L, const std::vector<double> & R) const;

     private:
      struct Impl;
      std::unique_ptr<Impl> impl_;
    };

    /// Throws ComplexOrbitalError unless eri (chemist, as cas::active_eri)
    /// has the real-orbital symmetry (tu|vw) = (ut|vw) to a relative `tol`.
    void require_real_eri(const std::vector<double> & eri, size_t n, double tol = 1e-10);

    /// The middle-index swap X[p,q,r,s] -> X[p,r,q,s] that converts
    /// chemist <-> physicist in either direction.
    std::vector<double> swap_middle(const std::vector<double> & X, size_t n);

  } // namespace cas
} // namespace helfem

#endif // HELFEM_CAS_CI_H
