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
    /// reci's CI is real, and that is all it needs to be. An orbital with
    /// m != 0 is complex in HelFEM's bases, but the integrals over such
    /// orbitals are still REAL numbers: the radial factors are real and so are
    /// the Gaunt coefficients of complex spherical harmonics. What complex
    /// orbitals cost is only the permutational symmetry (tu|vw) = (ut|vw);
    /// the two that a real symmetric Hamiltonian needs, (tu|vw) = (vw|tu) and
    /// (tu|vw) = (ut|wv), still hold. reci builds sigma from the full n^4
    /// tensor, so any active space works -- p, d, f shells included. The
    /// eri is passed through unsymmetrised; symmetrising it is exactly the
    /// real-orbital assumption that does NOT hold here.
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
      /// the cheap per-macroiteration call. Throws std::invalid_argument if
      /// the integrals are not those of a real symmetric Hamiltonian, which
      /// would mean a bug upstream, not a property of the orbitals.
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

      /// Coefficients in the CSFs of the target spin <-> in determinants.
      /// The CSFs are orthonormal combinations of determinants, so to_det is
      /// an isometry and to_csf(to_det(x)) == x.
      std::vector<double> to_det(const std::vector<double> & csf) const;
      std::vector<double> to_csf(const std::vector<double> & det) const;

      /// Exact diagonal of H in the CSF basis, E_inactive included.
      std::vector<double> hdiag_csf() const;

      /// Spin-free RDMs <L| ... |R>, chemist order. L == R gives the state
      /// RDMs.
      RDMs rdms(const std::vector<double> & L, const std::vector<double> & R) const;

     private:
      struct Impl;
      std::unique_ptr<Impl> impl_;
    };

    /// Largest violation of one permutational symmetry of a chemist tensor:
    /// which = 0: (tu|vw) = (vw|tu); 1: (tu|vw) = (ut|wv), both of which any
    /// real Hamiltonian has; 2: (tu|vw) = (ut|vw), which only real ORBITALS
    /// give.
    double eri_asymmetry(const std::vector<double> & eri, size_t n, int which);

    /// Throws std::invalid_argument unless eri has symmetries 0 and 1 above
    /// to a relative `tol`.
    void require_hermitian_eri(const std::vector<double> & eri, size_t n, double tol = 1e-10);

    /// The middle-index swap X[p,q,r,s] -> X[p,r,q,s] that converts
    /// chemist <-> physicist in either direction.
    std::vector<double> swap_middle(const std::vector<double> & X, size_t n);

  } // namespace cas
} // namespace helfem

#endif // HELFEM_CAS_CI_H
