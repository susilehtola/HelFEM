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
#include "cas_integrals.h"

#include <stdexcept>

namespace helfem {
  namespace cas {

    namespace {
      /// Row-major index into an (n,n,n,n) tensor, matching numpy C order and
      /// reci's Problem::eri.
      inline size_t idx4(size_t n, size_t t, size_t u, size_t v, size_t w) {
        return ((t * n + u) * n + v) * n + w;
      }
    } // namespace

    helfem::Matrix closed_shell_veff(const JKProvider & jk, const helfem::Matrix & P) {
      // exchange() wants a SPIN density and returns the signed contribution to
      // be added; for a closed shell the spin density is P/2.
      return jk.coulomb(P) + jk.exchange(0.5 * P);
    }

    helfem::Matrix inactive_fock(const JKProvider & jk, const helfem::Matrix & C,
                                 Eigen::Index ninact, double & E_inactive) {
      const helfem::Matrix Ci = C.leftCols(ninact);
      const helfem::Matrix P =
          (ninact > 0) ? helfem::Matrix(2.0 * Ci * Ci.transpose())
                       : helfem::Matrix(helfem::Matrix::Zero(C.rows(), C.rows()));
      const helfem::Matrix veff = closed_shell_veff(jk, P);
      E_inactive = (P * jk.hcore()).trace() + 0.5 * (P * veff).trace();
      return jk.hcore() + veff;
    }

    std::vector<double> active_eri(const JKProvider & jk, const helfem::Matrix & Ca) {
      const Eigen::Index n = Ca.cols();
      std::vector<double> eri((size_t) (n * n * n * n), 0.0);
      for (Eigen::Index t = 0; t < n; t++) {
        for (Eigen::Index u = t; u < n; u++) {
          // The pair density C_t C_u^T is passed BARE. It is not symmetric for
          // t != u, and that is the point: coulomb() is exactly linear and does
          // not symmetrise its input, so this returns (vw|tu) = (tu|vw) itself.
          //
          // Symmetrising to P + P^T instead -- which looks harmless, since a
          // physical density IS symmetric -- stores ((tu|vw) + (ut|vw))/2. For
          // REAL orbitals the two agree and nothing is lost, which is why every
          // m = 0 check passed. For COMPLEX orbitals, i.e. any m != 0, they
          // differ: a real antisymmetric density is a purely IMAGINARY density
          // (rho* = -rho), a channel the symmetrised call discards entirely.
          // Measured on an atomic lmax=1 basis, the antisymmetric part of
          // coulomb(C_t C_u^T) is as large as the symmetric part.
          const helfem::Matrix blk =
              Ca.transpose() * jk.coulomb(Ca.col(t) * Ca.col(u).transpose()) * Ca;
          for (Eigen::Index v = 0; v < n; v++)
            for (Eigen::Index w = 0; w < n; w++) {
              eri[idx4((size_t) n, (size_t) t, (size_t) u, (size_t) v, (size_t) w)] =
                  blk(v, w);
              // (ab|cd) = (ba|dc), so the (u,t) block is this one transposed in
              // (v,w) -- no second Fock build, and the cost stays n(n+1)/2.
              eri[idx4((size_t) n, (size_t) u, (size_t) t, (size_t) w, (size_t) v)] =
                  blk(v, w);
            }
        }
      }
      return eri;
    }

    ActiveHamiltonian active_hamiltonian(const JKProvider & jk,
                                         const helfem::Matrix & C,
                                         Eigen::Index ninact, Eigen::Index nact) {
      ActiveHamiltonian out;
      const helfem::Matrix FI = inactive_fock(jk, C, ninact, out.E_inactive);
      const helfem::Matrix Ca = C.middleCols(ninact, nact);
      out.h_eff = Ca.transpose() * FI * Ca;
      out.eri = active_eri(jk, Ca);
      return out;
    }

    helfem::Matrix generalized_fock(const JKProvider & jk, const helfem::Matrix & C,
                                    Eigen::Index ninact, Eigen::Index nact,
                                    const helfem::Matrix & D,
                                    const std::vector<double> & d,
                                    double core_occ) {
      const Eigen::Index nocc = ninact + nact;
      const Eigen::Index norb = C.cols();
      if (D.rows() != nact || D.cols() != nact)
        throw std::logic_error("generalized_fock: D is not (nact x nact).\n");
      if (d.size() != (size_t) (nact * nact * nact * nact))
        throw std::logic_error("generalized_fock: d is not nact^4.\n");

      const helfem::Matrix Ci = C.leftCols(ninact);
      const helfem::Matrix Ca = C.middleCols(ninact, nact);

      const helfem::Matrix PI =
          (ninact > 0) ? helfem::Matrix(2.0 * Ci * Ci.transpose())
                       : helfem::Matrix(helfem::Matrix::Zero(C.rows(), C.rows()));
      const helfem::Matrix FI =
          C.transpose() * (jk.hcore() + closed_shell_veff(jk, PI)) * C;

      const helfem::Matrix PA = Ca * D * Ca.transpose();
      const helfem::Matrix FA = C.transpose() * closed_shell_veff(jk, PA) * C;

      helfem::Matrix F = helfem::Matrix::Zero(norb, nocc);
      if (ninact > 0)
        F.leftCols(ninact) =
            core_occ * FI.leftCols(ninact) + 2.0 * FA.leftCols(ninact);
      // D is symmetric for a real CI vector, so D and D^T are interchangeable.
      F.middleCols(ninact, nact) = FI.middleCols(ninact, nact) * D;

      helfem::Matrix dtu(nact, nact);
      for (Eigen::Index t = 0; t < nact; t++) {
        for (Eigen::Index u = 0; u < nact; u++) {
          for (Eigen::Index v = 0; v < nact; v++)
            for (Eigen::Index w = 0; w < nact; w++)
              dtu(v, w) = d[idx4((size_t) nact, (size_t) t, (size_t) u,
                                 (size_t) v, (size_t) w)];
          const helfem::Matrix Ptu = Ca * dtu * Ca.transpose();
          // Passed BARE. An earlier version symmetrised this, on the grounds
          // that "(mn|vw) is symmetric in (v, w)" -- but that is a REAL-orbital
          // symmetry. For complex orbitals (any m != 0) the symmetrisation
          // silently drops sum_vw (d_tuvw - d_tuwv)(mu|vw)/2. coulomb() takes
          // arbitrary square input and is exactly linear, so nothing here
          // needs a symmetric argument. See active_eri for the same trap.
          const helfem::Matrix J = jk.coulomb(Ptu);
          F.col(ninact + t) += C.transpose() * (J * Ca.col(u));
        }
      }
      return F;
    }

    helfem::Matrix orbital_gradient(const JKProvider & jk, const helfem::Matrix & C,
                                    Eigen::Index ninact, Eigen::Index nact,
                                    const helfem::Matrix & D,
                                    const std::vector<double> & d) {
      const Eigen::Index norb = C.cols();
      helfem::Matrix Ff = helfem::Matrix::Zero(norb, norb);
      Ff.leftCols(ninact + nact) =
          generalized_fock(jk, C, ninact, nact, D, d);
      return 2.0 * (Ff - Ff.transpose());
    }


    helfem::Matrix fock_response_kappa(const JKProvider & jk,
                                       const helfem::Matrix & C,
                                       Eigen::Index ninact, Eigen::Index nact,
                                       const helfem::Matrix & D,
                                       const std::vector<double> & d,
                                       const helfem::Matrix & dkap) {
      const Eigen::Index nocc = ninact + nact;
      const Eigen::Index norb = C.cols();
      const helfem::Matrix Cx = C * dkap;            // dC/dt
      const helfem::Matrix Ci = C.leftCols(ninact);
      const helfem::Matrix Ca = C.middleCols(ninact, nact);
      const helfem::Matrix Cxi = Cx.leftCols(ninact);
      const helfem::Matrix Cxa = Cx.middleCols(ninact, nact);
      const helfem::Matrix zero = helfem::Matrix::Zero(C.rows(), C.rows());

      const helfem::Matrix PI =
          (ninact > 0) ? helfem::Matrix(2.0 * Ci * Ci.transpose()) : zero;
      const helfem::Matrix dPI =
          (ninact > 0)
              ? helfem::Matrix(2.0 * (Cxi * Ci.transpose() + Ci * Cxi.transpose()))
              : zero;
      const helfem::Matrix Vi = jk.hcore() + closed_shell_veff(jk, PI);
      const helfem::Matrix dVi = closed_shell_veff(jk, dPI);
      const helfem::Matrix FI = C.transpose() * Vi * C;
      const helfem::Matrix dFI = Cx.transpose() * Vi * C + C.transpose() * Vi * Cx
                                 + C.transpose() * dVi * C;

      const helfem::Matrix PA = Ca * D * Ca.transpose();
      const helfem::Matrix dPA =
          Cxa * D * Ca.transpose() + Ca * D * Cxa.transpose();
      const helfem::Matrix Va = closed_shell_veff(jk, PA);
      const helfem::Matrix dVa = closed_shell_veff(jk, dPA);
      const helfem::Matrix dFA = Cx.transpose() * Va * C + C.transpose() * Va * Cx
                                 + C.transpose() * dVa * C;

      helfem::Matrix dF = helfem::Matrix::Zero(norb, nocc);
      if (ninact > 0)
        dF.leftCols(ninact) =
            2.0 * (dFI.leftCols(ninact) + dFA.leftCols(ninact));
      dF.middleCols(ninact, nact) = dFI.middleCols(ninact, nact) * D;

      helfem::Matrix dtu(nact, nact);
      for (Eigen::Index t = 0; t < nact; t++) {
        for (Eigen::Index u = 0; u < nact; u++) {
          for (Eigen::Index v = 0; v < nact; v++)
            for (Eigen::Index w = 0; w < nact; w++)
              dtu(v, w) = d[idx4((size_t) nact, (size_t) t, (size_t) u,
                                 (size_t) v, (size_t) w)];
          const helfem::Matrix Ptu = Ca * dtu * Ca.transpose();
          const helfem::Matrix dPtu =
              Cxa * dtu * Ca.transpose() + Ca * dtu * Cxa.transpose();
          // Bare, for the reason given in generalized_fock: the (v, w)
          // symmetry of (mn|vw) holds only for real orbitals.
          const helfem::Matrix J = jk.coulomb(Ptu);
          const helfem::Matrix dJ = jk.coulomb(dPtu);
          dF.col(ninact + t) += Cx.transpose() * (J * Ca.col(u))
                                + C.transpose() * (dJ * Ca.col(u))
                                + C.transpose() * (J * Cxa.col(u));
        }
      }
      return dF;
    }

    helfem::Matrix hess_kappa_kappa_raw(const JKProvider & jk,
                                        const helfem::Matrix & C,
                                        Eigen::Index ninact, Eigen::Index nact,
                                        const helfem::Matrix & D,
                                        const std::vector<double> & d,
                                        const helfem::Matrix & dkap) {
      const Eigen::Index norb = C.cols();
      helfem::Matrix dFf = helfem::Matrix::Zero(norb, norb);
      dFf.leftCols(ninact + nact) =
          fock_response_kappa(jk, C, ninact, nact, D, d, dkap);
      return 2.0 * (dFf - dFf.transpose());
    }

    double hess_kappa_kappa(const JKProvider & jk, const helfem::Matrix & C,
                            Eigen::Index ninact, Eigen::Index nact,
                            const helfem::Matrix & D,
                            const std::vector<double> & d,
                            const helfem::Matrix & K1, const helfem::Matrix & K2) {
      const double a =
          (hess_kappa_kappa_raw(jk, C, ninact, nact, D, d, K2).cwiseProduct(K1)).sum();
      const double b =
          (hess_kappa_kappa_raw(jk, C, ninact, nact, D, d, K1).cwiseProduct(K2)).sum();
      // 1/2 for the symmetrisation, 1/2 because contracting the full
      // antisymmetric matrices counts every rotation pair twice.
      return 0.25 * (a + b);
    }

    double frozen_ci_energy(const JKProvider & jk, const helfem::Matrix & C,
                            Eigen::Index ninact, Eigen::Index nact,
                            const helfem::Matrix & D,
                            const std::vector<double> & d) {
      const ActiveHamiltonian ah = active_hamiltonian(jk, C, ninact, nact);
      double E = ah.E_inactive + (ah.h_eff.cwiseProduct(D)).sum();
      double two = 0.0;
      for (size_t i = 0; i < d.size(); i++) two += ah.eri[i] * d[i];
      return E + 0.5 * two;
    }

  } // namespace cas
} // namespace helfem
