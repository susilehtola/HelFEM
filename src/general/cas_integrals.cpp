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
      /// reci's storage order -- though not reci's index MEANING, which is
      /// physicist; see cas_ci.cpp.
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

    CASPoint::CASPoint(const JKProvider & jk, const helfem::Matrix & C,
                       Eigen::Index ninact, Eigen::Index nact)
        : jk_(jk), C_(C), ninact_(ninact), nact_(nact) {
      const Eigen::Index n = nact, nocc = ninact + nact;
      Vi_ = inactive_fock(jk, C, ninact, ah_.E_inactive);
      FIo_ = C.transpose() * (Vi_ * C.leftCols(nocc));

      // One Coulomb build per unordered active pair. J[P^T] = J[P]^T, since
      // the integrals are real: (mu nu|la si) = (nu mu|si la).
      const helfem::Matrix Ca = C.middleCols(ninact, n);
      L_.resize((size_t) (n * n));
      for (Eigen::Index v = 0; v < n; v++)
        for (Eigen::Index w = v; w < n; w++) {
          // Passed bare: see active_eri.
          const helfem::Matrix J = jk.coulomb(Ca.col(v) * Ca.col(w).transpose());
          L_[(size_t) (v * n + w)] = C.transpose() * (J * Ca);
          if (w != v)
            L_[(size_t) (w * n + v)] = C.transpose() * (J.transpose() * Ca);
        }

      ah_.h_eff = FIo_.block(ninact, ninact, n, n);
      ah_.eri.assign((size_t) (n * n * n * n), 0.0);
      for (Eigen::Index t = 0; t < n; t++)
        for (Eigen::Index u = 0; u < n; u++)
          for (Eigen::Index v = 0; v < n; v++)
            for (Eigen::Index w = 0; w < n; w++)
              ah_.eri[idx4((size_t) n, (size_t) t, (size_t) u, (size_t) v, (size_t) w)] =
                  L_[(size_t) (t * n + u)](ninact + v, w);
    }

    helfem::Matrix CASPoint::pair_block(const std::vector<double> & d, Eigen::Index v,
                                        Eigen::Index w) const {
      // T(u, t) = d_tuvw
      const Eigen::Index n = nact_;
      helfem::Matrix T(n, n);
      for (Eigen::Index t = 0; t < n; t++)
        for (Eigen::Index u = 0; u < n; u++)
          T(u, t) = d[idx4((size_t) n, (size_t) t, (size_t) u, (size_t) v, (size_t) w)];
      return T;
    }

    helfem::Matrix CASPoint::Q(const std::vector<double> & d) const {
      // Q[m][t] = sum_uvw d_tuvw (mu|vw)
      const Eigen::Index n = nact_;
      helfem::Matrix q = helfem::Matrix::Zero(C_.cols(), n);
      for (Eigen::Index v = 0; v < n; v++)
        for (Eigen::Index w = 0; w < n; w++)
          q += L_[(size_t) (v * n + w)] * pair_block(d, v, w);
      return q;
    }

    const helfem::Matrix & CASPoint::active_veff(const helfem::Matrix & D) const {
      // The state's is asked for at every Hessian-vector product; a
      // transition density's only once.
      if (Va_D_.size() == 0 || Va_D_ != D) {
        const helfem::Matrix Ca = C_.middleCols(ninact_, nact_);
        Va_ = closed_shell_veff(jk_, Ca * D * Ca.transpose());
        Va_D_ = D;
      }
      return Va_;
    }

    void CASPoint::check_rdms(const helfem::Matrix & D, const std::vector<double> & d) const {
      if (D.rows() != nact_ || D.cols() != nact_)
        throw std::logic_error("CASPoint: D is not (nact x nact).\n");
      if (d.size() != (size_t) (nact_ * nact_ * nact_ * nact_))
        throw std::logic_error("CASPoint: d is not nact^4.\n");
    }

    helfem::Matrix CASPoint::generalized_fock(const helfem::Matrix & D,
                                              const std::vector<double> & d,
                                              double core_occ) const {
      check_rdms(D, d);
      const Eigen::Index nocc = ninact_ + nact_;
      helfem::Matrix F(C_.cols(), nocc);
      if (ninact_ > 0)
        F.leftCols(ninact_) =
            core_occ * FIo_.leftCols(ninact_)
            + 2.0 * C_.transpose() * (active_veff(D) * C_.leftCols(ninact_));
      // D is symmetric for a real CI vector, so D and D^T are interchangeable.
      F.middleCols(ninact_, nact_) = FIo_.middleCols(ninact_, nact_) * D + Q(d);
      return F;
    }

    helfem::Matrix CASPoint::fock_response(const helfem::Matrix & D,
                                           const std::vector<double> & d,
                                           const helfem::Matrix & dkap,
                                           ActiveHamiltonian * dH) const {
      check_rdms(D, d);
      const Eigen::Index n = nact_, ni = ninact_, nocc = ni + n, norb = C_.cols();
      const helfem::Matrix Co = C_.leftCols(nocc);
      const helfem::Matrix Ci = C_.leftCols(ni), Ca = C_.middleCols(ni, n);
      // dC/dt for the occupied columns; the virtual ones enter only through
      // dkap^T below.
      const helfem::Matrix Cxo = C_ * dkap.leftCols(nocc);
      const helfem::Matrix Cxi = Cxo.leftCols(ni), Cxa = Cxo.middleCols(ni, n);

      helfem::Matrix dPI = 2.0 * Cxi * Ci.transpose();
      dPI += dPI.transpose().eval();
      const helfem::Matrix dVi =
          (ni > 0) ? closed_shell_veff(jk_, dPI) : helfem::Matrix(helfem::Matrix::Zero(norb, norb));
      helfem::Matrix dPA = Cxa * D * Ca.transpose();
      dPA += dPA.transpose().eval();
      const helfem::Matrix dVa = closed_shell_veff(jk_, dPA);
      const helfem::Matrix & Va = active_veff(D);

      // d(C^T V C)/dt = dkap^T (C^T V C) + C^T V (C dkap) + C^T dV C, over
      // the occupied columns.
      const helfem::Matrix dFI =
          dkap.transpose() * FIo_ + C_.transpose() * (Vi_ * Cxo + dVi * Co);
      helfem::Matrix dF(norb, nocc);
      if (ni > 0) {
        const helfem::Matrix FAi = C_.transpose() * (Va * Ci);
        const helfem::Matrix dFAi =
            dkap.transpose() * FAi + C_.transpose() * (Va * Cxi + dVa * Ci);
        dF.leftCols(ni) = 2.0 * (dFI.leftCols(ni) + dFAi);
      }

      // Q[m][t] = sum_uvw d_tuvw (mu|vw) moves with every index. The bra m
      // gives dkap^T Q; u gives the pair Coulomb matrices applied to dC_u,
      // and v, w the Coulomb matrices of the differentiated pair densities.
      // Both are rebuilt here, once per unordered pair, rather than stored:
      // over all orbitals they would not fit.
      helfem::Matrix acc = helfem::Matrix::Zero(norb, n);
      if (dH) {
        dH->eri.assign((size_t) (n * n * n * n), 0.0);
        // E_inactive = tr(P h) + tr(P veff[P])/2, veff linear and symmetric in
        // its argument: dE = tr(dP (h + veff[P])).
        dH->E_inactive = dPI.cwiseProduct(Vi_.transpose()).sum();
        dH->h_eff = Cxa.transpose() * (Vi_ * Ca) + Ca.transpose() * (Vi_ * Cxa)
                    + Ca.transpose() * (dVi * Ca);
      }
      for (Eigen::Index v = 0; v < n; v++)
        for (Eigen::Index w = v; w < n; w++) {
          const helfem::Matrix J = jk_.coulomb(Ca.col(v) * Ca.col(w).transpose());
          const helfem::Matrix Jx = jk_.coulomb(Cxa.col(v) * Ca.col(w).transpose()
                                                + Ca.col(v) * Cxa.col(w).transpose());
          const helfem::Matrix Tvw = pair_block(d, v, w);
          acc += J * (Cxa * Tvw) + Jx * (Ca * Tvw);
          if (w != v) {
            const helfem::Matrix Twv = pair_block(d, w, v);
            acc += J.transpose() * (Cxa * Twv) + Jx.transpose() * (Ca * Twv);
          }
          if (dH) {
            const helfem::Matrix JCa = J * Ca;
            const helfem::Matrix blk = Ca.transpose() * (Jx * Ca) + Cxa.transpose() * JCa
                                       + Ca.transpose() * (J * Cxa);
            for (Eigen::Index a = 0; a < n; a++)
              for (Eigen::Index b = 0; b < n; b++) {
                dH->eri[idx4((size_t) n, (size_t) v, (size_t) w, (size_t) a, (size_t) b)] = blk(a, b);
                dH->eri[idx4((size_t) n, (size_t) w, (size_t) v, (size_t) b, (size_t) a)] = blk(a, b);
              }
          }
        }
      dF.middleCols(ni, n) =
          dFI.middleCols(ni, n) * D + dkap.transpose() * Q(d) + C_.transpose() * acc;
      return dF;
    }

    helfem::Vector CASPoint::fock_diagonal(const helfem::Matrix & D) const {
      return C_.cwiseProduct((Vi_ + active_veff(D)) * C_).colwise().sum().transpose();
    }

    helfem::Matrix generalized_fock(const JKProvider & jk, const helfem::Matrix & C,
                                    Eigen::Index ninact, Eigen::Index nact,
                                    const helfem::Matrix & D,
                                    const std::vector<double> & d,
                                    double core_occ) {
      return CASPoint(jk, C, ninact, nact).generalized_fock(D, d, core_occ);
    }

    ActiveHamiltonian active_hamiltonian_response(const JKProvider & jk,
                                                  const helfem::Matrix & C,
                                                  Eigen::Index ninact,
                                                  Eigen::Index nact,
                                                  const helfem::Matrix & dkap) {
      // The Hamiltonian's response does not involve the RDMs; any will do.
      ActiveHamiltonian out;
      const helfem::Matrix D = helfem::Matrix::Zero(nact, nact);
      const std::vector<double> d((size_t) (nact * nact * nact * nact), 0.0);
      CASPoint(jk, C, ninact, nact).fock_response(D, d, dkap, &out);
      return out;
    }

    helfem::Matrix antisymmetrize_occupied(const helfem::Matrix & F) {
      const Eigen::Index nocc = F.cols();
      helfem::Matrix A = F;
      A.topRows(nocc) -= F.topRows(nocc).transpose();
      return A;
    }

    helfem::Matrix square_antisymmetric(const helfem::Matrix & F) {
      const Eigen::Index norb = F.rows(), nocc = F.cols();
      helfem::Matrix A = helfem::Matrix::Zero(norb, norb);
      A.leftCols(nocc) = F;
      return A - A.transpose();
    }

    helfem::Matrix orbital_gradient(const JKProvider & jk, const helfem::Matrix & C,
                                    Eigen::Index ninact, Eigen::Index nact,
                                    const helfem::Matrix & D,
                                    const std::vector<double> & d) {
      return 2.0 * square_antisymmetric(generalized_fock(jk, C, ninact, nact, D, d));
    }

    helfem::Matrix fock_response_kappa(const JKProvider & jk,
                                       const helfem::Matrix & C,
                                       Eigen::Index ninact, Eigen::Index nact,
                                       const helfem::Matrix & D,
                                       const std::vector<double> & d,
                                       const helfem::Matrix & dkap) {
      return CASPoint(jk, C, ninact, nact).fock_response(D, d, dkap);
    }

    helfem::Matrix hess_kappa_kappa_raw(const JKProvider & jk,
                                        const helfem::Matrix & C,
                                        Eigen::Index ninact, Eigen::Index nact,
                                        const helfem::Matrix & D,
                                        const std::vector<double> & d,
                                        const helfem::Matrix & dkap) {
      return 2.0 * square_antisymmetric(fock_response_kappa(jk, C, ninact, nact, D, d, dkap));
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
