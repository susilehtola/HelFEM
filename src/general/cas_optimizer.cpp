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
#include "cas_optimizer.h"
#include "trustregion_scf.h"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace helfem {
  namespace cas {

    std::vector<RotationPair> rotation_pairs(Eigen::Index norb, Eigen::Index ninact,
                                             Eigen::Index nact,
                                             const std::vector<int> & label) {
      if (!label.empty() && (Eigen::Index) label.size() != norb)
        throw std::invalid_argument("rotation_pairs: one label per orbital");
      const Eigen::Index nocc = ninact + nact;
      std::vector<RotationPair> pairs;
      for (Eigen::Index q = 0; q < nocc; q++)
        for (Eigen::Index p = std::max(q + 1, ninact); p < norb; p++) {
          if (q >= ninact && p < nocc)
            continue;  // active-active
          if (!label.empty() && label[(size_t) p] != label[(size_t) q])
            continue;
          pairs.push_back({p, q});
        }
      return pairs;
    }

    namespace {
      double dot(const std::vector<double> & a, const std::vector<double> & b) {
        double s = 0.0;
        for (size_t i = 0; i < a.size(); i++) s += a[i] * b[i];
        return s;
      }

      /// v minus its component along the unit vector u.
      std::vector<double> project_out(std::vector<double> v, const std::vector<double> & u) {
        const double a = dot(u, v);
        for (size_t i = 0; i < v.size(); i++) v[i] -= a * u[i];
        return v;
      }
    } // namespace

    CASSCF::CASSCF(const JKProvider & jk, const helfem::Matrix & S, CASCI & ci,
                   helfem::Matrix C, Eigen::Index ninact, Eigen::Index nact,
                   std::vector<int> label, Coupling coupling)
        : jk_(jk), S_(S), ci_(ci), ninact_(ninact), nact_(nact),
          pairs_(rotation_pairs(C.cols(), ninact, nact, label)), coupling_(coupling) {
      // Both parametrizations start from the CASCI ground state at C.
      std::vector<double> c;
      ci_.set_hamiltonian(active_hamiltonian(jk_, C, ninact_, nact_));
      ci_.solve_ground(c);
      adopt(C, ci_.to_csf(c));
    }

    helfem::Matrix CASSCF::kappa(const double * x) const {
      helfem::Matrix K = helfem::Matrix::Zero(C_.cols(), C_.cols());
      for (size_t k = 0; k < pairs_.size(); k++) {
        K(pairs_[k].p, pairs_[k].q) = x[k];
        K(pairs_[k].q, pairs_[k].p) = -x[k];
      }
      return K;
    }

    helfem::Matrix CASSCF::rotated(const double * x) const {
      return C_ * trscf::expm_skew(kappa(x));
    }

    std::vector<double> CASSCF::ci_at(const double * x) const {
      if (coupling_ == Coupling::TwoStep)
        return ccsf_;
      const std::vector<double> t =
          project_out(std::vector<double>(x + nrot(), x + nrot() + nci()), ccsf_);
      const double th = std::sqrt(dot(t, t));
      std::vector<double> c(ccsf_);
      if (th > 0.0)
        for (size_t i = 0; i < c.size(); i++)
          c[i] = std::cos(th) * ccsf_[i] + std::sin(th) / th * t[i];
      return c;
    }

    double CASSCF::expectation(const ActiveHamiltonian & ah, const std::vector<double> & ccsf) {
      ci_.set_hamiltonian(ah);
      const std::vector<double> c = ci_.to_det(ccsf);
      return dot(c, ci_.sigma(c));
    }

    void CASSCF::adopt(const helfem::Matrix & C, const std::vector<double> & ccsf) {
      // Loewdin, so that roundoff in many rotations cannot accumulate into
      // non-orthonormal orbitals; likewise the CI vector is renormalized.
      Eigen::SelfAdjointEigenSolver<helfem::Matrix> es(C.transpose() * S_ * C);
      C_ = C * (es.eigenvectors() * es.eigenvalues().array().rsqrt().matrix().asDiagonal()
                * es.eigenvectors().transpose());
      ah_ = active_hamiltonian(jk_, C_, ninact_, nact_);
      ci_.set_hamiltonian(ah_);
      if (coupling_ == Coupling::TwoStep) {
        E_ = ci_.solve_ground(c_);
        ccsf_ = ci_.to_csf(c_);
      } else {
        ccsf_ = ccsf;
        const double nrm = std::sqrt(dot(ccsf_, ccsf_));
        for (double & v : ccsf_) v /= nrm;
        c_ = ci_.to_det(ccsf_);
      }
      const std::vector<double> s = ci_.sigma(c_);
      E_ = dot(c_, s);
      sigma_ = ci_.to_csf(s);
      rdm_ = ci_.rdms(c_, c_);
      g_ = orbital_gradient(jk_, C_, ninact_, nact_, rdm_.D, rdm_.d);
    }

    double CASSCF::energy(const std::vector<double> & x) {
      const helfem::Matrix C = rotated(x.data());
      const ActiveHamiltonian ah = active_hamiltonian(jk_, C, ninact_, nact_);
      if (coupling_ == Coupling::TwoStep) {
        std::vector<double> c;
        ci_.set_hamiltonian(ah);
        return ci_.solve_ground(c);
      }
      return expectation(ah, ci_at(x.data()));
    }

    std::vector<double> CASSCF::gradient() const {
      std::vector<double> g(nparam());
      for (size_t k = 0; k < pairs_.size(); k++)
        g[k] = g_(pairs_[k].p, pairs_[k].q);
      // CI part: 2 P H|0> = 2 (sigma - E c), zero along |0> itself.
      for (size_t i = 0; i < nci(); i++)
        g[nrot() + i] = 2.0 * (sigma_[i] - E_ * ccsf_[i]);
      return g;
    }

    std::vector<double> CASSCF::hessian_vector(const std::vector<double> & x) {
      // hess_kappa_kappa_raw differentiates the gradient along a product of
      // exponentials, which differs from the Hessian by the commutator term
      // (1/2) g . [K, .]; taking it off leaves the symmetric Hessian.
      const helfem::Matrix K = kappa(x.data());
      helfem::Matrix M =
          hess_kappa_kappa_raw(jk_, C_, ninact_, nact_, rdm_.D, rdm_.d, K)
          - 0.5 * (g_ * K - K * g_);
      std::vector<double> hx(nparam(), 0.0);

      if (coupling_ == Coupling::Coupled) {
        const std::vector<double> t =
            project_out(std::vector<double>(x.begin() + nrot(), x.end()), ccsf_);
        const std::vector<double> tdet = ci_.to_det(t);

        // kappa-c: the orbital gradient is bilinear in the CI vector, so its
        // derivative along t is the gradient at the symmetrised transition
        // RDMs. Their inactive block vanishes because <t|0> = 0, hence
        // core_occ = 0.
        const RDMs a = ci_.rdms(tdet, c_), b = ci_.rdms(c_, tdet);
        std::vector<double> dt(a.d.size());
        for (size_t i = 0; i < dt.size(); i++) dt[i] = a.d[i] + b.d[i];
        helfem::Matrix Ft = helfem::Matrix::Zero(C_.cols(), C_.cols());
        Ft.leftCols(ninact_ + nact_) =
            generalized_fock(jk_, C_, ninact_, nact_, a.D + b.D, dt, 0.0);
        M += 2.0 * (Ft - Ft.transpose());

        // c-c: 2 P (H - E) t, with the reference Hamiltonian installed.
        ci_.set_hamiltonian(ah_);
        std::vector<double> hc = ci_.to_csf(ci_.sigma(tdet));
        for (size_t i = 0; i < hc.size(); i++) hc[i] = 2.0 * (hc[i] - E_ * t[i]);

        // c-kappa: 2 P H' |0>, H' the one-index-transformed Hamiltonian.
        ci_.set_hamiltonian(active_hamiltonian_response(jk_, C_, ninact_, nact_, K));
        const std::vector<double> sk = ci_.to_csf(ci_.sigma(c_));
        ci_.set_hamiltonian(ah_);
        for (size_t i = 0; i < hc.size(); i++) hc[i] += 2.0 * sk[i];

        hc = project_out(hc, ccsf_);
        std::copy(hc.begin(), hc.end(), hx.begin() + nrot());
      }
      for (size_t k = 0; k < pairs_.size(); k++)
        hx[k] = M(pairs_[k].p, pairs_[k].q);
      return hx;
    }

    std::vector<double> CASSCF::hessian_diagonal() const {
      // The usual Fock-matrix approximation to the diagonal, with F = F^I +
      // F^A; only the preconditioner sees it.
      double Ei;
      const helfem::Matrix Ca = C_.middleCols(ninact_, nact_);
      const helfem::Matrix F =
          C_.transpose()
          * (inactive_fock(jk_, C_, ninact_, Ei)
             + closed_shell_veff(jk_, Ca * rdm_.D * Ca.transpose()))
          * C_;
      const helfem::Matrix Fg = generalized_fock(jk_, C_, ninact_, nact_, rdm_.D, rdm_.d);

      std::vector<double> h(pairs_.size());
      for (size_t k = 0; k < pairs_.size(); k++) {
        const Eigen::Index p = pairs_[k].p, q = pairs_[k].q;
        const bool q_inactive = q < ninact_;
        const bool p_active = p < ninact_ + nact_;
        double v;
        if (q_inactive && !p_active) {
          v = 4.0 * (F(p, p) - F(q, q));
        } else if (q_inactive) {
          const double Dtt = rdm_.D(p - ninact_, p - ninact_);
          v = 4.0 * (F(p, p) - F(q, q)) + 2.0 * Dtt * F(q, q) - 2.0 * Fg(p, p);
        } else {
          const double Dtt = rdm_.D(q - ninact_, q - ninact_);
          v = 2.0 * Dtt * F(p, p) - 2.0 * Fg(q, q);
        }
        // Rotations into an empty or full active orbital are (nearly)
        // redundant and have a vanishing diagonal; keep the sign, bound
        // the magnitude away from zero.
        h[k] = std::copysign(std::max(std::abs(v), 1e-4), v);
      }
      if (nci()) {
        const std::vector<double> hd = ci_.hdiag_csf();
        for (size_t i = 0; i < nci(); i++)
          h.push_back(std::max(2.0 * (hd[i] - E_), 1e-4));
      }
      return h;
    }

    CASSCFResult CASSCF::run(const CASSCFOptions & opt) {
      CASSCFResult res;
      const int n = (int) nparam();
      if (opt.verbosity >= 1) {
        printf("%s CASSCF over %zu orbital rotations and %zu CI coefficients: "
               "%i inactive, %i active orbitals, %zu determinants, %zu CSFs\n",
               coupling_ == Coupling::Coupled ? "Coupled" : "Two-step",
               nrot(), nci(), (int) ninact_, (int) nact_, ci_.ndet(), ci_.ncsf());
        fflush(stdout);
      }

      otr::Callbacks cb;
      cb.update = [&](const double * x, double & f, double * grad, double * hdiag) {
        adopt(rotated(x), ci_at(x));
        res.n_update++;
        f = E_;
        const std::vector<double> g = gradient(), h = hessian_diagonal();
        std::copy(g.begin(), g.end(), grad);
        std::copy(h.begin(), h.end(), hdiag);
        return true;
      };
      cb.objective = [&](const double * x, double & f) {
        f = energy(std::vector<double>(x, x + n));
        res.n_objective++;
        return true;
      };
      cb.hessian_vector = [&](const double * x, double * hx) {
        const std::vector<double> v = hessian_vector(std::vector<double>(x, x + n));
        std::copy(v.begin(), v.end(), hx);
        res.n_hessian++;
        return true;
      };

      otr::Settings s = opt.otr;
      s.verbose = (opt.verbosity >= 1) ? std::max(3, opt.verbosity) : 0;
      otr::solve(n, cb, s);

      const std::vector<double> g = gradient();
      double g2 = 0.0;
      for (double v : g) g2 += v * v;
      res.energy = E_;
      res.grad_rms = n ? std::sqrt(g2 / n) : 0.0;
      if (opt.verbosity >= 1) {
        printf("CASSCF energy %.10f, RMS gradient %.3e: %zu updates, "
               "%zu objective evaluations, %zu Hessian-vector products\n",
               res.energy, res.grad_rms, res.n_update, res.n_objective, res.n_hessian);
        fflush(stdout);
      }
      return res;
    }

  } // namespace cas
} // namespace helfem
