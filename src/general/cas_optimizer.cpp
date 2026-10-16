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

    std::vector<Rotation> rotation_pairs(Eigen::Index norb, Eigen::Index ninact,
                                         Eigen::Index nact,
                                         const std::vector<int> & label,
                                         const std::vector<int> & shell) {
      if (!label.empty() && (Eigen::Index) label.size() != norb)
        throw std::invalid_argument("rotation_pairs: one label per orbital");
      if (!shell.empty() && (Eigen::Index) shell.size() != norb)
        throw std::invalid_argument("rotation_pairs: one shell per orbital");
      const Eigen::Index nocc = ninact + nact;
      auto lab = [&](Eigen::Index i) { return label.empty() ? 0 : label[(size_t) i]; };
      auto shl = [&](Eigen::Index i) { return shell.empty() ? (int) i : shell[(size_t) i]; };
      // 0 inactive, 1 active, 2 virtual
      auto cls = [&](Eigen::Index i) { return i < ninact ? 0 : (i < nocc ? 1 : 2); };

      // A shell's copies must all sit in the same class, or the tie would
      // rotate an active orbital together with an inactive one.
      for (Eigen::Index i = 0; i < norb; i++)
        for (Eigen::Index j = i + 1; j < norb; j++)
          if (shl(i) == shl(j) && cls(i) != cls(j))
            throw std::logic_error("CASSCF: the inactive/active/virtual partition "
                                   "splits a set of equivalent orbitals; choose the "
                                   "orbital counts so that whole shells fall on "
                                   "each side.\n");

      // The representative of a shell is its first copy, and the
      // representative block of a pair is the block of q's representative.
      auto first = [&](Eigen::Index i) {
        for (Eigen::Index j = 0; j < i; j++)
          if (shl(j) == shl(i)) return false;
        return true;
      };
      std::vector<Rotation> rots;
      for (Eigen::Index q = 0; q < nocc; q++) {
        if (!first(q))
          continue;
        for (Eigen::Index p = std::max(q + 1, ninact); p < norb; p++) {
          if (q >= ninact && p < nocc)
            continue;  // active-active
          if (lab(p) != lab(q))
            continue;
          Rotation r{{p, q}};
          // Images: in every other block holding a copy of q, the copy of p.
          for (Eigen::Index q2 = 0; q2 < norb; q2++) {
            if (q2 == q || shl(q2) != shl(q))
              continue;
            Eigen::Index p2 = -1;
            for (Eigen::Index i = 0; i < norb; i++)
              if (shl(i) == shl(p) && lab(i) == lab(q2)) p2 = i;
            if (p2 < 0)
              throw std::logic_error("rotation_pairs: equivalent blocks hold "
                                     "different shells");
            // Copies are contiguous and shells sorted, so the image keeps
            // the orientation; a flip would silently reverse its rotation.
            if (p2 < q2)
              throw std::logic_error("rotation_pairs: image pair out of order");
            r.push_back({p2, q2});
          }
          rots.push_back(r);
        }
      }
      return rots;
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

    void canonical_orbitals(const JKProvider & jk, const helfem::Matrix & S,
                            const helfem::Matrix & P,
                            const std::vector<std::vector<Eigen::Index>> & dsym,
                            const std::vector<helfem::Matrix> & Sinvh,
                            helfem::Matrix & C, helfem::Vector & energy,
                            std::vector<int> & label, std::vector<int> & shell) {
      const helfem::Matrix F = jk.hcore() + closed_shell_veff(jk, P);
      const helfem::Matrix SPS = S * P * S;
      const size_t nb = dsym.size();

      // Equivalent blocks: same size, identical overlap and core
      // Hamiltonian. That is the +m/-m pair of a diatomic and the 2l+1
      // blocks of one l in an atom, found without either geometry having to
      // say so.
      auto same = [&](const helfem::Matrix & A, const helfem::Matrix & B) {
        return (A - B).cwiseAbs().maxCoeff() <= 1e-12 * std::max(1.0, A.cwiseAbs().maxCoeff());
      };
      std::vector<int> group(nb, -1);
      std::vector<std::vector<size_t>> groups;
      for (size_t a = 0; a < nb; a++) {
        if (group[a] >= 0)
          continue;
        group[a] = (int) groups.size();
        groups.push_back({a});
        const helfem::Matrix Sa = S(dsym[a], dsym[a]), ha = jk.hcore()(dsym[a], dsym[a]);
        for (size_t b = a + 1; b < nb; b++)
          if (group[b] < 0 && dsym[b].size() == dsym[a].size()
              && same(Sa, S(dsym[b], dsym[b])) && same(ha, jk.hcore()(dsym[b], dsym[b]))) {
            group[b] = group[a];
            groups.back().push_back(b);
          }
      }

      struct Shell {
        double e;
        std::vector<std::pair<int, helfem::Vector>> copies;  // (block, AO vector)
      };
      std::vector<Shell> shells;
      for (const auto & g : groups) {
        // Averaging over the group symmetrises an operator built from a
        // density that was not itself symmetric, e.g. an open-shell SCF.
        helfem::Matrix Fg = helfem::Matrix::Zero(Sinvh[g[0]].cols(), Sinvh[g[0]].cols());
        for (size_t k : g)
          Fg += Sinvh[k].transpose() * F(dsym[k], dsym[k]) * Sinvh[k];
        Fg /= (double) g.size();
        // P's own occupied space, from its natural orbitals: the operator
        // above is not the one P is self-consistent for unless P is RHF, and
        // its eigenvectors would then mix P's occupied and empty orbitals --
        // after DFT, losing the Kohn-Sham occupied space the start was
        // meant to keep. So F is canonicalized within each part separately;
        // for a converged RHF that changes nothing, F and P commuting.
        helfem::Matrix Ng = helfem::Matrix::Zero(Fg.rows(), Fg.cols());
        for (size_t k : g)
          Ng += Sinvh[k].transpose() * SPS(dsym[k], dsym[k]) * Sinvh[k];
        Ng /= (double) g.size();
        Eigen::SelfAdjointEigenSolver<helfem::Matrix> ns(Ng);
        std::vector<Eigen::Index> occ_cols, virt_cols;
        for (Eigen::Index i = 0; i < ns.eigenvalues().size(); i++)
          (ns.eigenvalues()(i) > 1e-8 ? occ_cols : virt_cols).push_back(i);
        for (const auto * part : {&occ_cols, &virt_cols}) {
          if (part->empty())
            continue;
          const helfem::Matrix W = ns.eigenvectors()(Eigen::all, *part);
          Eigen::SelfAdjointEigenSolver<helfem::Matrix> es(W.transpose() * Fg * W);
          for (Eigen::Index i = 0; i < es.eigenvalues().size(); i++) {
            Shell sh{es.eigenvalues()(i), {}};
            for (size_t k : g) {
              const helfem::Vector ck = Sinvh[k] * (W * es.eigenvectors().col(i));
              helfem::Vector c = helfem::Vector::Zero(F.rows());
              for (size_t r = 0; r < dsym[k].size(); r++)
                c(dsym[k][r]) = ck((Eigen::Index) r);
              sh.copies.push_back({(int) k, c});
            }
            shells.push_back(sh);
          }
        }
      }
      // Stable, so equal energies keep their block order.
      std::stable_sort(shells.begin(), shells.end(),
                       [](const Shell & a, const Shell & b) { return a.e < b.e; });

      size_t norb = 0;
      for (const auto & sh : shells) norb += sh.copies.size();
      C.resize(F.rows(), (Eigen::Index) norb);
      energy.resize((Eigen::Index) norb);
      label.resize(norb);
      shell.resize(norb);
      size_t i = 0;
      for (size_t s = 0; s < shells.size(); s++)
        for (const auto & cp : shells[s].copies) {
          C.col((Eigen::Index) i) = cp.second;
          energy((Eigen::Index) i) = shells[s].e;
          label[i] = cp.first;
          shell[i] = (int) s;
          i++;
        }
    }

    void correlating_virtuals(const JKProvider & jk, const helfem::Matrix & S,
                              const helfem::Matrix & P, Eigen::Index ninact,
                              Eigen::Index nact, helfem::Matrix & C,
                              helfem::Vector & energy, std::vector<int> & label,
                              std::vector<int> & shell, std::vector<double> & weight) {
      // The copies of one shell are contiguous, in the same block order for
      // every shell of a group (canonical_orbitals builds them that way).
      struct Shell {
        double e, w = 0.0;
        bool occupied = false;
        std::vector<Eigen::Index> cols;
      };
      std::vector<Shell> sh;
      for (Eigen::Index i = 0; i < C.cols(); i++) {
        if (i == 0 || shell[(size_t) i] != shell[(size_t) i - 1])
          sh.push_back(Shell{energy(i), 0.0, false, {}});
        sh.back().cols.push_back(i);
      }
      const helfem::Matrix SPS = S * P * S;
      std::vector<double> occ((size_t) C.cols());
      for (Eigen::Index i = 0; i < C.cols(); i++) {
        occ[(size_t) i] = C.col(i).dot(SPS * C.col(i));
        // A shell is occupied if any copy is, so its copies stay together.
        if (occ[(size_t) i] > 1e-6)
          sh[(size_t) shell[(size_t) i]].occupied = true;
      }

      // Virtual shells per group of equivalent blocks.
      std::vector<std::vector<int>> group_blocks;
      std::vector<std::vector<size_t>> group_virt;
      for (size_t s = 0; s < sh.size(); s++) {
        if (sh[s].occupied)
          continue;
        std::vector<int> blocks;
        for (Eigen::Index c : sh[s].cols) blocks.push_back(label[(size_t) c]);
        size_t g = 0;
        while (g < group_blocks.size() && group_blocks[g] != blocks) g++;
        if (g == group_blocks.size()) {
          group_blocks.push_back(blocks);
          group_virt.push_back({});
        }
        group_virt[g].push_back(s);
      }

      // The virtuals of each copy, restricted to the rows of their block:
      // K is dense, and only that diagonal block of it is ever needed.
      struct Copy {
        std::vector<Eigen::Index> rows;
        helfem::Matrix V;
      };
      std::vector<std::vector<Copy>> copies(group_virt.size());
      for (size_t g = 0; g < group_virt.size(); g++) {
        const auto & v = group_virt[g];
        for (size_t j = 0; j < group_blocks[g].size(); j++) {
          Copy cp;
          for (Eigen::Index r = 0; r < C.rows(); r++)
            for (size_t a = 0; a < v.size(); a++)
              if (C(r, sh[v[a]].cols[j]) != 0.0) {
                cp.rows.push_back(r);
                break;
              }
          cp.V.resize((Eigen::Index) cp.rows.size(), (Eigen::Index) v.size());
          for (size_t a = 0; a < v.size(); a++)
            cp.V.col((Eigen::Index) a) = C(cp.rows, sh[v[a]].cols[j]);
          copies[g].push_back(std::move(cp));
        }
      }

      // A, averaged over the copies, one exchange build per active occupied
      // orbital. The inactive ones are the ninact lowest occupied; they are
      // core here, and what is wanted is what correlates the active ones.
      std::vector<helfem::Matrix> A(group_virt.size());
      for (size_t g = 0; g < A.size(); g++)
        A[g] = helfem::Matrix::Zero((Eigen::Index) group_virt[g].size(),
                                    (Eigen::Index) group_virt[g].size());
      Eigen::Index nocc_seen = 0;
      for (Eigen::Index i = 0; i < C.cols(); i++) {
        if (occ[(size_t) i] <= 1e-6 || nocc_seen++ < ninact)
          continue;
        // exchange() returns the signed Fock contribution, -K.
        const helfem::Matrix K = -jk.exchange(C.col(i) * C.col(i).transpose());
        for (size_t g = 0; g < A.size(); g++) {
          const auto & v = group_virt[g];
          helfem::Matrix Kv = helfem::Matrix::Zero(A[g].rows(), A[g].cols());
          for (const Copy & cp : copies[g])
            Kv += cp.V.transpose() * K(cp.rows, cp.rows) * cp.V;
          const double f = occ[(size_t) i] / (double) copies[g].size();
          for (size_t a = 0; a < v.size(); a++)
            for (size_t b = 0; b < v.size(); b++)
              A[g]((Eigen::Index) a, (Eigen::Index) b) +=
                  f * Kv((Eigen::Index) a, (Eigen::Index) b)
                  / (2.0 * energy(i) - sh[v[a]].e - sh[v[b]].e);
        }
      }
      // Kv is symmetric only up to rounding.
      for (auto & a : A) a = 0.5 * (a + a.transpose()).eval();

      const helfem::Matrix F = jk.hcore() + closed_shell_veff(jk, P);
      helfem::Matrix Cnew = C;
      for (size_t g = 0; g < A.size(); g++) {
        const auto & v = group_virt[g];
        Eigen::SelfAdjointEigenSolver<helfem::Matrix> es(A[g]);
        // The new vectors go into the same columns, so the copies keep their
        // blocks; which shell they belong to is redone by the sort below.
        for (size_t k = 0; k < v.size(); k++) {
          double e = 0.0;
          for (size_t j = 0; j < group_blocks[g].size(); j++) {
            helfem::Vector c = helfem::Vector::Zero(C.rows());
            for (size_t a = 0; a < v.size(); a++)
              c += es.eigenvectors()((Eigen::Index) a, (Eigen::Index) k) * C.col(sh[v[a]].cols[j]);
            Cnew.col(sh[v[k]].cols[j]) = c;
            e += c.dot(F * c);
          }
          sh[v[k]].e = e / (double) group_blocks[g].size();
          sh[v[k]].w = es.eigenvalues()((Eigen::Index) k);
        }
      }

      // The active space takes the most strongly correlating virtual shells,
      // whole shells, until it is full.
      Eigen::Index nvirt_active = ninact + nact;
      for (const Shell & x : sh)
        if (x.occupied)
          nvirt_active -= (Eigen::Index) x.cols.size();
      std::vector<size_t> vorder;
      for (size_t s = 0; s < sh.size(); s++)
        if (!sh[s].occupied)
          vorder.push_back(s);
      std::stable_sort(vorder.begin(), vorder.end(),
                       [&](size_t a, size_t b) { return sh[a].w < sh[b].w; });
      std::vector<bool> chosen(sh.size(), false);
      Eigen::Index taken = 0;
      for (size_t s : vorder) {
        if (taken >= nvirt_active)
          break;
        chosen[s] = true;
        taken += (Eigen::Index) sh[s].cols.size();
      }

      // The virtuals left outside are canonicalized again, within the span
      // the chosen ones leave in each group: the CASSCF preconditioner takes
      // its orbital Hessian diagonal from the Fock matrix, which assumes
      // that it is diagonal there. Correlating functions are not, and a
      // first macroiteration of O2 CAS(2,2) took 34 Hessian-vector products
      // over them.
      for (size_t g = 0; g < group_virt.size(); g++) {
        std::vector<size_t> rest;
        for (size_t s : group_virt[g])
          if (!chosen[s])
            rest.push_back(s);
        if (rest.empty())
          continue;
        const Eigen::Index nr = (Eigen::Index) rest.size();
        const size_t ncopy = group_blocks[g].size();
        std::vector<helfem::Matrix> Vr(ncopy);
        helfem::Matrix Fr = helfem::Matrix::Zero(nr, nr);
        for (size_t j = 0; j < ncopy; j++) {
          const auto & rows = copies[g][j].rows;
          Vr[j].resize((Eigen::Index) rows.size(), nr);
          for (Eigen::Index a = 0; a < nr; a++)
            Vr[j].col(a) = Cnew(rows, sh[rest[(size_t) a]].cols[j]);
          Fr += Vr[j].transpose() * F(rows, rows) * Vr[j];
        }
        Fr /= (double) ncopy;
        Eigen::SelfAdjointEigenSolver<helfem::Matrix> es(0.5 * (Fr + Fr.transpose()));
        for (size_t j = 0; j < ncopy; j++) {
          const helfem::Matrix U = Vr[j] * es.eigenvectors();
          for (Eigen::Index a = 0; a < nr; a++) {
            const Eigen::Index c = sh[rest[(size_t) a]].cols[j];
            Cnew.col(c).setZero();
            Cnew(copies[g][j].rows, c) = U.col(a);
          }
        }
        for (Eigen::Index a = 0; a < nr; a++) {
          sh[rest[(size_t) a]].e = es.eigenvalues()(a);
          sh[rest[(size_t) a]].w = 0.0;
        }
      }

      // Occupied by energy, then the chosen virtuals by weight, most negative
      // first, then the rest by energy.
      auto rank = [&](size_t s) { return sh[s].occupied ? 0 : chosen[s] ? 1 : 2; };
      std::vector<size_t> order(sh.size());
      for (size_t s = 0; s < order.size(); s++) order[s] = s;
      std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        if (rank(a) != rank(b))
          return rank(a) < rank(b);
        return rank(a) == 1 ? sh[a].w < sh[b].w : sh[a].e < sh[b].e;
      });
      helfem::Matrix Cout(C.rows(), C.cols());
      helfem::Vector eout(C.cols());
      std::vector<int> lout(label.size()), sout(shell.size());
      weight.assign((size_t) C.cols(), 0.0);
      Eigen::Index col = 0;
      for (size_t s = 0; s < order.size(); s++)
        for (Eigen::Index c : sh[order[s]].cols) {
          Cout.col(col) = Cnew.col(c);
          eout(col) = sh[order[s]].e;
          lout[(size_t) col] = label[(size_t) c];
          sout[(size_t) col] = (int) s;
          weight[(size_t) col] = sh[order[s]].w;
          col++;
        }
      C = Cout;
      energy = eout;
      label = lout;
      shell = sout;
    }

    CASSCF::CASSCF(const JKProvider & jk, const helfem::Matrix & S, CASCI & ci,
                   helfem::Matrix C, Eigen::Index ninact, Eigen::Index nact,
                   const std::vector<int> & label, const std::vector<int> & shell,
                   Coupling coupling)
        : jk_(jk), S_(S), ci_(ci), ninact_(ninact), nact_(nact),
          pairs_(rotation_pairs(C.cols(), ninact, nact, label, shell)), coupling_(coupling) {
      // Both parametrizations start from the CASCI ground state at C.
      C = orthonormalized(C);
      trial_ = std::make_unique<CASPoint>(jk_, C, ninact_, nact_);
      std::vector<double> c;
      ci_.set_hamiltonian(trial_->hamiltonian());
      ci_.solve_ground(c);
      adopt(C, ci_.to_csf(c));
    }

    helfem::Matrix CASSCF::kappa(const double * x) const {
      helfem::Matrix K = helfem::Matrix::Zero(C_.cols(), C_.cols());
      for (size_t k = 0; k < pairs_.size(); k++)
        for (const RotationPair & r : pairs_[k]) {
          K(r.p, r.q) = x[k];
          K(r.q, r.p) = -x[k];
        }
      return K;
    }

    helfem::Matrix CASSCF::orthonormalized(const helfem::Matrix & C) const {
      // So that roundoff in many rotations cannot accumulate into
      // non-orthonormal orbitals. Loewdin over the occupied orbitals, and
      // the virtuals made orthogonal to them: the rotations are exact
      // exponentials, so what drifts is roundoff, and a Loewdin over all
      // orbitals would be a dense eigensolve of the whole basis (4004
      // orbitals for Cr2) at every step.
      const Eigen::Index nocc = ninact_ + nact_, nvirt = C.cols() - nocc;
      helfem::Matrix out = C;
      const helfem::Matrix Co = C.leftCols(nocc);
      Eigen::SelfAdjointEigenSolver<helfem::Matrix> es(Co.transpose() * S_ * Co);
      out.leftCols(nocc) = Co * (es.eigenvectors()
                                 * es.eigenvalues().array().rsqrt().matrix().asDiagonal()
                                 * es.eigenvectors().transpose());
      if (nvirt > 0) {
        const helfem::Matrix SCo = S_ * out.leftCols(nocc);
        out.rightCols(nvirt) -= out.leftCols(nocc) * (SCo.transpose() * C.rightCols(nvirt));
      }
      return out;
    }

    helfem::Matrix CASSCF::rotated(const double * x) const {
      // Every rotation has its lower index among the occupied orbitals.
      return orthonormalized(trscf::apply_kappa(C_, kappa(x), ninact_ + nact_));
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
      // C comes from rotated(), already orthonormal. The optimizer normally
      // adopts the point whose energy it evaluated last, so those integrals
      // are taken over rather than rebuilt.
      C_ = C;
      if (trial_ && trial_->orbitals() == C)
        pt_ = std::move(trial_);
      else
        pt_ = std::make_unique<CASPoint>(jk_, C_, ninact_, nact_);
      trial_.reset();
      ci_.set_hamiltonian(pt_->hamiltonian());
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
      Fg_ = pt_->generalized_fock(rdm_.D, rdm_.d);
      g_ = 2.0 * square_antisymmetric(Fg_);
    }

    double CASSCF::energy(const std::vector<double> & x) {
      trial_ = std::make_unique<CASPoint>(jk_, rotated(x.data()), ninact_, nact_);
      const ActiveHamiltonian & ah = trial_->hamiltonian();
      if (coupling_ == Coupling::TwoStep) {
        std::vector<double> c;
        ci_.set_hamiltonian(ah);
        return ci_.solve_ground(c);
      }
      return expectation(ah, ci_at(x.data()));
    }

    std::vector<double> CASSCF::gradient() const {
      std::vector<double> g(nparam(), 0.0);
      // A tied parameter moves every image, so its derivative is the sum.
      for (size_t k = 0; k < pairs_.size(); k++)
        for (const RotationPair & r : pairs_[k])
          g[k] += g_(r.p, r.q);
      // CI part: 2 P H|0> = 2 (sigma - E c), zero along |0> itself.
      for (size_t i = 0; i < nci(); i++)
        g[nrot() + i] = 2.0 * (sigma_[i] - E_ * ccsf_[i]);
      return g;
    }

    std::vector<double> CASSCF::hessian_vector(const std::vector<double> & x) {
      // hess_kappa_kappa_raw differentiates the gradient along a product of
      // exponentials, which differs from the Hessian by the commutator term
      // (1/2) g . [K, .]; taking it off leaves the symmetric Hessian.
      // Only the occupied columns of M are parameters.
      const Eigen::Index nocc = ninact_ + nact_;
      const helfem::Matrix K = kappa(x.data());
      ActiveHamiltonian dH;
      const helfem::Matrix dF = pt_->fock_response(
          rdm_.D, rdm_.d, K, coupling_ == Coupling::Coupled ? &dH : nullptr);
      helfem::Matrix M = 2.0 * antisymmetrize_occupied(dF)
                         - 0.5 * (g_ * K.leftCols(nocc) - K * g_.leftCols(nocc));
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
        M += 2.0 * antisymmetrize_occupied(pt_->generalized_fock(a.D + b.D, dt, 0.0));

        // c-c: 2 P (H - E) t, with the reference Hamiltonian installed.
        ci_.set_hamiltonian(pt_->hamiltonian());
        std::vector<double> hc = ci_.to_csf(ci_.sigma(tdet));
        for (size_t i = 0; i < hc.size(); i++) hc[i] = 2.0 * (hc[i] - E_ * t[i]);

        // c-kappa: 2 P H' |0>, H' the one-index-transformed Hamiltonian.
        ci_.set_hamiltonian(dH);
        const std::vector<double> sk = ci_.to_csf(ci_.sigma(c_));
        ci_.set_hamiltonian(pt_->hamiltonian());
        for (size_t i = 0; i < hc.size(); i++) hc[i] += 2.0 * sk[i];

        hc = project_out(hc, ccsf_);
        std::copy(hc.begin(), hc.end(), hx.begin() + nrot());
      }
      for (size_t k = 0; k < pairs_.size(); k++)
        for (const RotationPair & r : pairs_[k])
          hx[k] += M(r.p, r.q);
      return hx;
    }

    std::vector<double> CASSCF::hessian_diagonal() const {
      // The usual Fock-matrix approximation to the diagonal, with F = F^I +
      // F^A; only the preconditioner sees it.
      const helfem::Vector Fd = pt_->fock_diagonal(rdm_.D);
      const helfem::Matrix & Fg = Fg_;

      std::vector<double> h(pairs_.size());
      for (size_t k = 0; k < pairs_.size(); k++) {
        // Equivalent images have equal diagonals; the tied parameter's is
        // their sum, i.e. the representative's times the number of images.
        const Eigen::Index p = pairs_[k][0].p, q = pairs_[k][0].q;
        const double nimg = (double) pairs_[k].size();
        const bool q_inactive = q < ninact_;
        const bool p_active = p < ninact_ + nact_;
        double v;
        if (q_inactive && !p_active) {
          v = 4.0 * (Fd(p) - Fd(q));
        } else if (q_inactive) {
          const double Dtt = rdm_.D(p - ninact_, p - ninact_);
          v = 4.0 * (Fd(p) - Fd(q)) + 2.0 * Dtt * Fd(q) - 2.0 * Fg(p, p);
        } else {
          const double Dtt = rdm_.D(q - ninact_, q - ninact_);
          v = 2.0 * Dtt * Fd(p) - 2.0 * Fg(q, q);
        }
        // Rotations into an empty or full active orbital are (nearly)
        // redundant and have a vanishing diagonal; keep the sign, bound
        // the magnitude away from zero.
        v *= nimg;
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

    CASSCFResult casscf_from_scf(const JKProvider & jk, const helfem::Matrix & S,
                                 const helfem::Matrix & P,
                                 const std::vector<std::vector<Eigen::Index>> & dsym,
                                 const std::vector<helfem::Matrix> & Sinvh,
                                 const std::vector<std::string> & block_name,
                                 int Ntot, const CASRequest & req) {
      const int nactel = Ntot - 2 * req.ninact;
      if (req.twoS < 0 || nactel < 0 || req.twoS > nactel || (nactel - req.twoS) % 2 != 0
          || req.twoS > 2 * req.nact - nactel)
        throw std::logic_error("CASSCF: " + std::to_string(nactel) + " active electrons "
                               "cannot form multiplicity " + std::to_string(req.twoS + 1)
                               + " in " + std::to_string(req.nact) + " orbitals.\n");

      helfem::Matrix C;
      helfem::Vector eorb;
      std::vector<int> label, shell;
      std::vector<double> weight;
      canonical_orbitals(jk, S, P, dsym, Sinvh, C, eorb, label, shell);
      correlating_virtuals(jk, S, P, req.ninact, req.nact, C, eorb, label, shell, weight);
      if (req.ninact + req.nact > C.cols())
        throw std::logic_error("CASSCF: more inactive and active orbitals than the "
                               "basis has.\n");
      if (req.options.verbosity >= 1) {
        printf("\nCASSCF(%i,%i), multiplicity %i, %i inactive orbitals; starting "
               "orbitals (energy, and correlating weight of an empty one):\n",
               nactel, req.nact, req.twoS + 1, req.ninact);
        for (int i = 0; i < req.ninact + req.nact; i++) {
          printf("  %3i %-10s % .8f %s", i, block_name[(size_t) label[(size_t) i]].c_str(),
                 eorb(i), i < req.ninact ? "inactive" : "active");
          if (weight[(size_t) i] != 0.0)
            printf("  % .3e", weight[(size_t) i]);
          printf("\n");
        }
        fflush(stdout);
      }

      CASCI ci(req.nact, (nactel + req.twoS) / 2, (nactel - req.twoS) / 2);
      CASSCF cas(jk, S, ci, C, req.ninact, req.nact, label, shell, req.coupling);
      const CASSCFResult res = cas.run(req.options);
      // Checked here rather than trusted to OpenTrustRegion's exit status: the
      // drivers print this energy as the result of the run.
      if (!(res.grad_rms <= req.options.otr.conv_tol)) {
        char msg[160];
        snprintf(msg, sizeof msg, "CASSCF did not converge: RMS gradient %.3e "
                 "above the threshold %.1e", res.grad_rms, req.options.otr.conv_tol);
        throw std::runtime_error(msg);
      }
      return res;
    }

  } // namespace cas
} // namespace helfem
