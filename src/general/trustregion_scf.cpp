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
#include "trustregion_scf.h"

#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>

namespace helfem {
  namespace trscf {

    namespace {
      /// An occupation this close to empty or full counts as integer: the
      /// filling coordinate of such a block sits on a kink (an orbital
      /// just filled) or a bound (the block is empty), not in the smooth
      /// interior, so it is not a free parameter.
      const double occ_tol = 1e-6;
      /// Floor on |lambda - mu| in the preconditioner, matching
      /// OpenTrustRegion's own precond_floor.
      const double precond_floor = 1e-10;

      /// exp(A) for a real antisymmetric A. i*A is Hermitian, so
      /// diagonalizing it exponentiates the eigenvalues exactly and the
      /// result is orthogonal by construction -- no scaling-and-squaring
      /// truncation, and no dependence on Eigen's unsupported modules.
      helfem::Matrix expm_skew(const helfem::Matrix &A) {
        const Eigen::Index n = A.rows();
        if (n == 0)
          return helfem::Matrix(0, 0);
        const std::complex<double> im(0.0, 1.0);
        Eigen::MatrixXcd H = im * A.cast<std::complex<double>>();
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(H);
        Eigen::VectorXcd ph =
            (-im * es.eigenvalues().cast<std::complex<double>>()).array().exp();
        return (es.eigenvectors() * ph.asDiagonal() *
                es.eigenvectors().adjoint())
            .real();
      }

      /// U * exp(kappa), where kappa is antisymmetric and supported only
      /// on rows and columns 0..m-1 (the occupied and fractionally
      /// occupied orbitals; rotations among the virtuals are redundant
      /// and are not parameters).
      ///
      /// That support means kappa has rank at most 2m: its range lies in
      /// span{e_0..e_{m-1}} plus the virtual part of its first m columns.
      /// Projecting onto an orthonormal basis Q of those 2m directions
      /// turns the exponential into a 2m x 2m problem,
      /// exp(kappa) = I + Q (exp(Q'kappa Q) - I) Q', which costs O(n^2 m)
      /// instead of the O(n^3) a dense exponential would -- and this is
      /// called once per block per objective evaluation, so the
      /// difference is the difference between a usable optimizer and an
      /// unusable one.
      helfem::Matrix apply_kappa(const helfem::Matrix &U,
                                 const helfem::Matrix &kappa, Eigen::Index m) {
        const Eigen::Index n = kappa.rows();
        if (kappa.cwiseAbs().maxCoeff() == 0.0)
          return U;
        if (2 * m >= n)
          // The projection would not shrink anything.
          return U * expm_skew(kappa);

        // Orthonormal basis of the virtual part of the first m columns.
        Eigen::HouseholderQR<helfem::Matrix> qr(kappa.bottomLeftCorner(n - m, m));
        const helfem::Matrix Zq =
            qr.householderQ() * helfem::Matrix::Identity(n - m, m);

        helfem::Matrix Q = helfem::Matrix::Zero(n, 2 * m);
        Q.topLeftCorner(m, m) = helfem::Matrix::Identity(m, m);
        Q.bottomRightCorner(n - m, m) = Zq;

        helfem::Matrix kt = Q.transpose() * kappa * Q;
        // Antisymmetrize away the roundoff, so expm_skew's Hermitian
        // eigensolver sees exactly what it expects.
        kt = 0.5 * (kt - kt.transpose());
        helfem::Matrix E = expm_skew(kt);
        E.diagonal().array() -= 1.0;

        return U + (U * Q) * E * Q.transpose();
      }

      /// Symmetric (Loewdin) reorthonormalization, to stop the reference
      /// orbitals drifting off the orthogonal manifold over many
      /// macroiterations of rounded exponentials.
      void reorthonormalize(helfem::Matrix &U) {
        const helfem::Matrix S = U.transpose() * U;
        Eigen::SelfAdjointEigenSolver<helfem::Matrix> es(S);
        const helfem::Vector isq = es.eigenvalues().array().rsqrt();
        U = U * (es.eigenvectors() * isq.asDiagonal() *
                 es.eigenvectors().transpose());
      }

      /// Euclidean projection onto the simplex {q >= 0, sum q = total}.
      void project_simplex(std::vector<double> &q, double total) {
        const size_t n = q.size();
        if (n == 0)
          return;

        // A feasible point is its own projection, so return it untouched
        // rather than round-tripping it through the arithmetic below: the
        // sum is conserved by construction, but only to rounding, and the
        // shift that leaks out of that is applied to every filling on
        // every objective evaluation. The steps this guards are almost
        // never needed -- a free block leaving q >= 0 is a pattern change,
        // not a normal step -- so the common path should be exactly the
        // identity, not the identity plus 1e-16.
        bool feasible = true;
        for (size_t i = 0; i < n; i++)
          if (q[i] < 0.0) {
            feasible = false;
            break;
          }
        if (feasible)
          return;

        std::vector<double> u(q);
        std::sort(u.begin(), u.end(), std::greater<double>());
        double csum = 0.0, theta = 0.0;
        size_t rho = 0;
        for (size_t j = 0; j < n; j++) {
          csum += u[j];
          const double t = (csum - total) / (double)(j + 1);
          if (u[j] - t > 0.0) {
            rho = j + 1;
            theta = t;
          }
        }
        if (rho == 0)
          theta = (csum - total) / (double)n;
        for (size_t i = 0; i < n; i++)
          q[i] = std::max(q[i] - theta, 0.0);
      }

      /// Orthonormal basis of the sum-zero subspace of R^n (Helmert).
      /// Column m has m ones, then -m, then zeros. Orthonormal, so the
      /// parameter metric stays isotropic instead of singling out one
      /// block the way eliminating q_0 would.
      helfem::Matrix helmert(size_t n) {
        if (n < 2)
          return helfem::Matrix::Zero((Eigen::Index)n, 0);
        helfem::Matrix U =
            helfem::Matrix::Zero((Eigen::Index)n, (Eigen::Index)n - 1);
        for (Eigen::Index m = 1; m < (Eigen::Index)n; m++) {
          const double nrm = std::sqrt((double)m * (m + 1));
          for (Eigen::Index i = 0; i < m; i++)
            U(i, m - 1) = 1.0 / nrm;
          U(m, m - 1) = -(double)m / nrm;
        }
        return U;
      }
    } // namespace

    Optimizer::Optimizer(const helfem::Matrix &Sinvh,
                         const std::vector<double> &maxocc,
                         const std::vector<size_t> &blocks_per_particle,
                         FockBuilder fock, ResponseBuilder response)
        : Optimizer(std::vector<helfem::Matrix>(maxocc.size(), Sinvh), maxocc,
                    blocks_per_particle, fock, response) {}

    Optimizer::Optimizer(const std::vector<helfem::Matrix> &Sinvh,
                         const std::vector<double> &maxocc,
                         const std::vector<size_t> &blocks_per_particle,
                         FockBuilder fock, ResponseBuilder response)
        : Sinvh_(Sinvh), maxocc_(maxocc), fock_(fock), response_(response) {
      if (Sinvh_.size() != maxocc_.size()) {
        std::ostringstream oss;
        oss << "Got " << Sinvh_.size() << " orthonormalizers for "
            << maxocc_.size() << " blocks.\n";
        throw std::logic_error(oss.str());
      }
      size_t b = 0;
      for (size_t p = 0; p < blocks_per_particle.size(); p++) {
        std::vector<size_t> blocks;
        for (size_t i = 0; i < blocks_per_particle[p]; i++, b++) {
          blocks.push_back(b);
          particle_.push_back(p);
        }
        pblocks_.push_back(blocks);
      }
      if (b != maxocc_.size()) {
        std::ostringstream oss;
        oss << "Block count mismatch: " << b << " blocks across the particle "
            << "types but " << maxocc_.size() << " maximum occupations.\n";
        throw std::logic_error(oss.str());
      }
    }

    void Optimizer::fill_occupations(size_t b, double q, helfem::Vector &f,
                                     Eigen::Index &k) const {
      const Eigen::Index n = norb(b);
      const double w = maxocc_[b];
      f = helfem::Vector::Zero(n);
      for (Eigen::Index i = 0; i < n; i++)
        f(i) = std::min(std::max(q - (double)i * w, 0.0), w);

      // The active orbital is the first one that is not full to within
      // the tolerance -- found by scanning rather than as floor(q/w),
      // which lands on the orbital BELOW at an exact shell filling as
      // soon as q is a rounded 11.999999999 rather than 12. That
      // off-by-one would make a closed shell look partly open, and would
      // make the aufbau check read the filled orbital's energy instead of
      // the empty one's.
      k = n - 1;
      for (Eigen::Index i = 0; i < n; i++)
        if (f(i) < w - occ_tol) {
          k = i;
          break;
        }
    }

    void Optimizer::set_reference(const std::vector<helfem::Matrix> &orbs,
                                  const std::vector<helfem::Vector> &occs) {
      if (orbs.size() != nblock() || occs.size() != nblock())
        throw std::logic_error(
            "Wrong number of blocks handed to the second-order optimizer.\n");

      U_ = orbs;
      q_.assign(nblock(), 0.0);
      f_.resize(nblock());
      active_.assign(nblock(), 0);
      C_.resize(nblock());

      for (size_t b = 0; b < nblock(); b++) {
        if (orbs[b].rows() != norb(b) || orbs[b].cols() != norb(b)) {
          std::ostringstream oss;
          oss << "Block " << b << " has " << orbs[b].rows() << "x"
              << orbs[b].cols() << " orbitals, expected " << norb(b) << "x"
              << norb(b) << ".\n";
          throw std::logic_error(oss.str());
        }
        // The occupations are read only as a block electron count: which
        // orbital is the fractional one follows from the ordering, which
        // the first-order solver leaves in ascending orbital energy.
        q_[b] = occs[b].sum();
        for (Eigen::Index i = 1; i < occs[b].size(); i++)
          if (occs[b](i) > occs[b](i - 1) + occ_tol) {
            std::ostringstream oss;
            oss << "Block " << b << " has occupation " << occs[b](i)
                << " above orbital " << i - 1 << "'s " << occs[b](i - 1)
                << ". The second-order parametrization fills each block in "
                << "orbital order, so the orbitals must arrive sorted by "
                << "energy.\n";
            throw std::logic_error(oss.str());
          }
        fill_occupations(b, q_[b], f_[b], active_[b]);
        C_[b] = Sinvh_[b] * U_[b];
      }
    }

    void Optimizer::build_layout() {
      pair_i_.assign(nblock(), std::vector<Eigen::Index>());
      pair_j_.assign(nblock(), std::vector<Eigen::Index>());
      koff_.assign(nblock(), 0);

      size_t off = 0;
      for (size_t b = 0; b < nblock(); b++) {
        const Eigen::Index n = norb(b);
        koff_[b] = off;
        const Eigen::Index k = active_[b];
        // A rotation changes the density only if the two orbitals differ
        // in occupation, so the non-redundant set is exactly the pairs
        // with f_i != f_j. Under aufbau filling that is: both below k
        // means both full; both above k means both empty; and -- the case
        // that is easy to miss -- if orbital k itself carries nothing,
        // which is what a block sitting at an exact shell filling or an
        // empty block looks like, then (k, j > k) is empty-against-empty
        // too.
        //
        // Missing that last one made every empty block contribute Nrad-1
        // parameters that were all exactly redundant: zero gradient, zero
        // Hessian. They cost more than their arithmetic. They inflate the
        // parameter count, they seed the Davidson subspace with null
        // directions, and they make the trust-region norm count distances
        // along coordinates that do not move the energy at all.
        const bool active_occupied = (f_[b](k) > occ_tol);
        for (Eigen::Index i = 0; i < n; i++)
          for (Eigen::Index j = i + 1; j < n; j++) {
            if (i < k && j < k)
              continue;
            if (i > k && j > k)
              continue;
            if (i == k && !active_occupied)
              continue;
            pair_i_[b].push_back(i);
            pair_j_[b].push_back(j);
          }
        off += pair_i_[b].size();
      }
      qoff_ = off;

      // Free occupation coordinates: the blocks whose fractionally
      // occupied orbital really is fractional. A block at an exact shell
      // filling sits on the kink where one orbital has just filled and
      // the next has not started, and an empty block sits on q = 0; in
      // neither case is the filling a smooth coordinate, and in neither
      // case should a local second-order step be the thing that decides
      // to open a closed shell. That decision belongs to the first-order
      // phase that hands over the occupation pattern.
      sumzero_.assign(pblocks_.size(), helfem::Matrix());
      free_.assign(pblocks_.size(), std::vector<size_t>());
      for (size_t p = 0; p < pblocks_.size(); p++) {
        for (size_t b : pblocks_[p]) {
          // The active orbital is never full by construction, so a block
          // is free exactly when that orbital carries something.
          if (f_[b](active_[b]) > occ_tol)
            free_[p].push_back(b);
        }
        sumzero_[p] = helmert(free_[p].size());
        off += (size_t)sumzero_[p].cols();
      }
      nparam_ = off;

      entry_active_ = active_;
      pattern_changed_ = false;
    }

    bool Optimizer::pattern_stale() const {
      // Only the free blocks can move: the others have no parameter to
      // move them. A free block leaves the pattern when its fractional
      // orbital fills or empties, at which point the pair list built
      // around that orbital, and the sum-zero basis built around the set
      // of free blocks, both describe a different problem.
      // The active index is checked for EVERY block, not just the free
      // ones: a frozen block moving would mean something changed an
      // occupation that has no parameter, and that is worth catching
      // rather than trusting it cannot happen.
      for (size_t b = 0; b < nblock(); b++)
        if (active_[b] != entry_active_[b])
          return true;
      for (size_t p = 0; p < pblocks_.size(); p++)
        for (size_t b : free_[p])
          if (f_[b](active_[b]) <= occ_tol ||
              f_[b](active_[b]) >= maxocc_[b] - occ_tol)
            return true;
      return false;
    }

    void Optimizer::report_conditioning(int verbosity) const {
      if (verbosity < 1 || nparam_ == 0)
        return;

      // The orbital part of the uncoupled diagonal. Its spread IS the
      // conditioning of the model: a rotation whose (f_i-f_j)(F_jj-F_ii)
      // is tiny is a direction the quadratic model barely sees.
      helfem::Vector d(hdiag_.head((Eigen::Index)qoff_).cwiseAbs());
      if (qoff_ > 0) {
        std::vector<double> s(d.data(), d.data() + d.size());
        std::sort(s.begin(), s.end());
        const double top = s.back();
        size_t below3 = 0, below6 = 0;
        for (double v : s) {
          if (v < 1e-3 * top) below3++;
          if (v < 1e-6 * top) below6++;
        }
        printf("  orbital Hessian diagonal: %.3e to %.3e, median %.3e\n",
               s.front(), top, s[s.size() / 2]);
        printf("  %i of %i rotations below 1e-3 of the largest, %i below "
               "1e-6: those are the directions the model cannot resolve\n",
               (int)below3, (int)qoff_, (int)below6);
      }

      if (exact_val_.size())
        printf("  exactly built subspace: %i parameters, eigenvalues %.3e to "
               "%.3e\n",
               (int)exact_val_.size(), exact_val_.minCoeff(),
               exact_val_.maxCoeff());

      // How much room each free filling has before it hits a shell edge.
      // That headroom is the radius within which the energy is smooth in
      // this coordinate: past it the gradient jumps from one orbital
      // energy to the next, and a step that crosses it is being scored by
      // a quadratic model of a function that has a corner in the way.
      for (size_t p = 0; p < pblocks_.size(); p++)
        for (size_t b : free_[p]) {
          const double fk = f_[b](active_[b]);
          printf("  block %i filling: orbital %i holds %.6f of %.1f, so the "
                 "smooth range runs %.6f down and %.6f up\n",
                 (int)b, (int)active_[b], fk, maxocc_[b], fk, maxocc_[b] - fk);
        }
      fflush(stdout);
    }

    bool Optimizer::release_kkt_violation(int verbosity) {
      // The KKT conditions of  min E(n)  s.t.  0 <= n_i <= w,  sum n_i = N
      // are, with eps_i = dE/dn_i by Janak's theorem,
      //
      //     0 < n_i < w   =>   eps_i == eps_F
      //         n_i = w   =>   eps_i <= eps_F
      //         n_i = 0   =>   eps_i >= eps_F
      //
      // A block frozen at an integer occupation satisfies the second or
      // third of those only by luck. When it does not, the point is a
      // stationary point of the RESTRICTED problem -- the one with that
      // block pinned -- and not of the problem being solved, which is why
      // the gradient can be tiny there and the energy still wrong.
      //
      // The fix is not to report it. Multiplier signs say which way charge
      // wants to move, so move a little that way: the block then has a
      // fractional occupation, becomes a free coordinate when the layout is
      // rebuilt, and the optimizer can carry the rest.
      const double nudge = 1e-3;

      for (size_t p = 0; p < pblocks_.size(); p++) {
        if (free_[p].empty())
          continue;
        double efermi = -std::numeric_limits<double>::max();
        for (size_t b : free_[p])
          efermi = std::max(efermi, F_[b](active_[b], active_[b]));

        for (size_t b : pblocks_[p]) {
          if (std::find(free_[p].begin(), free_[p].end(), b) != free_[p].end())
            continue;

          // Frozen full, but its topmost filled orbital sits above the
          // Fermi level: it should give charge up.
          if (active_[b] > 0 &&
              F_[b](active_[b] - 1, active_[b] - 1) > efermi + 1e-8) {
            size_t to = free_[p][0];
            for (size_t c : free_[p])
              if (F_[c](active_[c], active_[c]) < F_[to](active_[to], active_[to]))
                to = c;
            if (verbosity >= 1)
              printf("\nBlock %i is pinned full at % .6f, above the Fermi level "
                     "% .6f, so the solution\n         satisfies the "
                     "stationarity of a restricted problem and not the KKT "
                     "conditions of this one.\n         Releasing it towards "
                     "block %i.\n",
                     (int)b, F_[b](active_[b] - 1, active_[b] - 1), efermi,
                     (int)to);
            q_[b] -= nudge;
            q_[to] += nudge;
            return true;
          }

          // Frozen empty (or at a shell edge), but the orbital that would
          // fill next sits below the Fermi level: it should take charge on.
          if (F_[b](active_[b], active_[b]) < efermi - 1e-8) {
            size_t from = free_[p][0];
            for (size_t c : free_[p])
              if (F_[c](active_[c], active_[c]) > F_[from](active_[from], active_[from]))
                from = c;
            if (q_[from] <= nudge)
              continue;
            if (verbosity >= 1)
              printf("\nBlock %i is pinned with its next orbital at % .6f, below "
                     "the Fermi level % .6f,\n         which violates the KKT "
                     "conditions. Releasing it from block %i.\n",
                     (int)b, F_[b](active_[b], active_[b]), efermi, (int)from);
            q_[b] += nudge;
            q_[from] -= nudge;
            return true;
          }
        }
      }
      return false;
    }

    void Optimizer::check_aufbau(int verbosity) const {
      // The blocks held frozen are frozen because their topmost orbital
      // is exactly full or exactly empty. That is only the right answer
      // if the orbital that would fill next lies above the Fermi level;
      // otherwise the solution is stationary but not aufbau, and the
      // occupation pattern handed over was wrong.
      double efermi = -std::numeric_limits<double>::max();
      bool have_fermi = false;
      for (size_t p = 0; p < pblocks_.size(); p++)
        for (size_t b : free_[p]) {
          efermi = std::max(efermi, F_[b](active_[b], active_[b]));
          have_fermi = true;
        }
      if (!have_fermi)
        return;

      // A block frozen at an integer occupation can be wrong in two ways,
      // and by Janak's theorem dE/dn_i = eps_i both are read off the same
      // orbital energies. It may want to RECEIVE charge, if the orbital
      // that would fill next lies below the Fermi level. Or it may want to
      // GIVE UP charge, if its topmost filled orbital lies above it --
      // which is the more dangerous of the two, because the solve then
      // converges tidily to the pure-state solution and reports a
      // perfectly small gradient. Measured on Fe handed over at 20
      // first-order iterations: 4s full at -0.163 against 3d at -0.248,
      // converged to 2.2e-9, and 0.032 Eh above the ensemble minimum with
      // nothing in the output to say so.
      for (size_t p = 0; p < pblocks_.size(); p++)
        for (size_t b : pblocks_[p]) {
          if (std::find(free_[p].begin(), free_[p].end(), b) != free_[p].end())
            continue;
          // active_ is the first orbital that is not full, i.e. the one
          // that would receive charge if this block opened up.
          const double enext = F_[b](active_[b], active_[b]);
          if (enext < efermi - 1e-8 && verbosity >= 1)
            printf("WARNING: block %i is held at an integer occupation, but its "
                   "next orbital lies at % .6f, below the Fermi level % .6f.\n"
                   "         The occupation pattern is not aufbau-stable; the "
                   "first-order phase handed over the wrong one.\n",
                   (int)b, enext, efermi);
          // ... and the topmost orbital it actually holds.
          if (active_[b] > 0) {
            const double etop = F_[b](active_[b] - 1, active_[b] - 1);
            if (etop > efermi + 1e-8 && verbosity >= 1)
              printf("WARNING: block %i is held full up to an orbital at % .6f, "
                     "above the Fermi level % .6f.\n"
                     "         Moving charge out of it would lower the energy, "
                     "so this is a pure-state solution and not the ensemble "
                     "one.\n         Hand over from a better converged "
                     "first-order solution (--preiter).\n",
                     (int)b, etop, efermi);
          }
        }
    }

    void Optimizer::project_fillings(std::vector<double> &q) const {
      // Only the free fillings are projected. They are the only ones a
      // step can move, and they conserve their own sum, so the frozen
      // blocks must come through untouched -- projecting over all of them
      // would let a clamp on one free block shift charge into a frozen
      // one, which changes an occupation nobody is watching: pattern_stale
      // would not see it, because the block has no parameter to have moved.
      for (size_t p = 0; p < pblocks_.size(); p++) {
        if (free_[p].empty())
          continue;
        double total = 0.0;
        std::vector<double> sub;
        for (size_t b : free_[p]) {
          sub.push_back(q[b]);
          total += q_[b];
        }
        project_simplex(sub, total);
        for (size_t i = 0; i < free_[p].size(); i++)
          q[free_[p][i]] = sub[i];
      }
    }

    void Optimizer::decode(const double *x, std::vector<helfem::Matrix> &kappa,
                           std::vector<double> &dq) const {
      kappa.resize(nblock());
      for (size_t b = 0; b < nblock(); b++)
        kappa[b] = helfem::Matrix::Zero(norb(b), norb(b));
      for (size_t b = 0; b < nblock(); b++)
        for (size_t p = 0; p < pair_i_[b].size(); p++) {
          const double v = x[koff_[b] + p];
          kappa[b](pair_i_[b][p], pair_j_[b][p]) = v;
          kappa[b](pair_j_[b][p], pair_i_[b][p]) = -v;
        }

      dq.assign(nblock(), 0.0);
      size_t off = qoff_;
      for (size_t p = 0; p < pblocks_.size(); p++) {
        const Eigen::Index ny = sumzero_[p].cols();
        if (ny <= 0)
          continue;
        Eigen::Map<const helfem::Vector> y(x + off, ny);
        const helfem::Vector d = sumzero_[p] * y;
        for (size_t i = 0; i < free_[p].size(); i++)
          dq[free_[p][i]] = d((Eigen::Index)i);
        off += (size_t)ny;
      }
    }

    void Optimizer::displace(const std::vector<helfem::Matrix> &kappa,
                             const std::vector<double> &dq,
                             std::vector<helfem::Matrix> &U,
                             std::vector<double> &q) const {
      U.resize(nblock());
      q.resize(nblock());
      for (size_t b = 0; b < nblock(); b++) {
        U[b] = apply_kappa(U_[b], kappa[b], active_[b] + 1);
        q[b] = q_[b] + dq[b];
      }
      project_fillings(q);
    }

    void Optimizer::densities(const std::vector<helfem::Matrix> &U,
                              const std::vector<double> &q,
                              helfem::Cube &P) const {
      P.resize(nblock());
      for (size_t b = 0; b < nblock(); b++)
        P[b] = helfem::Matrix::Zero(nfem(b), nfem(b));
      for (size_t b = 0; b < nblock(); b++) {
        helfem::Vector f;
        Eigen::Index k;
        fill_occupations(b, q[b], f, k);
        // Only orbitals up to the fractionally occupied one carry any
        // density, so the density build is O(Nfem^2 * nocc), not O(N^3).
        const Eigen::Index nocc = k + 1;
        const helfem::Matrix Cocc = Sinvh_[b] * U[b].leftCols(nocc);
        P[b] = Cocc * f.head(nocc).asDiagonal() * Cocc.transpose();
      }
    }

    void Optimizer::gradient(double *g) const {
      for (size_t b = 0; b < nblock(); b++) {
        const helfem::Matrix &F = F_[b];
        const helfem::Vector &f = f_[b];
        for (size_t p = 0; p < pair_i_[b].size(); p++) {
          const Eigen::Index i = pair_i_[b][p], j = pair_j_[b][p];
          g[koff_[b] + p] = 2.0 * F(i, j) * (f(j) - f(i));
        }
      }

      // dE/dq_b is the Fock expectation value of the orbital whose
      // occupation the filling coordinate moves.
      size_t off = qoff_;
      for (size_t p = 0; p < pblocks_.size(); p++) {
        const Eigen::Index ny = sumzero_[p].cols();
        if (ny <= 0)
          continue;
        helfem::Vector gq((Eigen::Index)free_[p].size());
        for (size_t i = 0; i < free_[p].size(); i++) {
          const size_t b = free_[p][i];
          gq((Eigen::Index)i) = F_[b](active_[b], active_[b]);
        }
        const helfem::Vector gy = sumzero_[p].transpose() * gq;
        for (Eigen::Index i = 0; i < ny; i++)
          g[off + (size_t)i] = gy(i);
        off += (size_t)ny;
      }
    }

    helfem::Matrix Optimizer::occupation_hessian() const {
      // The exact second derivative of the energy with respect to the
      // block fillings. Since the density is linear in the occupations at
      // fixed orbitals, there is no d^2P/dq^2 term and the whole thing is
      // the kernel sandwiched between the two active orbital densities:
      //     H(b,c) = <k_b k_b | W | k_c k_c>,  W = Coulomb + f_xc.
      // One probe per block gives a whole column, and the probes are
      // batched so the grid work is done once.
      const size_t nb = nblock();
      helfem::Cube P;
      densities(U_, q_, P);

      std::vector<helfem::Cube> probe(nb), resp;
      for (size_t c = 0; c < nb; c++) {
        probe[c].resize(nb);
        for (size_t b = 0; b < nb; b++)
          probe[c][b] = helfem::Matrix::Zero(nfem(b), nfem(b));
        const helfem::Vector ck = C_[c].col(active_[c]);
        probe[c][c] = ck * ck.transpose();
      }
      response_(P, probe, resp);
      stats_.n_response++;

      helfem::Matrix H((Eigen::Index)nb, (Eigen::Index)nb);
      for (size_t c = 0; c < nb; c++)
        for (size_t b = 0; b < nb; b++) {
          const helfem::Vector cb = C_[b].col(active_[b]);
          H((Eigen::Index)b, (Eigen::Index)c) = cb.dot(resp[c][b] * cb);
        }
      return 0.5 * (H + H.transpose());
    }

    void Optimizer::hessian(const std::vector<helfem::Vector> &X,
                            std::vector<helfem::Vector> &HX) const {
      const size_t nt = X.size();
      HX.assign(nt, helfem::Vector::Zero((Eigen::Index)nparam_));
      if (nt == 0)
        return;

      helfem::Cube P;
      densities(U_, q_, P);

      // First-order density change of each trial vector,
      //     D = [kappa, diag f] + diag(df),
      // pushed to the FEM basis for the response builder.
      std::vector<std::vector<helfem::Matrix>> D(nt);
      std::vector<std::vector<helfem::Matrix>> kap(nt);
      std::vector<std::vector<double>> dq(nt);
      std::vector<helfem::Cube> dP(nt);
      for (size_t t = 0; t < nt; t++) {
        decode(X[t].data(), kap[t], dq[t]);
        D[t].assign(nblock(), helfem::Matrix());
        dP[t].assign(nblock(), helfem::Matrix());
        for (size_t b = 0; b < nblock(); b++) {
          helfem::Matrix Db = kap[t][b] * f_[b].asDiagonal();
          Db -= f_[b].asDiagonal() * kap[t][b];
          Db(active_[b], active_[b]) += dq[t][b];
          D[t][b] = Db;
          dP[t][b] = C_[b] * Db * C_[b].transpose();
        }
      }

      std::vector<helfem::Cube> dF;
      response_(P, dP, dF);
      stats_.n_response++;
      stats_.n_hessian += nt;

      for (size_t t = 0; t < nt; t++) {
        std::vector<double> hq(nblock(), 0.0);
        for (size_t b = 0; b < nblock(); b++) {
          const helfem::Matrix &F = F_[b];
          const helfem::Vector &f = f_[b];
          const helfem::Matrix &kb = kap[t][b];
          // Response Fock matrix back in the MO basis
          const helfem::Matrix dFmo = C_[b].transpose() * dF[t][b] * C_[b];

          // Uncoupled part: the second-order term of exp(kappa) acting on
          // the reference density, d/dkappa of Tr(F [kappa,[kappa,diag f]])/2.
          const helfem::Matrix K = kb * f.asDiagonal() - f.asDiagonal() * kb;
          const helfem::Matrix S = F * kb - kb * F;
          const helfem::Matrix A =
              (F * K - K * F) + (S * f.asDiagonal() - f.asDiagonal() * S);

          const double dfk = dq[t][b];
          const Eigen::Index k = active_[b];
          for (size_t p = 0; p < pair_i_[b].size(); p++) {
            const Eigen::Index i = pair_i_[b][p], j = pair_j_[b][p];
            // df is nonzero only at the active orbital
            const double ddf = (j == k ? dfk : 0.0) - (i == k ? dfk : 0.0);
            HX[t][(Eigen::Index)(koff_[b] + p)] =
                A(i, j) + 2.0 * F(i, j) * ddf +
                2.0 * dFmo(i, j) * (f(j) - f(i));
          }
          // Occupation direction: the mixed kappa-occupation term plus
          // the kernel coupling.
          hq[b] = S(k, k) + dFmo(k, k);
        }

        size_t off = qoff_;
        for (size_t p = 0; p < pblocks_.size(); p++) {
          const Eigen::Index ny = sumzero_[p].cols();
          if (ny <= 0)
            continue;
          helfem::Vector v((Eigen::Index)free_[p].size());
          for (size_t i = 0; i < free_[p].size(); i++)
            v((Eigen::Index)i) = hq[free_[p][i]];
          const helfem::Vector hy = sumzero_[p].transpose() * v;
          for (Eigen::Index i = 0; i < ny; i++)
            HX[t][(Eigen::Index)(off + (size_t)i)] = hy(i);
          off += (size_t)ny;
        }
      }
    }

    void Optimizer::build_exact_subspace() {
      // Every occupation parameter, and only those. Their uncoupled
      // diagonal is exactly zero, so the diagonal preconditioner has
      // nothing whatever to say about the directions that matter most.
      exact_idx_.clear();
      for (size_t i = qoff_; i < nparam_; i++)
        exact_idx_.push_back(i);

      const Eigen::Index ns = (Eigen::Index)exact_idx_.size();
      if (ns == 0) {
        exact_vec_ = helfem::Matrix();
        exact_val_ = helfem::Vector();
        return;
      }

      std::vector<helfem::Vector> probes(exact_idx_.size());
      for (size_t s = 0; s < exact_idx_.size(); s++) {
        probes[s] = helfem::Vector::Zero((Eigen::Index)nparam_);
        probes[s]((Eigen::Index)exact_idx_[s]) = 1.0;
      }
      std::vector<helfem::Vector> HP;
      hessian(probes, HP);

      helfem::Matrix Hs(ns, ns);
      for (Eigen::Index c = 0; c < ns; c++)
        for (Eigen::Index r = 0; r < ns; r++)
          Hs(r, c) = HP[(size_t)c]((Eigen::Index)exact_idx_[(size_t)r]);
      Hs = 0.5 * (Hs + Hs.transpose());

      Eigen::SelfAdjointEigenSolver<helfem::Matrix> es(Hs);
      exact_val_ = es.eigenvalues();
      exact_vec_ = es.eigenvectors();
    }

    void Optimizer::precondition(const double *r, double mu,
                                 double *out) const {
      // Diagonal everywhere except the exactly built subspace, where the
      // shifted block is inverted in its own eigenbasis.
      //
      // OpenTrustRegion drives one callback from two places with two
      // different contracts, distinguished only by the shift it passes.
      // At mu = 0 it is standing in for abs_diag_precond, which is
      // explicitly a POSITIVE-DEFINITE preconditioner: the truncated
      // conjugate gradient path takes sqrt(x . M x) with it, so a
      // negative eigenvalue there produces a NaN step length and the
      // solve dies on its first macroiteration. At a nonzero shift it is
      // standing in for level_shifted_diag_precond, which wants the
      // signed inverse of H - mu. Hence the absolute value below,
      // matching what those two routines do elementwise.
      const bool absolute = (mu == 0.0);
      auto invert = [&](double d) {
        if (absolute)
          return 1.0 / std::max(std::abs(d), precond_floor);
        if (std::abs(d) < precond_floor)
          d = (d < 0.0) ? -precond_floor : precond_floor;
        return 1.0 / d;
      };

      for (size_t i = 0; i < nparam_; i++)
        out[i] = r[i] * invert(hdiag_((Eigen::Index)i) - mu);

      const Eigen::Index ns = (Eigen::Index)exact_idx_.size();
      if (ns == 0)
        return;

      helfem::Vector rs(ns);
      for (Eigen::Index i = 0; i < ns; i++)
        rs(i) = r[exact_idx_[(size_t)i]];
      helfem::Vector ys = exact_vec_.transpose() * rs;
      for (Eigen::Index i = 0; i < ns; i++)
        ys(i) *= invert(exact_val_(i) - mu);
      rs = exact_vec_ * ys;
      for (Eigen::Index i = 0; i < ns; i++)
        out[exact_idx_[(size_t)i]] = rs(i);
    }

    void Optimizer::adopt(const std::vector<helfem::Matrix> &U,
                          const std::vector<double> &q, double *grad,
                          double *h_diag) {
      U_ = U;
      q_ = q;
      for (size_t b = 0; b < nblock(); b++) {
        reorthonormalize(U_[b]);
        fill_occupations(b, q_[b], f_[b], active_[b]);
        C_[b] = Sinvh_[b] * U_[b];
      }

      helfem::Cube P, F;
      densities(U_, q_, P);
      energy_ = fock_(P, F);
      stats_.n_update++;
      // A step was accepted, so the runaway counter starts over.
      hess_this_macro_ = 0;
      best_energy_ = energy_;
      best_step_ = helfem::Vector();

      F_.resize(nblock());
      for (size_t b = 0; b < nblock(); b++)
        F_[b] = C_[b].transpose() * F[b] * C_[b];

      if (grad)
        gradient(grad);

      // Uncoupled Hessian diagonal. Along the occupation directions it is
      // identically zero, which is exactly why they are handled by the
      // exact block below rather than by this.
      hdiag_ = helfem::Vector::Zero((Eigen::Index)nparam_);
      for (size_t b = 0; b < nblock(); b++) {
        const helfem::Matrix &F2 = F_[b];
        const helfem::Vector &f = f_[b];
        for (size_t p = 0; p < pair_i_[b].size(); p++) {
          const Eigen::Index i = pair_i_[b][p], j = pair_j_[b][p];
          hdiag_((Eigen::Index)(koff_[b] + p)) =
              2.0 * (f(i) - f(j)) * (F2(j, j) - F2(i, i));
        }
      }

      if (nparam_ > qoff_) {
        const helfem::Matrix Hqq = occupation_hessian();
        size_t off = qoff_;
        for (size_t p = 0; p < pblocks_.size(); p++) {
          const Eigen::Index ny = sumzero_[p].cols();
          if (ny <= 0)
            continue;
          helfem::Matrix sub((Eigen::Index)free_[p].size(),
                             (Eigen::Index)free_[p].size());
          for (size_t i = 0; i < free_[p].size(); i++)
            for (size_t j = 0; j < free_[p].size(); j++)
              sub((Eigen::Index)i, (Eigen::Index)j) =
                  Hqq((Eigen::Index)free_[p][i], (Eigen::Index)free_[p][j]);
          const helfem::Matrix red =
              sumzero_[p].transpose() * sub * sumzero_[p];
          for (Eigen::Index i = 0; i < ny; i++)
            hdiag_((Eigen::Index)(off + (size_t)i)) = red(i, i);
          off += (size_t)ny;
        }
      }

      build_exact_subspace();
      if (pattern_stale())
        pattern_changed_ = true;
      if (stats_.n_update == 1)
        report_conditioning(report_verbosity_);

      if (h_diag)
        for (size_t i = 0; i < nparam_; i++)
          h_diag[i] = hdiag_((Eigen::Index)i);
    }

    Result Optimizer::run(const Options &opts) {
      if (U_.empty())
        throw std::logic_error(
            "The second-order optimizer has no reference to start from.\n");

      report_verbosity_ = opts.verbosity;
      stats_ = Result();

      otr::Callbacks cb;
      cb.update = [&](const double *x, double &func, double *grad,
                      double *h_diag) {
        std::vector<helfem::Matrix> kappa;
        std::vector<double> dq;
        decode(x, kappa, dq);
        std::vector<helfem::Matrix> U;
        std::vector<double> q;
        displace(kappa, dq, U, q);
        adopt(U, q, grad, h_diag);
        func = energy_;
        return true;
      };
      cb.objective = [&](const double *x, double &func) {
        std::vector<helfem::Matrix> kappa;
        std::vector<double> dq;
        decode(x, kappa, dq);
        std::vector<helfem::Matrix> U;
        std::vector<double> q;
        displace(kappa, dq, U, q);
        helfem::Cube P, F;
        densities(U, q, P);
        func = fock_(P, F);
        stats_.n_objective++;
        // Remember the best point offered. A stalling solve is not failing
        // to FIND a good step, it is refusing to accept one: the step it
        // discards was measured lowering the energy by 5.3e-4 and landing
        // on the minimum. Keeping it costs one vector.
        if (func < best_energy_) {
          best_energy_ = func;
          best_step_ = Eigen::Map<const helfem::Vector>(x, (Eigen::Index)nparam_);
        }
        return true;
      };
      cb.hessian_vector = [&](const double *x, double *hx) {
        // The ceiling has to be enforced here, not between
        // macroiterations, because the runaway it guards against happens
        // inside a single one: OpenTrustRegion re-enters its
        // microiteration loop until a step is accepted, and never asks the
        // convergence callback in between. Reporting failure is the clean
        // way out -- an exception would have to unwind through Fortran.
        if (hess_budget_ && hess_this_macro_ >= hess_budget_) {
          budget_spent_ = true;
          return false;
        }
        hess_this_macro_++;
        std::vector<helfem::Vector> X(
            1, Eigen::Map<const helfem::Vector>(x, (Eigen::Index)nparam_));
        std::vector<helfem::Vector> HX;
        hessian(X, HX);
        for (size_t i = 0; i < nparam_; i++)
          hx[i] = HX[0]((Eigen::Index)i);
        return true;
      };
      if (opts.exact_precond)
        cb.precondition = [&](const double *r, double mu, double *out) {
          precondition(r, mu, out);
          return true;
        };
      hess_budget_ = (opts.max_hessian > 0)
                         ? (size_t)opts.max_hessian
                         : 5 * (size_t)std::max(1, opts.otr.n_micro);
      const size_t hess_budget = hess_budget_;
      cb.converged = [&](bool &stop) {
        // Not a convergence test but a bail-out, on either of two counts:
        // the reference has left the occupation pattern the
        // parametrization was built around, so continuing would optimize
        // the wrong coordinates; or the work ceiling is spent, which means
        // the rejection loop is running away and more of it will not help.
        // Either way the gradient check after the solve reports honestly
        // what was actually reached.
        stop = pattern_changed_;
        return true;
      };

      // The occupation pattern is discrete and the parametrization is
      // smooth only inside one; a shell closing under the optimizer is a
      // change of pattern, not of coordinates. So the solve runs inside a
      // loop that rebuilds the parametrization around whatever pattern the
      // reference has arrived at.
      const otr::Settings otrset = opts.otr;
      for (int pass = 0;; pass++) {
        build_layout();

        if (opts.verbosity >= 1) {
          size_t nfree = 0;
          for (const auto &fp : free_)
            nfree += fp.size();
          printf("\n%s over %i parameters: %i orbital rotations and %i "
                 "occupation transfers among %i fractionally occupied "
                 "blocks.\n",
                 pass ? "Restarting the second-order optimization"
                      : "\nSecond-order optimization",
                 (int)nparam_, (int)qoff_, (int)(nparam_ - qoff_), (int)nfree);
          if (nparam_ == qoff_)
            printf("No block carries a fractional occupation, so only the "
                   "orbitals are optimized.\n");
          fflush(stdout);
        }

        budget_spent_ = false;
        try {
          otr::solve((int)nparam_, cb, otrset);
        } catch (std::exception &) {
          // A spent budget comes back as a callback failure, which the
          // library turns into an error return. Any other failure is real
          // and is left to propagate. The reference is whatever the last
          // accepted step installed either way, so the energy and gradient
          // reported below are a real point on the surface -- just not a
          // converged one.
          if (!budget_spent_)
            throw;
        }
        // A stalled solve gets its own step back. Retrying was tried twice
        // and failed twice -- a fresh subspace reproduced the stall
        // exactly, and a progressively looser residual target recovered
        // worse than simply starting loose -- but both of those retried
        // from where the stall left the reference, which is the wrong
        // place. The right place is the best point the stalled pass
        // actually visited: OpenTrustRegion evaluated it, saw it lower the
        // energy, and threw it away for missing a residual target it never
        // had to meet to be an improvement.
        // Whether the solve stopped because it spent its budget or
        // because OpenTrustRegion declared maximum precision, the question
        // is the same: did it decline a step that was an improvement? Both
        // stall modes end that way, and the second one does not even reach
        // the budget -- Curium stops after 87 products with the gradient
        // still at 1.2e-3.
        helfem::Vector gpass((Eigen::Index)nparam_);
        gradient(gpass.data());
        const bool pass_converged =
            gpass.norm() / std::sqrt((double)nparam_) <= otrset.conv_tol;
        if (!pass_converged && best_step_.size() &&
            best_energy_ < energy_ - 1e-12 * std::abs(energy_) &&
            pass < opts.max_restarts) {
          std::vector<helfem::Matrix> kappa;
          std::vector<double> dq;
          decode(best_step_.data(), kappa, dq);
          std::vector<helfem::Matrix> U;
          std::vector<double> q;
          displace(kappa, dq, U, q);
          if (opts.verbosity >= 1) {
            printf("\nA macroiteration stalled having found a step worth "
                   "%.3e in energy and declined it.\n         Taking that "
                   "step and continuing from there.\n",
                   energy_ - best_energy_);
            fflush(stdout);
          }
          adopt(U, q, nullptr, nullptr);
          stats_.n_restart++;
          continue;
        }
        // Before giving up on a pass, ask whether the point it reached is
        // a KKT point at all. If a frozen block is pinned on the wrong side
        // of the Fermi level, release it and go again: the restricted
        // problem it solved is not the problem.
        if (!pattern_changed_ && pass < opts.max_restarts &&
            release_kkt_violation(opts.verbosity)) {
          std::vector<helfem::Matrix> U = U_;
          std::vector<double> q = q_;
          adopt(U, q, nullptr, nullptr);
          stats_.n_restart++;
          continue;
        }
        if (budget_spent_ || !pattern_changed_ || pass >= opts.max_restarts)
          break;
        stats_.n_restart++;
        if (opts.verbosity >= 1) {
          printf("\nThe occupation pattern changed under the optimizer -- a "
                 "shell filled or emptied -- so the parametrization is "
                 "rebuilt around the new one.\n");
          fflush(stdout);
        }
      }

      // Report from the converged reference rather than trusting the
      // library's own bookkeeping: OpenTrustRegion returns success when
      // its line search stalls, so a zero error code is not by itself a
      // statement that the gradient came down.
      helfem::Vector g((Eigen::Index)nparam_);
      gradient(g.data());
      stats_.energy = energy_;
      stats_.grad_rms = g.norm() / std::sqrt((double)nparam_);
      stats_.converged = (stats_.grad_rms <= opts.otr.conv_tol);

      if (!stats_.converged && opts.verbosity >= 1) {
        printf("\nWARNING: the second-order optimization stopped with RMS "
               "gradient %.3e, above the requested %.3e.\n",
               stats_.grad_rms, opts.otr.conv_tol);
        if (budget_spent_)
          printf("         One macroiteration spent %i Hessian-vector products "
                 "without accepting a step,\n         so the trust region was "
                 "rejecting everything it was offered. Hand over from a\n"
                 "         different point (--preiter) or raise --somaxhess.\n",
                 (int)hess_budget);
        fflush(stdout);
      }
      check_aufbau(opts.verbosity);

      return stats_;
    }

    bool Optimizer::verify(double step, double tol, int verbosity,
                           bool exact_hessian) {
      build_layout();
      report_verbosity_ = verbosity;
      stats_ = Result();

      // Establish the reference, its energy, gradient and Hessian data.
      helfem::Vector g((Eigen::Index)nparam_), hd((Eigen::Index)nparam_);
      adopt(U_, q_, g.data(), hd.data());
      const double E0 = energy_;

      // Energy at a displacement from the FIXED reference. Nothing here
      // may adopt: the finite differences have to be taken in one
      // coordinate system.
      const std::vector<helfem::Matrix> U0 = U_;
      const std::vector<double> q0 = q_;
      auto energy_at = [&](const helfem::Vector &x) {
        std::vector<helfem::Matrix> kappa;
        std::vector<double> dq;
        decode(x.data(), kappa, dq);
        std::vector<helfem::Matrix> U;
        std::vector<double> q;
        displace(kappa, dq, U, q);
        helfem::Cube P, F;
        densities(U, q, P);
        return fock_(P, F);
      };

      std::mt19937 rng(20250824);
      std::normal_distribution<double> gauss(0.0, 1.0);
      auto random_dir = [&]() {
        helfem::Vector d((Eigen::Index)nparam_);
        for (Eigen::Index i = 0; i < d.size(); i++)
          d(i) = gauss(rng);
        d /= d.norm();
        return d;
      };

      // What the finite differences can actually resolve. A central
      // difference of the energy carries a roundoff error eps*|E|/h and a
      // truncation error of order h^2 times the third derivative, for
      // which h^2*|d.H.d| is the right order-of-magnitude stand-in.
      // Without this floor the check fails on any reference that is
      // already converged, where the directional derivatives are
      // legitimately at the noise level and a RELATIVE comparison says
      // nothing about whether the analytic derivatives are right.
      const double eps = std::numeric_limits<double>::epsilon();
      const double roundoff = eps * std::abs(E0) / step;
      auto agrees = [&](double ana, double num, double atol) {
        return std::abs(num - ana) <=
               tol * std::max(std::abs(ana), std::abs(num)) + atol;
      };
      auto relerr = [&](double ana, double num) {
        return std::abs(num - ana) /
               std::max(1e-10, std::abs(num) + std::abs(ana));
      };

      bool ok = true;
      if (verbosity >= 1)
        printf("\nFinite-difference check of the second-order derivatives, "
               "step %.1e over %i parameters\n", step, (int)nparam_);

      // --- gradient: directional derivative of the energy. Each
      // direction needs its own curvature for the truncation floor, so
      // the Hessian-vector products come first.
      std::vector<helfem::Vector> D;
      D.push_back(helfem::Vector(g / g.norm()));
      for (int i = 0; i < 3; i++)
        D.push_back(random_dir());
      std::vector<helfem::Vector> HD;
      hessian(D, HD);

      for (size_t trial = 0; trial < 3; trial++) {
        const helfem::Vector &d = D[trial];
        const double num = (energy_at(step * d) - energy_at(-step * d)) /
                           (2.0 * step);
        const double ana = g.dot(d);
        const double atol =
            10.0 * (roundoff + step * step * std::abs(d.dot(HD[trial])));
        const bool pass = agrees(ana, num, atol);
        if (verbosity >= 1)
          printf("  gradient . d%-2zu  analytic % .10e  numerical % .10e  "
                 "rel.err %.2e  noise %.1e%s\n",
                 trial, ana, num, relerr(ana, num), atol, pass ? "" : "   ***");
        ok = ok && pass;
      }

      // --- Hessian: the bilinear form, by four-point polarization. This
      // checks the whole H d vector against the energy surface rather
      // than only the quadratic form along one direction.
      const double hess_atol = roundoff / step;
      double worst_hess = 0.0;
      const std::pair<int, int> pairs[] = {{0, 0}, {1, 1}, {1, 2}, {2, 3}};
      for (const auto &pr : pairs) {
        const helfem::Vector &d1 = D[(size_t)pr.first];
        const helfem::Vector &d2 = D[(size_t)pr.second];
        const helfem::Vector sum = d1 + d2, dif = d1 - d2;
        const double num = (energy_at(step * sum) + energy_at(-step * sum) -
                            energy_at(step * dif) - energy_at(-step * dif)) /
                           (4.0 * step * step);
        const double ana = d1.dot(HD[(size_t)pr.second]);
        const bool pass = agrees(ana, num, hess_atol);
        worst_hess = std::max(worst_hess, relerr(ana, num));
        if (verbosity >= 1)
          printf("  d%i . H d%-2i     analytic % .10e  numerical % .10e  "
                 "rel.err %.2e  noise %.1e%s\n",
                 pr.first, pr.second, ana, num, relerr(ana, num), hess_atol,
                 (pass || !exact_hessian) ? "" : "   ***");
        ok = ok && (pass || !exact_hessian);
      }

      // --- and the symmetry of the analytic Hessian itself, which the
      // polarization above cannot see.
      for (size_t a = 1; a < D.size(); a++)
        for (size_t b = a + 1; b < D.size(); b++) {
          const double ab = D[a].dot(HD[b]), ba = D[b].dot(HD[a]);
          const bool pass = agrees(ab, ba, 0.0);
          if (verbosity >= 2)
            printf("  symmetry d%zu/d%zu  % .10e vs % .10e  rel.err %.2e%s\n",
                   a, b, ab, ba, relerr(ab, ba), pass ? "" : "   ***");
          ok = ok && pass;
        }

      if (verbosity >= 1) {
        printf("  reference energy % .10f\n", E0);
        if (!exact_hessian)
          printf("  The model Hessian is built from the density-density kernel "
                 "block alone,\n  and differs from the true one by %.2f%% "
                 "here. That costs iterations, not\n  correctness: every step "
                 "is still validated against the exact energy.\n",
                 100.0 * worst_hess);
        printf("%s\n", ok ? "  All derivatives agree."
                          : "  DERIVATIVES DISAGREE -- the second-order "
                            "optimizer is not trustworthy here.");
        fflush(stdout);
      }
      return ok;
    }

  } // namespace trscf
} // namespace helfem
