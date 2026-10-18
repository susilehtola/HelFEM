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
#include "cas_ci.h"

// reci's headers stay in this translation unit: they need C++20 and Kokkos,
// and keeping them out of cas_ci.h keeps both requirements off every HelFEM
// translation unit that merely uses a CASCI.
#include <Kokkos_Core.hpp>
#include <ci_solver.h>

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>
#include <span>
#include <string>

namespace helfem {
  namespace cas {

    KokkosScope::KokkosScope() {
      if (!Kokkos::is_initialized() && !Kokkos::is_finalized()) {
        Kokkos::initialize();
        owner_ = true;
      }
    }

    KokkosScope::~KokkosScope() {
      if (owner_)
        Kokkos::finalize();
    }

    std::vector<double> swap_middle(const std::vector<double> & X, size_t n) {
      if (X.size() != n * n * n * n)
        throw std::logic_error("swap_middle: tensor is not n^4");
      std::vector<double> Y(X.size());
      for (size_t p = 0; p < n; p++)
        for (size_t q = 0; q < n; q++)
          for (size_t r = 0; r < n; r++)
            for (size_t s = 0; s < n; s++)
              Y[((p * n + r) * n + q) * n + s] = X[((p * n + q) * n + r) * n + s];
      return Y;
    }

    double eri_asymmetry(const std::vector<double> & eri, size_t n, int which) {
      double asym = 0.0;
      auto at = [&](size_t t, size_t u, size_t v, size_t w) {
        return eri[((t * n + u) * n + v) * n + w];
      };
      for (size_t t = 0; t < n; t++)
        for (size_t u = 0; u < n; u++)
          for (size_t v = 0; v < n; v++)
            for (size_t w = 0; w < n; w++) {
              const double a = at(t, u, v, w);
              const double b = which == 0 ? at(v, w, t, u)
                             : which == 1 ? at(u, t, w, v)
                                          : at(u, t, v, w);
              asym = std::max(asym, std::abs(a - b));
            }
      return asym;
    }

    void require_hermitian_eri(const std::vector<double> & eri, size_t n, double tol) {
      double scale = 0.0;
      for (double x : eri) scale = std::max(scale, std::abs(x));
      const double lim = tol * std::max(scale, 1.0);
      const double a0 = eri_asymmetry(eri, n, 0), a1 = eri_asymmetry(eri, n, 1);
      if (a0 > lim || a1 > lim)
        throw std::invalid_argument(
            "active-space integrals are not those of a real symmetric "
            "Hamiltonian: (tu|vw) = (vw|tu) off by " + std::to_string(a0)
            + ", (tu|vw) = (ut|wv) off by " + std::to_string(a1));
    }

    namespace {
      /// The M_L sector as reci sees it: m mod n in Z_n, with n larger than
      /// twice the largest |M_L| the electrons can reach, so that no two
      /// reachable M_L share a label.
      std::optional<CISymmetry> ml_sector(const std::vector<int> & m, int nalpha, int nbeta,
                                          int ML) {
        if (m.empty())
          return std::nullopt;
        std::vector<int> am;
        for (int x : m) am.push_back(std::abs(x));
        std::sort(am.rbegin(), am.rend());
        int Mmax = 0;
        for (int k = 0; k < nalpha; k++) Mmax += am[(size_t) k];
        for (int k = 0; k < nbeta; k++) Mmax += am[(size_t) k];
        const int n = 2 * std::max(Mmax, std::abs(ML)) + 1;
        auto label = [n](int x) { return (x % n + n) % n; };
        CISymmetry sym{reci::SymmetryGroup::cyclic((size_t) n), {}, label(ML)};
        for (int x : m) sym.orbital_irreps.push_back(label(x));
        return sym;
      }

      CISolverOptions solver_options() {
        // Chosen explicitly rather than inherited from reci's defaults, so a
        // change of default there cannot silently change the CI here.
        CISolverOptions o;
        o.solver = "davidson";
        o.sigma_algorithm = "kh";
        // The heat-bath guess, seeded inside the M_L sector: H never couples
        // two sectors, so the seed root's M_L is the answer's.
        o.guess = "hci";
        o.guess_subspace = 1000;
        o.hci_eps = 1e-3;
        o.hci_seed = 200;
        o.tol = 1e-8;
        o.max_iter = 500;
        o.memory_mb = 1024;
        return o;
      }
    } // namespace

    struct CASCI::Impl {
      size_t nact;
      // FourFold always: complex active orbitals lack (tu|vw) = (ut|vw), and
      // an EightFold declaration lets Knowles-Handy drop the antisymmetric
      // pair channel those integrals need. For real orbitals the channel
      // is merely zero, so FourFold costs time, never correctness.
      CISolver solver;
      bool have_h = false;

      Impl(int nact_, int na, int nb, const std::vector<int> & m, int ML)
          : nact((size_t) nact_),
            solver((size_t) nact_, (size_t) na, (size_t) nb, na - nb,
                   reci::ERISymmetry::FourFold, solver_options(), ml_sector(m, na, nb, ML)) {}

      const reci::CISigmaBuilder<double> & builder() const {
        if (!have_h)
          throw std::logic_error("CASCI: set_hamiltonian has not been called");
        return solver.sigma_builder();
      }
    };

    CASCI::CASCI(int nact, int nalpha, int nbeta, const std::vector<int> & m, int ML) {
      if (!Kokkos::is_initialized())
        throw std::logic_error("CASCI: Kokkos is not initialized; hold a "
                               "helfem::cas::KokkosScope in main()");
      if (nalpha < nbeta)
        throw std::invalid_argument("CASCI: nalpha < nbeta; the target is M_S = S >= 0");
      if (nact < 1 || nalpha > nact || nbeta > nact)
        throw std::invalid_argument("CASCI: electrons do not fit the active space");
      if (!m.empty() && m.size() != (size_t) nact)
        throw std::invalid_argument("CASCI: give the m of every active orbital, or none");
      impl_ = std::make_unique<Impl>(nact, nalpha, nbeta, m, ML);
      if (ncsf() == 0)
        throw std::invalid_argument("CASCI: no configuration of the active electrons has M_L = "
                                    + std::to_string(ML));
    }

    CASCI::~CASCI() = default;

    size_t CASCI::ndet() const { return impl_->solver.ndet(); }
    size_t CASCI::ncsf() const { return impl_->solver.ncsf(); }

    void CASCI::set_hamiltonian(const ActiveHamiltonian & ah) {
      const size_t n = impl_->nact;
      if ((size_t) ah.h_eff.rows() != n || (size_t) ah.h_eff.cols() != n
          || ah.eri.size() != n * n * n * n)
        throw std::invalid_argument("CASCI: Hamiltonian does not match the active space");
      require_hermitian_eri(ah.eri, n);

      std::vector<double> h(n * n);
      for (size_t p = 0; p < n; p++)
        for (size_t q = 0; q < n; q++)
          h[p * n + q] = ah.h_eff((Eigen::Index) p, (Eigen::Index) q);
      // chemist (tu|vw) -> reci's physicist <pq|rs> = (pr|qs)
      impl_->solver.set_Hamiltonian(ah.E_inactive, std::move(h), swap_middle(ah.eri, n));
      impl_->have_h = true;
    }

    std::vector<double> CASCI::sigma(const std::vector<double> & c) const {
      if (c.size() != ndet())
        throw std::invalid_argument("CASCI::sigma: vector length is not ndet");
      std::vector<double> s(c.size());
      impl_->builder().Hamiltonian(c.data(), s.data(), c.size());
      return s;
    }

    double CASCI::solve_ground(std::vector<double> & civec) const {
      impl_->builder();  // throws before the Hamiltonian is installed
      // reci's solver, with its own guess, preconditioner and eigensolver;
      // the CI problem is solved there and nowhere here.
      const CISolveResult r = impl_->solver.solve(1);
      if (!r.converged || r.energies.empty())
        throw std::runtime_error("CASCI: reci's CI solver did not converge");

      civec = to_det(r.csf_vectors);
      const double nrm = std::sqrt(std::inner_product(civec.begin(), civec.end(), civec.begin(), 0.0));
      for (double & x : civec) x /= nrm;
      const std::vector<double> sg = sigma(civec);
      const double E = std::inner_product(civec.begin(), civec.end(), sg.begin(), 0.0);
      if (std::abs(E - r.energies[0]) > 1e-8 * std::max(1.0, std::abs(E)))
        throw std::logic_error("CASCI: reci's eigenvector does not reproduce its eigenvalue");
      return E;
    }

    std::vector<double> CASCI::to_det(const std::vector<double> & csf) const {
      if (csf.size() != ncsf())
        throw std::invalid_argument("CASCI::to_det: vector length is not ncsf");
      return impl_->solver.csf_to_det(csf);
    }

    std::vector<double> CASCI::to_csf(const std::vector<double> & det) const {
      if (det.size() != ndet())
        throw std::invalid_argument("CASCI::to_csf: vector length is not ndet");
      return impl_->solver.det_to_csf(det);
    }

    std::vector<double> CASCI::hdiag_csf() const {
      impl_->builder();  // throws before the Hamiltonian is installed
      return impl_->solver.sigma_builder().build_Hdiag_csf_exact(impl_->solver.spin_adapter());
    }

    RDMs CASCI::rdms(const std::vector<double> & L, const std::vector<double> & R) const {
      const auto & sb = impl_->builder();
      if (L.size() != ndet() || R.size() != ndet())
        throw std::invalid_argument("CASCI::rdms: vector length is not ndet");
      // reci takes mutable spans; work on copies so the callers' vectors are
      // const here as they are in fact.
      std::vector<double> l(L), r(R);
      const size_t n = impl_->nact;

      const std::vector<double> g1 = sb.compute_sf_1rdm(std::span(l), std::span(r));
      RDMs out;
      out.D.resize((Eigen::Index) n, (Eigen::Index) n);
      for (size_t p = 0; p < n; p++)
        for (size_t q = 0; q < n; q++)
          out.D((Eigen::Index) p, (Eigen::Index) q) = g1[p * n + q];

      // reci's G[p,q,r,s] = <a+p a+q a_s a_r>; chemist d[t,u,v,w] =
      // <a+t a+v a_w a_u> = G[t,v,u,w], the same middle swap as the integrals.
      out.d = swap_middle(sb.compute_sf_2rdm(std::span(l), std::span(r)), n);
      return out;
    }

  } // namespace cas
} // namespace helfem
