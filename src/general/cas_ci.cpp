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
#include <reci/ci_sigma_builder.h>
#include <reci/ci_spin_adapter.h>
#include <reci/ci_strings.h>

#include <Eigen/Eigenvalues>

#include <cmath>
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

    void require_real_eri(const std::vector<double> & eri, size_t n, double tol) {
      double scale = 0.0, asym = 0.0;
      for (size_t t = 0; t < n; t++)
        for (size_t u = 0; u < n; u++)
          for (size_t v = 0; v < n; v++)
            for (size_t w = 0; w < n; w++) {
              const double a = eri[((t * n + u) * n + v) * n + w];
              const double b = eri[((u * n + t) * n + v) * n + w];
              scale = std::max(scale, std::abs(a));
              asym = std::max(asym, std::abs(a - b));
            }
      if (asym > tol * std::max(scale, 1.0))
        throw ComplexOrbitalError(
            "active-space integrals violate (tu|vw) = (ut|vw) by "
            + std::to_string(asym) + ": an active orbital is complex (m != 0 "
            "in the atomic basis), which the real CI in reci cannot represent");
    }

    struct CASCI::Impl {
      size_t nact;
      int twoS;
      reci::CIStrings strings;
      reci::CISpinAdapter<double> spin;
      // Built at the first set_hamiltonian: CISigmaBuilder takes integrals at
      // construction, and holds a reference to `strings`, which is why Impl
      // owns both and is never moved.
      std::unique_ptr<reci::CISigmaBuilder<double>> sb;

      Impl(int nact_, int na, int nb)
          : nact((size_t) nact_), twoS(na - nb),
            strings((size_t) na, (size_t) nb, (size_t) nact_),
            spin(na - nb, na - nb, nact_) {
        spin.prepare_couplings(strings);
      }

      const reci::CISigmaBuilder<double> & builder() const {
        if (!sb)
          throw std::logic_error("CASCI: set_hamiltonian has not been called");
        return *sb;
      }
    };

    CASCI::CASCI(int nact, int nalpha, int nbeta) {
      if (!Kokkos::is_initialized())
        throw std::logic_error("CASCI: Kokkos is not initialized; hold a "
                               "helfem::cas::KokkosScope in main()");
      if (nalpha < nbeta)
        throw std::invalid_argument("CASCI: nalpha < nbeta; the target is M_S = S >= 0");
      if (nact < 1 || nalpha > nact || nbeta > nact)
        throw std::invalid_argument("CASCI: electrons do not fit the active space");
      impl_ = std::make_unique<Impl>(nact, nalpha, nbeta);
    }

    CASCI::~CASCI() = default;

    size_t CASCI::ndet() const { return impl_->strings.ndet(); }
    size_t CASCI::ncsf() const { return impl_->spin.ncsf(); }

    void CASCI::set_hamiltonian(const ActiveHamiltonian & ah) {
      const size_t n = impl_->nact;
      if ((size_t) ah.h_eff.rows() != n || (size_t) ah.h_eff.cols() != n
          || ah.eri.size() != n * n * n * n)
        throw std::invalid_argument("CASCI: Hamiltonian does not match the active space");
      require_real_eri(ah.eri, n);

      std::vector<double> h(n * n);
      for (size_t p = 0; p < n; p++)
        for (size_t q = 0; q < n; q++)
          h[p * n + q] = ah.h_eff((Eigen::Index) p, (Eigen::Index) q);
      // chemist (tu|vw) -> reci's physicist <pq|rs> = (pr|qs)
      std::vector<double> V = swap_middle(ah.eri, n);

      if (!impl_->sb) {
        impl_->sb = std::make_unique<reci::CISigmaBuilder<double>>(
            impl_->strings, ah.E_inactive, std::move(h), std::move(V), /*log_level=*/0);
        // Chosen explicitly rather than inherited from reci's defaults, so a
        // change of default there cannot silently change the CI here.
        impl_->sb->set_algorithm("kh");
        impl_->sb->set_memory(1024);
      } else {
        impl_->sb->set_Hamiltonian(ah.E_inactive, std::move(h), std::move(V));
      }
    }

    std::vector<double> CASCI::sigma(const std::vector<double> & c) const {
      if (c.size() != ndet())
        throw std::invalid_argument("CASCI::sigma: vector length is not ndet");
      std::vector<double> s(c.size());
      impl_->builder().Hamiltonian(c.data(), s.data(), c.size());
      return s;
    }

    double CASCI::solve_ground(std::vector<double> & civec, size_t max_csf) const {
      const auto & sb = impl_->builder();
      const size_t nc = ncsf(), nd = ndet();
      if (nc == 0)
        throw std::runtime_error("CASCI: no CSFs of the target spin");
      if (nc > max_csf)
        throw std::runtime_error("CASCI: " + std::to_string(nc) + " CSFs exceed the "
                                 "dense solver's limit of " + std::to_string(max_csf));

      // H in the CSF basis, one column per CSF: CSF -> determinants -> sigma
      // -> back to CSFs. The CSFs are an orthonormal combination of
      // determinants, so this is T^T H T and symmetric up to roundoff.
      helfem::Matrix Hc((Eigen::Index) nc, (Eigen::Index) nc);
      std::vector<double> e(nc, 0.0), det(nd), sig(nd), col(nc);
      for (size_t k = 0; k < nc; k++) {
        e[k] = 1.0;
        impl_->spin.csf_C_to_det_C(e, det);
        sb.Hamiltonian(det.data(), sig.data(), nd);
        impl_->spin.det_C_to_csf_C(sig, col);
        for (size_t j = 0; j < nc; j++)
          Hc((Eigen::Index) j, (Eigen::Index) k) = col[j];
        e[k] = 0.0;
      }
      const double asym = (Hc - Hc.transpose()).cwiseAbs().maxCoeff();
      if (asym > 1e-9 * std::max(1.0, Hc.cwiseAbs().maxCoeff()))
        throw std::runtime_error("CASCI: CSF-basis Hamiltonian is not symmetric ("
                                 + std::to_string(asym) + ")");

      Eigen::SelfAdjointEigenSolver<helfem::Matrix> es(0.5 * (Hc + Hc.transpose()));
      std::vector<double> c0(nc);
      for (size_t j = 0; j < nc; j++)
        c0[j] = es.eigenvectors()((Eigen::Index) j, 0);
      civec.assign(nd, 0.0);
      impl_->spin.csf_C_to_det_C(c0, civec);
      return es.eigenvalues()(0);
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
