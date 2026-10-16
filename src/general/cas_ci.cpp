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
      require_hermitian_eri(ah.eri, n);

      std::vector<double> h(n * n);
      for (size_t p = 0; p < n; p++)
        for (size_t q = 0; q < n; q++)
          h[p * n + q] = ah.h_eff((Eigen::Index) p, (Eigen::Index) q);
      // chemist (tu|vw) -> reci's physicist <pq|rs> = (pr|qs)
      std::vector<double> V = swap_middle(ah.eri, n);

      // FourFold always: complex active orbitals lack (tu|vw) = (ut|vw), and
      // an EightFold declaration lets Knowles-Handy drop the antisymmetric
      // pair channel those integrals need. For real orbitals the channel
      // is merely zero, so FourFold costs time, never correctness.
      if (!impl_->sb) {
        impl_->sb = std::make_unique<reci::CISigmaBuilder<double>>(
            impl_->strings, ah.E_inactive, std::move(h), std::move(V),
            reci::ERISymmetry::FourFold, /*log_level=*/0);
        // Chosen explicitly rather than inherited from reci's defaults, so a
        // change of default there cannot silently change the CI here.
        impl_->sb->set_algorithm("kh");
        impl_->sb->set_memory(1024);
      } else {
        impl_->sb->set_Hamiltonian(ah.E_inactive, std::move(h), std::move(V),
                                   reci::ERISymmetry::FourFold);
      }
    }

    std::vector<double> CASCI::sigma(const std::vector<double> & c) const {
      if (c.size() != ndet())
        throw std::invalid_argument("CASCI::sigma: vector length is not ndet");
      std::vector<double> s(c.size());
      impl_->builder().Hamiltonian(c.data(), s.data(), c.size());
      return s;
    }

    double CASCI::solve_ground(std::vector<double> & civec, size_t max_dense) const {
      const auto & sb = impl_->builder();
      const size_t nc = ncsf(), nd = ndet();
      if (nc == 0)
        throw std::runtime_error("CASCI: no CSFs of the target spin");
      if (nc > max_dense)
        return davidson(civec);

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

    double CASCI::davidson(std::vector<double> & civec) const {
      const auto & sb = impl_->builder();
      const size_t nc = ncsf(), nd = ndet();
      const Eigen::Index n = (Eigen::Index) nc;
      const std::vector<double> hd = hdiag_csf();
      auto apply = [&](const helfem::Vector & x) {
        std::vector<double> xc(x.data(), x.data() + nc), det(nd), sig(nd), out(nc);
        impl_->spin.csf_C_to_det_C(xc, det);
        sb.Hamiltonian(det.data(), sig.data(), nd);
        impl_->spin.det_C_to_csf_C(sig, out);
        return helfem::Vector(Eigen::Map<helfem::Vector>(out.data(), n));
      };

      // Start from the CSF with the lowest diagonal element.
      helfem::Vector x = helfem::Vector::Zero(n);
      x((Eigen::Index) (std::min_element(hd.begin(), hd.end()) - hd.begin())) = 1.0;

      // Collapsed to the current and the previous Ritz vector when full, so
      // the subspace keeps the direction the iteration is moving in.
      const Eigen::Index maxdim = 24;
      const int maxit = 1000;
      const double tol = 1e-9;
      helfem::Matrix V(n, maxdim), AV(n, maxdim);
      V.col(0) = x;
      AV.col(0) = apply(x);
      Eigen::Index m = 1;
      helfem::Vector xprev = x;
      double theta = 0.0, rnorm = 0.0;
      for (int it = 0; it < maxit; it++) {
        const helfem::Matrix Hs = V.leftCols(m).transpose() * AV.leftCols(m);
        Eigen::SelfAdjointEigenSolver<helfem::Matrix> es(0.5 * (Hs + Hs.transpose()));
        theta = es.eigenvalues()(0);
        const helfem::Vector y = es.eigenvectors().col(0);
        x = V.leftCols(m) * y;
        helfem::Vector r = AV.leftCols(m) * y - theta * x;
        rnorm = r.norm();
        if (rnorm < tol) {
          civec.assign(nd, 0.0);
          std::vector<double> xc(x.data(), x.data() + nc);
          impl_->spin.csf_C_to_det_C(xc, civec);
          return theta;
        }
        // Diagonal correction, kept away from the pole.
        for (Eigen::Index i = 0; i < n; i++) {
          double d = theta - hd[(size_t) i];
          if (std::abs(d) < 1e-4) d = d < 0.0 ? -1e-4 : 1e-4;
          r(i) /= d;
        }
        if (m == maxdim) {
          helfem::Vector p = xprev - x * x.dot(xprev);
          V.col(0) = x;
          AV.col(0) = AV.leftCols(m) * y;
          m = 1;
          if (p.norm() > 1e-8) {
            p.normalize();
            V.col(1) = p;
            AV.col(1) = apply(p);
            m = 2;
          }
        }
        xprev = x;
        for (int pass = 0; pass < 2; pass++)
          r -= V.leftCols(m) * (V.leftCols(m).transpose() * r);
        const double nr = r.norm();
        if (nr < 1e-14)
          break;
        V.col(m) = r / nr;
        AV.col(m) = apply(V.col(m));
        m++;
      }
      throw std::runtime_error("CASCI: Davidson did not converge, residual "
                               + std::to_string(rnorm) + " after " + std::to_string(maxit)
                               + " iterations");
    }

    std::vector<double> CASCI::to_det(const std::vector<double> & csf) const {
      if (csf.size() != ncsf())
        throw std::invalid_argument("CASCI::to_det: vector length is not ncsf");
      std::vector<double> det(ndet());
      impl_->spin.csf_C_to_det_C(csf, det);
      return det;
    }

    std::vector<double> CASCI::to_csf(const std::vector<double> & det) const {
      if (det.size() != ndet())
        throw std::invalid_argument("CASCI::to_csf: vector length is not ndet");
      std::vector<double> csf(ncsf());
      impl_->spin.det_C_to_csf_C(det, csf);
      return csf;
    }

    std::vector<double> CASCI::hdiag_csf() const {
      impl_->builder();  // throws before the Hamiltonian is installed
      return impl_->sb->build_Hdiag_csf_exact(impl_->spin);
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
