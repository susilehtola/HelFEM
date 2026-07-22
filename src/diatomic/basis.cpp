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
#include "basis.h"
#include "quadrature.h"
#include "PolynomialBasis.h"
#include "chebyshev.h"
#include "lobatto.h"
#include <Eigen/Eigenvalues>
#include "../general/angular_index_helpers.h"
#include <cstring>
#include "../general/spherical_harmonics.h"
#include "../general/gaunt.h"
#include "../general/gsz.h"
#include "utils.h"
#include "../general/timer.h"
#include "../general/scf_helpers.h"
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <complex>
#include <limits>
#include <map>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace helfem {
  namespace diatomic {
    namespace basis {

      // --------------------------------------------------------------------
      // Auto-converging quadrature for the in-element two-electron kernel.
      //
      // Mirrors the atomic Stage-1 helper (libhelfem/src/RadialBasis.cpp,
      // converge_rule): recompute a probe block at a rising quadrature order
      // until it stops changing, so the accuracy of the two-electron radial
      // integrals is set by the floating-point type (here double's eps)
      // instead of a user-supplied --nquad. The diatomic path is double-only,
      // so this is a plain double specialization rather than a template.
      //
      // The probe builds its own rule, and the FAMILY is chosen per
      // integrand class (see libhelfem/src/RadialBasis.cpp for the full
      // analysis of why this matters):
      //   * Gauss-Lobatto for the analytic blocks -- radial_integral's
      //     sinh^m cosh^n weights, kinetic's sinh weight, and the
      //     cross-basis overlap projection. These integrands are entire (or
      //     polynomial x entire), so a polynomial-exact rule converges
      //     geometrically and exit (1) fires after a doubling or two. The
      //     modified Gauss-Chebyshev rule, being the trapezoid rule under
      //     the sin^4-Jacobian Perez-Jorda transformation, has a FIXED
      //     Euler-Maclaurin order O(n^-10) no matter how smooth the
      //     integrand -- ample for double, but needlessly slow to refine.
      //   * Gauss-Chebyshev for the P_L/Q_L Green's-function blocks
      //     (Plm/Qlm integrals, the in-element two-electron kernel). Over
      //     the element touching mu=0 the Q_L(cosh mu) weight is endpoint-
      //     singular (Q_L(1) = +inf), which a Lobatto endpoint node would
      //     evaluate directly, and for endpoint-singular integrands the
      //     Chebyshev rule's sin^4 node clustering is the right treatment
      //     anyway.
      //
      // TWO stopping conditions, both meaning "all of double's precision has
      // been extracted":
      //   (1) True eps convergence: the block stops changing to 8*eps, the
      //       same criterion FiniteElementBasis's 1e matrix_element uses.
      //   (2) Roundoff-floor stall: the 2e Coulomb P_L(cosh mu_<)Q_L(cosh mu_>)
      //       Green's function is NOT polynomial-exact, so (unlike the 1e
      //       Gauss-Lobatto block) the block difference converges only down to
      //       the assembly roundoff floor and then wobbles rather than
      //       collapsing to zero. Once it is deep in the asymptotic regime
      //       (diff <= sqrt(eps)*(scale+tol)) and a doubling no longer at least
      //       halves it (diff > 0.5*prevdiff), it has converged: return
      //       quietly. Without this every production element would grind to the
      //       cap and print a spurious warning.
      //
      // nmax cap = 512 with a one-shot warning is a genuine backstop, not the
      // common path; the start order is seeded from --nquad so the common case
      // converges in 1-2 steps.
      // --------------------------------------------------------------------
      namespace {
        /// Order cap for the refinement loop.
        const int twoe_nmax = 512;
        /// Warn at most once per process if the refinement hits the order cap.
        bool twoe_cap_warned = false;

        /// Refine `probe(n)` -- which must rebuild its block using its own
        /// n-point Gauss-Chebyshev rule -- by doubling n from nstart until the
        /// block is stable, and return the converged block. If `nconv` is
        /// non-null it receives the (finer) converged order, so a caller can
        /// rebuild a heavier object (e.g. a TwoElectronElement) once at that
        /// order. The block's shape is independent of n, so the
        /// block-difference comparison is well defined.
        ///
        /// `seed_fallback` selects what happens if the order cap is reached
        /// without convergence. The disjoint Q_L integral over the element that
        /// touches mu=0 is genuinely NON-convergent for |M| >= 2 -- there
        /// Q_{L,|M|}(cosh mu) ~ mu^{-|M|}, so the bare integral (without the
        /// companion P_{L,|M|} ~ mu^{|M|} that regularizes it inside the
        /// in-element kernel) diverges. That block is never actually used (the
        /// innermost element is never the OUTER element of a disjoint pair), so
        /// forcing it is pointless. With seed_fallback the routine then returns
        /// the block at the seed order -- exactly the fixed --nquad value the
        /// pre-auto-convergence code produced -- quietly. Without it (the
        /// in-element kernel, which IS convergent) a cap is a real anomaly and
        /// gets the one-shot warning + best estimate.
        template <typename Fn>
        helfem::Matrix converge_block(const Fn & probe, int nstart,
                                      const char * what, int * nconv = nullptr,
                                      bool seed_fallback = false) {
          const double eps     = std::numeric_limits<double>::epsilon();
          const double tol     = 8.0 * eps;
          const double sqrteps = std::sqrt(eps);
          // Machine-precision floor. These blocks are not polynomial-exact and
          // the LIP assembly carries a roundoff floor of a small multiple of
          // eps (empirically ~1e-15 relative), so the block difference
          // plateaus there rather than collapsing to 8*eps. 256*eps ~ 5.7e-14
          // is comfortably above that floor yet far below both the cd_thresh
          // (1e-12) the kernel is later factorized to and the 1e-10 total-
          // energy tolerance: once the difference reaches it, all of double's
          // precision has been extracted.
          const double floor_rel = 256.0 * eps;

          helfem::Matrix prev, cur, seed;
          bool have = false;
          double prevdiff = -1.0;
          int n = std::max(nstart, 2);
          for(;;) {
            cur = probe(n);
            if(!have)
              seed = cur;   // the fixed --nquad value, for seed_fallback
            if(have) {
              const double diff  = (cur - prev).cwiseAbs().maxCoeff();
              const double scale = cur.cwiseAbs().maxCoeff();
              // (1) true eps convergence (well-conditioned, polynomial-exact
              // blocks reach this).
              if(diff <= tol * (scale + tol)) {
                if(nconv) *nconv = n;
                return cur;
              }
              // (2) roundoff-floor stall: the Coulomb P_L(cosh mu_<)Q_L(cosh
              // mu_>) Green's function is not polynomial-exact, and over the
              // element that touches mu=0 the Q_L weight is only C^0 (a
              // mu*ln(mu) endpoint kink, Q_L(1)=+inf). So the block difference
              // converges only down to the assembly roundoff floor and then
              // wobbles. Once it is deep in the asymptotic regime
              // (diff <= sqrt(eps)*scale) AND it has either reached the
              // machine-precision floor (diff <= floor_rel*scale) or a doubling
              // no longer at least halves it, all of double's precision has
              // been extracted: stop quietly. The floor test makes this robust
              // when only a couple of doublings separate the seed from the cap.
              if(diff <= sqrteps * (scale + tol) &&
                 (diff <= floor_rel * (scale + tol) ||
                  (prevdiff >= 0.0 && diff > 0.5 * prevdiff))) {
                if(nconv) *nconv = n;
                return cur;
              }
              prevdiff = diff;
            }
            prev = cur;
            have = true;
            if(n >= twoe_nmax) {
              if(seed_fallback) {
                // Non-convergent integrand (the divergent innermost-element Q_L
                // for |M| >= 2). Fall back to the requested --nquad order
                // quietly -- the same value the fixed-order code produced.
                if(nconv) *nconv = std::max(nstart, 2);
                return seed;
              }
              if(!twoe_cap_warned) {
                twoe_cap_warned = true;
                printf("Warning: diatomic %s hit the quadrature order cap"
                       " (n=%d) without converging to eps(double); using best"
                       " estimate.\n", what, twoe_nmax);
                fflush(stdout);
              }
              if(nconv) *nconv = n;
              return cur;
            }
            n = std::min(2 * n, twoe_nmax);
          }
        }
      }

      RadialBasis::RadialBasis() {
      }

      RadialBasis::RadialBasis(const polynomial_basis::FiniteElementBasis & fem_, int n_quad) : fem(fem_) {
        // Get quadrature rule
        chebyshev::chebyshev<double>(n_quad,xq,wq);
        for(Eigen::Index i=0;i<xq.size();i++) {
          if(!std::isfinite(xq[i]))
            printf("xq[%i]=%e\n",(int) i, xq[i]);
          if(!std::isfinite(wq[i]))
            printf("wq[%i]=%e\n",(int) i, wq[i]);
        }
      }

      RadialBasis::~RadialBasis() {
      }

      int RadialBasis::get_nquad() const {
        return (int) xq.size();
      }

      helfem::Vector RadialBasis::get_bval() const {
        return fem.get_bval();
      }

      int RadialBasis::get_poly_id() const {
        return fem.get_poly_id();
      }

      int RadialBasis::get_poly_nnodes() const {
        return fem.get_poly_nnodes();
      }

      size_t RadialBasis::Nel() const {
        // Number of elements is
        return fem.get_nelem();
      }

      size_t RadialBasis::Nbf() const {
        // Number of basis functions is Nbf*Nel - (Nel-1)*Noverlap - Noverlap
        return fem.get_nbf();
      }

      size_t RadialBasis::Nprim(size_t iel) const {
	return fem.get_nprim(iel);
      }

      size_t RadialBasis::max_Nprim() const {
        return fem.get_max_nprim();
      }

      void RadialBasis::get_idx(size_t iel, size_t & ifirst, size_t & ilast) const {
        fem.get_idx(iel, ifirst, ilast);
      }

      helfem::Matrix RadialBasis::radial_integral(int m, int n) const {
        std::function<double(double)> chsh;
        if(m!=0 && n!=0) {
          chsh = [m, n](double mu) { return std::pow(std::sinh(mu), m)*std::pow(std::cosh(mu), n); };
        } else if(m!=0 && n==0) {
          chsh = [m](double mu) { return std::pow(std::sinh(mu), m); };
        } else if(m==0 && n!=0) {
          chsh = [n](double mu) { return std::pow(std::cosh(mu), n); };
        }
        // Auto-converging: refine the quadrature order until the block is
        // stable to eps(double), so accuracy follows the FP type instead of
        // the caller's --nquad. The sinh/cosh weight is entire, so the
        // polynomial-exact Gauss-Lobatto rule converges geometrically (see
        // the family note atop converge_block) and no seed fallback is
        // needed.
        return converge_block(
            [&](int nq) {
              helfem::Vector x, w;
              lobatto::lobatto_compute<double>(nq, x, w);
              return fem.matrix_element(false, false, x, w, chsh);
            },
            std::max((int) xq.size(), 5), "radial_integral");
      }

      helfem::Matrix RadialBasis::overlap(const RadialBasis & rh, int n) const {
        // Find element pairs that share any mu range (independent of the
        // quadrature order).
        std::vector<std::vector<size_t>> overlap(fem.get_nelem());
        for(size_t iel=0;iel<fem.get_nelem();iel++) {
          double istart(fem.element_begin(iel));
          double iend(fem.element_end(iel));
          for(size_t jel=0;jel<rh.fem.get_nelem();jel++) {
            double jstart(rh.fem.element_begin(jel));
            double jend(rh.fem.element_end(jel));
            if((jstart >= istart && jstart<iend) || (istart >= jstart && istart < jend))
              overlap[iel].push_back(jel);
          }
        }

        // Auto-converging projection: refine the quadrature order until the
        // whole (Nbf x rh.Nbf) block is stable to eps(double), so the accuracy
        // follows the FP type rather than either basis's --nquad. Seeded from
        // the finer of the two bases. The integrand (polynomials x the entire
        // sinh cosh^n weight) is analytic, so the Gauss-Lobatto rule converges
        // geometrically (see the family note atop converge_block).
        return converge_block(
            [&](int nq) {
              helfem::Vector xproj, wproj;
              lobatto::lobatto_compute<double>(nq, xproj, wproj);

              helfem::Matrix S(helfem::Matrix::Zero(Nbf(), rh.Nbf()));
              for(size_t iel=0;iel<fem.get_nelem();iel++) {
                for(size_t jj=0;jj<overlap[iel].size();jj++) {
                  size_t jel=overlap[iel][jj];
                  // FE-side element ranges.
                  double imin(fem.element_begin(iel));
                  double imax(fem.element_end(iel));
                  double jmin(rh.fem.element_begin(jel));
                  double jmax(rh.fem.element_end(jel));
                  // Shared range.
                  double intstart(std::max(imin, jmin));
                  double intend(std::min(imax, jmax));
                  double intmid(0.5*(intend+intstart));
                  double intlen(0.5*(intend-intstart));

                  // The Lobatto rule includes x = +-1, whose mapped image
                  // intmid +- intlen can round a few ulps past the exact
                  // interval ends and trip eval_prim's bounds check; clamp
                  // to the interval, whose ends are exact element
                  // boundaries of one basis and interior to the other.
                  helfem::Vector mu((intmid*helfem::Vector::Ones(xproj.size())+intlen*xproj)
                                        .cwiseMax(intstart).cwiseMin(intend));
                  helfem::Vector xi(fem.eval_prim(mu, iel));
                  helfem::Vector xj(rh.fem.eval_prim(mu, jel));

                  size_t ifirst, ilast;
                  get_idx(iel, ifirst, ilast);
                  size_t jfirst, jlast;
                  rh.get_idx(jel, jfirst, jlast);

                  helfem::Vector wtot(wproj*intlen);
                  wtot.array() *= mu.array().sinh();
                  if(n!=0) wtot.array() *= mu.array().cosh().pow((double) n);
                  helfem::Matrix ibf(fem.eval_dnf(xi, 0, iel));
                  helfem::Matrix jbf(rh.fem.eval_dnf(xj, 0, jel));
                  helfem::Matrix s(ibf.transpose()*wtot.asDiagonal()*jbf);
                  S.block(ifirst, jfirst, ilast-ifirst+1, jlast-jfirst+1) += s;
                }
              }
              return S;
            },
            std::max({(int) xq.size(), (int) rh.xq.size(), 5}), "overlap");
      }

      helfem::Matrix RadialBasis::Plm_integral(int k, size_t iel, int L, int M, const legendretable::LegendreTable & legtab) const {
        std::function<double(double)> Plm;
        if(k!=0) {
          Plm = [legtab, k, L, M](double mu) { return std::sinh(mu)*std::pow(std::cosh(mu), k)*legtab.get_Plm(L,M,cosh(mu)); };
        } else {
          Plm = [legtab, L, M](double mu) { return std::sinh(mu)*legtab.get_Plm(L,M,cosh(mu)); };
        }
        // Auto-converging disjoint (cross-element) 2e integral. The Gauss-
        // Chebyshev order is refined until the block is stable, so the accuracy
        // follows the FP type instead of the caller's --nquad. We keep the
        // Gauss-Chebyshev rule (not the FE's Gauss-Lobatto auto-overload)
        // deliberately: over the element that touches mu=0 the companion Q_L
        // weight is singular there (Q_L(1)=+inf), and Chebyshev's endpoint
        // clustering handles that far better than Lobatto's endpoint nodes.
        // converge_block carries the roundoff-floor stall that lets that
        // element converge quietly rather than grind to the order cap.
        return converge_block(
            [&](int n) {
              helfem::Vector x, w;
              chebyshev::chebyshev<double>(n, x, w);
              return fem.matrix_element(iel, false, false, x, w, Plm);
            },
            std::max((int) xq.size(), 5), "Plm_integral", nullptr, true);
      }

      helfem::Matrix RadialBasis::Qlm_integral(int k, size_t iel, int L, int M, const legendretable::LegendreTable & legtab) const {
        std::function<double(double)> Qlm;
        if(k!=0) {
          Qlm = [legtab, k, L, M](double mu) { return std::sinh(mu)*std::pow(std::cosh(mu), k)*legtab.get_Qlm(L,M,cosh(mu)); };
        } else {
          Qlm = [legtab, L, M](double mu) { return std::sinh(mu)*legtab.get_Qlm(L,M,cosh(mu)); };
        }
        // Auto-converging disjoint 2e integral (see Plm_integral above). This
        // is the branch whose weight Q_L(cosh mu) is singular on the mu=0
        // element -- the Chebyshev rule + roundoff-floor stall handle it.
        return converge_block(
            [&](int n) {
              helfem::Vector x, w;
              chebyshev::chebyshev<double>(n, x, w);
              return fem.matrix_element(iel, false, false, x, w, Qlm);
            },
            std::max((int) xq.size(), 5), "Qlm_integral", nullptr, true);
      }

      helfem::Matrix RadialBasis::kinetic() const {
        std::function<double(double)> sinhmu = [](double mu) {return std::sinh(mu);};
        // Auto-converging (see radial_integral): refine the Gauss-Lobatto
        // order until the derivative block is stable to eps(double); the
        // sinh weight is entire, so the refinement is geometric.
        return converge_block(
            [&](int nq) {
              helfem::Vector x, w;
              lobatto::lobatto_compute<double>(nq, x, w);
              return fem.matrix_element(true, true, x, w, sinhmu);
            },
            std::max((int) xq.size(), 5), "kinetic");
      }

      helfem::Matrix RadialBasis::twoe_integral(int alpha, int beta, size_t iel, int L, int M, const legendretable::LegendreTable & legtab) const {
        double mumin=fem.element_begin(iel);
        double mumax=fem.element_end(iel);

        // Integral by quadrature
        std::shared_ptr<const helfem::polynomial_basis::PolynomialBasis> p(fem.get_basis(iel));
        helfem::Matrix tei(quadrature::twoe_integral(mumin,mumax,alpha,beta,xq,wq,p,L,M,legtab));

        return tei;
      }

      quadrature::TwoElectronElement RadialBasis::twoe_element(size_t iel) const {
        // Build at the stored quadrature order.
        return twoe_element(iel, (int) xq.size());
      }

      quadrature::TwoElectronElement RadialBasis::twoe_element(size_t iel, int n) const {
        // Build at a specified Gauss-Chebyshev order: a fresh n-point rule
        // instead of the stored (xq, wq). This is what lets compute_tei refine
        // the in-element two-electron kernel until it is converged to
        // eps(double).
        helfem::Vector x, w;
        chebyshev::chebyshev<double>(n, x, w);
        std::shared_ptr<const helfem::polynomial_basis::PolynomialBasis> p(fem.get_basis(iel));
        return quadrature::twoe_element(fem.element_begin(iel), fem.element_end(iel), x, w, p);
      }

      helfem::Matrix RadialBasis::twoe_integral(int alpha, int beta, const quadrature::TwoElectronElement & el, int L, int M, const legendretable::LegendreTable & legtab) const {
        return quadrature::twoe_integral(el, alpha, beta, L, M, legtab);
      }

      helfem::Vector RadialBasis::get_chmu_quad() const {
        // Quadrature points for normal integrals
        helfem::Vector muq(fem.get_nelem()*xq.size()*(xq.size()+1));
        size_t ioff=0;

        for(size_t iel=0;iel<fem.get_nelem();iel++) {
          // Element ranges from
          double mumin0=fem.element_begin(iel);
          double mumax0=fem.element_end(iel);

          // Midpoint is at
          double mumid0(0.5*(mumax0+mumin0));
          // and half-length of interval is
          double mulen0(0.5*(mumax0-mumin0));
          // mu values are then
          helfem::Vector mu0(mumid0*helfem::Vector::Ones(xq.size())+mulen0*xq);

          // Store values
          muq.segment(ioff,mu0.size())=mu0;
          ioff+=mu0.size();

          // Subintervals for in-element two-electron integrals
          for(Eigen::Index isub=0;isub<xq.size();isub++) {
            double mumin = (isub==0) ? mumin0 : mu0(isub-1);
            double mumax = mu0(isub);

            double mumid(0.5*(mumax+mumin));
            double mulen(0.5*(mumax-mumin));
            helfem::Vector mu(mumid*helfem::Vector::Ones(xq.size())+mulen*xq);
            muq.segment(ioff,mu.size())=mu;
            ioff+=mu.size();
          }
        }

        // Sort ascending, then take cosh
        std::sort(muq.data(), muq.data()+muq.size());
        return muq.array().cosh().matrix();
      }

      helfem::Matrix RadialBasis::get_bf(size_t iel) const {
        return get_bf(iel, xq);
      }

      helfem::Matrix RadialBasis::get_bf(size_t iel, const helfem::Vector & x) const {
        return fem.eval_dnf(x, 0, iel);
      }

      helfem::Matrix RadialBasis::get_df(size_t iel) const {
        return fem.eval_dnf(xq, 1, iel);
      }

      helfem::Matrix RadialBasis::get_d2f(size_t iel) const {
        return fem.eval_dnf(xq, 2, iel);
      }

      helfem::Vector RadialBasis::get_wrad(size_t iel) const {
        // This is just the radial rule, no r^2 factor included here
        return fem.scaling_factor(iel)*wq;
      }

      helfem::Vector RadialBasis::get_r(size_t iel) const {
        return fem.eval_coord(xq, iel);
      }

      void lm_to_l_m(const Eigen::VectorXi & lmax, Eigen::VectorXi & lval, Eigen::VectorXi & mval) {
        {
          std::vector<int> lv, mv;
          for(int mabs=0;mabs<lmax.size();mabs++)
            for(int l=mabs;l<=lmax(mabs);l++) {
              lv.push_back(l);
              mv.push_back(mabs);
              if(mabs>0) {
                lv.push_back(l);
                mv.push_back(-mabs);
              }
            }
          lval=Eigen::Map<const Eigen::VectorXi>(lv.data(), (Eigen::Index) lv.size());
          mval=Eigen::Map<const Eigen::VectorXi>(mv.data(), (Eigen::Index) mv.size());
        }
      }

      TwoDBasis::TwoDBasis() {
      }

      TwoDBasis::TwoDBasis(int Z1_, int Z2_, double Rhalf_, const std::shared_ptr<const polynomial_basis::PolynomialBasis> &poly, int n_quad, const helfem::Vector & bval, const Eigen::VectorXi & lval_, const Eigen::VectorXi & mval_, bool legendre) {
        // Nuclear charge
        Z1=Z1_;
        Z2=Z2_;
        Rhalf=Rhalf_;

        // Construct radial basis
        bool zero_func_left=false; // sigma orbitals are allowed to reach the nucleus; this is cleaned up for non-sigma orbitals elsewhere in the code
        bool zero_deriv_left=false;
        bool zero_func_right=true;
        bool zero_deriv_right=true;
        polynomial_basis::FiniteElementBasis fem(poly, bval, zero_func_left, zero_deriv_left, zero_func_right, zero_deriv_right);
        radial=RadialBasis(fem, n_quad);
        // Angular basis.
        lval=lval_;
        mval=mval_;

        // Gaunt coefficients
        int gmax(lval.maxCoeff()+2);

        // Legendre function values
        if(legendre) {
          int Lmax=0;
          int Mmax=0;

          // Form L|M| and LM maps
          lm_map.clear();
          LM_map.clear();
          for(size_t iang=0;iang<lval.size();iang++) {
            for(size_t jang=0;jang<lval.size();jang++) {
              // l and m values
              int li(lval(iang));
              int mi(mval(iang));
              int lj(lval(jang));
              int mj(mval(jang));
              // LH m value
              int M(mj-mi);

              int Lstart=std::max(std::abs(lj-li)-2,abs(M));
              int Lend=lj+li+2;
              for(int L=Lstart;L<=Lend;L++) {
                lmidx_t p;
                p.first=L;

                // Check maxima
                Lmax=std::max(Lmax,L);
                Mmax=std::max(Mmax,std::abs(M));

                // L|M|. lmind returns idx == lm_map.size() when the
                // requested (L, |M|) sorts AFTER every existing entry
                // (e.g. right after the first push_back). Do not touch
                // lm_map[idx] in that case -- under GCC 16's hardened
                // std::vector::operator[] that is a hard abort; under
                // older libstdc++ it silently returned garbage that
                // happened to compare unequal to p, so the map was
                // built correctly by luck.
                p.second=std::abs(M);
                if(!lm_map.size()) {
                  lm_map.push_back(p);
                } else {
                  size_t idx=lmind(L,M,false);
                  if (idx == lm_map.size())
                    lm_map.push_back(p);
                  else if (!(lm_map[idx]==p))
                    lm_map.insert(lm_map.begin()+idx,p);
                }

                // LM (same guard as above).
                p.second=M;
                if(!LM_map.size()) {
                  LM_map.push_back(p);
                } else {
                  size_t idx=LMind(L,M,false);
                  if (idx == LM_map.size())
                    LM_map.push_back(p);
                  else if (!(LM_map[idx]==p))
                    LM_map.insert(LM_map.begin()+idx,p);
                }
              }
            }
          }

          // One-electron matrices need gmax,5,gmax
          // Two-electron matrices need Lmax+2,Lmax,Lmax+2
          int lrval(std::max(Lmax+2,gmax));
          int midval(std::max(Lmax,5));

          Timer t;
          printf("Computing Gaunt coefficients ... ");
          fflush(stdout);
          gaunt=gaunt::Gaunt(lrval,midval,lrval);
          printf("done (% .3f s)\n",t.get());
          fflush(stdout);

          t.set();
          printf("Computing Legendre function values ... ");
          fflush(stdout);

          // Fill table with necessary values
          legtab=legendretable::LegendreTable(Lmax,Mmax);
          helfem::Vector chmu(radial.get_chmu_quad());
          for(Eigen::Index i=0;i<chmu.size();i++)
            legtab.compute(chmu(i));
          printf("done (% .3f s)\n",t.get());
          fflush(stdout);

        } else {
          // One-electron matrices need gmax,5,gmax
          int lrval(gmax);
          int midval(5);
          int Mmax=mval.maxCoeff()-mval.minCoeff();

          gaunt=gaunt::Gaunt(lrval,midval,lrval);
        }

        // Cache the real->dummy index map once. pure_indices() rebuilds it on
        // every call, and boundary expansion / removal runs on every Fock build.
        pure_idx = pure_indices();
      }

      TwoDBasis::~TwoDBasis() {
      }

      int TwoDBasis::get_Z1() const {
        return Z1;
      }

      int TwoDBasis::get_Z2() const {
        return Z2;
      }

      double TwoDBasis::get_Rhalf() const {
        return Rhalf;
      }

      Eigen::VectorXi TwoDBasis::get_lval() const {
        return lval;
      }

      Eigen::VectorXi TwoDBasis::get_mval() const {
        return mval;
      }

      int TwoDBasis::get_nquad() const {
        return radial.get_nquad();
      }

      helfem::Vector TwoDBasis::get_bval() const {
        return radial.get_bval();
      }

      double TwoDBasis::get_mumax() const {
        helfem::Vector bval(radial.get_bval());
        return bval(bval.size()-1);
      }

      int TwoDBasis::get_poly_id() const {
        return radial.get_poly_id();
      }

      int TwoDBasis::get_poly_nnodes() const {
        return radial.get_poly_nnodes();
      }

      size_t TwoDBasis::Ndummy() const {
        return lval.size()*radial.Nbf();
      }

      size_t TwoDBasis::Nbf() const {
        // Count total number of basis functions
        size_t nbf=0;
        for(size_t i=0;i<mval.size();i++) {
          nbf+=radial.Nbf();
          if(mval(i)!=0)
            // Remove first function
            nbf--;
        }

        return nbf;
      }

      size_t TwoDBasis::Nrad() const {
        return radial.Nbf();
      }

      size_t TwoDBasis::Nang() const {
        return lval.size();
      }

      std::vector<Eigen::Index> TwoDBasis::pure_indices() const {
        // Indices of the pure functions
        std::vector<Eigen::Index> idx(Nbf());

        size_t ioff=0;
        for(size_t i=0;i<(size_t) mval.size();i++) {
          if(mval(i)==0) {
            for(size_t j=0;j<radial.Nbf();j++)
              idx[ioff+j]=(Eigen::Index) (i*radial.Nbf()+j);
            ioff+=radial.Nbf();
          } else {
	    // Just drop the first function
            for(size_t j=0;j<radial.Nbf()-1;j++)
              idx[ioff+j]=(Eigen::Index) (i*radial.Nbf()+1+j);
            ioff+=radial.Nbf()-1;
          }
        }

        return idx;
      }

      helfem::Matrix TwoDBasis::in_element_kernel(size_t iel, int L, int M) const {
        // Use the same auto-converged order as compute_tei so this diagnostic
        // reflects the kernel that was actually factorized.
        const quadrature::TwoElectronElement el(radial.twoe_element(iel, converged_twoe_order(iel)));
        const helfem::Matrix T00(quadrature::twoe_integral(el, 0, 0, L, M, legtab));
        const helfem::Matrix T02(quadrature::twoe_integral(el, 0, 2, L, M, legtab));
        const helfem::Matrix T22(quadrature::twoe_integral(el, 2, 2, L, M, legtab));

        const Eigen::Index n = T00.rows();
        helfem::Matrix W(2*n, 2*n);
        W.topLeftCorner(n,n)     =  T00;
        W.topRightCorner(n,n)    = -T02;
        W.bottomLeftCorner(n,n)  = -T02.transpose();
        W.bottomRightCorner(n,n) =  T22;
        // Symmetrize away roundoff
        return helfem::Matrix(0.5*(W + W.transpose()));
      }

      helfem::Matrix TwoDBasis::in_element_kernel_exchange(size_t iel, int L, int M) const {
        const quadrature::TwoElectronElement el(radial.twoe_element(iel, converged_twoe_order(iel)));
        const size_t Ni(radial.Nprim(iel));

        const helfem::Matrix T00(quadrature::twoe_integral(el, 0, 0, L, M, legtab));
        const helfem::Matrix T02(quadrature::twoe_integral(el, 0, 2, L, M, legtab));
        const helfem::Matrix T20(quadrature::twoe_integral(el, 2, 0, L, M, legtab));
        const helfem::Matrix T22(quadrature::twoe_integral(el, 2, 2, L, M, legtab));

        const helfem::Matrix K00(utils::exchange_tei(T00,Ni,Ni,Ni,Ni));
        const helfem::Matrix K02(utils::exchange_tei(T02,Ni,Ni,Ni,Ni));
        const helfem::Matrix K20(utils::exchange_tei(T20,Ni,Ni,Ni,Ni));
        const helfem::Matrix K22(utils::exchange_tei(T22,Ni,Ni,Ni,Ni));

        const Eigen::Index n = K00.rows();
        helfem::Matrix Kcat(n, 4*n);
        Kcat.block(0,0*n,n,n) = K00;
        Kcat.block(0,1*n,n,n) = K02;
        Kcat.block(0,2*n,n,n) = K20;
        Kcat.block(0,3*n,n,n) = K22;
        return Kcat;
      }

      std::pair<double,double> TwoDBasis::check_cd(size_t iel, int L, int M) const {
        const size_t Ni(radial.Nprim(iel));
        const size_t ilm(lmind(L,M));
        const size_t Nel(radial.Nel());

        // Exact kernel and the exact exchange-ordered tensors. Build at the
        // same auto-converged order compute_tei used, so the reference W here
        // matches the one that produced cd_B / cd_sigma (otherwise kerr would
        // pick up the order mismatch rather than the factorization error).
        const quadrature::TwoElectronElement el(radial.twoe_element(iel, converged_twoe_order(iel)));
        const helfem::Matrix T00(quadrature::twoe_integral(el,0,0,L,M,legtab));
        const helfem::Matrix T02(quadrature::twoe_integral(el,0,2,L,M,legtab));
        const helfem::Matrix T20(quadrature::twoe_integral(el,2,0,L,M,legtab));
        const helfem::Matrix T22(quadrature::twoe_integral(el,2,2,L,M,legtab));

        const Eigen::Index n = T00.rows();
        helfem::Matrix W(2*n,2*n);
        W.topLeftCorner(n,n)     =  T00;
        W.topRightCorner(n,n)    = -T02;
        W.bottomLeftCorner(n,n)  = -T02.transpose();
        W.bottomRightCorner(n,n) =  T22;
        W = (0.5*(W + W.transpose())).eval();

        // 1) Does B diag(sigma) B' reproduce W?
        const helfem::Matrix & B = cd_B[ilm*Nel+iel];
        const helfem::Vector & sigma = cd_sigma[ilm*Nel+iel];
        helfem::Matrix Wrec(helfem::Matrix::Zero(2*n,2*n));
        for(Eigen::Index p=0;p<B.cols();p++)
          Wrec += sigma(p)*(B.col(p)*B.col(p).transpose());
        const double kerr = (W - Wrec).norm() / W.norm();

        // 2) Does the RI-K contraction match the exact exchange-ordered one?
        //    Use deterministic pseudo-random R blocks.
        helfem::Matrix R00(Ni,Ni), R02(Ni,Ni), R20(Ni,Ni), R22(Ni,Ni);
        for(size_t i=0;i<Ni;i++)
          for(size_t j=0;j<Ni;j++) {
            const double t = (double)(i*Ni + j + 1);
            R00(i,j) = std::sin(0.7*t);
            R02(i,j) = std::cos(1.3*t);
            R20(i,j) = std::sin(2.1*t + 0.4);
            R22(i,j) = std::cos(0.9*t + 1.1);
          }

        // Exact: Ksub = sum_ab ktei_ab * vec(R_ab)
        const helfem::Matrix K00(utils::exchange_tei(T00,Ni,Ni,Ni,Ni));
        const helfem::Matrix K02(utils::exchange_tei(T02,Ni,Ni,Ni,Ni));
        const helfem::Matrix K20(utils::exchange_tei(T20,Ni,Ni,Ni,Ni));
        const helfem::Matrix K22(utils::exchange_tei(T22,Ni,Ni,Ni,Ni));
        helfem::Vector kex(helfem::Vector::Zero(Ni*Ni));
        kex += K00*Eigen::Map<const helfem::Vector>(R00.data(),R00.size());
        kex += K02*Eigen::Map<const helfem::Vector>(R02.data(),R02.size());
        kex += K20*Eigen::Map<const helfem::Vector>(R20.data(),R20.size());
        kex += K22*Eigen::Map<const helfem::Vector>(R22.data(),R22.size());
        Eigen::Map<const helfem::Matrix> Kexact(kex.data(),Ni,Ni);

        // RI-K, exactly as exchange() does it
        helfem::Matrix Kri(helfem::Matrix::Zero(Ni,Ni));
        for(Eigen::Index p=0;p<B.cols();p++) {
          Eigen::Map<const helfem::Matrix> M0(B.col(p).data(),     Ni, Ni);
          Eigen::Map<const helfem::Matrix> M2(B.col(p).data() + n, Ni, Ni);
          const helfem::Matrix M0t(M0.transpose());
          const helfem::Matrix M2t(M2.transpose());
          Kri += sigma(p)*helfem::Matrix(  M0t*R00*M0t
                                         - M0t*R02*M2t
                                         - M2t*R20*M0t
                                         + M2t*R22*M2t );
        }

        const double rerr = (Kexact - Kri).norm() / Kexact.norm();
        return std::make_pair(kerr, rerr);
      }

      std::vector<Eigen::Index> TwoDBasis::m_indices(int m) const {
        return helfem::collect_shell_indices(mval.size(),
            [&](size_t i) { return (mval(i) == 0) ? radial.Nbf() : radial.Nbf() - 1; },
            [&](size_t i) { return mval(i) == m; });
      }

      std::vector<Eigen::Index> TwoDBasis::m_indices(int m, bool odd) const {
        return helfem::collect_shell_indices(mval.size(),
            [&](size_t i) { return (mval(i) == 0) ? radial.Nbf() : radial.Nbf() - 1; },
            [&](size_t i) { return mval(i) == m && (lval(i) % 2 == odd); });
      }

      std::vector<std::vector<Eigen::Index>> TwoDBasis::get_sym_idx(int symm) const {
        std::vector<std::vector<Eigen::Index>> idx;
        if(symm==0) {
          idx.resize(1);
          idx[0].resize(Nbf());
          for(Eigen::Index i=0;i<(Eigen::Index) Nbf();i++)
            idx[0][i]=i;
        } else if(symm==1) {
          // Unique m values in ascending order (matches arma::find_unique + sort).
          std::vector<int> mv;
          for(size_t i=0;i<mval.size();i++)
            if(std::find(mv.begin(),mv.end(),mval(i))==mv.end())
              mv.push_back(mval(i));
          std::sort(mv.begin(),mv.end());

          idx.resize(mv.size());
          for(size_t i=0;i<mv.size();i++)
            idx[i]=m_indices(mv[i]);
        } else if(symm==2) {
          // Unique m values in ascending order (matches arma::find_unique + sort).
          std::vector<int> mv;
          for(size_t i=0;i<mval.size();i++)
            if(std::find(mv.begin(),mv.end(),mval(i))==mv.end())
              mv.push_back(mval(i));
          std::sort(mv.begin(),mv.end());

          idx.resize(2*mv.size());
          for(size_t i=0;i<mv.size();i++) {
            idx[2*i]=m_indices(mv[i],false);
            idx[2*i+1]=m_indices(mv[i],true);
          }
        } else
          throw std::logic_error("Unknown symmetry\n");

        return idx;
      }

      helfem::Matrix TwoDBasis::Sinvh(bool chol, int sym) const {
        // Form overlap matrix
        helfem::Matrix S(overlap());

        // Half-inverse is
        if(sym==0) {
          return scf::form_Sinvh(S, chol);
        } else {
          // Get basis function indices
          std::vector<std::vector<Eigen::Index>> midx(get_sym_idx(sym));
          // Construct Sinvh in each subblock
          helfem::Matrix Sinvh(helfem::Matrix::Zero(Nbf(),Nbf()));
          size_t ioff=0;
          for(size_t i=0;i<midx.size();i++) {
            if(midx[i].empty())
              continue;

            // Gather the S(midx,midx) subblock
            const size_t n(midx[i].size());
            helfem::Matrix Ssub(n,n);
            for(size_t a=0;a<n;a++)
              for(size_t b=0;b<n;b++)
                Ssub(a,b)=S(midx[i][a], midx[i][b]);

            helfem::Matrix block(scf::form_Sinvh(Ssub,chol));

            // Scatter into Sinvh(midx[i], ioff..ioff+n-1)
            for(size_t a=0;a<n;a++)
              for(size_t b=0;b<n;b++)
                Sinvh(midx[i][a], ioff+b)=block(a,b);
            // Increment offset
            ioff += n;
          }
          return Sinvh;
        }
      }

      void TwoDBasis::set_sub(helfem::Matrix & M, size_t iang, size_t jang, const helfem::Matrix & Mrad) const {
        const size_t N(radial.Nbf());
        M.block(iang*N,jang*N,N,N)=Mrad;
      }

      void TwoDBasis::add_sub(helfem::Matrix & M, size_t iang, size_t jang, const helfem::Matrix & Mrad) const {
        const size_t N(radial.Nbf());
        M.block(iang*N,jang*N,N,N)+=Mrad;
      }

      helfem::Matrix TwoDBasis::overlap() const {
        // Build radial matrix elements
        helfem::Matrix I10(radial.radial_integral(1,0));
        helfem::Matrix I12(radial.radial_integral(1,2));

        // Full overlap matrix
        helfem::Matrix S(helfem::Matrix::Zero(Ndummy(),Ndummy()));
        // Fill elements
        for(size_t iang=0;iang<lval.size();iang++) {
          int li(lval(iang));
          int mi(mval(iang));

          for(size_t jang=0;jang<lval.size();jang++) {
            int lj(lval(jang));
            int mj(mval(jang));

            // Calculate coupling
            if(mi==mj) {
              if(li==lj)
                set_sub(S,iang,jang,I12);

              // We can also couple through the cos^2 term
              double cpl(gaunt.cosine2_coupling(lj,mj,li,mi));
              if(cpl!=0.0)
                add_sub(S,iang,jang,-cpl*I10);
            }
          }
        }

        // Plug in prefactor
        S*=std::pow(Rhalf,3);

        return remove_boundaries(S);
      }

      helfem::Matrix TwoDBasis::overlap(const TwoDBasis & rh) const {
        // Cross-basis overlap. Same coupling structure as overlap() but
        // with rh's radial basis on the right.
        helfem::Matrix I10(radial.overlap(rh.radial, 0));
        helfem::Matrix I12(radial.overlap(rh.radial, 2));

        const size_t Ni(radial.Nbf());
        const size_t Nj(rh.radial.Nbf());
        helfem::Matrix S(helfem::Matrix::Zero(Ndummy(), rh.Ndummy()));
        for(size_t iang=0;iang<lval.size();iang++) {
          int li(lval(iang));
          int mi(mval(iang));
          for(size_t jang=0;jang<rh.lval.size();jang++) {
            int lj(rh.lval(jang));
            int mj(rh.mval(jang));
            if(mi==mj) {
              if(li==lj)
                S.block(iang*Ni, jang*Nj, Ni, Nj) = I12;
              double cpl(gaunt.cosine2_coupling(lj, mj, li, mi));
              if(cpl!=0.0)
                S.block(iang*Ni, jang*Nj, Ni, Nj) -= cpl*I10;
            }
          }
        }
        S *= std::pow(Rhalf, 3);
        // Trim boundaries on both sides so shapes align with Sinvh_new /
        // Sinvh_old. pure_indices() gives the gather lists.
        std::vector<Eigen::Index> ridx(pure_indices());
        std::vector<Eigen::Index> cidx(rh.pure_indices());
        helfem::Matrix Strim(ridx.size(), cidx.size());
        for(size_t a=0;a<ridx.size();a++)
          for(size_t b=0;b<cidx.size();b++)
            Strim(a,b)=S(ridx[a], cidx[b]);
        return Strim;
      }

      helfem::Matrix TwoDBasis::kinetic() const {
        // Build radial kinetic energy matrix
        helfem::Matrix Trad(radial.kinetic());
        helfem::Matrix Ip1(radial.radial_integral(1,0));
        helfem::Matrix Im1(radial.radial_integral(-1,0));

        // Full kinetic energy matrix
        helfem::Matrix T(helfem::Matrix::Zero(Ndummy(),Ndummy()));
        // Fill elements
        for(size_t iang=0;iang<lval.size();iang++) {
          set_sub(T,iang,iang,Trad);
          if(lval(iang)!=0) {
            // We also get the l(l+1) term
            add_sub(T,iang,iang,(double) (lval(iang)*(lval(iang)+1))*Ip1);
          }
          if(mval(iang)!=0) {
            // We also get the m^2 term
            add_sub(T,iang,iang,(double) (mval(iang)*mval(iang))*Im1);
          }
        }

        // Plug in prefactor
        T*=Rhalf/2.0;

        return remove_boundaries(T);
      }

      helfem::Matrix TwoDBasis::nuclear() const {
        // Build radial matrices
        helfem::Matrix I10(radial.radial_integral(1,0));
        helfem::Matrix I11(radial.radial_integral(1,1));

        // Full nuclear attraction matrix
        helfem::Matrix V(helfem::Matrix::Zero(Ndummy(),Ndummy()));

        // Fill elements
        for(size_t iang=0;iang<lval.size();iang++) {
          int li(lval(iang));
          int mi(mval(iang));

          for(size_t jang=0;jang<lval.size();jang++) {
            int lj(lval(jang));
            int mj(mval(jang));

            // Calculate coupling
            if(mi==mj) {
              if(li==lj)
                set_sub(V,iang,jang,(double) (Z1+Z2)*I11);

              // We can also couple through the cos term
	      if(Z1!=Z2) {
		double cpl(gaunt.cosine_coupling(lj,mj,li,mi));
		if(cpl!=0.0)
		  add_sub(V,iang,jang,((double) (Z2-Z1)*cpl)*I10);
	      }
            }
          }
        }

        // Plug in prefactor
        V*=-std::pow(Rhalf,2);

        return remove_boundaries(V);
      }

      helfem::Matrix TwoDBasis::dipole_z() const {
        // Full electric couplings
        helfem::Matrix V(helfem::Matrix::Zero(Ndummy(),Ndummy()));

        // Build radial matrix elements
        helfem::Matrix I11(radial.radial_integral(1,1));
        helfem::Matrix I13(radial.radial_integral(1,3));

        // Fill elements
        for(size_t iang=0;iang<lval.size();iang++) {
          int li(lval(iang));
          int mi(mval(iang));

          for(size_t jang=0;jang<lval.size();jang++) {
            int lj(lval(jang));
            int mj(mval(jang));

            // Calculate coupling
            if(mi==mj) {
              // Coupling through the cos term
              double cpl1(gaunt.cosine_coupling(lj,mj,li,mi));
              if(cpl1!=0.0)
                add_sub(V,iang,jang,cpl1*I13);

              // We can also couple through the cos^3 term
              double cpl3(gaunt.cosine3_coupling(lj,mj,li,mi));
              if(cpl3!=0.0)
                add_sub(V,iang,jang,-cpl3*I11);
            }
          }
        }

        // Plug in prefactors
        V*=std::pow(Rhalf,4);

        return remove_boundaries(V);
      }

      helfem::Matrix TwoDBasis::quadrupole_zz() const {
        // Full electric couplings
        helfem::Matrix V(helfem::Matrix::Zero(Ndummy(),Ndummy()));

        // Build radial matrix elements
        helfem::Matrix I10(radial.radial_integral(1,0));
        helfem::Matrix I12(radial.radial_integral(1,2));
        helfem::Matrix I14(radial.radial_integral(1,4));

        // Fill elements
        for(size_t iang=0;iang<lval.size();iang++) {
          int li(lval(iang));
          int mi(mval(iang));

          for(size_t jang=0;jang<lval.size();jang++) {
            int lj(lval(jang));
            int mj(mval(jang));

            // Calculate coupling
            if(mi==mj) {
              // Coupling through the cos^4 term
              double cpl4(gaunt.cosine4_coupling(lj,mj,li,mi));
              if(cpl4!=0.0)
                add_sub(V,iang,jang,cpl4*(I10-3.0*I12));

              // We can also couple through the cos^2 term
              double cpl2(gaunt.cosine2_coupling(lj,mj,li,mi));
              if(cpl2!=0.0)
                add_sub(V,iang,jang,cpl2*(3.0*I14-I10));

              // or the delta term
              if(li==lj)
                add_sub(V,iang,jang,I12-I14);
            }
          }
        }

        // Plug in prefactors
        V*=std::pow(Rhalf,5)/2;

        return remove_boundaries(V);
      }

      helfem::Matrix TwoDBasis::Bz_field(double B) const {
        // Full couplings
        helfem::Matrix V(helfem::Matrix::Zero(Ndummy(),Ndummy()));

        // Build radial matrix elements
        helfem::Matrix I10(radial.radial_integral(1,0)*std::pow(Rhalf,3));
        helfem::Matrix I12(radial.radial_integral(1,2)*std::pow(Rhalf,3));
        helfem::Matrix I30(radial.radial_integral(3,0)*std::pow(Rhalf,5));
        helfem::Matrix I32(radial.radial_integral(3,2)*std::pow(Rhalf,5));

        // Fill elements
        for(size_t iang=0;iang<lval.size();iang++) {
          int li(lval(iang));
          int mi(mval(iang));

          for(size_t jang=0;jang<lval.size();jang++) {
            int lj(lval(jang));
            int mj(mval(jang));

            // Calculate coupling
            if(mi==mj) {
              // Coupling strength
              double cs(B*B/8.0);

              // Coupling through the sin^2 term
              double cpl2(gaunt.sine2_coupling(lj,mj,li,mi));
              if(cpl2!=0.0)
                add_sub(V,iang,jang,cs*cpl2*I32);

              // We can also couple through the sin^2 cos^2 term
              double cpl22(gaunt.cosine2_sine2_coupling(lj,mj,li,mi));
              if(cpl22!=0.0)
                add_sub(V,iang,jang,-cs*cpl22*I30);

              // m term
              double ds(-0.5*mj*B);
              if(ds!=0.0) {
                if(li==lj)
                  add_sub(V,iang,jang,ds*I12);

                // We can also couple through the cos^2 term
                double cpl(gaunt.cosine2_coupling(lj,mj,li,mi));
                if(cpl!=0.0)
                  add_sub(V,iang,jang,(-ds*cpl)*I10);
              }
            }
          }
        }

        return remove_boundaries(V);
      }


      bool operator<(const lmidx_t & lh, const lmidx_t & rh) {
        if(lh.first < rh.first)
          return true;
        if(lh.first > rh.first)
          return false;

        if(lh.second < rh.second)
          return true;
        if(lh.second > rh.second)
          return false;

        return false;
      }

      bool operator==(const lmidx_t & lh, const lmidx_t & rh) {
        return (lh.first == rh.first) && (lh.second == rh.second);
      }

      int TwoDBasis::converged_twoe_order(size_t iel) const {
        // Seed the refinement from the requested --nquad so the common case
        // converges in 1-2 steps; the loop, not the seed, sets the accuracy.
        int nstart = radial.get_nquad();
        if(nstart < 5) nstart = 5;
        if(nstart > twoe_nmax) nstart = twoe_nmax;

        // Nothing to probe: no multipoles requested. Fall back to the seed.
        if(lm_map.empty())
          return nstart;

        // Probe the HARDEST multipole. lm_map is sorted ascending by (L, |M|)
        // (see operator< on lmidx_t), so its back() is the largest L and, for
        // that L, the largest |M| -- the sharpest Green's function and hence
        // the one that needs the most quadrature points. Converging it
        // converges every lower multipole.
        const int L = lm_map.back().first;
        const int M = lm_map.back().second;

        int nconv = nstart;
        converge_block(
            [&](int n) {
              const quadrature::TwoElectronElement el(radial.twoe_element(iel, n));
              const helfem::Matrix T00(quadrature::twoe_integral(el, 0, 0, L, M, legtab));
              const helfem::Matrix T02(quadrature::twoe_integral(el, 0, 2, L, M, legtab));
              const helfem::Matrix T22(quadrature::twoe_integral(el, 2, 2, L, M, legtab));
              // Assemble the 2-channel kernel W exactly as compute_tei does, so
              // the probe measures the quantity that is actually factorized.
              const Eigen::Index nn = T00.rows();
              helfem::Matrix W(2*nn, 2*nn);
              W.topLeftCorner(nn,nn)     =  T00;
              W.topRightCorner(nn,nn)    = -T02;
              W.bottomLeftCorner(nn,nn)  = -T02.transpose();
              W.bottomRightCorner(nn,nn) =  T22;
              return helfem::Matrix(0.5*(W + W.transpose()));
            },
            nstart, "in-element two-electron kernel", &nconv);
        return nconv;
      }

      void TwoDBasis::compute_tei(bool exchange) {
        // Number of distinct L values is
        size_t Nel(radial.Nel());

        // Compute disjoint integrals
        disjoint_P0.resize(Nel*lm_map.size());
        disjoint_P2.resize(Nel*lm_map.size());
        disjoint_Q0.resize(Nel*lm_map.size());
        disjoint_Q2.resize(Nel*lm_map.size());
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(size_t ilm=0;ilm<lm_map.size();ilm++) {
          int L(lm_map[ilm].first);
          int M(lm_map[ilm].second);
          for(size_t iel=0;iel<Nel;iel++) {
            disjoint_P0[ilm*Nel+iel]=radial.Plm_integral(0,iel,L,M,legtab);
            disjoint_P2[ilm*Nel+iel]=radial.Plm_integral(2,iel,L,M,legtab);
            disjoint_Q0[ilm*Nel+iel]=radial.Qlm_integral(0,iel,L,M,legtab);
            disjoint_Q2[ilm*Nel+iel]=radial.Qlm_integral(2,iel,L,M,legtab);
          }
        }

        // Factorize the in-element two-electron integrals.
        //
        // The kernel is the symmetric 2-channel matrix
        //     W = [  T00  -T02 ]
        //         [ -T02'  T22 ]
        // (T20 = T02' by construction, since twoe_integral(a,b) is the
        // transpose of twoe_integral(b,a)). It is 2*Nprim^2 square but has a
        // numerical rank of only ~2*Nprim, because the P_L(cosh mu_<)
        // Q_L(cosh mu_>) Green's function is semi-separable. Keep the
        // eigenvectors above a relative threshold and store
        //     W = B diag(sigma) B',  sigma = +-1,
        // which is a Cholesky-type factorization that also tolerates the
        // indefiniteness the odd-|M| blocks show.
        //
        // This puts the integrals in DF/RI form, so exchange comes out of the
        // standard RI-K contraction and no exchange-ordered tensor is needed.
        cd_thresh = 1e-12;
        cd_B.assign(Nel*lm_map.size(), helfem::Matrix());
        cd_sigma.assign(Nel*lm_map.size(), helfem::Vector());

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(size_t iel=0;iel<Nel;iel++) {
          // Auto-convergence: refine the Gauss-Chebyshev order on the hardest
          // multipole, then build the element ONCE at the converged order and
          // reuse it across all (L, |M|). The order is set by double's eps, not
          // by --nquad.
          const int nconv = converged_twoe_order(iel);
          const quadrature::TwoElectronElement eldata(radial.twoe_element(iel, nconv));

          for(size_t ilm=0;ilm<lm_map.size();ilm++) {
            int L(lm_map[ilm].first);
            int M(lm_map[ilm].second);

            const helfem::Matrix T00(radial.twoe_integral(0,0,eldata,L,M,legtab));
            const helfem::Matrix T02(radial.twoe_integral(0,2,eldata,L,M,legtab));
            const helfem::Matrix T22(radial.twoe_integral(2,2,eldata,L,M,legtab));

            const Eigen::Index n = T00.rows();
            helfem::Matrix W(2*n, 2*n);
            W.topLeftCorner(n,n)     =  T00;
            W.topRightCorner(n,n)    = -T02;
            W.bottomLeftCorner(n,n)  = -T02.transpose();
            W.bottomRightCorner(n,n) =  T22;
            // Symmetrize away roundoff
            W = (0.5*(W + W.transpose())).eval();

            // Diagonal-pivoted, sign-aware Cholesky: the usual Cholesky
            // decomposition of the integrals, but picking the pivot by largest
            // |diagonal| and carrying its sign, so the indefinite odd-|M|
            // blocks are handled too. Costs O(n^2 r) rather than the O(n^3) of
            // a full eigendecomposition, which matters because this runs for
            // every (element, L, |M|).
            const Eigen::Index N = 2*n;
            helfem::Vector d(W.diagonal());
            const double dmax0 = d.cwiseAbs().maxCoeff();

            std::vector<helfem::Vector> Bcols;
            std::vector<double> sgn;
            for(Eigen::Index p=0;p<N;p++) {
              // Largest remaining diagonal, by magnitude
              Eigen::Index piv = 0;
              const double dpiv = d.cwiseAbs().maxCoeff(&piv);
              if(dmax0 <= 0.0 || dpiv <= cd_thresh*dmax0)
                break;

              const double dp = d(piv);
              const double s = (dp >= 0.0) ? 1.0 : -1.0;

              // Residual column: W(:,piv) with the vectors found so far removed
              helfem::Vector col(W.col(piv));
              for(size_t q=0;q<Bcols.size();q++)
                col -= sgn[q]*Bcols[q](piv)*Bcols[q];
              col /= std::sqrt(std::abs(dp));

              // Deflate the diagonal
              d.array() -= s*col.array().square();
              d(piv) = 0.0;

              Bcols.push_back(col);
              sgn.push_back(s);
            }

            const Eigen::Index r = (Eigen::Index) Bcols.size();
            helfem::Matrix B(N, r);
            helfem::Vector sigma(r);
            for(Eigen::Index p=0;p<r;p++) {
              B.col(p) = Bcols[p];
              sigma(p) = sgn[p];
            }

            cd_B[ilm*Nel+iel]     = B;
            cd_sigma[ilm*Nel+iel] = sigma;
          }
        }

        // No exchange-ordered tensor is built. The exchange PAIRING of these
        // integrals is full rank -- 225 of 225 for 15-node LIPs -- so it
        // cannot be compressed directly. Instead the factorization above puts
        // the integrals in DF/RI form, and exchange() gets K from the standard
        // RI-K contraction, which needs only cd_B / cd_sigma.
        (void) exchange;
      }

      std::vector<double> TwoDBasis::build_LMfac_abs() const {
        const double Rhalf5_4pi = 4.0 * M_PI * std::pow(Rhalf, 5);
        std::vector<double> tbl(lm_map.size());
        for(size_t i = 0; i < lm_map.size(); ++i) {
          const int L  = lm_map[i].first;
          const int Ma = lm_map[i].second;  // |M|
          double fr = 1.0;
          for(int p = L + Ma; p > L - Ma; --p) fr *= p;
          tbl[i] = Rhalf5_4pi / fr;
        }
        return tbl;
      }

      size_t TwoDBasis::lmind(int L, int M, bool check) const {
        // Switch to |M|
        M=std::abs(M);
        // Find index in the L,|M| table
        lmidx_t p(L,M);
        std::vector<lmidx_t>::const_iterator low(std::lower_bound(lm_map.begin(),lm_map.end(),p));
        if(check && low == lm_map.end()) {
          std::ostringstream oss;
          oss << "Could not find L=" << p.first << ", |M|= " << p.second << " on the list!\n";
          throw std::logic_error(oss.str());
        }
        // Index is
        size_t idx(low-lm_map.begin());
        // When check==false callers use idx == lm_map.size() as a
        // "not-found sentinel". Do not touch lm_map[idx] in that path
        // -- with libstdc++'s hardened operator[] (GCC 16 default) an
        // OOB access on operator[] is a hard abort even though the value
        // would just be compared and discarded.
        if (idx == lm_map.size())
          return idx;
        if(check && (lm_map[idx].first != L || lm_map[idx].second != M)) {
          std::ostringstream oss;
          oss << "Map error: tried to get L = " << L << ", M = " << M << " but got instead L = " << lm_map[idx].first << ", M = " << lm_map[idx].second << "!\n";
          throw std::logic_error(oss.str());
        }

        return idx;
      }

      size_t TwoDBasis::LMind(int L, int M, bool check) const {
        // Find index in the L,M table
        lmidx_t p(L,M);
        std::vector<lmidx_t>::const_iterator low(std::lower_bound(LM_map.begin(),LM_map.end(),p));
        if(check && low == LM_map.end()) {
          std::ostringstream oss;
          oss << "Could not find L=" << p.first << ", M= " << p.second << " on the list!\n";
          throw std::logic_error(oss.str());
        }
        // Index is
        size_t idx(low-LM_map.begin());
        // See lmind() above: guard the operator[] access when idx == size.
        if (idx == LM_map.size())
          return idx;
        if(check && (LM_map[idx].first != L || LM_map[idx].second != M)) {
          std::ostringstream oss;
          oss << "Map error: tried to get L = " << L << ", M = " << M << " but got instead L = " << LM_map[idx].first << ", M = " << LM_map[idx].second << "!\n";
          throw std::logic_error(oss.str());
        }

        return idx;
      }

      static double factorial_ratio(int pmax, int pmin) {
        // Check consistency of arguments
        if(pmax < pmin)
          return 1.0/factorial_ratio(pmin, pmax);

        // Calculate ratio
        double r=1.0;
        for(int p=pmax;p>pmin;p--)
          r*=p;

        return r;
      }

      helfem::Matrix TwoDBasis::coulomb(const helfem::Matrix & P_in) const {
        if(!cd_B.size())
          throw std::logic_error("Primitive teis have not been computed!\n");

        // Eigen throughout: no arma bridge on the Fock path.
        const helfem::Matrix P(expand_boundaries(P_in));

        // Number of radial elements
        size_t Nel(radial.Nel());
        // Number of radial functions
        size_t Nrad(radial.Nbf());

        // Radial helper matrices
        std::vector<helfem::Matrix> Paux0(LM_map.size());
        std::vector<helfem::Matrix> Paux2(LM_map.size());
        for(size_t i=0;i<Paux0.size();i++) {
          Paux0[i]=helfem::Matrix::Zero(Nrad,Nrad);
          Paux2[i]=helfem::Matrix::Zero(Nrad,Nrad);
        }

        // Form radial helpers: contract ket
        for(size_t kang=0;kang<lval.size();kang++) {
          for(size_t lang=0;lang<lval.size();lang++) {
            // l and m values
            int lk(lval(kang));
            int mk(mval(kang));
            int ll(lval(lang));
            int ml(mval(lang));
            // RH m value
            int M(mk-ml);
            // M values match. Loop over possible couplings
            int Lmin=std::max(std::abs(lk-ll)-2,abs(M));
            int Lmax=lk+ll+2;
            for(int L=Lmin;L<=Lmax;L++) {
              const size_t iLM(LMind(L,M));
              // Calculate coupling coefficients
              double cpl0(gaunt.mod_coeff(lk,mk,L,M,ll,ml));
              double cpl2(gaunt.coeff(lk,mk,L,M,ll));
              // Increment
              helfem::Matrix Prad(P.block(kang*Nrad,lang*Nrad,Nrad,Nrad));
              if(cpl0!=0.0)
                Paux0[iLM]+=cpl0*Prad;
              if(cpl2!=0.0)
                Paux2[iLM]+=cpl2*Prad;
            }
          }
        }

        // Coulomb helpers
        std::vector<helfem::Matrix> Jaux0(LM_map.size());
        std::vector<helfem::Matrix> Jaux2(LM_map.size());
        for(size_t i=0;i<Jaux0.size();i++) {
          Jaux0[i]=helfem::Matrix::Zero(Nrad,Nrad);
          Jaux2[i]=helfem::Matrix::Zero(Nrad,Nrad);
        }
        // Cache the angular prefactor 4*pi*Rhalf^5 / ((L+|M|)!/(L-|M|)!) for
        // every (L, |M|) in the lm_map. The (-1)^M sign is applied at the
        // lookup site since lm_map is indexed by |M| only. pow(Rhalf,5) and
        // the factorial loop both dominated the inner-loop cost otherwise.
        const std::vector<double> LMfac_abs = build_LMfac_abs();
        for(size_t iLM=0;iLM<LM_map.size();iLM++) {
          // Values of L and M
          int L(LM_map[iLM].first);
          int M(LM_map[iLM].second);

          // Helpers
          const size_t ilm(lmind(L,M));
          const double signM = (M & 1) ? -1.0 : 1.0;
          const double LMfac(signM * LMfac_abs[ilm]);

          // Loop over input elements
          for(size_t jel=0;jel<Nel;jel++) {
            size_t jfirst, jlast;
            radial.get_idx(jel,jfirst,jlast);
            size_t Nj(jlast-jfirst+1);

            // Get density submatrices
            helfem::Matrix Psub0(Paux0[iLM].block(jfirst,jfirst,Nj,Nj));
            helfem::Matrix Psub2(Paux2[iLM].block(jfirst,jfirst,Nj,Nj));

            // Contract integrals
            double jsmall0 = LMfac*(disjoint_P0[ilm*Nel+jel]*Psub0).trace();
            double jbig0 = LMfac*(disjoint_Q0[ilm*Nel+jel]*Psub0).trace();
            double jsmall2 = LMfac*(disjoint_P2[ilm*Nel+jel]*Psub2).trace();
            double jbig2 = LMfac*(disjoint_Q2[ilm*Nel+jel]*Psub2).trace();

            // Increment J: jel>iel
            double ifac0(jbig0 - jbig2);
            double ifac2(-jbig0 + jbig2);
            for(size_t iel=0;iel<jel;iel++) {
              size_t ifirst, ilast;
              radial.get_idx(iel,ifirst,ilast);
              size_t Ni(ilast-ifirst+1);

              const helfem::Matrix & iint0=disjoint_P0[ilm*Nel+iel];
              const helfem::Matrix & iint2=disjoint_P2[ilm*Nel+iel];
              Jaux0[iLM].block(ifirst,ifirst,Ni,Ni)+=ifac0*iint0;
              Jaux2[iLM].block(ifirst,ifirst,Ni,Ni)+=ifac2*iint2;
            }

            // Increment J: jel<iel
            ifac0=jsmall0 - jsmall2;
            ifac2=-jsmall0 + jsmall2;
            for(size_t iel=jel+1;iel<Nel;iel++) {
              size_t ifirst, ilast;
              radial.get_idx(iel,ifirst,ilast);
              size_t Ni(ilast-ifirst+1);

              const helfem::Matrix & iint0=disjoint_Q0[ilm*Nel+iel];
              const helfem::Matrix & iint2=disjoint_Q2[ilm*Nel+iel];
              Jaux0[iLM].block(ifirst,ifirst,Ni,Ni)+=ifac0*iint0;
              Jaux2[iLM].block(ifirst,ifirst,Ni,Ni)+=ifac2*iint2;
            }

            // In-element contribution
            {
              size_t iel=jel;
              size_t ifirst=jfirst;
              size_t Ni=Nj;

              // Column-major vectorisation of the (Nj x Nj) density blocks.
              // Psub0/Psub2 are materialised contiguous matrices, so a Map
              // reproduces arma::vectorise exactly.
              Eigen::Map<const helfem::Vector> pv0(Psub0.data(), Psub0.size());
              Eigen::Map<const helfem::Vector> pv2(Psub2.data(), Psub2.size());

              // In-element block, from the low-rank factorization
              //   W = B diag(sigma) B',
              // applied to the 2-channel density [pv0; pv2]. This is the same
              // contraction the four T00/T02/T20/T22 matvecs used to do, at
              // O(Nprim^2 * r) instead of O(Nprim^4).
              const helfem::Matrix & B = cd_B[ilm*Nel+iel];
              const helfem::Vector & sigma = cd_sigma[ilm*Nel+iel];
              const Eigen::Index nn = (Eigen::Index) (Ni*Ni);

              helfem::Vector p2ch(2*nn);
              p2ch.head(nn) = pv0;
              p2ch.tail(nn) = pv2;

              helfem::Vector c = B.transpose()*p2ch;
              c.array() *= sigma.array();
              const helfem::Vector jv2ch = LMfac*(B*c);

              const helfem::Vector jv0 = jv2ch.head(nn);
              const helfem::Vector jv2 = jv2ch.tail(nn);

              // Reshape back to (Ni x Ni), column-major
              Eigen::Map<const helfem::Matrix> Jsub0(jv0.data(), Ni, Ni);
              Eigen::Map<const helfem::Matrix> Jsub2(jv2.data(), Ni, Ni);

              // Increment global Coulomb matrix
              Jaux0[iLM].block(ifirst,ifirst,Ni,Ni)+=Jsub0;
              Jaux2[iLM].block(ifirst,ifirst,Ni,Ni)+=Jsub2;
            }
          }
        }

        // Full Coulomb matrix
        helfem::Matrix J(helfem::Matrix::Zero(Ndummy(),Ndummy()));
        for(size_t iang=0;iang<lval.size();iang++) {
          for(size_t jang=0;jang<lval.size();jang++) {
            // l and m values
            int li(lval(iang));
            int mi(mval(iang));
            int lj(lval(jang));
            int mj(mval(jang));
            // LH m value
            int M(mj-mi);

            int Lmin=std::max(std::abs(lj-li)-2,abs(M));
            int Lmax=lj+li+2;
            for(int L=Lmin;L<=Lmax;L++) {
              const size_t iLM(LMind(L,M));

              // Couplings
              double cpl0(gaunt.mod_coeff(lj,mj,L,M,li,mi));
              if(cpl0!=0.0) {
                J.block(iang*Nrad,jang*Nrad,Nrad,Nrad)+=cpl0*Jaux0[iLM];
              }

              double cpl2(gaunt.coeff(lj,mj,L,M,li));
              if(cpl2!=0.0) {
                J.block(iang*Nrad,jang*Nrad,Nrad,Nrad)+=cpl2*Jaux2[iLM];
              }
            }
          }
        }

        return remove_boundaries(J);
      }

      helfem::Matrix TwoDBasis::exchange(const helfem::Matrix & P_in) const {
        if(!cd_B.size())
          throw std::logic_error("Primitive teis have not been computed!\n");

        // Eigen throughout: no arma bridge on the Fock path.
        const helfem::Matrix P(expand_boundaries(P_in));

        // Number of radial elements
        size_t Nel(radial.Nel());
        // Number of radial basis functions
        size_t Nrad(radial.Nbf());

        // Pre-compute the (L, |M|) angular prefactor table once per call so
        // the inner loop reduces to a vector lookup + sign branch.
        const std::vector<double> LMfac_abs = build_LMfac_abs();

        // Full exchange matrix. Each (jang,kang) writes a disjoint block, so
        // the parallel accumulation into K is race-free.
        helfem::Matrix K(helfem::Matrix::Zero(Ndummy(),Ndummy()));

#ifdef _OPENMP
#pragma omp parallel
#endif
        {
          // Increment
#ifdef _OPENMP
#pragma omp for collapse(2)
#endif
          for(size_t jang=0;jang<lval.size();jang++) {
            for(size_t kang=0;kang<lval.size();kang++) {
              int lj(lval(jang));
              int mj(mval(jang));

              int lk(lval(kang));
              int mk(mval(kang));

              // Form radial helpers
              std::vector<helfem::Matrix> Rmat00(lm_map.size());
              std::vector<helfem::Matrix> Rmat02(lm_map.size());
              std::vector<helfem::Matrix> Rmat20(lm_map.size());
              std::vector<helfem::Matrix> Rmat22(lm_map.size());
              for(size_t i=0;i<lm_map.size();i++) {
                Rmat00[i]=helfem::Matrix::Zero(Nrad,Nrad);
                Rmat02[i]=helfem::Matrix::Zero(Nrad,Nrad);
                Rmat20[i]=helfem::Matrix::Zero(Nrad,Nrad);
                Rmat22[i]=helfem::Matrix::Zero(Nrad,Nrad);
              }
              // Is there a coupling to the channel?
              std::vector<bool> couple(lm_map.size(),false);

              // Perform angular sums
              for(size_t iang=0;iang<lval.size();iang++) {
                int li(lval(iang));
                int mi(mval(iang));

                for(size_t lang=0;lang<lval.size();lang++) {
                  int ll(lval(lang));
                  int ml(mval(lang));

                  // LH m value
                  int M(mj-mi);
                  // RH m value
                  int Mp(mk-ml);
                  if(M!=Mp)
                    continue;

                  // Do we have any density in this block?
                  double bdens(P.block(iang*Nrad,lang*Nrad,Nrad,Nrad).norm());
                  //printf("(%i %i) (%i %i) density block norm %e\n",li,mi,ll,ml,bdens);
                  if(bdens<10*DBL_EPSILON)
                    continue;

                  // M values match. Loop over possible couplings
                  int Lmin=std::max(std::max(std::abs(li-lj),std::abs(lk-ll))-2,abs(M));
                  int Lmax=std::min(li+lj,lk+ll)+2;

                  for(int L=Lmin;L<=Lmax;L++) {
                    // Calculate total coupling coefficient
                    double cpl00(gaunt.mod_coeff(lj,mj,L,M,li,mi)*gaunt.mod_coeff(lk,mk,L,M,ll,ml));
                    double cpl02(-gaunt.mod_coeff(lj,mj,L,M,li,mi)*gaunt.coeff(lk,mk,L,M,ll));
                    double cpl20(-gaunt.coeff(lj,mj,L,M,li)*gaunt.mod_coeff(lk,mk,L,M,ll,ml));
                    double cpl22(gaunt.coeff(lj,mj,L,M,li)*gaunt.coeff(lk,mk,L,M,ll));

                    // Is there any coupling?
                    if(cpl00==0.0 && cpl02==0.0 && cpl20==0.0 && cpl22==0.0)
                      continue;

                    // Index in the L,|M| table
                    const size_t ilm(lmind(L,M));
                    const double signM = (M & 1) ? -1.0 : 1.0;
                    const double LMfac(signM * LMfac_abs[ilm]);

                    helfem::Matrix Psub(P.block(iang*Nrad,lang*Nrad,Nrad,Nrad));

                    Rmat00[ilm]+=(LMfac*cpl00)*Psub;
                    Rmat02[ilm]+=(LMfac*cpl02)*Psub;
                    Rmat20[ilm]+=(LMfac*cpl20)*Psub;
                    Rmat22[ilm]+=(LMfac*cpl22)*Psub;
                    couple[ilm]=true;
                  }
                }
              }

              // Loop over elements: output
              for(size_t iel=0;iel<Nel;iel++) {
                size_t ifirst, ilast;
                radial.get_idx(iel,ifirst,ilast);

                // Input
                for(size_t jel=0;jel<Nel;jel++) {
                  size_t jfirst, jlast;
                  radial.get_idx(jel,jfirst,jlast);

                  // Number of functions in the two elements
                  size_t Ni(ilast-ifirst+1);
                  size_t Nj(jlast-jfirst+1);

                  if(iel == jel) {
                    /*
                      The exchange matrix is given by
                      K(jk) = (ij|kl) P(il)
                      i.e. the complex conjugation hits i and l as
                      in the density matrix.

                      To get this in the proper order, we permute the integrals
                      K(jk) = (jk;il) P(il)
                    */

                    // RI-K contraction. The in-element integrals are stored as
                    //   T_ab[(i,j),(k,l)] = s_ab sum_P sigma_P M^P_a[i,j] M^P_b[k,l]
                    // with s_00 = s_22 = +1, s_02 = s_20 = -1 (the signs of the
                    // 2-channel kernel W), and M^P_a the a-channel part of
                    // Cholesky vector P reshaped to (Ni x Ni), column-major --
                    // the same (i,j) -> j*Ni+i order the tensors used.
                    //
                    // K(j,k) = sum_ab s_ab sum_il T_ab[(i,j),(k,l)] R_ab(i,l)
                    //        = sum_P sigma_P sum_ab s_ab M^P_a' R_ab M^P_b'
                    //
                    // i.e. the standard RI-K: two (Ni x Ni) matrix products per
                    // Cholesky vector. No exchange-ordered tensor is needed --
                    // which is just as well, since that pairing is full rank.
                    helfem::Matrix Ksubm(helfem::Matrix::Zero(Ni,Nj));

                    for(size_t ilm=0;ilm<lm_map.size();ilm++) {
                      if(!couple[ilm])
                        continue;

                      const helfem::Matrix & B = cd_B[ilm*Nel+iel];
                      const helfem::Vector & sigma = cd_sigma[ilm*Nel+iel];
                      const Eigen::Index nn = (Eigen::Index) (Ni*Ni);

                      const helfem::Matrix Rb00(Rmat00[ilm].block(ifirst,jfirst,Ni,Nj));
                      const helfem::Matrix Rb02(Rmat02[ilm].block(ifirst,jfirst,Ni,Nj));
                      const helfem::Matrix Rb20(Rmat20[ilm].block(ifirst,jfirst,Ni,Nj));
                      const helfem::Matrix Rb22(Rmat22[ilm].block(ifirst,jfirst,Ni,Nj));

                      for(Eigen::Index p=0;p<B.cols();p++) {
                        // Channel blocks of Cholesky vector p, as (Ni x Ni)
                        Eigen::Map<const helfem::Matrix> M0(B.col(p).data(),      Ni, Ni);
                        Eigen::Map<const helfem::Matrix> M2(B.col(p).data() + nn, Ni, Ni);

                        const helfem::Matrix M0t(M0.transpose());
                        const helfem::Matrix M2t(M2.transpose());

                        helfem::Matrix contrib(  M0t*Rb00*M0t
                                               - M0t*Rb02*M2t
                                               - M2t*Rb20*M0t
                                               + M2t*Rb22*M2t );
                        Ksubm += sigma(p)*contrib;
                      }
                    }

                    // Increment global exchange matrix
                    K.block(jang*Nrad+ifirst,kang*Nrad+jfirst,Ni,Nj)-=Ksubm;

                  } else {
                    helfem::Matrix Ksub(helfem::Matrix::Zero(Ni,Nj));
                    for(size_t ilm=0;ilm<lm_map.size();ilm++) {
                      if(!couple[ilm])
                        continue;
                      // Disjoint integrals. When r(iel)>r(jel), iel gets Q, jel gets P.
                      const helfem::Matrix & iint0=(iel>jel) ? disjoint_Q0[ilm*Nel+iel] : disjoint_P0[ilm*Nel+iel];
                      const helfem::Matrix & iint2=(iel>jel) ? disjoint_Q2[ilm*Nel+iel] : disjoint_P2[ilm*Nel+iel];
                      const helfem::Matrix & jint0=(iel>jel) ? disjoint_P0[ilm*Nel+jel] : disjoint_Q0[ilm*Nel+jel];
                      const helfem::Matrix & jint2=(iel>jel) ? disjoint_P2[ilm*Nel+jel] : disjoint_Q2[ilm*Nel+jel];

                      // (Niel x Njel) = (Niel x Njel) x (Njel x Njel)
                      helfem::Matrix T(Rmat00[ilm].block(ifirst,jfirst,Ni,Nj)*jint0.transpose() + Rmat02[ilm].block(ifirst,jfirst,Ni,Nj)*jint2.transpose());
                      Ksub-=iint0*T;

                      T=Rmat20[ilm].block(ifirst,jfirst,Ni,Nj)*jint0.transpose() + Rmat22[ilm].block(ifirst,jfirst,Ni,Nj)*jint2.transpose();
                      Ksub-=iint2*T;
                    }

                    // Increment global exchange matrix
                    K.block(jang*Nrad+ifirst,kang*Nrad+jfirst,Ni,Nj)+=Ksub;
                  }
                }
              }
            }
          }
        }

        return remove_boundaries(K);
      }

      helfem::Matrix TwoDBasis::remove_boundaries(const helfem::Matrix & Fnob) const {
        const Eigen::Index N = (Eigen::Index) Ndummy();
        if(Fnob.rows() != N || Fnob.cols() != N) {
          std::ostringstream oss;
          oss << "Matrix does not have expected size! Got " << Fnob.rows() << " x " << Fnob.cols() << ", expected " << N << " x " << N << "!\n";
          throw std::logic_error(oss.str());
        }
        const Eigen::Index n = (Eigen::Index) pure_idx.size();
        helfem::Matrix Fpure(n, n);
        for(Eigen::Index j=0;j<n;j++)
          for(Eigen::Index i=0;i<n;i++)
            Fpure(i,j) = Fnob(pure_idx[i], pure_idx[j]);
        return Fpure;
      }

      helfem::Matrix TwoDBasis::expand_boundaries(const helfem::Matrix & Ppure) const {
        const Eigen::Index n = (Eigen::Index) pure_idx.size();
        if(Ppure.rows() != n || Ppure.cols() != n) {
          std::ostringstream oss;
          oss << "Matrix does not have expected size! Got " << Ppure.rows() << " x " << Ppure.cols() << ", expected " << n << " x " << n << "!\n";
          throw std::logic_error(oss.str());
        }
        helfem::Matrix Pnob(helfem::Matrix::Zero((Eigen::Index) Ndummy(), (Eigen::Index) Ndummy()));
        for(Eigen::Index j=0;j<n;j++)
          for(Eigen::Index i=0;i<n;i++)
            Pnob(pure_idx[i], pure_idx[j]) = Ppure(i,j);
        return Pnob;
      }

      Eigen::MatrixXcd TwoDBasis::eval_bf(size_t iel, size_t irad, double cth, double phi) const {
        // Evaluate spherical harmonics
        Eigen::VectorXcd sph(lval.size());
        for(size_t i=0;i<(size_t) lval.size();i++)
          sph(i)=::spherical_harmonics(lval(i),mval(i),cth,phi);

        // Evaluate radial functions (single quadrature point, 1 x Nc row)
        const helfem::Matrix rad(radial.get_bf(iel).row(irad));
        const Eigen::Index Nc = rad.cols();

        // Form supermatrix. sph(i) is complex, rad is real -> cast rad.
        Eigen::MatrixXcd bf(rad.rows(),lval.size()*Nc);
        for(size_t i=0;i<(size_t) lval.size();i++)
          bf.middleCols(i*Nc,Nc)=sph(i)*rad.cast<std::complex<double>>();

        return bf;
      }

      helfem::Matrix TwoDBasis::eval_bf(size_t iel, size_t irad, double cth, int m) const {
        return eval_bf(iel, irad, cth, m, radial.get_bf(iel));
      }

      helfem::Matrix TwoDBasis::eval_bf(size_t iel, size_t irad, double cth, int m, const helfem::Matrix & rad_all) const {
        // Figure out list of functions
        std::vector<size_t> flist;
        for(size_t i=0;i<(size_t) mval.size();i++)
          if(mval(i)==m)
            flist.push_back(i);

        // Evaluate spherical harmonics
        helfem::Vector sph(flist.size());
        for(size_t i=0;i<flist.size();i++)
          sph(i)=std::real(::spherical_harmonics(lval(flist[i]),mval(flist[i]),cth,0.0));

        // Radial functions at this quadrature point (1 x Nc row)
        const helfem::Matrix rad(rad_all.row(irad));
        const Eigen::Index Nc = rad.cols();

        // Form supermatrix
        helfem::Matrix bf(rad.rows(),flist.size()*Nc);
        for(size_t i=0;i<flist.size();i++)
          bf.middleCols(i*Nc,Nc)=sph(i)*rad;

        return bf;
      }

      Eigen::VectorXcd TwoDBasis::eval_bf(double mu, double cth, double phi) const {
	// Find out which element mu belongs to
	const helfem::Vector bval(radial.get_bval());
	Eigen::Index iel;
	for(iel=0;iel<bval.size()-1;iel++)
	  if(bval(iel)<=mu && mu<=bval(iel+1))
	    break;
	if(iel==bval.size()-1) {
	  std::ostringstream oss;
	  oss << "mu value " << mu << " not found!\n";
	  throw std::logic_error(oss.str());
	}

	// x value is then
	helfem::Vector x(1);
	x(0)=2.0*(mu-bval(iel))/(bval(iel+1)-bval(iel)) - 1.0;

	// Evaluate spherical harmonics
        Eigen::VectorXcd sph(lval.size());
        for(size_t i=0;i<(size_t) lval.size();i++)
          sph(i)=::spherical_harmonics(lval(i),mval(i),cth,phi);
        // Evaluate radial functions (1 x Nc row)
        const helfem::Matrix rad(radial.get_bf(iel,x));

	// Get indices of radial functions
	size_t ifirst, ilast;
        radial.get_idx(iel,ifirst,ilast);

        // Form supermatrix. arma used trans() on a real matrix = plain
        // transpose; sph(i) is complex so cast the transposed radial row.
	Eigen::VectorXcd bf(Eigen::VectorXcd::Zero(Ndummy()));
	for(size_t i=0;i<(size_t) lval.size();i++) {
	  bf.segment(i*radial.Nbf()+ifirst,ilast-ifirst+1)=sph(i)*rad.transpose().cast<std::complex<double>>();
	}

	return bf(pure_idx);
      }

      void TwoDBasis::eval_df(size_t iel, size_t irad, double cth, double phi, Eigen::MatrixXcd & dr, Eigen::MatrixXcd & dth, Eigen::MatrixXcd & dphi) const {
        // Evaluate spherical harmonics
        Eigen::VectorXcd sph(lval.size());
        for(size_t i=0;i<(size_t) lval.size();i++)
          sph(i)=::spherical_harmonics(lval(i),mval(i),cth,phi);

        // Evaluate radial functions (single quadrature point, 1 x Nc rows)
        const helfem::Matrix frad(radial.get_bf(iel).row(irad));
        const helfem::Matrix drad(radial.get_df(iel).row(irad));
        const Eigen::Index Nc = frad.cols();

        // Form supermatrices
        dr = Eigen::MatrixXcd::Zero(frad.rows(),lval.size()*Nc);
        dth = Eigen::MatrixXcd::Zero(frad.rows(),lval.size()*Nc);
        dphi = Eigen::MatrixXcd::Zero(frad.rows(),lval.size()*Nc);

        // Radial one is easy (complex scalar * real matrix -> cast)
        for(size_t i=0;i<(size_t) lval.size();i++)
          dr.middleCols(i*Nc,Nc)=sph(i)*drad.cast<std::complex<double>>();
        // and so is phi
        for(size_t i=0;i<(size_t) lval.size();i++)
          dphi.middleCols(i*Nc,Nc)=(std::complex<double>(0.0,mval(i))*sph(i))*frad.cast<std::complex<double>>();
        // sin^2(theta) = (1 - cth)(1 + cth) avoids the catastrophic
        // cancellation in 1 - cth*cth when cth is close to +/- 1.
        const double sinth = std::sqrt(std::max((1.0-cth)*(1.0+cth), 0.0));
        const double cotth = (sinth > 0.0) ? cth/sinth : 0.0;

        // but theta is nastier
        for(size_t i=0;i<(size_t) lval.size();i++) {
          int l(lval(i));
          int m(mval(i));

          // Angular factor
          std::complex<double> angfac(m*cotth*sph(i));
          if(mval(i)<lval(i))
            angfac+=sqrt((l-m)*(l+m+1))*std::exp(std::complex<double>(0,-phi))*::spherical_harmonics(lval(i),mval(i)+1,cth,phi);

          dth.middleCols(i*Nc,Nc)=angfac*frad.cast<std::complex<double>>();
        }
      }

      void TwoDBasis::eval_df(size_t iel, size_t irad, double cth, int m, helfem::Matrix & dr, helfem::Matrix & dth) const {
        // Functions belonging to this m block (same selection as
        // eval_bf(iel,irad,cth,m), so the columns line up).
        std::vector<size_t> flist;
        for(size_t i=0;i<(size_t) mval.size();i++)
          if(mval(i)==m)
            flist.push_back(i);

        // sin^2(theta) = (1 - cth)(1 + cth) avoids the catastrophic
        // cancellation in 1 - cth*cth when cth is close to +/- 1.
        const double sinth = std::sqrt(std::max((1.0-cth)*(1.0+cth), 0.0));
        const double cotth = (sinth > 0.0) ? cth/sinth : 0.0;

        // Angular factors, evaluated at phi = 0 where they are real.
        helfem::Vector sph(flist.size()), dsph(flist.size());
        for(size_t i=0;i<flist.size();i++) {
          const int l(lval(flist[i]));
          const int mm(mval(flist[i]));
          sph(i)=std::real(::spherical_harmonics(l,mm,cth,0.0));
          // d Y_l^m / d theta = m cot(th) Y_l^m + sqrt((l-m)(l+m+1)) Y_l^{m+1}
          // (the e^{-i phi} of the raising term is unity at phi = 0).
          double angfac = mm*cotth*sph(i);
          if(mm < l)
            angfac += std::sqrt((double)(l-mm)*(double)(l+mm+1))
                       * std::real(::spherical_harmonics(l,mm+1,cth,0.0));
          dsph(i)=angfac;
        }

        // Radial functions and their derivatives at the single radial point
        const helfem::Matrix frad(radial.get_bf(iel).row(irad));
        const helfem::Matrix drad(radial.get_df(iel).row(irad));
        const Eigen::Index Nc = frad.cols();

        dr = helfem::Matrix::Zero(frad.rows(),flist.size()*Nc);
        dth = helfem::Matrix::Zero(frad.rows(),flist.size()*Nc);
        for(size_t i=0;i<flist.size();i++) {
          dr.middleCols(i*Nc,Nc)=sph(i)*drad;
          dth.middleCols(i*Nc,Nc)=dsph(i)*frad;
        }
      }

      void TwoDBasis::eval_lf(size_t iel, size_t irad, double cth, int m, helfem::Matrix & lf) const {
        // Same function selection as eval_bf(iel,irad,cth,m) so the columns
        // line up.
        std::vector<size_t> flist;
        for(size_t i=0;i<(size_t) mval.size();i++)
          if(mval(i)==m)
            flist.push_back(i);

        // Geometry at this (mu, nu) point
        const double mu(radial.get_r(iel)(irad));
        const double shmu(std::sinh(mu)), chmu(std::cosh(mu));
        // sin^2(nu), written to avoid cancellation near |cth| = 1
        const double sth2(std::max((1.0-cth)*(1.0+cth), 0.0));
        // h^2 = Rhalf^2 (sinh^2 mu + sin^2 nu)
        const double h2(Rhalf*Rhalf*(shmu*shmu + sth2));
        const double cothmu((shmu>0.0) ? chmu/shmu : 0.0);
        // m^2/sinh^2(mu). Regular on the quadrature grid, where mu > 0; for
        // m != 0 the basis functions vanish as sinh^|m|(mu) on the axis anyway.
        const double m2_sh2((shmu>0.0) ? (m*m)/(shmu*shmu) : 0.0);

        // Radial functions and their first two mu derivatives (1 x Nc rows)
        const helfem::Matrix frad(radial.get_bf(iel).row(irad));
        const helfem::Matrix drad(radial.get_df(iel).row(irad));
        const helfem::Matrix d2rad(radial.get_d2f(iel).row(irad));
        const Eigen::Index Nc = frad.cols();

        // R'' + coth(mu) R' is common to every l in the block
        const helfem::Matrix radop(d2rad + cothmu*drad);

        lf = helfem::Matrix::Zero(frad.rows(),flist.size()*Nc);
        for(size_t i=0;i<flist.size();i++) {
          const int l(lval(flist[i]));
          const double sph(std::real(::spherical_harmonics(l,m,cth,0.0)));
          const double lfac(l*(l+1) + m2_sh2);
          lf.middleCols(i*Nc,Nc)
            = (sph/h2)*(radop - lfac*frad);
        }
      }

      std::vector<Eigen::Index> TwoDBasis::dummy_idx_to_real_idx(const std::vector<Eigen::Index> & idx) const {
        for(size_t i=0;i<idx.size();i++)
          if((size_t) idx[i]>=Ndummy())
            throw std::logic_error("Invalid index vector!\n");

        // idx is a subset of dummy indices, which need to be
        // converted to real indices.  we just need to build the map
        // from dummy to real indices. pure_indices() is the list of
        // real (dummy) indices, in real-index order.
        const std::vector<Eigen::Index> real_idx(pure_indices());
        // Now the list has all the info needed to construct the
        // mapping between the two
        std::map<Eigen::Index, Eigen::Index> mapping;
        for(size_t i=0;i<real_idx.size();i++) {
          mapping[real_idx[i]] = (Eigen::Index) i;
        }

        // Mapped indices
        std::vector<Eigen::Index> mapidx;
        for(size_t i=0;i<idx.size();i++) {
          // Try to find the function
          std::map<Eigen::Index, Eigen::Index>::const_iterator pos(mapping.find(idx[i]));
          // Dummy functions are not on the map
          if(pos == mapping.end())
            continue;
          // If we are here, the function is real
          mapidx.push_back(pos->second);
        }
        return mapidx;
      }

      std::vector<Eigen::Index> TwoDBasis::bf_list_dummy(size_t iel) const {
        // Radial functions in element
        size_t ifirst, ilast;
        radial.get_idx(iel,ifirst,ilast);
        // Number of radial functions in element
        size_t Nr(ilast-ifirst+1);

        // Total number of radial functions
        size_t Nrad(radial.Nbf());

        // List of functions in the element
        std::vector<Eigen::Index> idx(Nr*lval.size());
        for(size_t iam=0;iam<(size_t) lval.size();iam++)
          for(size_t j=0;j<Nr;j++)
            idx[iam*Nr+j]=(Eigen::Index) (Nrad*iam+ifirst+j);

        return idx;
      }

      std::vector<Eigen::Index> TwoDBasis::bf_list(size_t iel) const {
        return dummy_idx_to_real_idx(bf_list_dummy(iel));
      }

      std::vector<Eigen::Index> TwoDBasis::bf_list_dummy(size_t iel, int m) const {
        // Radial functions in element
        size_t ifirst, ilast;
        radial.get_idx(iel,ifirst,ilast);
        // Number of radial functions in element
        size_t Nr(ilast-ifirst+1);

        // Total number of radial functions
        size_t Nrad(radial.Nbf());

        // List of functions in the element
        std::vector<Eigen::Index> idx;
        for(size_t iam=0;iam<(size_t) lval.size();iam++)
          if(mval(iam)==m)
            for(size_t j=0;j<Nr;j++)
              idx.push_back((Eigen::Index) (Nrad*iam+ifirst+j));

        return idx;
      }

      size_t TwoDBasis::get_rad_Nel() const {
        return radial.Nel();
      }

      helfem::Matrix TwoDBasis::get_rad_bf(size_t iel) const {
        return radial.get_bf(iel);
      }

      helfem::Matrix TwoDBasis::get_rad_df(size_t iel) const {
        return radial.get_df(iel);
      }

      helfem::Matrix TwoDBasis::get_rad_d2f(size_t iel) const {
        return radial.get_d2f(iel);
      }

      helfem::Vector TwoDBasis::get_wrad(size_t iel) const {
        return radial.get_wrad(iel);
      }

      helfem::Vector TwoDBasis::get_r(size_t iel) const {
        return radial.get_r(iel);
      }

    }
  }
}
