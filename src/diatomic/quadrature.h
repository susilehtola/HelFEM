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
#ifndef INTEGRALS_H
#define INTEGRALS_H

#include <Matrix.h>
#include <memory>
#include "../general/legendretable.h"
#include "../legendre/Legendre.h"
#include <unordered_map>
#include <cmath>
#include "PolynomialBasis.h"
#include <vector>

namespace helfem {
  namespace diatomic {
    namespace quadrature {

    // Associated Legendre values at the quadrature points of ONE element
    // and ONE |M|, shared across that group's L values.
    //
    // The recurrence runs forward in L at fixed m, so a single pass yields
    // every L of an |M| for what one (L, |M|) costs. The disjoint-integral
    // build asks for one (L, |M|) at a time, though, and each ask used to
    // rebuild the whole table and keep one entry -- 58.7 million times in a
    // 30 s H2 run, against 395 thousand hits on the shared LegendreTable,
    // whose stored points are those of the BASE rule and so never match the
    // auto-converging refinement's.
    //
    // Caching here rather than in LegendreTable is what makes the scope
    // right: the entries are exactly the points this (element, |M|) group
    // asks for, they are reused by every L in the group and by all four
    // integral families, and they are released when the group is done. One
    // instance per (element, |M|) is built by the caller, so it is
    // thread-local by construction and needs no locking.
    class MLegendreCache {
      int Mabs, Lmax;
      std::unordered_map<double, std::vector<double>> Ptab, Qtab;

      const std::vector<double> & fetch(bool wantP, double chmu) {
        std::unordered_map<double, std::vector<double>> & tab = wantP ? Ptab : Qtab;
        auto it = tab.find(chmu);
        if(it != tab.end())
          return it->second;
        // One recurrence pass; keep this |M|'s column.
        std::vector<double> full((size_t) (Lmax+1)*(Mabs+1), 0.0);
        if(wantP)
          ::helfem::legendre::plm(full.data(), Lmax, Mabs, chmu);
        else
          ::helfem::legendre::qlm(full.data(), Lmax, Mabs, chmu);
        std::vector<double> col((size_t) (Lmax+1));
        for(int L=0;L<=Lmax;L++) {
          double v = full[(size_t) L + (size_t) Mabs*(Lmax+1)];
          if(v!=0.0 && !std::isnormal(v))
            v=0.0;                      // matches LegendreTable's filtering
          col[(size_t) L] = v;
        }
        return tab.emplace(chmu, std::move(col)).first->second;
      }

    public:
      MLegendreCache(int Mabs_, int Lmax_) : Mabs(Mabs_), Lmax(Lmax_) {}
      /// Re-point at another |M| group, dropping the previous group's
      /// entries: the caller walks |M| runs in order and never returns to
      /// one, so nothing is lost and the footprint stays that of a single
      /// group.
      void reset(int Mabs_, int Lmax_) {
        Mabs=Mabs_; Lmax=Lmax_; Ptab.clear(); Qtab.clear();
      }
      /// Q is logarithmically singular at chmu == 1; the table reports zero.
      double P(int L, double chmu) { return fetch(true,  chmu)[(size_t) L]; }
      double Q(int L, double chmu) { return (chmu==1.0) ? 0.0 : fetch(false, chmu)[(size_t) L]; }
    };

      /**
       * Computes the inner in-element two-electron integral:
       * \f$ \phi^{l,LM}(\mu) = \int_{0}^{\mu}d\mu'\cosh^{l}\mu'\sinh\mu'B_{\gamma}(\mu')B_{\delta}(\mu')P_{L,|M|}(\cosh\mu') \f$
       */
      helfem::Matrix twoe_inner_integral(double mumin, double mumax, int l, const helfem::Vector & x, const helfem::Vector & wx, const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly, int L, int M, MLegendreCache & tab);

      /**
       * Computes a primitive two-electron in-element integral.
       * Cross-element integrals reduce to products of radial integrals.
       * Note that the routine needs the polynomial representation.
       */
      helfem::Matrix twoe_integral(double rmin, double rmax, int k, int l, const helfem::Vector & x, const helfem::Vector & wx, const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly, int L, int M, MLegendreCache & tab);

      /**
       * Everything in the two-electron in-element integrals that depends only
       * on the ELEMENT and the quadrature rule -- not on (k, l, L, M).
       *
       * The polynomials are the expensive part (profiling put the FEM
       * evaluation at ~35% of a run), and they were being re-evaluated inside
       * the (L, M) x (alpha, beta) loops: compute_tei asks for 4 (alpha,beta)
       * combinations per (element, L, M), each of which runs twoe_integral_wrk
       * twice, and each of those re-evaluated the outer basis, the outer
       * product table, and one inner basis per subinterval. None of that
       * depends on k, l, L or M.
       *
       * Build this once per element and hand it to twoe_integral() below.
       */
      struct TwoElectronElement {
        /// Quadrature nodes and weights on [-1, 1]
        helfem::Vector x, wx;
        /// Half-length of the element in mu
        double mulen;
        /// Outer quadrature: mu, cosh(mu), sinh(mu) at the element's points
        helfem::Vector mu, chmu, shmu;
        /// Outer basis functions, (nquad x nbf)
        helfem::Matrix bf;
        /// Outer product table B_i(mu) B_j(mu), (nquad x nbf^2)
        helfem::Matrix bfprod;

        /// Per-subinterval data for the cumulative inner integral. Subinterval
        /// ip runs from mu(ip-1) to mu(ip) (and from mumin to mu(0) for ip=0),
        /// and each uses a fresh set of nquad points.
        struct Subinterval {
          /// Half-length of the subinterval
          double mulen;
          /// cosh(mu), sinh(mu) at the subinterval's points
          helfem::Vector chmu, shmu;
          /// Basis functions there, (nquad x nbf)
          helfem::Matrix bf;
        };
        std::vector<Subinterval> sub;
      };

      /// Build the element-only data above. Call once per element.
      TwoElectronElement twoe_element(double mumin, double mumax, const helfem::Vector & x, const helfem::Vector & wx, const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly);

      /// Primitive two-electron in-element integral, reusing precomputed
      /// element data. Equivalent to the twoe_integral() above, but without
      /// re-evaluating any polynomial.
      helfem::Matrix twoe_integral(const TwoElectronElement & el, int k, int l, int L, int M, MLegendreCache & tab);
    }
  }
}

#endif
