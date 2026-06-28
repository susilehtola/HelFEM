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
#ifndef ATOMIC_BASIS_STO_H
#define ATOMIC_BASIS_STO_H

#include "RadialBasis.h"
#include "NAORadialBasis.h"
#include "FiniteElementBasis.h"
#include "PolynomialBasis.h"
#include <armadillo>
#include <cmath>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace helfem {
  // Forward declaration: helfem::chebyshev::chebyshev (defined in
  // libhelfem/src/chebyshev.h; not exposed via libhelfem/include/).
  namespace chebyshev {
    void chebyshev(int n, arma::vec & x, arma::vec & w);
  }
  namespace atomic {
    namespace basis {

      /// A single Slater-type primitive
      ///   R_i(r) = N_i * r^{n - 1} * exp(-zeta * r)
      ///   u_i(r) = r * R_i(r) = N_i * r^n * exp(-zeta * r)
      /// with n >= 1, zeta > 0. The normalisation N_i is chosen so the
      /// individual primitive has unit radial self-overlap
      /// integral_0^inf u_i^2 dr = 1, i.e.
      ///   N_i = sqrt((2 zeta)^{2n+1} / (2n)!).
      struct STOPrimitive {
        int    n;
        double zeta;
      };

      /// A contracted STO basis function: f(r) = sum_i c_i R_i(r),
      /// each R_i a STOPrimitive. The contraction is unrenormalised --
      /// callers can pass arbitrary coefficients and the factory below
      /// will produce an L2-orthonormal NAO basis from them.
      struct STOContracted {
        std::vector<STOPrimitive> primitives;
        arma::vec                 contraction;
      };

      namespace detail_sto {

        /// Per-primitive norm: N = sqrt((2 zeta)^{2n+1} / (2n)!).
        inline double primitive_norm(const STOPrimitive & p) {
          double f = 1.0;
          for (int k = 2; k <= 2 * p.n; ++k) f *= k;
          return std::sqrt(std::pow(2.0 * p.zeta, 2 * p.n + 1) / f);
        }

        /// u(r) = r * R(r) for the contracted basis function.
        inline double eval_u(const STOContracted & c, double r) {
          double sum = 0.0;
          for (size_t i = 0; i < c.primitives.size(); ++i) {
            const auto & p = c.primitives[i];
            sum += c.contraction(i) * primitive_norm(p) *
                   std::pow(r, p.n) * std::exp(-p.zeta * r);
          }
          return sum;
        }

        /// Radial extent: smallest r > r_peak at which |u(r)| <= tol * |u_peak|.
        /// Done per primitive (the contracted function's tail is dominated by
        /// the smallest-zeta primitive); the basis-wide r_max is the max
        /// over all primitives over all contracted basis functions.
        ///
        /// For u(r) = N r^n e^{-zeta r}:
        ///   r_peak = n / zeta,  u_peak = N (n/zeta)^n e^{-n}
        ///   u(r)/u_peak = (r/r_peak)^n * exp(n - zeta r) -- monotonically
        ///   decreasing for r > r_peak, so bisect for the threshold.
        inline double primitive_r_extent(const STOPrimitive & p, double tol) {
          const double r_peak = double(p.n) / p.zeta;
          auto u_over_peak = [&](double r) {
            return std::pow(r / r_peak, p.n) *
                   std::exp(double(p.n) - p.zeta * r);
          };
          double lo = r_peak;
          double hi = r_peak + 200.0 / p.zeta;
          while (u_over_peak(hi) > tol) hi *= 2.0;
          for (int it = 0; it < 200; ++it) {
            double mid = 0.5 * (lo + hi);
            if (u_over_peak(mid) > tol) lo = mid; else hi = mid;
            if ((hi - lo) < 1e-12 * std::max(1.0, r_peak)) break;
          }
          return hi;
        }

      } // namespace detail_sto

      /// Build a NAORadialBasis whose NAO functions are FE-projected
      /// representations of the input STO basis.
      ///
      /// The underlying FE basis is constructed to:
      ///   - extend from 0 to r_max, where r_max is chosen so that EVERY
      ///     input primitive's value at r_max is at most `tol` times its
      ///     peak value (default tol = 1e-12 ~ sqrt(double eps));
      ///   - have `nelem` exponentially-distributed elements with
      ///     `nnodes`-node LIPs.
      /// Each NAO column is the L2 projection of one input STO onto the
      /// FE basis: solve S c = <u_i, u_STO> via fem.vector_element.
      ///
      /// All matrix elements on the returned NAORadialBasis are then
      /// computed by the FE quadrature -- since the FE basis is dense
      /// enough to represent the STOs to ~tol precision, the integrals
      /// are tight to that precision too. Convergence is systematic:
      /// tighten tol or increase nelem/nnodes -> tighter integrals.
      inline NAORadialBasis make_nao_sto(
          const std::vector<STOContracted> & basis,
          double tol     = 1e-12,
          int    nelem   = 30,
          int    nnodes  = 15) {
        if (basis.empty())
          throw std::logic_error("make_nao_sto: empty basis.\n");
        for (const auto & c : basis) {
          if (c.primitives.empty() ||
              c.contraction.n_elem != c.primitives.size())
            throw std::logic_error("make_nao_sto: malformed STOContracted.\n");
          for (const auto & p : c.primitives) {
            if (p.n < 1)
              throw std::logic_error("make_nao_sto: STO n must be >= 1.\n");
            if (p.zeta <= 0.0)
              throw std::logic_error("make_nao_sto: STO zeta must be > 0.\n");
          }
        }
        // Pick r_max as the largest primitive r_extent across the basis.
        double r_max = 0.0;
        for (const auto & c : basis)
          for (const auto & p : c.primitives)
            r_max = std::max(r_max, detail_sto::primitive_r_extent(p, tol));
        // Exponential grid from 0 to r_max: bval(k) = r_max * (exp(2*k/nelem) - 1) /
        // (exp(2) - 1). Same shape used elsewhere in the test suite.
        arma::vec bval(nelem + 1);
        const double denom = std::exp(2.0) - 1.0;
        for (int k = 0; k <= nelem; ++k)
          bval(k) = r_max * (std::exp(2.0 * k / nelem) - 1.0) / denom;
        // FE basis with Dirichlet BCs at both ends -- consistent with
        // u(0) = u(r_max) = 0 for STOs (vanishes at origin since u = r * R
        // with R(0) finite; vanishes at r_max by construction).
        auto poly = std::shared_ptr<const polynomial_basis::PolynomialBasis>(
            polynomial_basis::get_basis(/*primbas=*/4, nnodes));
        polynomial_basis::FiniteElementBasis fem(poly, bval,
            /*zero_func_left*/true,  /*zero_deriv_left*/false,
            /*zero_func_right*/true, /*zero_deriv_right*/false);
        atomic::basis::FEMRadialBasis radial(fem, 5 * poly->get_nbf());
        const arma::mat S = radial.overlap();
        // Quadrature points + weights for the projection integrals.
        arma::vec xq, wq;
        helfem::chebyshev::chebyshev(5 * poly->get_nbf(), xq, wq);
        // Project each STO onto the FE basis: c = S^{-1} <u_i, u_STO>.
        arma::mat C(S.n_rows, basis.size(), arma::fill::zeros);
        for (size_t k = 0; k < basis.size(); ++k) {
          const STOContracted & sto = basis[k];
          auto f = [&sto](double r) { return detail_sto::eval_u(sto, r); };
          arma::vec b = fem.vector_element(/*der=*/0, xq, wq, f);
          C.col(k) = arma::solve(S, b);
        }
        return NAORadialBasis::from_owned_radial(std::move(radial), std::move(C));
      }

    } // namespace basis
  } // namespace atomic
} // namespace helfem

#endif
