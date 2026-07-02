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
#ifndef ATOMIC_BASIS_GTO_H
#define ATOMIC_BASIS_GTO_H

#include "RadialBasis.h"
#include "ArmaEigen.h"
#include "NAORadialBasis.h"
#include "FiniteElementBasis.h"
#include "PolynomialBasis.h"
#include <lib1dfem/chebyshev.h>
#include <Eigen/Cholesky>
#include <cmath>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace helfem {
  namespace atomic {
    namespace basis {

      /// A single Gaussian-type primitive
      ///   R_i(r) = N_i * r^n * exp(-alpha * r^2)
      ///   u_i(r) = r * R_i(r) = N_i * r^{n+1} * exp(-alpha * r^2)
      /// with n >= 0 (typically n = l for a primitive of angular
      /// momentum l), alpha > 0. The normalisation N_i is chosen so
      /// integral_0^inf u_i^2 dr = 1, i.e.
      ///   N_i = 1 / sqrt( (1/2) Gamma(n + 3/2) / (2 alpha)^{n + 3/2} ).
      struct GTOPrimitive {
        int    n;
        double alpha;
      };

      /// A contracted GTO basis function: f(r) = sum_i c_i R_i(r).
      struct GTOContracted {
        std::vector<GTOPrimitive> primitives;
        /// Phase 5.27: Eigen at the consumer boundary. Users can
        /// construct via `helfem::Vector::Map(data, n)` from any
        /// double* range without an armadillo dependency.
        helfem::Vector            contraction;
      };

      namespace detail_gto {

        /// (1/2) Gamma((p+1)/2) / a^{(p+1)/2}, p >= 0.
        inline double I(int p, double a) {
          const double half = 0.5 * (p + 1);
          return 0.5 * std::tgamma(half) / std::pow(a, half);
        }
        /// Per-primitive norm: N = 1 / sqrt(I(2n+2, 2 alpha)).
        inline double primitive_norm(const GTOPrimitive & p) {
          return 1.0 / std::sqrt(I(2 * p.n + 2, 2.0 * p.alpha));
        }
        /// u(r) = r * R(r) for the contracted basis function.
        inline double eval_u(const GTOContracted & c, double r) {
          double sum = 0.0;
          for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(c.primitives.size()); ++i) {
            const auto & p = c.primitives[i];
            sum += c.contraction(i) * primitive_norm(p) *
                   std::pow(r, p.n + 1) * std::exp(-p.alpha * r * r);
          }
          return sum;
        }

        /// Radial extent for u = N r^{n+1} e^{-alpha r^2}.
        /// Peak at r_peak = sqrt((n+1)/(2 alpha)); u_peak well-defined.
        /// u(r)/u_peak monotonically decreasing past r_peak, bisect.
        inline double primitive_r_extent(const GTOPrimitive & p, double tol) {
          const int m = p.n + 1;            // power of r in u
          const double r_peak = std::sqrt(double(m) / (2.0 * p.alpha));
          auto u_over_peak = [&](double r) {
            // (r/r_peak)^m * exp((alpha)(r_peak^2 - r^2))
            return std::pow(r / r_peak, m) *
                   std::exp(p.alpha * (r_peak * r_peak - r * r));
          };
          double lo = r_peak;
          double hi = r_peak + 20.0 / std::sqrt(p.alpha);
          while (u_over_peak(hi) > tol) hi *= 2.0;
          for (int it = 0; it < 200; ++it) {
            double mid = 0.5 * (lo + hi);
            if (u_over_peak(mid) > tol) lo = mid; else hi = mid;
            if ((hi - lo) < 1e-12 * std::max(1.0, r_peak)) break;
          }
          return hi;
        }

      } // namespace detail_gto

      /// Build a NAORadialBasis whose NAO functions are FE-projected
      /// representations of the input GTO basis. Same algorithm as
      /// make_nao_sto -- only the primitive form differs.
      inline NAORadialBasis make_nao_gto(
          const std::vector<GTOContracted> & basis,
          double tol     = 1e-12,
          int    nelem   = 30,
          int    nnodes  = 15) {
        if (basis.empty())
          throw std::logic_error("make_nao_gto: empty basis.\n");
        for (const auto & c : basis) {
          if (c.primitives.empty() ||
              static_cast<size_t>(c.contraction.size()) != c.primitives.size())
            throw std::logic_error("make_nao_gto: malformed GTOContracted.\n");
          for (const auto & p : c.primitives) {
            if (p.n < 0)
              throw std::logic_error("make_nao_gto: GTO n must be >= 0.\n");
            if (p.alpha <= 0.0)
              throw std::logic_error("make_nao_gto: GTO alpha must be > 0.\n");
          }
        }
        double r_max = 0.0;
        for (const auto & c : basis)
          for (const auto & p : c.primitives)
            r_max = std::max(r_max, detail_gto::primitive_r_extent(p, tol));
        helfem::Vector bval(nelem + 1);
        const double denom = std::exp(2.0) - 1.0;
        for (int k = 0; k <= nelem; ++k)
          bval(k) = r_max * (std::exp(2.0 * k / nelem) - 1.0) / denom;
        auto poly = std::shared_ptr<const polynomial_basis::PolynomialBasis>(
            polynomial_basis::get_basis(/*primbas=*/4, nnodes));
        polynomial_basis::FiniteElementBasis fem(poly, bval,
            /*zero_func_left*/true,  /*zero_deriv_left*/false,
            /*zero_func_right*/true, /*zero_deriv_right*/false);
        atomic::basis::FEMRadialBasis radial(fem, 5 * poly->get_nbf());
        const helfem::Matrix S = radial.overlap();
        // Quadrature points + weights for the projection integrals. Go
        // straight to lib1dfem's Eigen-typed templated chebyshev to skip
        // libhelfem's legacy arma-shim.
        helfem::Vector xq, wq;
        helfem::lib1dfem::chebyshev::chebyshev<double>(
            5 * poly->get_nbf(), xq, wq);
        helfem::Matrix C = helfem::Matrix::Zero(S.rows(), basis.size());
        // Overlap is symmetric positive definite -- one LDLT solves all.
        Eigen::LDLT<helfem::Matrix> S_ldlt(S);
        for (size_t k = 0; k < basis.size(); ++k) {
          const GTOContracted & gto = basis[k];
          auto f = [&gto](double r) { return detail_gto::eval_u(gto, r); };
          const helfem::Vector be = fem.vector_element(/*der=*/0, xq, wq, f);
          C.col(k) = S_ldlt.solve(be);
        }
        // NAORadialBasis::from_owned_radial still takes arma::mat for
        // its C_ storage; bridge once at the boundary.
        return NAORadialBasis::from_owned_radial(std::move(radial),
                                                  helfem::to_arma(C));
      }

    } // namespace basis
  } // namespace atomic
} // namespace helfem

#endif
