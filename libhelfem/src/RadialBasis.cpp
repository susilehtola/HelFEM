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
#include "RadialBasis.h"
#include "ArmaEigen.h"
#include "RadialPotential.h"
#include "chebyshev.h"
#include "quadrature.h"
#include "utils.h"
#include <cfloat>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace helfem {
  namespace atomic {
    namespace basis {
      namespace {
        // Phase 5.3/5.4 helpers: FiniteElementBasis methods are now
        // Eigen-typed. RadialBasis keeps its arma::vec/arma::mat surface
        // for downstream chemistry code; bridge inline at every fem.*
        // call site via these one-line memcpy converters.
        inline helfem::Vector arma_to_eigen_vec(const arma::vec & v) {
          helfem::Vector e(v.n_elem);
          std::memcpy(e.data(), v.memptr(), sizeof(double) * v.n_elem);
          return e;
        }
        inline arma::mat eigen_mat_to_arma(const helfem::Matrix & m) {
          arma::mat out(m.rows(), m.cols());
          std::memcpy(out.memptr(), m.data(),
                      sizeof(double) * static_cast<size_t>(m.size()));
          return out;
        }
        inline arma::vec eigen_vec_to_arma(const helfem::Vector & v) {
          arma::vec out(v.size());
          std::memcpy(out.memptr(), v.data(), sizeof(double) * v.size());
          return out;
        }
      } // namespace

      FEMRadialBasis::FEMRadialBasis() {}

      FEMRadialBasis::FEMRadialBasis(const polynomial_basis::FiniteElementBasis & fem_, int n_quad) : fem(fem_) {
        // Phase 5.6: xq/wq are Eigen. Call lib1dfem::chebyshev directly
        // (the libhelfem chebyshev shim only has an arma::vec overload).
        helfem::lib1dfem::chebyshev::chebyshev<double>(n_quad, xq, wq);
        for (Eigen::Index i = 0; i < xq.size(); ++i) {
          if (!std::isfinite(xq(i)))
            printf("xq[%lld]=%e\n", (long long) i, xq(i));
          if (!std::isfinite(wq(i)))
            printf("wq[%lld]=%e\n", (long long) i, wq(i));
        }
      }

      FEMRadialBasis::~FEMRadialBasis() {}

      int FEMRadialBasis::get_nquad() const {
        return (int) xq.size();
      }

      helfem::Vector FEMRadialBasis::get_xq() const {
        // Phase 5.20: xq storage is Eigen; return it directly at the public boundary.
        return xq;
      }

      size_t FEMRadialBasis::Nbf() const {
        return fem.get_nbf();
      }

      size_t FEMRadialBasis::Nel() const {
        return fem.get_nelem();
      }

      size_t FEMRadialBasis::Nprim(size_t iel) const {
        return fem.get_nprim(iel);
      }

      size_t FEMRadialBasis::max_Nprim() const {
        return fem.get_max_nprim();
      }

      void FEMRadialBasis::get_idx(size_t iel, size_t &ifirst, size_t &ilast) const {
        fem.get_idx(iel, ifirst, ilast);
      }

      helfem::Vector FEMRadialBasis::get_bval() const {
        // Phase 5.20: fem.get_bval() is Eigen; return directly.
        return fem.get_bval();
      }

      int FEMRadialBasis::get_poly_id() const {
        return fem.get_poly_id();
      }

      int FEMRadialBasis::get_poly_nnodes() const {
        return fem.get_poly_nnodes();
      }

      helfem::Matrix FEMRadialBasis::radial_integral(int Rexp, size_t iel, double x_left, double x_right) const {
        // <R | r^Rexp | R> with the FE-natural dr measure means weight r^(Rexp+2):
        //   integral B(r)^2 r^(Rexp+2) / r^2  dr  =  integral R(r)^2 r^(Rexp+2)  dr
        // = QM <r^Rexp> = integral R^2 r^Rexp r^2 dr.
        const std::function<double(double)> rpowL =
            [Rexp](double r){ return std::pow(r, Rexp + 2); };
        helfem::Matrix ret = matrix_element(iel, BasisKind::R0, BasisKind::R0,
                                            rpowL, x_left, x_right);
        if (ret.array().isNaN().any())
          printf("radial_integral(%i,%i) has NaN!\n", Rexp, (int)iel);
        return ret;
      }

      // Phase 5.4: matrix_element fn-pointer is now Eigen-typed.
      // get_bf/df/lf are still arma; bridge there per call.
      static std::function<helfem::Matrix(helfem::Vector, size_t)>
      make_evaluator(const FEMRadialBasis * rb, FEMRadialBasis::BasisKind k) {
        using BK = FEMRadialBasis::BasisKind;
        switch (k) {
          case BK::B0: return [rb](helfem::Vector x, size_t iel) {
            return rb->get_fem().eval_f(x, iel);
          };
          case BK::B1: return [rb](helfem::Vector x, size_t iel) {
            return rb->get_fem().eval_df(x, iel);
          };
          case BK::B2: return [rb](helfem::Vector x, size_t iel) {
            return rb->get_fem().eval_d2f(x, iel);
          };
          case BK::R0: return [rb](helfem::Vector x, size_t iel) {
            return rb->get_bf(x, iel);
          };
          case BK::R1: return [rb](helfem::Vector x, size_t iel) {
            return rb->get_df(x, iel);
          };
          case BK::R2: return [rb](helfem::Vector x, size_t iel) {
            return rb->get_lf(x, iel);
          };
        }
        throw std::logic_error("FEMRadialBasis::matrix_element: unknown BasisKind\n");
      }

      helfem::Matrix FEMRadialBasis::matrix_element(
          size_t iel, BasisKind bra, BasisKind ket,
          const std::function<double(double)> & weight) const {
        // Phase 5.4: fem.matrix_element is Eigen; xq, wq members
        // are still arma -- bridge.
        auto lhs = make_evaluator(this, bra);
        auto rhs = make_evaluator(this, ket);
        return fem.matrix_element(iel, lhs, rhs, xq, wq, weight);
      }

      helfem::Matrix FEMRadialBasis::matrix_element(
          BasisKind bra, BasisKind ket,
          const std::function<double(double)> & weight) const {
        auto lhs = make_evaluator(this, bra);
        auto rhs = make_evaluator(this, ket);
        return fem.matrix_element(lhs, rhs, xq, wq, weight);
      }

      helfem::Matrix FEMRadialBasis::matrix_element(
          size_t iel, BasisKind bra, BasisKind ket,
          const std::function<double(double)> & weight,
          double x_left, double x_right) const {
        auto lhs = make_evaluator(this, bra);
        auto rhs = make_evaluator(this, ket);
        return fem.matrix_element(iel, lhs, rhs, xq, wq, weight, x_left, x_right);
      }

      helfem::Matrix FEMRadialBasis::bessel_il_integral(int L, double lambda, size_t iel) const {
        return matrix_element(iel, BasisKind::B0, BasisKind::B0,
                              [L, lambda](double r){ return utils::bessel_il(r * lambda, L); });
      }

      helfem::Matrix FEMRadialBasis::bessel_kl_integral(int L, double lambda, size_t iel) const {
        return matrix_element(iel, BasisKind::B0, BasisKind::B0,
                              [L, lambda](double r){ return utils::bessel_kl(r * lambda, L); });
      }

      // Per-element B-evaluator for cross-basis quadrature. Only B-kinds
      // are meaningful cross-basis (R-kinds carry a per-element 1/r factor
      // that's tied to a single basis's element-length scaling).
      // Phase 5.8: cross-basis B-evaluator rewritten in native Eigen.
      static helfem::Matrix eval_B_at(const polynomial_basis::FiniteElementBasis & fem,
                                       const helfem::Vector & x, size_t iel,
                                       FEMRadialBasis::BasisKind k) {
        using BK = FEMRadialBasis::BasisKind;
        switch (k) {
          case BK::B0: return fem.eval_f  (x, iel);
          case BK::B1: return fem.eval_df (x, iel);
          case BK::B2: return fem.eval_d2f(x, iel);
          case BK::R0: case BK::R1: case BK::R2:
            throw std::logic_error(
                "FEMRadialBasis::matrix_element(cross-basis): R-kinds are not "
                "supported (R = B/r is tied to one basis's element-length "
                "scaling; use B-kinds + an explicit weight function instead).\n");
        }
        throw std::logic_error("eval_B_at: unknown BasisKind\n");
      }

      helfem::Matrix FEMRadialBasis::matrix_element(
          const FEMRadialBasis & rh,
          BasisKind bra, BasisKind ket,
          const std::function<double(double)> & weight) const {
        // Pick a quadrature rule sized for the finer of the two bases.
        const Eigen::Index n_quad = std::max(xq.size(), rh.xq.size());
        helfem::Vector xproj, wproj;
        helfem::lib1dfem::chebyshev::chebyshev<double>((int) n_quad, xproj, wproj);

        // List overlapping (iel, jel) element pairs.
        std::vector<std::vector<size_t>> overlap(fem.get_nelem());
        for (size_t iel = 0; iel < fem.get_nelem(); ++iel) {
          const double istart = fem.element_begin(iel);
          const double iend   = fem.element_end(iel);
          for (size_t jel = 0; jel < rh.fem.get_nelem(); ++jel) {
            const double jstart = rh.fem.element_begin(jel);
            const double jend   = rh.fem.element_end(jel);
            if ((jstart >= istart && jstart < iend) ||
                (istart >= jstart && istart < jend))
              overlap[iel].push_back(jel);
          }
        }

        helfem::Matrix S = helfem::Matrix::Zero(Nbf(), rh.Nbf());
        for (size_t iel = 0; iel < fem.get_nelem(); ++iel) {
          for (size_t jel : overlap[iel]) {
            const double intstart = std::max(fem.element_begin(iel),
                                             rh.fem.element_begin(jel));
            const double intend   = std::min(fem.element_end(iel),
                                             rh.fem.element_end(jel));
            const double intmid   = 0.5 * (intend + intstart);
            const double intlen   = 0.5 * (intend - intstart);

            const helfem::Vector r =
                helfem::Vector::Constant(xproj.size(), intmid) + intlen * xproj;

            const helfem::Vector xi = fem.eval_prim(r, iel);
            const helfem::Vector xj = rh.fem.eval_prim(r, jel);

            helfem::Vector wtot = wproj * intlen;
            if (weight)
              for (Eigen::Index i = 0; i < r.size(); ++i)
                wtot(i) *= weight(r(i));

            const helfem::Matrix ifunc = eval_B_at(fem,    xi, iel, bra);
            const helfem::Matrix jfunc = eval_B_at(rh.fem, xj, jel, ket);

            size_t ifirst, ilast; get_idx(iel, ifirst, ilast);
            size_t jfirst, jlast; rh.get_idx(jel, jfirst, jlast);
            const Eigen::Index Ni = ilast - ifirst + 1;
            const Eigen::Index Nj = jlast - jfirst + 1;
            // ifunc^T * diag(wtot) * jfunc = (ifunc^T * wtot.asDiagonal()) * jfunc.
            S.block((Eigen::Index) ifirst, (Eigen::Index) jfirst, Ni, Nj)
                += ifunc.transpose() * wtot.asDiagonal() * jfunc;
          }
        }
        return S;
      }

      helfem::Matrix FEMRadialBasis::radial_integral(const FEMRadialBasis &rh, int n,
                                                     bool lhder, bool rhder) const {
        modelpotential::RadialPotential rad(n);
        return model_potential(rh, &rad, lhder, rhder);
      }

      helfem::Matrix FEMRadialBasis::model_potential(const FEMRadialBasis &rh,
                                                     const modelpotential::ModelPotential *model,
                                                     bool lhder, bool rhder) const {
        return matrix_element(rh,
                              lhder ? BasisKind::B1 : BasisKind::B0,
                              rhder ? BasisKind::B1 : BasisKind::B0,
                              [model](double r){ return model->V(r); });
      }

      helfem::Matrix FEMRadialBasis::overlap(const FEMRadialBasis &rh) const {
        return radial_integral(rh, 0);
      }

      // The four named matrix elements below are the proof-of-concept
      // migration onto the BasisKind dispatcher introduced above. Math is
      // unchanged; each routine collapses to a single composition of
      // (bra evaluator, ket evaluator, weight function).
      //
      //   overlap     = integral B(r) B(r) dr                    -> B0 * B0 * 1
      //   kinetic     = (1/2) integral B'(r) B'(r) dr            -> B1 * B1 * 1
      //   kinetic_l   = (1/2) integral R(r) R(r) dr              -> R0 * R0 * 1
      //                (= (1/2) <u | 1/r^2 | u> = half-centrifugal per l(l+1))
      //   nuclear     = - integral R(r) * r * R(r) dr            -> R0 * R0 * r
      //                (= - <u | 1/r | u>; sign included, Z=1 implicit)

      static const std::function<double(double)> kNoWeight;  // default-constructed = identity
      static const std::function<double(double)> kRWeight = [](double r){ return r; };

      // Phase 2a: matrix_element returns helfem::Matrix natively, so the
      // four virtuals are now one-line direct expressions (Eigen scalar
      // arithmetic works on the return value with no manual wrapping).
      // Per-element variants stay arma-typed; bridge with to_arma at the
      // matrix_element call site.
      helfem::Matrix FEMRadialBasis::overlap()      const { return matrix_element(BasisKind::B0, BasisKind::B0, kNoWeight); }
      helfem::Matrix FEMRadialBasis::overlap(size_t iel) const { return matrix_element(iel, BasisKind::B0, BasisKind::B0, kNoWeight); }

      helfem::Matrix FEMRadialBasis::kinetic()      const { return 0.5 * matrix_element(BasisKind::B1, BasisKind::B1, kNoWeight); }
      helfem::Matrix FEMRadialBasis::kinetic(size_t iel) const { return 0.5 * matrix_element(iel, BasisKind::B1, BasisKind::B1, kNoWeight); }

      helfem::Matrix FEMRadialBasis::kinetic_l()    const { return 0.5 * matrix_element(BasisKind::R0, BasisKind::R0, kNoWeight); }
      helfem::Matrix FEMRadialBasis::kinetic_l(size_t iel) const { return 0.5 * matrix_element(iel, BasisKind::R0, BasisKind::R0, kNoWeight); }

      helfem::Matrix FEMRadialBasis::nuclear()      const { return -matrix_element(BasisKind::R0, BasisKind::R0, kRWeight); }
      helfem::Matrix FEMRadialBasis::nuclear(size_t iel) const { return -matrix_element(iel, BasisKind::R0, BasisKind::R0, kRWeight); }

      helfem::Matrix FEMRadialBasis::polynomial_confinement(size_t iel, int N, double shift_pot) const {
        return matrix_element(iel, BasisKind::R0, BasisKind::R0,
                              [N, shift_pot](double r) {
                                return (r < shift_pot)
                                    ? 0.0
                                    : std::pow(r - shift_pot, N + 2);
                              });
      }

      helfem::Matrix FEMRadialBasis::exponential_confinement(size_t iel, int N, double r_0, double shift_pot) const {
        return matrix_element(iel, BasisKind::B0, BasisKind::B0,
                              [r_0, N, shift_pot](double r) {
                                if (r < shift_pot) return 0.0;
                                const double r_ratio = (r - shift_pot) / r_0;
                                double fact = 1.0;
                                double V = 0.0;
                                double r_ratio_pow_k = 1.0;
                                for (int k = 0; k < N; ++k) {
                                  V -= r_ratio_pow_k / fact;
                                  fact *= k + 1;
                                  r_ratio_pow_k *= r_ratio;
                                }
                                V += std::exp(r_ratio);
                                V *= fact;
                                return V;
                              });
      }

      helfem::Matrix FEMRadialBasis::barrier_confinement(size_t iel, double V, double shift_pot) const {
        return matrix_element(iel, BasisKind::B0, BasisKind::B0,
                              [V, shift_pot](double r) {
                                return (r < shift_pot) ? 0.0 : V;
                              });
      }

      helfem::Matrix FEMRadialBasis::junq_confinement(size_t iel, int N, double V0, double r_c, double shift_pot) const {
        return matrix_element(iel, BasisKind::B0, BasisKind::B0,
                              [N, r_c, V0, shift_pot](double r) {
                                if (r < shift_pot) return 0.0;
                                const double denominator  = std::pow(r_c - r, N);
                                const double exponential  = std::exp(-(r_c - shift_pot) / (r - shift_pot));
                                return V0 * exponential / denominator;
                              });
      }

      helfem::Matrix FEMRadialBasis::confinement_potential(size_t iel, int N, double r_0, int iconf, double V, double shift_pot) const {
	// Attractive potential does not make sense for shift_pot != 0

	// sign of r0 controls if the potential is attractive or repulsive
	int sign = (r_0<0) ? -1 : 1;
	r_0 = std::abs(r_0);

	if(iconf==1) {
          printf("Polynomial confinement, r_0 = %e N = %i shift = %e \n",r_0,N,shift_pot);
	  if(N<0) {
	    if(shift_pot != 0.0)
	      throw std::logic_error("Cannot have a divergent potential with a shift!\n");
	    return sign*std::pow(r_0, N)*polynomial_confinement(iel, N, shift_pot);
	  } else {
	    return sign*std::pow(r_0, -N)*polynomial_confinement(iel, N, shift_pot);
	  }

	} else if(iconf==2) {
          printf("Exponential confinement, r_0 = %e N = %i shift = %e \n",r_0,N,shift_pot);

	  if(N<0)
	    throw std::logic_error("Exponential confinement potential does not make sense with negative N!\n");
	  if(N==0)
	    throw std::logic_error("Exponential confinement potential requires N >= 1!");

	  return exponential_confinement(iel, N, r_0, shift_pot);

	} else if(iconf==3) {
	  if(V<0)
	    throw std::logic_error("Cannot have attractive barrier!\n");

          printf("Barrier confinement, V = %e shift = %e \n",V,shift_pot);
	  return barrier_confinement(iel, V, shift_pot);

	} else if(iconf==4) {
          printf("Junquera-type confinement, r_0 = %e N = %i V = %e shift = %e \n",r_0,N,V,shift_pot);
	  if(N<=0)
	    throw std::logic_error("Junquera confinement potential requires N >= 1!");
	  if(V<=0)
	    throw std::logic_error("Cannot have attractive Junquera potential!\n");
	  return junq_confinement(iel, N, V, get_bval().maxCoeff(), shift_pot);
	} else
	  throw std::logic_error("Case not implemented!\n");
      }


      helfem::Matrix FEMRadialBasis::model_potential(const modelpotential::ModelPotential *model,
                                                     size_t iel) const {
        return matrix_element(iel, BasisKind::B0, BasisKind::B0,
                              [model](double r){ return model->V(r); });
      }

      helfem::Matrix FEMRadialBasis::nuclear_offcenter(size_t iel, double Rhalf, int L) const {
        if (fem.element_begin(iel) <= Rhalf) {
          return -sqrt(4.0 * M_PI / (2 * L + 1)) *
                  radial_integral(-L - 1, iel) *
                  std::pow(Rhalf, L);
        } else if (fem.element_end(iel) >= Rhalf) {
          return -sqrt(4.0 * M_PI / (2 * L + 1)) *
                  radial_integral(L, iel) *
                  std::pow(Rhalf, -L - 1);
        } else {
          throw std::logic_error("Nucleus placed within element!\n");
        }
      }

      helfem::Matrix FEMRadialBasis::twoe_integral(int L, size_t iel) const {
        double Rmin(fem.element_begin(iel));
        double Rmax(fem.element_end(iel));

        // Integral by quadrature
        std::shared_ptr<const polynomial_basis::PolynomialBasis> p(fem.get_basis(iel));
        // Phase 5.7: quadrature::twoe_integral is now Eigen.
        helfem::Matrix tei = quadrature::twoe_integral(Rmin, Rmax, xq, wq, p, L);
        if (tei.array().isNaN().any())
          printf("twoe_integral(%i,%i) has NaN!\n", L, (int) iel);
        return tei;
      }

      // Pivoted Cholesky with truncation. Returns Lout of shape
      // (n x r) such that Lout * Lout^T == A up to abs tolerance on the
      // residual diagonal. Standard textbook algorithm (Higham, Sec 10.3);
      // pivots greedily on the largest remaining diagonal each step.
      // Phase 5.9: native Eigen.
      static helfem::Matrix pivoted_cholesky_(const helfem::Matrix & A, double tol) {
        const Eigen::Index n = A.rows();
        if (A.cols() != n)
          throw std::logic_error("pivoted_cholesky: input must be square.\n");
        helfem::Vector D = A.diagonal();
        std::vector<unsigned char> done(n, 0);
        // Build columns one at a time; concat at end.
        std::vector<helfem::Vector> Lcols;
        for (Eigen::Index k = 0; k < n; ++k) {
          // Pivot on largest remaining diagonal residual.
          Eigen::Index pivot = n;
          double pivot_val = tol;
          for (Eigen::Index i = 0; i < n; ++i)
            if (!done[i] && D(i) > pivot_val) {
              pivot = i;
              pivot_val = D(i);
            }
          if (pivot == n) break;
          done[pivot] = 1;
          const double sqrt_d = std::sqrt(pivot_val);
          helfem::Vector col(n);
          for (Eigen::Index i = 0; i < n; ++i) {
            if (done[i] && i != pivot) { col(i) = 0.0; continue; }
            double s = A(i, pivot);
            // s -= sum_{j<k} L(i, j) * L(pivot, j)
            for (size_t j = 0; j < Lcols.size(); ++j)
              s -= Lcols[j](i) * Lcols[j](pivot);
            col(i) = s / sqrt_d;
          }
          col(pivot) = sqrt_d;
          Lcols.push_back(col);
          for (Eigen::Index i = 0; i < n; ++i)
            if (!done[i]) D(i) -= col(i) * col(i);
        }
        helfem::Matrix L(n, (Eigen::Index) Lcols.size());
        for (size_t j = 0; j < Lcols.size(); ++j)
          L.col((Eigen::Index) j) = Lcols[j];
        return L;
      }

      helfem::Matrix FEMRadialBasis::twoe_integral_cholesky(int L, size_t iel,
                                                            double tol) const {
        // Phase 5.9: pivoted_cholesky_ is native Eigen; no bridge.
        return pivoted_cholesky_(twoe_integral(L, iel), tol);
      }

      helfem::Matrix FEMRadialBasis::yukawa_integral_cholesky(int L, double lambda,
                                                              size_t iel,
                                                              double tol) const {
        return pivoted_cholesky_(yukawa_integral(L, lambda, iel), tol);
      }


      helfem::Matrix FEMRadialBasis::yukawa_integral(int L, double lambda, size_t iel) const {
        double Rmin(fem.element_begin(iel));
        double Rmax(fem.element_end(iel));

        // Integral by quadrature
        std::shared_ptr<const polynomial_basis::PolynomialBasis> p(fem.get_basis(iel));
        return quadrature::yukawa_integral(Rmin, Rmax, xq, wq, p, L, lambda);
      }

      helfem::Matrix FEMRadialBasis::erfc_integral(int L, double mu, size_t iel, size_t kel) const {
        // Number of quadrature points
        size_t Nq = (size_t) xq.size();
        // Number of subintervals
        size_t Nint;

        if (iel == kel) {
          // Intraelement integral is harder to converge due to the
          // electronic cusp, so let's do the same trick as in the
          // separable case and use a tighter grid for the "inner"
          // integral.
          Nint = Nq;

        } else {
          // A single interval suffices since there's no cusp.
          Nint = 1;
        }

        // Phase 5.8: local scratch is now native Eigen.
        helfem::Vector xi, wi;
        helfem::lib1dfem::chebyshev::chebyshev<double>((int) Nq, xi, wi);
        helfem::Matrix ibf = fem.eval_f(xi, iel);
        const double Rmini = fem.element_begin(iel);
        const double Rmaxi = fem.element_end(iel);

        // Rh quadrature points: Nint copies of xi, mapped to subintervals.
        helfem::Vector xk(Nq * Nint), wk(Nq * Nint);
        for (size_t ii = 0; ii < Nint; ++ii) {
          const double istart = ii * 2.0 / Nint - 1.0;
          const double iend   = (ii + 1) * 2.0 / Nint - 1.0;
          const double imid   = 0.5 * (iend + istart);
          const double ilen   = 0.5 * (iend - istart);
          xk.segment((Eigen::Index)(ii * Nq), (Eigen::Index) Nq)
              = helfem::Vector::Constant((Eigen::Index) Nq, imid) + xi * ilen;
          wk.segment((Eigen::Index)(ii * Nq), (Eigen::Index) Nq) = wi * ilen;
        }
        const helfem::Matrix kbf = fem.eval_f(xk, kel);
        const double Rmink = fem.element_begin(kel);
        const double Rmaxk = fem.element_end(kel);

        helfem::Matrix tei = quadrature::erfc_integral(
            Rmini, Rmaxi, ibf, xi, wi,
            Rmink, Rmaxk, kbf, xk, wk, L, mu);
        // Symmetrize just to be sure (quadrature points differ across iel/kel).
        if (iel == kel)
          tei = 0.5 * (tei + tei.transpose()).eval();

        return tei;
      }

      helfem::Matrix FEMRadialBasis::spherical_potential(size_t iel) const {
        double Rmin(fem.element_begin(iel));
        double Rmax(fem.element_end(iel));

        // Integral by quadrature
        std::shared_ptr<polynomial_basis::PolynomialBasis> p(fem.get_basis(iel));
        return quadrature::spherical_potential(Rmin, Rmax, xq, wq, p);
      }

      helfem::Matrix FEMRadialBasis::get_bf(size_t iel) const {
        return get_bf(xq, iel);
      }

      // get_taylor() has been removed in favour of fem.eval_over_r().

      helfem::Vector FEMRadialBasis::eval_orbs(const helfem::Matrix & C, double r) const {
        if(r > fem.element_end(fem.get_nelem()-1)) {
          // Wave function is zero beyond the practical infinity.
          return helfem::Vector::Zero(C.cols());
        }
        // Find the element and evaluate the primitive coordinate.
        const size_t iel = fem.find_element(r);
        helfem::Vector r_e(1); r_e(0) = r;
        const helfem::Vector xe = fem.eval_prim(r_e, iel);

        // Basis functions in the element -- Eigen throughout after Phase 5.24.
        const helfem::Matrix val = get_bf(xe, iel);

        // Slice C over the element's basis-function index range and
        // return the row-vector product transposed to a column.
        size_t ifirst, ilast;
        get_idx(iel, ifirst, ilast);
        const Eigen::Index n = static_cast<Eigen::Index>(ilast - ifirst + 1);
        const helfem::Matrix Csub = C.block(ifirst, 0, n, C.cols());
        return (val * Csub).transpose();
      }

      helfem::Matrix FEMRadialBasis::get_bf(const helfem::Vector & x, size_t iel) const {
        // Phase 5.24: input is Eigen; no arma bridge at either end.
        if (iel == 0)
          return fem.eval_over_r(x, 0, iel);
        helfem::Matrix val = fem.eval_f(x, iel);
        const helfem::Vector r = fem.eval_coord(x, iel);
        for (Eigen::Index ifun = 0; ifun < val.cols(); ++ifun)
          for (Eigen::Index ir = 0; ir < val.rows(); ++ir)
            val(ir, ifun) /= r(ir);
        return val;
      }

      helfem::Matrix FEMRadialBasis::get_df(size_t iel) const {
        return get_df(xq, iel);
      }

      helfem::Matrix FEMRadialBasis::get_df(const helfem::Vector & x, size_t iel) const {
        if (iel == 0)
          return fem.eval_over_r(x, 1, iel);
        helfem::Matrix fval = fem.eval_f (x, iel);
        helfem::Matrix dval = fem.eval_df(x, iel);
        const helfem::Vector r = fem.eval_coord(x, iel);
        helfem::Matrix der(fval.rows(), fval.cols());
        for (Eigen::Index ifun = 0; ifun < der.cols(); ++ifun)
          for (Eigen::Index ir = 0; ir < der.rows(); ++ir) {
            const double invr = 1.0 / r(ir);
            der(ir, ifun) = (-fval(ir, ifun) * invr + dval(ir, ifun)) * invr;
          }
        return der;
      }

      helfem::Matrix FEMRadialBasis::get_lf(size_t iel) const {
        return get_lf(xq, iel);
      }

      helfem::Matrix FEMRadialBasis::get_lf(const helfem::Vector & x, size_t iel) const {
        if (iel == 0)
          return fem.eval_over_r(x, 2, iel);
        helfem::Matrix fval = fem.eval_f  (x, iel);
        helfem::Matrix dval = fem.eval_df (x, iel);
        helfem::Matrix lval = fem.eval_d2f(x, iel);
        const helfem::Vector r = fem.eval_coord(x, iel);
        helfem::Matrix lapl(fval.rows(), fval.cols());
        for (Eigen::Index ifun = 0; ifun < lapl.cols(); ++ifun)
          for (Eigen::Index ir = 0; ir < lapl.rows(); ++ir) {
            const double invr = 1.0 / r(ir);
            lapl(ir, ifun) = ((2.0 * fval(ir, ifun) * invr - 2.0 * dval(ir, ifun)) * invr
                              + lval(ir, ifun)) * invr;
          }
        return lapl;
      }

      helfem::Vector FEMRadialBasis::get_wrad(size_t iel) const {
        // Phase 5.20: internal wq is Eigen; scale in Eigen and return.
        return fem.scaling_factor(iel) * wq;
      }

      helfem::Vector FEMRadialBasis::get_wrad(const arma::vec & w, size_t iel) const {
        // Chemistry-layer callers still pass arma::vec weights; bridge
        // once to Eigen and scale. Return type is Eigen per Phase 5.20.
        const double s = fem.scaling_factor(iel);
        helfem::Vector out(w.n_elem);
        for (arma::uword i = 0; i < w.n_elem; ++i) out(i) = s * w(i);
        return out;
      }

      helfem::Vector FEMRadialBasis::get_r(size_t iel) const {
        // Phase 5.20: eval_coord returns Eigen; return directly.
        return fem.eval_coord(xq, iel);
      }

      helfem::Vector FEMRadialBasis::get_r(const arma::vec & x, size_t iel) const {
        // Bridge input to Eigen once; return Eigen per Phase 5.20.
        return fem.eval_coord(arma_to_eigen_vec(x), iel);
      }

      double FEMRadialBasis::get_r(double x, size_t iel) const {
        return fem.eval_coord(x, iel);
      }

      // Nuclear coordinate: primitive basis polynomials belong to [-1,1],
      // and the physical r=0 corresponds to x=-1 in the first element.
      // Building this single-x-point vector as an Eigen 1-vector avoids
      // the arma bridge that Phase 5.3-5.6 introduced.
      static helfem::Vector nuclear_x() {
        helfem::Vector x(1); x(0) = -1.0;
        return x;
      }

      double FEMRadialBasis::nuclear_density(const helfem::Matrix &Prad) const {
        if (static_cast<size_t>(Prad.rows()) != Nbf() ||
            static_cast<size_t>(Prad.cols()) != Nbf())
          throw std::logic_error("nuclear_density expects a radial density matrix\n");

        // Derivative at nucleus.
        const helfem::Matrix der = fem.eval_df(nuclear_x(), (size_t) 0);
        // First-element radial index range.
        size_t ifirst, ilast;
        get_idx(0, ifirst, ilast);
        const Eigen::Index n = static_cast<Eigen::Index>(ilast - ifirst + 1);
        // Density submatrix.
        const helfem::Matrix Psub = Prad.block(ifirst, ifirst, n, n);
        // P_uv B_u'(0) B_v'(0) -- one number.
        return (der * Psub * der.transpose()).value();
      }

      double FEMRadialBasis::nuclear_density_gradient(const helfem::Matrix &Prad) const {
        if (static_cast<size_t>(Prad.rows()) != Nbf() ||
            static_cast<size_t>(Prad.cols()) != Nbf())
          throw std::logic_error("nuclear_density_gradient expects a radial density matrix\n");

        const helfem::Vector xn = nuclear_x();
        const helfem::Matrix der  = fem.eval_df (xn, (size_t) 0);
        const helfem::Matrix lapl = fem.eval_d2f(xn, (size_t) 0);
        size_t ifirst, ilast;
        get_idx(0, ifirst, ilast);
        const Eigen::Index n = static_cast<Eigen::Index>(ilast - ifirst + 1);
        const helfem::Matrix Psub = Prad.block(ifirst, ifirst, n, n);
        // P_uv B_u'(0) B_v''(0) -- one number.
        return (der * Psub * lapl.transpose()).value();
      }

      Eigen::RowVectorXd FEMRadialBasis::nuclear_orbital(const helfem::Matrix &C) const {
        const helfem::Matrix der = fem.eval_df(nuclear_x(), (size_t) 0);
        size_t ifirst, ilast;
        get_idx(0, ifirst, ilast);
        const Eigen::Index n = static_cast<Eigen::Index>(ilast - ifirst + 1);
        const helfem::Matrix Csub = C.block(ifirst, 0, n, C.cols());
        // C_ui B_u'(0) -- row vector of length C.cols().
        return der * Csub;
      }
    } // namespace basis
  } // namespace atomic
} // namespace helfem
