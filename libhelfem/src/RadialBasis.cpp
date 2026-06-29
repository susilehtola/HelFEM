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
        // Get quadrature rule
        chebyshev::chebyshev(n_quad, xq, wq);
        for (size_t i = 0; i < xq.n_elem; i++) {
          if (!std::isfinite(xq[i]))
            printf("xq[%i]=%e\n", (int)i, xq[i]);
          if (!std::isfinite(wq[i]))
            printf("wq[%i]=%e\n", (int)i, wq[i]);
        }
      }

      FEMRadialBasis::~FEMRadialBasis() {}

      int FEMRadialBasis::get_nquad() const {
        return (int)xq.n_elem;
      }

      arma::vec FEMRadialBasis::get_xq() const {
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

      arma::vec FEMRadialBasis::get_bval() const {
        // Phase 5.4: fem.get_bval() is Eigen; bridge here -- RadialBasis
        // keeps its arma::vec surface for downstream chemistry-layer code.
        return eigen_vec_to_arma(fem.get_bval());
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
            return helfem::to_eigen(rb->get_bf(eigen_vec_to_arma(x), iel));
          };
          case BK::R1: return [rb](helfem::Vector x, size_t iel) {
            return helfem::to_eigen(rb->get_df(eigen_vec_to_arma(x), iel));
          };
          case BK::R2: return [rb](helfem::Vector x, size_t iel) {
            return helfem::to_eigen(rb->get_lf(eigen_vec_to_arma(x), iel));
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
        return fem.matrix_element(iel, lhs, rhs,
                                  arma_to_eigen_vec(xq),
                                  arma_to_eigen_vec(wq), weight);
      }

      helfem::Matrix FEMRadialBasis::matrix_element(
          BasisKind bra, BasisKind ket,
          const std::function<double(double)> & weight) const {
        auto lhs = make_evaluator(this, bra);
        auto rhs = make_evaluator(this, ket);
        return fem.matrix_element(lhs, rhs,
                                  arma_to_eigen_vec(xq),
                                  arma_to_eigen_vec(wq), weight);
      }

      helfem::Matrix FEMRadialBasis::matrix_element(
          size_t iel, BasisKind bra, BasisKind ket,
          const std::function<double(double)> & weight,
          double x_left, double x_right) const {
        auto lhs = make_evaluator(this, bra);
        auto rhs = make_evaluator(this, ket);
        return fem.matrix_element(iel, lhs, rhs,
                                  arma_to_eigen_vec(xq),
                                  arma_to_eigen_vec(wq), weight, x_left, x_right);
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
      static arma::mat eval_B_at(const polynomial_basis::FiniteElementBasis & fem,
                                 const arma::vec & x, size_t iel,
                                 FEMRadialBasis::BasisKind k) {
        using BK = FEMRadialBasis::BasisKind;
        const helfem::Vector xe = arma_to_eigen_vec(x);
        switch (k) {
          case BK::B0: return eigen_mat_to_arma(fem.eval_f  (xe, iel));
          case BK::B1: return eigen_mat_to_arma(fem.eval_df (xe, iel));
          case BK::B2: return eigen_mat_to_arma(fem.eval_d2f(xe, iel));
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
        // Pick a quadrature rule sized for the finer of the two bases so the
        // projection is well resolved on every overlap interval.
        const size_t n_quad = std::max(xq.n_elem, rh.xq.n_elem);
        arma::vec xproj, wproj;
        chebyshev::chebyshev(n_quad, xproj, wproj);

        // For each element on the bra side, list every ket-side element it
        // overlaps in r-space; the product of two FE shape functions is
        // nonzero only there.
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

        arma::mat S(Nbf(), rh.Nbf(), arma::fill::zeros);
        for (size_t iel = 0; iel < fem.get_nelem(); ++iel) {
          for (size_t jel : overlap[iel]) {
            // Restrict the quadrature to the actual overlap interval.
            const double intstart = std::max(fem.element_begin(iel),
                                             rh.fem.element_begin(jel));
            const double intend   = std::min(fem.element_end(iel),
                                             rh.fem.element_end(jel));
            const double intmid   = 0.5 * (intend + intstart);
            const double intlen   = 0.5 * (intend - intstart);

            const arma::vec r =
                intmid * arma::ones<arma::vec>(xproj.n_elem) + intlen * xproj;

            // Back-transform r into each basis's reference coords
            // (Phase 5.3: eval_prim is Eigen; bridge).
            const helfem::Vector r_e = arma_to_eigen_vec(r);
            const helfem::Vector xi_e = fem.eval_prim(r_e, iel);
            const helfem::Vector xj_e = rh.fem.eval_prim(r_e, jel);
            arma::vec xi(xi_e.size()); std::memcpy(xi.memptr(), xi_e.data(), sizeof(double) * xi_e.size());
            arma::vec xj(xj_e.size()); std::memcpy(xj.memptr(), xj_e.data(), sizeof(double) * xj_e.size());

            arma::vec wtot = wproj * intlen;
            if (weight)
              for (arma::uword i = 0; i < r.n_elem; ++i)
                wtot(i) *= weight(r(i));

            const arma::mat ifunc = eval_B_at(fem,    xi, iel, bra);
            const arma::mat jfunc = eval_B_at(rh.fem, xj, jel, ket);

            size_t ifirst, ilast; get_idx(iel, ifirst, ilast);
            size_t jfirst, jlast; rh.get_idx(jel, jfirst, jlast);
            S.submat(ifirst, jfirst, ilast, jlast) +=
                arma::trans(ifunc) * arma::diagmat(wtot) * jfunc;
          }
        }
        return helfem::to_eigen(S);
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
	  return junq_confinement(iel, N, V, arma::max(get_bval()), shift_pot);
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
        arma::mat tei(quadrature::twoe_integral(Rmin, Rmax, xq, wq, p, L));
        if(tei.has_nan()) {
          printf("twoe_integral(%i,%i) has NaN!\n",L,(int) iel);
        }
        return helfem::to_eigen(tei);
      }

      // Pivoted Cholesky with truncation. Returns Lout of shape
      // (n x r) such that Lout * Lout^T == A up to abs tolerance on the
      // residual diagonal. Standard textbook algorithm (Higham, Sec 10.3);
      // pivots greedily on the largest remaining diagonal each step.
      static arma::mat pivoted_cholesky_(const arma::mat & A, double tol) {
        const arma::uword n = A.n_rows;
        if (A.n_cols != n)
          throw std::logic_error("pivoted_cholesky: input must be square.\n");
        arma::vec  D    = A.diag();
        arma::uvec done(n, arma::fill::zeros);
        arma::mat  L(n, 0);
        for (arma::uword k = 0; k < n; ++k) {
          // Pivot on the largest remaining diagonal residual.
          arma::uword pivot = n;
          double pivot_val = tol;
          for (arma::uword i = 0; i < n; ++i)
            if (!done(i) && D(i) > pivot_val) {
              pivot = i;
              pivot_val = D(i);
            }
          if (pivot == n) break;            // truncated -- rank reached
          done(pivot) = 1;
          const double sqrt_d = std::sqrt(pivot_val);
          arma::vec col(n);
          // col(i) = (A(i, pivot) - sum_{j<k} L(i,j) L(pivot,j)) / sqrt_d.
          // For already-pivoted rows (i with done(i)==1, i != pivot) the
          // residual is 0 by construction; skip the computation cleanly
          // by zeroing.
          for (arma::uword i = 0; i < n; ++i) {
            if (done(i) && i != pivot) { col(i) = 0.0; continue; }
            double s = A(i, pivot);
            if (L.n_cols > 0)
              s -= arma::dot(L.row(i), L.row(pivot));
            col(i) = s / sqrt_d;
          }
          col(pivot) = sqrt_d;
          L.insert_cols(L.n_cols, col);
          // Update residual diagonal for the not-yet-pivoted rows.
          for (arma::uword i = 0; i < n; ++i)
            if (!done(i)) D(i) -= col(i) * col(i);
        }
        return L;
      }

      helfem::Matrix FEMRadialBasis::twoe_integral_cholesky(int L, size_t iel,
                                                            double tol) const {
        // Phase 2a: twoe_integral now returns helfem::Matrix; bridge here
        // -- pivoted_cholesky_ is arma-based.
        return helfem::to_eigen(
            pivoted_cholesky_(helfem::to_arma(twoe_integral(L, iel)), tol));
      }

      helfem::Matrix FEMRadialBasis::yukawa_integral_cholesky(int L, double lambda,
                                                              size_t iel,
                                                              double tol) const {
        return helfem::to_eigen(
            pivoted_cholesky_(helfem::to_arma(yukawa_integral(L, lambda, iel)), tol));
      }


      helfem::Matrix FEMRadialBasis::yukawa_integral(int L, double lambda, size_t iel) const {
        double Rmin(fem.element_begin(iel));
        double Rmax(fem.element_end(iel));

        // Integral by quadrature
        std::shared_ptr<const polynomial_basis::PolynomialBasis> p(fem.get_basis(iel));
        arma::mat tei(quadrature::yukawa_integral(Rmin, Rmax, xq, wq, p, L, lambda));

        return helfem::to_eigen(tei);
      }

      helfem::Matrix FEMRadialBasis::erfc_integral(int L, double mu, size_t iel, size_t kel) const {
        // Number of quadrature points
        size_t Nq = xq.n_elem;
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

        // Get lh quadrature points
        arma::vec xi, wi;
        chebyshev::chebyshev(Nq, xi, wi);
        // and basis function values (Phase 5.3 bridge).
        arma::mat ibf(eigen_mat_to_arma(fem.eval_f(arma_to_eigen_vec(xi), iel)));
        double Rmini(fem.element_begin(iel));
        double Rmaxi(fem.element_end(iel));

        // Rh quadrature points
        arma::vec xk(Nq * Nint);
        arma::vec wk(Nq * Nint);
        for (size_t ii = 0; ii < Nint; ii++) {
          // Interval starts at
          double istart = ii * 2.0 / Nint - 1.0;
          double iend = (ii + 1) * 2.0 / Nint - 1.0;
          // Midpoint and half-length of interval
          double imid = 0.5 * (iend + istart);
          double ilen = 0.5 * (iend - istart);

          // Place quadrature points at
          xk.subvec(ii * Nq, (ii + 1) * Nq - 1) = arma::ones<arma::vec>(Nq) * imid + xi * ilen;
          // which have the renormalized weights
          wk.subvec(ii * Nq, (ii + 1) * Nq - 1) = wi * ilen;
        }
        // and basis function values (Phase 5.3 bridge).
        arma::mat kbf(eigen_mat_to_arma(fem.eval_f(arma_to_eigen_vec(xk), kel)));
        double Rmink(fem.element_begin(kel));
        double Rmaxk(fem.element_end(kel));

        // Evaluate integral
        arma::mat tei(quadrature::erfc_integral(Rmini, Rmaxi, ibf, xi, wi, Rmink,
                                                Rmaxk, kbf, xk, wk, L, mu));
        // Symmetrize just to be sure, since quadrature points were
        // different
        if (iel == kel)
          tei = 0.5 * (tei + tei.t());

        return helfem::to_eigen(tei);
      }

      helfem::Matrix FEMRadialBasis::spherical_potential(size_t iel) const {
        double Rmin(fem.element_begin(iel));
        double Rmax(fem.element_end(iel));

        // Integral by quadrature
        std::shared_ptr<polynomial_basis::PolynomialBasis> p(fem.get_basis(iel));
        arma::mat pot(quadrature::spherical_potential(Rmin, Rmax, xq, wq, p));

        return helfem::to_eigen(pot);
      }

      arma::mat FEMRadialBasis::get_bf(size_t iel) const {
        return get_bf(xq, iel);
      }

      // get_taylor() has been removed in favour of fem.eval_over_r().

      arma::vec FEMRadialBasis::eval_orbs(const arma::mat & C, double r) const {
        if(r > fem.element_end(fem.get_nelem()-1)) {
          // The wave function is zero here
          arma::vec val(C.n_cols, arma::fill::zeros);
          return val;
        } else {
          // Find the element we are in
          size_t iel = fem.find_element(r);
          // Find the value of the primitive coordinate
          helfem::Vector r_e(1); r_e(0) = r;
          helfem::Vector xe = fem.eval_prim(r_e, iel);
          arma::vec x(xe.size()); std::memcpy(x.memptr(), xe.data(), sizeof(double) * xe.size());

          // Evaluate the basis functions in the element
          arma::mat val(get_bf(x, iel));
          // Figure out the corresponding indices of the basis functions
          size_t ifirst, ilast;
          get_idx(iel, ifirst, ilast);
          arma::mat Csub(C.rows(ifirst, ilast));
          return (val*Csub).t();
        }
      }

      arma::mat FEMRadialBasis::get_bf(const arma::vec & x, size_t iel) const {
        // Phase 5.3: fem.eval_* / eval_coord are Eigen; bridge.
        const helfem::Vector xe = arma_to_eigen_vec(x);
        if (iel == 0)
          return eigen_mat_to_arma(fem.eval_over_r(xe, 0, iel));
        arma::mat val(eigen_mat_to_arma(fem.eval_f(xe, iel)));
        const helfem::Vector r_e = fem.eval_coord(xe, iel);
        arma::vec r(r_e.size()); std::memcpy(r.memptr(), r_e.data(), sizeof(double) * r_e.size());
        for (size_t ifun = 0; ifun < val.n_cols; ifun++)
          for (size_t ir = 0; ir < x.n_elem; ir++)
            val(ir, ifun) /= r(ir);
        return val;
      }

      arma::mat FEMRadialBasis::get_df(size_t iel) const {
        return get_df(xq ,iel);
      }

      arma::mat FEMRadialBasis::get_df(const arma::vec & x, size_t iel) const {
        const helfem::Vector xe = arma_to_eigen_vec(x);
        if (iel == 0)
          return eigen_mat_to_arma(fem.eval_over_r(xe, 1, iel));
        arma::mat fval(eigen_mat_to_arma(fem.eval_f (xe, iel)));
        arma::mat dval(eigen_mat_to_arma(fem.eval_df(xe, iel)));
        const helfem::Vector r_e = fem.eval_coord(xe, iel);
        arma::vec r(r_e.size()); std::memcpy(r.memptr(), r_e.data(), sizeof(double) * r_e.size());
        arma::mat der(fval.n_rows, fval.n_cols);
        for (size_t ifun = 0; ifun < der.n_cols; ifun++)
          for (size_t ir = 0; ir < x.n_elem; ir++) {
            const double invr = 1.0 / r(ir);
            der(ir, ifun) = (-fval(ir, ifun) * invr + dval(ir, ifun)) * invr;
          }
        return der;
      }

      arma::mat FEMRadialBasis::get_lf(size_t iel) const {
        return get_lf(xq, iel);
      }

      arma::mat FEMRadialBasis::get_lf(const arma::vec & x, size_t iel) const {
        const helfem::Vector xe = arma_to_eigen_vec(x);
        if (iel == 0)
          return eigen_mat_to_arma(fem.eval_over_r(xe, 2, iel));
        arma::mat fval(eigen_mat_to_arma(fem.eval_f  (xe, iel)));
        arma::mat dval(eigen_mat_to_arma(fem.eval_df (xe, iel)));
        arma::mat lval(eigen_mat_to_arma(fem.eval_d2f(xe, iel)));
        const helfem::Vector r_e = fem.eval_coord(xe, iel);
        arma::vec r(r_e.size()); std::memcpy(r.memptr(), r_e.data(), sizeof(double) * r_e.size());
        arma::mat lapl(fval.n_rows, fval.n_cols);
        for (size_t ifun = 0; ifun < lapl.n_cols; ifun++)
          for (size_t ir = 0; ir < x.n_elem; ir++) {
            const double invr = 1.0 / r(ir);
            lapl(ir, ifun) = ((2.0 * fval(ir, ifun) * invr - 2.0 * dval(ir, ifun)) * invr
                              + lval(ir, ifun)) * invr;
          }
        return lapl;
      }

      arma::vec FEMRadialBasis::get_wrad(size_t iel) const {
        return get_wrad(wq, iel);
      }

      arma::vec FEMRadialBasis::get_wrad(const arma::vec & w, size_t iel) const {
        // This is just the radial rule, no r^2 factor included here
        return fem.scaling_factor(iel) * w;
      }

      arma::vec FEMRadialBasis::get_r(size_t iel) const {
        return get_r(xq, iel);
      }

      arma::vec FEMRadialBasis::get_r(const arma::vec & x, size_t iel) const {
        const helfem::Vector r_e = fem.eval_coord(arma_to_eigen_vec(x), iel);
        arma::vec r(r_e.size()); std::memcpy(r.memptr(), r_e.data(), sizeof(double) * r_e.size());
        return r;
      }

      double FEMRadialBasis::get_r(double x, size_t iel) const {
        return fem.eval_coord(x, iel);
      }

      double FEMRadialBasis::nuclear_density(const arma::mat &Prad) const {
        if (Prad.n_rows != Nbf() || Prad.n_cols != Nbf())
          throw std::logic_error("nuclear_density expects a radial density matrix\n");

        // Nuclear coordinate
        arma::vec x(1);
        // Remember that the primitive basis polynomials belong to [-1,1]
        x(0) = -1.0;

        // Evaluate derivative at nucleus
        arma::mat der(eigen_mat_to_arma(fem.eval_df(arma_to_eigen_vec(x), (size_t) 0)));

        // Radial functions in element
        size_t ifirst, ilast;
        get_idx(0, ifirst, ilast);
        // Density submatrix
        arma::mat Psub(Prad.submat(ifirst, ifirst, ilast, ilast));
        // P_uv B_u'(0) B_v'(0)
        double den(arma::as_scalar(der * Psub * arma::trans(der)));

        return den;
      }

      double FEMRadialBasis::nuclear_density_gradient(const arma::mat &Prad) const {
        if (Prad.n_rows != Nbf() || Prad.n_cols != Nbf())
          throw std::logic_error("nuclear_density_gradient expects a radial density matrix\n");

        // Nuclear coordinate
        arma::vec x(1);
        // Remember that the primitive basis polynomials belong to [-1,1]
        x(0) = -1.0;

        // Evaluate derivative at nucleus
        arma::mat der(eigen_mat_to_arma(fem.eval_df(arma_to_eigen_vec(x), (size_t) 0)));
        arma::mat lapl(eigen_mat_to_arma(fem.eval_d2f(arma_to_eigen_vec(x), (size_t) 0)));

        // Radial functions in element
        size_t ifirst, ilast;
        get_idx(0, ifirst, ilast);
        // Density submatrix
        arma::mat Psub(Prad.submat(ifirst, ifirst, ilast, ilast));
        // P_uv B_u'(0) B_v''(0)
        double den(arma::as_scalar(der * Psub * arma::trans(lapl)));

        return den;
      }

      arma::rowvec FEMRadialBasis::nuclear_orbital(const arma::mat &C) const {
        // Nuclear coordinate
        arma::vec x(1);
        // Remember that the primitive basis polynomials belong to [-1,1]
        x(0) = -1.0;

        // Derivative
        arma::mat der(eigen_mat_to_arma(fem.eval_df(arma_to_eigen_vec(x), (size_t) 0)));
        // Radial functions in element
        size_t ifirst, ilast;
        get_idx(0, ifirst, ilast);
        // Density submatrix
        arma::mat Csub(C.rows(ifirst, ilast));

        // C_ui B_u'(0)
        return der * Csub;
      }
    } // namespace basis
  } // namespace atomic
} // namespace helfem
