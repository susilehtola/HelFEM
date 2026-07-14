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
#include "RadialPotential.h"
#include "quadrature.h"
#include "utils.h"
#include <lib1dfem/chebyshev.h>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace helfem {
  namespace atomic {
    namespace basis {

      template <typename T>
      FEMRadialBasisT<T>::FEMRadialBasisT() {}

      template <typename T>
      FEMRadialBasisT<T>::FEMRadialBasisT(const polynomial_basis::FiniteElementBasisT<T> & fem_, int n_quad) : fem(fem_) {
        helfem::lib1dfem::chebyshev::chebyshev<T>(n_quad, xq, wq);
        for (Eigen::Index i = 0; i < xq.size(); ++i) {
          // printf is a double-only boundary; cast there and nowhere else.
          if (!std::isfinite(xq(i)))
            printf("xq[%lld]=%e\n", (long long) i, (double) xq(i));
          if (!std::isfinite(wq(i)))
            printf("wq[%lld]=%e\n", (long long) i, (double) wq(i));
        }
      }

      template <typename T>
      FEMRadialBasisT<T>::~FEMRadialBasisT() {}

      template <typename T>
      int FEMRadialBasisT<T>::get_nquad() const {
        return (int) xq.size();
      }

      template <typename T>
      helfem::Vec<T> FEMRadialBasisT<T>::get_xq() const {
        // Phase 5.20: xq storage is Eigen; return it directly at the public boundary.
        return xq;
      }

      template <typename T>
      size_t FEMRadialBasisT<T>::Nbf() const {
        return fem.get_nbf();
      }

      template <typename T>
      size_t FEMRadialBasisT<T>::Nel() const {
        return fem.get_nelem();
      }

      template <typename T>
      size_t FEMRadialBasisT<T>::Nprim(size_t iel) const {
        return fem.get_nprim(iel);
      }

      template <typename T>
      size_t FEMRadialBasisT<T>::max_Nprim() const {
        return fem.get_max_nprim();
      }

      template <typename T>
      void FEMRadialBasisT<T>::get_idx(size_t iel, size_t &ifirst, size_t &ilast) const {
        fem.get_idx(iel, ifirst, ilast);
      }

      template <typename T>
      helfem::Vec<T> FEMRadialBasisT<T>::get_bval() const {
        // Phase 5.20: fem.get_bval() is Eigen; return directly.
        return fem.get_bval();
      }

      template <typename T>
      int FEMRadialBasisT<T>::get_poly_id() const {
        return fem.get_poly_id();
      }

      template <typename T>
      int FEMRadialBasisT<T>::get_poly_nnodes() const {
        return fem.get_poly_nnodes();
      }

      template <typename T>
      helfem::Mat<T> FEMRadialBasisT<T>::radial_integral(int Rexp, size_t iel, T x_left, T x_right) const {
        // <R | r^Rexp | R> with the FE-natural dr measure means weight r^(Rexp+2):
        //   integral B(r)^2 r^(Rexp+2) / r^2  dr  =  integral R(r)^2 r^(Rexp+2)  dr
        // = QM <r^Rexp> = integral R^2 r^Rexp r^2 dr.
        const std::function<T(T)> rpowL =
            [Rexp](T r){ return std::pow(r, Rexp + 2); };
        helfem::Mat<T> ret = matrix_element(iel, BasisKind::R0, BasisKind::R0,
                                            rpowL, x_left, x_right);
        if (ret.array().isNaN().any())
          printf("radial_integral(%i,%i) has NaN!\n", Rexp, (int)iel);
        return ret;
      }

      template <typename T>
      static std::function<helfem::Mat<T>(helfem::Vec<T>, size_t)>
      make_evaluator(const FEMRadialBasisT<T> * rb, typename FEMRadialBasisT<T>::BasisKind k) {
        using BK = typename FEMRadialBasisT<T>::BasisKind;
        switch (k) {
          case BK::B0: return [rb](helfem::Vec<T> x, size_t iel) {
            return rb->get_fem().eval_f(x, iel);
          };
          case BK::B1: return [rb](helfem::Vec<T> x, size_t iel) {
            return rb->get_fem().eval_df(x, iel);
          };
          case BK::B2: return [rb](helfem::Vec<T> x, size_t iel) {
            return rb->get_fem().eval_d2f(x, iel);
          };
          case BK::R0: return [rb](helfem::Vec<T> x, size_t iel) {
            return rb->get_bf(x, iel);
          };
          case BK::R1: return [rb](helfem::Vec<T> x, size_t iel) {
            return rb->get_df(x, iel);
          };
          case BK::R2: return [rb](helfem::Vec<T> x, size_t iel) {
            return rb->get_lf(x, iel);
          };
        }
        throw std::logic_error("FEMRadialBasisT::matrix_element: unknown BasisKind\n");
      }

      template <typename T>
      helfem::Mat<T> FEMRadialBasisT<T>::matrix_element(
          size_t iel, BasisKind bra, BasisKind ket,
          const std::function<T(T)> & weight) const {
        auto lhs = make_evaluator<T>(this, bra);
        auto rhs = make_evaluator<T>(this, ket);
        return fem.matrix_element(iel, lhs, rhs, xq, wq, weight);
      }

      template <typename T>
      helfem::Mat<T> FEMRadialBasisT<T>::matrix_element(
          BasisKind bra, BasisKind ket,
          const std::function<T(T)> & weight) const {
        auto lhs = make_evaluator<T>(this, bra);
        auto rhs = make_evaluator<T>(this, ket);
        return fem.matrix_element(lhs, rhs, xq, wq, weight);
      }

      template <typename T>
      helfem::Mat<T> FEMRadialBasisT<T>::matrix_element(
          size_t iel, BasisKind bra, BasisKind ket,
          const std::function<T(T)> & weight,
          T x_left, T x_right) const {
        auto lhs = make_evaluator<T>(this, bra);
        auto rhs = make_evaluator<T>(this, ket);
        return fem.matrix_element(iel, lhs, rhs, xq, wq, weight, x_left, x_right);
      }

      template <typename T>
      helfem::Mat<T> FEMRadialBasisT<T>::bessel_il_integral(int L, T lambda, size_t iel) const {
        return matrix_element(iel, BasisKind::B0, BasisKind::B0,
                              [L, lambda](T r){ return utils::bessel_il<T>(r * lambda, L); });
      }

      template <typename T>
      helfem::Mat<T> FEMRadialBasisT<T>::bessel_kl_integral(int L, T lambda, size_t iel) const {
        return matrix_element(iel, BasisKind::B0, BasisKind::B0,
                              [L, lambda](T r){ return utils::bessel_kl<T>(r * lambda, L); });
      }

      // Per-element B-evaluator for cross-basis quadrature. Only B-kinds
      // are meaningful cross-basis (R-kinds carry a per-element 1/r factor
      // that's tied to a single basis's element-length scaling).
      // Phase 5.8: cross-basis B-evaluator rewritten in native Eigen.
      template <typename T>
      static helfem::Mat<T> eval_B_at(const polynomial_basis::FiniteElementBasisT<T> & fem,
                                      const helfem::Vec<T> & x, size_t iel,
                                      typename FEMRadialBasisT<T>::BasisKind k) {
        using BK = typename FEMRadialBasisT<T>::BasisKind;
        switch (k) {
          case BK::B0: return fem.eval_f  (x, iel);
          case BK::B1: return fem.eval_df (x, iel);
          case BK::B2: return fem.eval_d2f(x, iel);
          case BK::R0: case BK::R1: case BK::R2:
            throw std::logic_error(
                "FEMRadialBasisT::matrix_element(cross-basis): R-kinds are not "
                "supported (R = B/r is tied to one basis's element-length "
                "scaling; use B-kinds + an explicit weight function instead).\n");
        }
        throw std::logic_error("eval_B_at: unknown BasisKind\n");
      }

      template <typename T>
      helfem::Mat<T> FEMRadialBasisT<T>::matrix_element(
          const FEMRadialBasisT<T> & rh,
          BasisKind bra, BasisKind ket,
          const std::function<T(T)> & weight) const {
        // Pick a quadrature rule sized for the finer of the two bases.
        const Eigen::Index n_quad = std::max(xq.size(), rh.xq.size());
        helfem::Vec<T> xproj, wproj;
        helfem::lib1dfem::chebyshev::chebyshev<T>((int) n_quad, xproj, wproj);

        // List overlapping (iel, jel) element pairs.
        std::vector<std::vector<size_t>> overlap(fem.get_nelem());
        for (size_t iel = 0; iel < fem.get_nelem(); ++iel) {
          const T istart = fem.element_begin(iel);
          const T iend   = fem.element_end(iel);
          for (size_t jel = 0; jel < rh.fem.get_nelem(); ++jel) {
            const T jstart = rh.fem.element_begin(jel);
            const T jend   = rh.fem.element_end(jel);
            if ((jstart >= istart && jstart < iend) ||
                (istart >= jstart && istart < jend))
              overlap[iel].push_back(jel);
          }
        }

        helfem::Mat<T> S = helfem::Mat<T>::Zero(Nbf(), rh.Nbf());
        for (size_t iel = 0; iel < fem.get_nelem(); ++iel) {
          for (size_t jel : overlap[iel]) {
            const T intstart = std::max(fem.element_begin(iel),
                                        rh.fem.element_begin(jel));
            const T intend   = std::min(fem.element_end(iel),
                                        rh.fem.element_end(jel));
            const T intmid   = T(0.5) * (intend + intstart);
            const T intlen   = T(0.5) * (intend - intstart);

            const helfem::Vec<T> r =
                helfem::Vec<T>::Constant(xproj.size(), intmid) + intlen * xproj;

            const helfem::Vec<T> xi = fem.eval_prim(r, iel);
            const helfem::Vec<T> xj = rh.fem.eval_prim(r, jel);

            helfem::Vec<T> wtot = wproj * intlen;
            if (weight)
              for (Eigen::Index i = 0; i < r.size(); ++i)
                wtot(i) *= weight(r(i));

            const helfem::Mat<T> ifunc = eval_B_at<T>(fem,    xi, iel, bra);
            const helfem::Mat<T> jfunc = eval_B_at<T>(rh.fem, xj, jel, ket);

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

      template <typename T>
      helfem::Mat<T> FEMRadialBasisT<T>::radial_integral(const FEMRadialBasisT<T> &rh, int n,
                                                         bool lhder, bool rhder) const {
        modelpotential::RadialPotentialT<T> rad(n);
        return model_potential(rh, &rad, lhder, rhder);
      }

      template <typename T>
      helfem::Mat<T> FEMRadialBasisT<T>::model_potential(const FEMRadialBasisT<T> &rh,
                                                         const modelpotential::ModelPotentialT<T> *model,
                                                         bool lhder, bool rhder) const {
        return matrix_element(rh,
                              lhder ? BasisKind::B1 : BasisKind::B0,
                              rhder ? BasisKind::B1 : BasisKind::B0,
                              [model](T r){ return model->V(r); });
      }

      template <typename T>
      helfem::Mat<T> FEMRadialBasisT<T>::overlap(const FEMRadialBasisT<T> &rh) const {
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

      namespace {
        /// default-constructed = identity weight
        template <typename T>
        const std::function<T(T)> & kNoWeight() {
          static const std::function<T(T)> f;
          return f;
        }
        template <typename T>
        const std::function<T(T)> & kRWeight() {
          static const std::function<T(T)> f = [](T r){ return r; };
          return f;
        }
      }

      template <typename T>
      helfem::Mat<T> FEMRadialBasisT<T>::overlap() const { return matrix_element(BasisKind::B0, BasisKind::B0, kNoWeight<T>()); }
      template <typename T>
      helfem::Mat<T> FEMRadialBasisT<T>::overlap(size_t iel) const { return matrix_element(iel, BasisKind::B0, BasisKind::B0, kNoWeight<T>()); }

      template <typename T>
      helfem::Mat<T> FEMRadialBasisT<T>::kinetic() const { return T(0.5) * matrix_element(BasisKind::B1, BasisKind::B1, kNoWeight<T>()); }
      template <typename T>
      helfem::Mat<T> FEMRadialBasisT<T>::kinetic(size_t iel) const { return T(0.5) * matrix_element(iel, BasisKind::B1, BasisKind::B1, kNoWeight<T>()); }

      template <typename T>
      helfem::Mat<T> FEMRadialBasisT<T>::kinetic_l() const { return T(0.5) * matrix_element(BasisKind::R0, BasisKind::R0, kNoWeight<T>()); }
      template <typename T>
      helfem::Mat<T> FEMRadialBasisT<T>::kinetic_l(size_t iel) const { return T(0.5) * matrix_element(iel, BasisKind::R0, BasisKind::R0, kNoWeight<T>()); }

      template <typename T>
      helfem::Mat<T> FEMRadialBasisT<T>::nuclear() const { return -matrix_element(BasisKind::R0, BasisKind::R0, kRWeight<T>()); }
      template <typename T>
      helfem::Mat<T> FEMRadialBasisT<T>::nuclear(size_t iel) const { return -matrix_element(iel, BasisKind::R0, BasisKind::R0, kRWeight<T>()); }

      template <typename T>
      helfem::Mat<T> FEMRadialBasisT<T>::polynomial_confinement(size_t iel, int N, T shift_pot) const {
        return matrix_element(iel, BasisKind::R0, BasisKind::R0,
                              [N, shift_pot](T r) {
                                return (r < shift_pot)
                                    ? T(0)
                                    : std::pow(r - shift_pot, N + 2);
                              });
      }

      template <typename T>
      helfem::Mat<T> FEMRadialBasisT<T>::exponential_confinement(size_t iel, int N, T r_0, T shift_pot) const {
        return matrix_element(iel, BasisKind::B0, BasisKind::B0,
                              [r_0, N, shift_pot](T r) {
                                if (r < shift_pot) return T(0);
                                const T r_ratio = (r - shift_pot) / r_0;
                                T fact = T(1);
                                T V = T(0);
                                T r_ratio_pow_k = T(1);
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

      template <typename T>
      helfem::Mat<T> FEMRadialBasisT<T>::barrier_confinement(size_t iel, T V, T shift_pot) const {
        return matrix_element(iel, BasisKind::B0, BasisKind::B0,
                              [V, shift_pot](T r) {
                                return (r < shift_pot) ? T(0) : V;
                              });
      }

      template <typename T>
      helfem::Mat<T> FEMRadialBasisT<T>::junq_confinement(size_t iel, int N, T V0, T r_c, T shift_pot) const {
        return matrix_element(iel, BasisKind::B0, BasisKind::B0,
                              [N, r_c, V0, shift_pot](T r) {
                                if (r < shift_pot) return T(0);
                                const T denominator  = std::pow(r_c - r, N);
                                const T exponential  = std::exp(-(r_c - shift_pot) / (r - shift_pot));
                                return V0 * exponential / denominator;
                              });
      }

      template <typename T>
      helfem::Mat<T> FEMRadialBasisT<T>::confinement_potential(size_t iel, int N, T r_0, int iconf, T V, T shift_pot) const {
	// Attractive potential does not make sense for shift_pot != 0

	// sign of r0 controls if the potential is attractive or repulsive
	int sign = (r_0<0) ? -1 : 1;
	r_0 = std::abs(r_0);

	if(iconf==1) {
          printf("Polynomial confinement, r_0 = %e N = %i shift = %e \n",(double) r_0,N,(double) shift_pot);
	  if(N<0) {
	    if(shift_pot != T(0))
	      throw std::logic_error("Cannot have a divergent potential with a shift!\n");
	    return T(sign)*std::pow(r_0, N)*polynomial_confinement(iel, N, shift_pot);
	  } else {
	    return T(sign)*std::pow(r_0, -N)*polynomial_confinement(iel, N, shift_pot);
	  }

	} else if(iconf==2) {
          printf("Exponential confinement, r_0 = %e N = %i shift = %e \n",(double) r_0,N,(double) shift_pot);

	  if(N<0)
	    throw std::logic_error("Exponential confinement potential does not make sense with negative N!\n");
	  if(N==0)
	    throw std::logic_error("Exponential confinement potential requires N >= 1!");

	  return exponential_confinement(iel, N, r_0, shift_pot);

	} else if(iconf==3) {
	  if(V<0)
	    throw std::logic_error("Cannot have attractive barrier!\n");

          printf("Barrier confinement, V = %e shift = %e \n",(double) V,(double) shift_pot);
	  return barrier_confinement(iel, V, shift_pot);

	} else if(iconf==4) {
          printf("Junquera-type confinement, r_0 = %e N = %i V = %e shift = %e \n",(double) r_0,N,(double) V,(double) shift_pot);
	  if(N<=0)
	    throw std::logic_error("Junquera confinement potential requires N >= 1!");
	  if(V<=0)
	    throw std::logic_error("Cannot have attractive Junquera potential!\n");
	  return junq_confinement(iel, N, V, get_bval().maxCoeff(), shift_pot);
	} else
	  throw std::logic_error("Case not implemented!\n");
      }


      template <typename T>
      helfem::Mat<T> FEMRadialBasisT<T>::model_potential(const modelpotential::ModelPotentialT<T> *model,
                                                         size_t iel) const {
        return matrix_element(iel, BasisKind::B0, BasisKind::B0,
                              [model](T r){ return model->V(r); });
      }

      template <typename T>
      helfem::Mat<T> FEMRadialBasisT<T>::nuclear_offcenter(size_t iel, T Rhalf, int L) const {
        // Multipole term of 1/|r - Rhalf| over element iel. For an element
        // entirely outside Rhalf (r >= Rhalf) the source is enclosed, so
        // r_< = Rhalf, r_> = r  ->  Rhalf^L * <r^{-L-1}>. For an element
        // entirely inside (r <= Rhalf), r_< = r, r_> = Rhalf  ->
        // <r^L> * Rhalf^{-L-1}. The comparison operators were inverted in
        // an earlier migration, which for r>Rhalf elements evaluated
        // <r^L> out to Rmax (~40^L) and blew up the SCF; restored here.
        if (fem.element_begin(iel) >= Rhalf) {
          return -std::sqrt(T(4) * utils::pi<T>() / T(2 * L + 1)) *
                  radial_integral(-L - 1, iel) *
                  std::pow(Rhalf, L);
        } else if (fem.element_end(iel) <= Rhalf) {
          return -std::sqrt(T(4) * utils::pi<T>() / T(2 * L + 1)) *
                  radial_integral(L, iel) *
                  std::pow(Rhalf, -L - 1);
        } else {
          throw std::logic_error("Nucleus placed within element!\n");
        }
      }

      template <typename T>
      helfem::Mat<T> FEMRadialBasisT<T>::twoe_integral(int L, size_t iel) const {
        T Rmin(fem.element_begin(iel));
        T Rmax(fem.element_end(iel));

        // Integral by quadrature
        std::shared_ptr<const helfem::lib1dfem::polynomial_basis::PolynomialBasis<T>> p(fem.get_basis(iel));
        // Phase 5.7: quadrature::twoe_integral is now Eigen.
        helfem::Mat<T> tei = quadrature::twoe_integral<T>(Rmin, Rmax, xq, wq, p, L);
        if (tei.array().isNaN().any())
          printf("twoe_integral(%i,%i) has NaN!\n", L, (int) iel);
        return tei;
      }

      // Pivoted Cholesky with truncation. Returns Lout of shape
      // (n x r) such that Lout * Lout^T == A up to abs tolerance on the
      // residual diagonal. Standard textbook algorithm (Higham, Sec 10.3);
      // pivots greedily on the largest remaining diagonal each step.
      // Phase 5.9: native Eigen.
      template <typename T>
      static helfem::Mat<T> pivoted_cholesky_(const helfem::Mat<T> & A, T tol) {
        const Eigen::Index n = A.rows();
        if (A.cols() != n)
          throw std::logic_error("pivoted_cholesky: input must be square.\n");
        helfem::Vec<T> D = A.diagonal();
        std::vector<unsigned char> done(n, 0);
        // Build columns one at a time; concat at end.
        std::vector<helfem::Vec<T>> Lcols;
        for (Eigen::Index k = 0; k < n; ++k) {
          // Pivot on largest remaining diagonal residual.
          Eigen::Index pivot = n;
          T pivot_val = tol;
          for (Eigen::Index i = 0; i < n; ++i)
            if (!done[i] && D(i) > pivot_val) {
              pivot = i;
              pivot_val = D(i);
            }
          if (pivot == n) break;
          done[pivot] = 1;
          const T sqrt_d = std::sqrt(pivot_val);
          helfem::Vec<T> col(n);
          for (Eigen::Index i = 0; i < n; ++i) {
            if (done[i] && i != pivot) { col(i) = T(0); continue; }
            T s = A(i, pivot);
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
        helfem::Mat<T> L(n, (Eigen::Index) Lcols.size());
        for (size_t j = 0; j < Lcols.size(); ++j)
          L.col((Eigen::Index) j) = Lcols[j];
        return L;
      }

      template <typename T>
      helfem::Mat<T> FEMRadialBasisT<T>::twoe_integral_cholesky(int L, size_t iel,
                                                                T tol) const {
        // Phase 5.9: pivoted_cholesky_ is native Eigen; no bridge.
        return pivoted_cholesky_<T>(twoe_integral(L, iel), tol);
      }

      template <typename T>
      helfem::Mat<T> FEMRadialBasisT<T>::yukawa_integral_cholesky(int L, T lambda,
                                                                  size_t iel,
                                                                  T tol) const {
        return pivoted_cholesky_<T>(yukawa_integral(L, lambda, iel), tol);
      }


      template <typename T>
      helfem::Mat<T> FEMRadialBasisT<T>::yukawa_integral(int L, T lambda, size_t iel) const {
        T Rmin(fem.element_begin(iel));
        T Rmax(fem.element_end(iel));

        // Integral by quadrature
        std::shared_ptr<const helfem::lib1dfem::polynomial_basis::PolynomialBasis<T>> p(fem.get_basis(iel));
        return quadrature::yukawa_integral<T>(Rmin, Rmax, xq, wq, p, L, lambda);
      }

      template <typename T>
      helfem::Mat<T> FEMRadialBasisT<T>::erfc_integral(int L, T mu, size_t iel, size_t kel) const {
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
        helfem::Vec<T> xi, wi;
        helfem::lib1dfem::chebyshev::chebyshev<T>((int) Nq, xi, wi);
        helfem::Mat<T> ibf = fem.eval_f(xi, iel);
        const T Rmini = fem.element_begin(iel);
        const T Rmaxi = fem.element_end(iel);

        // Rh quadrature points: Nint copies of xi, mapped to subintervals.
        helfem::Vec<T> xk(Nq * Nint), wk(Nq * Nint);
        for (size_t ii = 0; ii < Nint; ++ii) {
          const T istart = ii * T(2) / T(Nint) - T(1);
          const T iend   = (ii + 1) * T(2) / T(Nint) - T(1);
          const T imid   = T(0.5) * (iend + istart);
          const T ilen   = T(0.5) * (iend - istart);
          xk.segment((Eigen::Index)(ii * Nq), (Eigen::Index) Nq)
              = helfem::Vec<T>::Constant((Eigen::Index) Nq, imid) + xi * ilen;
          wk.segment((Eigen::Index)(ii * Nq), (Eigen::Index) Nq) = wi * ilen;
        }
        const helfem::Mat<T> kbf = fem.eval_f(xk, kel);
        const T Rmink = fem.element_begin(kel);
        const T Rmaxk = fem.element_end(kel);

        helfem::Mat<T> tei = quadrature::erfc_integral<T>(
            Rmini, Rmaxi, ibf, xi, wi,
            Rmink, Rmaxk, kbf, xk, wk, L, mu);
        // Symmetrize just to be sure (quadrature points differ across iel/kel).
        if (iel == kel)
          tei = T(0.5) * (tei + tei.transpose()).eval();

        return tei;
      }

      template <typename T>
      helfem::Mat<T> FEMRadialBasisT<T>::spherical_potential(size_t iel) const {
        T Rmin(fem.element_begin(iel));
        T Rmax(fem.element_end(iel));

        // Integral by quadrature
        std::shared_ptr<helfem::lib1dfem::polynomial_basis::PolynomialBasis<T>> p(fem.get_basis(iel));
        return quadrature::spherical_potential<T>(Rmin, Rmax, xq, wq, p);
      }

      template <typename T>
      helfem::Mat<T> FEMRadialBasisT<T>::get_bf(size_t iel) const {
        return get_bf(xq, iel);
      }

      // get_taylor() has been removed in favour of fem.eval_over_r().

      template <typename T>
      helfem::Vec<T> FEMRadialBasisT<T>::eval_orbs(const helfem::Mat<T> & C, T r) const {
        if(r > fem.element_end(fem.get_nelem()-1)) {
          // Wave function is zero beyond the practical infinity.
          return helfem::Vec<T>::Zero(C.cols());
        }
        // Find the element and evaluate the primitive coordinate.
        const size_t iel = fem.find_element(r);
        helfem::Vec<T> r_e(1); r_e(0) = r;
        const helfem::Vec<T> xe = fem.eval_prim(r_e, iel);

        // Basis functions in the element -- Eigen throughout after Phase 5.24.
        const helfem::Mat<T> val = get_bf(xe, iel);

        // Slice C over the element's basis-function index range and
        // return the row-vector product transposed to a column.
        size_t ifirst, ilast;
        get_idx(iel, ifirst, ilast);
        const Eigen::Index n = static_cast<Eigen::Index>(ilast - ifirst + 1);
        const helfem::Mat<T> Csub = C.block(ifirst, 0, n, C.cols());
        return (val * Csub).transpose();
      }

      template <typename T>
      helfem::Mat<T> FEMRadialBasisT<T>::get_bf(const helfem::Vec<T> & x, size_t iel) const {
        if (iel == 0)
          return fem.eval_over_r(x, 0, iel);
        helfem::Mat<T> val = fem.eval_f(x, iel);
        const helfem::Vec<T> r = fem.eval_coord(x, iel);
        for (Eigen::Index ifun = 0; ifun < val.cols(); ++ifun)
          for (Eigen::Index ir = 0; ir < val.rows(); ++ir)
            val(ir, ifun) /= r(ir);
        return val;
      }

      template <typename T>
      helfem::Mat<T> FEMRadialBasisT<T>::get_df(size_t iel) const {
        return get_df(xq, iel);
      }

      template <typename T>
      helfem::Mat<T> FEMRadialBasisT<T>::get_df(const helfem::Vec<T> & x, size_t iel) const {
        if (iel == 0)
          return fem.eval_over_r(x, 1, iel);
        helfem::Mat<T> fval = fem.eval_f (x, iel);
        helfem::Mat<T> dval = fem.eval_df(x, iel);
        const helfem::Vec<T> r = fem.eval_coord(x, iel);
        helfem::Mat<T> der(fval.rows(), fval.cols());
        for (Eigen::Index ifun = 0; ifun < der.cols(); ++ifun)
          for (Eigen::Index ir = 0; ir < der.rows(); ++ir) {
            const T invr = T(1) / r(ir);
            der(ir, ifun) = (-fval(ir, ifun) * invr + dval(ir, ifun)) * invr;
          }
        return der;
      }

      template <typename T>
      helfem::Mat<T> FEMRadialBasisT<T>::get_lf(size_t iel) const {
        return get_lf(xq, iel);
      }

      template <typename T>
      helfem::Mat<T> FEMRadialBasisT<T>::get_lf(const helfem::Vec<T> & x, size_t iel) const {
        if (iel == 0)
          return fem.eval_over_r(x, 2, iel);
        helfem::Mat<T> fval = fem.eval_f  (x, iel);
        helfem::Mat<T> dval = fem.eval_df (x, iel);
        helfem::Mat<T> lval = fem.eval_d2f(x, iel);
        const helfem::Vec<T> r = fem.eval_coord(x, iel);
        helfem::Mat<T> lapl(fval.rows(), fval.cols());
        for (Eigen::Index ifun = 0; ifun < lapl.cols(); ++ifun)
          for (Eigen::Index ir = 0; ir < lapl.rows(); ++ir) {
            const T invr = T(1) / r(ir);
            lapl(ir, ifun) = ((T(2) * fval(ir, ifun) * invr - T(2) * dval(ir, ifun)) * invr
                              + lval(ir, ifun)) * invr;
          }
        return lapl;
      }

      template <typename T>
      helfem::Vec<T> FEMRadialBasisT<T>::get_wrad(size_t iel) const {
        return get_wrad(wq, iel);
      }

      template <typename T>
      helfem::Vec<T> FEMRadialBasisT<T>::get_wrad(const helfem::Vec<T> & w, size_t iel) const {
        return fem.scaling_factor(iel) * w;
      }

      template <typename T>
      helfem::Vec<T> FEMRadialBasisT<T>::get_r(size_t iel) const {
        return get_r(xq, iel);
      }

      template <typename T>
      helfem::Vec<T> FEMRadialBasisT<T>::get_r(const helfem::Vec<T> & x, size_t iel) const {
        return fem.eval_coord(x, iel);
      }

      template <typename T>
      T FEMRadialBasisT<T>::get_r(T x, size_t iel) const {
        return fem.eval_coord(x, iel);
      }

      // Nuclear coordinate: primitive basis polynomials belong to [-1,1],
      // and the physical r=0 corresponds to x=-1 in the first element.
      template <typename T>
      static helfem::Vec<T> nuclear_x() {
        helfem::Vec<T> x(1); x(0) = T(-1);
        return x;
      }

      template <typename T>
      T FEMRadialBasisT<T>::nuclear_density(const helfem::Mat<T> &Prad) const {
        if (static_cast<size_t>(Prad.rows()) != Nbf() ||
            static_cast<size_t>(Prad.cols()) != Nbf())
          throw std::logic_error("nuclear_density expects a radial density matrix\n");

        // Derivative at nucleus.
        const helfem::Mat<T> der = fem.eval_df(nuclear_x<T>(), (size_t) 0);
        // First-element radial index range.
        size_t ifirst, ilast;
        get_idx(0, ifirst, ilast);
        const Eigen::Index n = static_cast<Eigen::Index>(ilast - ifirst + 1);
        // Density submatrix.
        const helfem::Mat<T> Psub = Prad.block(ifirst, ifirst, n, n);
        // P_uv B_u'(0) B_v'(0) -- one number.
        return (der * Psub * der.transpose()).value();
      }

      template <typename T>
      T FEMRadialBasisT<T>::nuclear_density_gradient(const helfem::Mat<T> &Prad) const {
        if (static_cast<size_t>(Prad.rows()) != Nbf() ||
            static_cast<size_t>(Prad.cols()) != Nbf())
          throw std::logic_error("nuclear_density_gradient expects a radial density matrix\n");

        const helfem::Vec<T> xn = nuclear_x<T>();
        const helfem::Mat<T> der  = fem.eval_df (xn, (size_t) 0);
        const helfem::Mat<T> lapl = fem.eval_d2f(xn, (size_t) 0);
        size_t ifirst, ilast;
        get_idx(0, ifirst, ilast);
        const Eigen::Index n = static_cast<Eigen::Index>(ilast - ifirst + 1);
        const helfem::Mat<T> Psub = Prad.block(ifirst, ifirst, n, n);
        // P_uv B_u'(0) B_v''(0) -- one number.
        return (der * Psub * lapl.transpose()).value();
      }

      template <typename T>
      helfem::RowVec<T> FEMRadialBasisT<T>::nuclear_orbital(const helfem::Mat<T> &C) const {
        const helfem::Mat<T> der = fem.eval_df(nuclear_x<T>(), (size_t) 0);
        size_t ifirst, ilast;
        get_idx(0, ifirst, ilast);
        const Eigen::Index n = static_cast<Eigen::Index>(ilast - ifirst + 1);
        const helfem::Mat<T> Csub = C.block(ifirst, 0, n, C.cols());
        // C_ui B_u'(0) -- row vector of length C.cols().
        return der * Csub;
      }

      // Explicit instantiations. The scalar types HelFEM is built for.
      template class FEMRadialBasisT<double>;
      template class FEMRadialBasisT<long double>;
#ifdef HELFEM_HAVE_FLOAT128
      template class FEMRadialBasisT<_Float128>;
#endif
    } // namespace basis
  } // namespace atomic
} // namespace helfem
