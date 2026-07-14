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
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "gaunt.h"
#include "utils.h"
#include "wignernj.hpp"

#ifdef HELFEM_HAVE_FLOAT128
// Declares `__float128 gaunt_q(int, int, int, int, int, int)`. Note that
// __float128 (the GCC extension) and _Float128 (the C++23 standard extended
// floating-point type) are DISTINCT types even though they have identical
// layout; the bridge below converts between them explicitly.
#include "wignernj_quadmath.h"
#endif

namespace helfem {
  namespace gaunt {

    namespace {
      // Dispatch to libwignernj at the requested scalar type.
      //
      // libwignernj evaluates every coupling coefficient from an EXACT
      // prime-factorised rational and rounds only at the very end, so each of
      // these entry points returns the correctly rounded value AT ITS OWN
      // precision -- the Gaunt coefficients are never the accuracy bottleneck,
      // at any T. (Checked against the closed form gaunt(0,0,1,0,1,0) =
      // 1/sqrt(4 pi): double errs by 3.8e-18, long double by 1.2e-20, and
      // _Float128 reproduces it exactly to 36 digits.)
      template <typename T> struct wigner_gaunt;

      template <> struct wigner_gaunt<double> {
        static double eval(int tl1, int tm1, int tl2, int tm2, int tl3, int tm3) {
          return wignernj::gaunt<double>(tl1, tm1, tl2, tm2, tl3, tm3);
        }
      };

      template <> struct wigner_gaunt<long double> {
        static long double eval(int tl1, int tm1, int tl2, int tm2, int tl3, int tm3) {
          return wignernj::gaunt<long double>(tl1, tm1, tl2, tm2, tl3, tm3);
        }
      };

#ifdef HELFEM_HAVE_FLOAT128
      template <> struct wigner_gaunt<_Float128> {
        static _Float128 eval(int tl1, int tm1, int tl2, int tm2, int tl3, int tm3) {
          // gaunt_q is a C entry point returning __float128. The two types are
          // bit-identical (IEEE binary128) but not interconvertible in C++, so
          // reinterpret the bits rather than cast.
          const __float128 v = ::gaunt_q(tl1, tm1, tl2, tm2, tl3, tm3);
          _Float128 out;
          static_assert(sizeof(v) == sizeof(out),
                        "__float128 and _Float128 must share a layout");
          std::memcpy(&out, &v, sizeof(out));
          return out;
        }
      };
#endif
    } // anonymous namespace

    template <typename T>
    T gaunt_coefficient_T(int L, int M, int l, int m, int lp, int mp) {
      // gaunt_coefficient(L,M,l,m,lp,mp) = integral Y_L^M* Y_l^m Y_lp^mp dOmega.
      // Y_L^M* = (-1)^M Y_L^{-M}, so this equals
      //   (-1)^M * integral Y_L^{-M} Y_l^m Y_lp^mp dOmega = (-1)^M * wignernj::gaunt(L,-M,l,m,lp,mp).
      const T sign = (M & 1) ? T(-1) : T(1);
      return sign * wigner_gaunt<T>::eval(2*L, -2*M, 2*l, 2*m, 2*lp, 2*mp);
    }

    double gaunt_coefficient(int L, int M, int l, int m, int lp, int mp) {
      return gaunt_coefficient_T<double>(L, M, l, m, lp, mp);
    }

    double modified_gaunt_coefficient(int lj, int mj, int L, int M, int li, int mi) {
      static const double const0(2.0/3.0*sqrt(M_PI));
      static const double const2(4.0/15.0*sqrt(5.0*M_PI));

      // Coupling through Y_0^0
      double cpl0(gaunt_coefficient(L,M,0,0,L,M)*gaunt_coefficient(lj,mj,li,mi,L,M));

      // Coupling through Y_2^0
      double cpl2=0.0;
      for(int Lp=std::max(std::max(L-2,0),std::abs(M));Lp<=L+2;Lp++)
        cpl2+=gaunt_coefficient(Lp,M,2,0,L,M)*gaunt_coefficient(lj,mj,li,mi,Lp,M);

      return const0*cpl0+const2*cpl2;
    }

    // (L l lp; 0 0 0) is zero unless L+l+lp is even and the triangle inequality
    // holds, so the whole (M, m) sweep can be skipped for failing triples.
    static inline bool gaunt_triple_nonzero(int L, int l, int lp) {
      if((L + l + lp) % 2 != 0) return false;
      if(lp < std::abs(L - l) || lp > L + l) return false;
      return true;
    }

    // The angular prefactors below expand cos^n(theta) / sin^n(theta) in
    // Y_n^0. They were double-precision literals (M_PI is a double macro),
    // which would have silently pinned a higher-precision table to double.
    // utils::pi<T>() is a long-double literal that rounds to exactly M_PI at
    // T = double, and the arithmetic is otherwise identical, so the T = double
    // instantiation is bit-for-bit what it was before.
    template <typename T> static inline T sqrt_pi()      { return std::sqrt(utils::pi<T>()); }
    template <typename T> static inline T sqrt_5pi()     { return std::sqrt(T(5)*utils::pi<T>()); }
    template <typename T> static inline T sqrt_7pi()     { return std::sqrt(T(7)*utils::pi<T>()); }
    template <typename T> static inline T sqrt_3pi()     { return std::sqrt(T(3)*utils::pi<T>()); }
    template <typename T> static inline T sqrt_11pi()    { return std::sqrt(T(11)*utils::pi<T>()); }
    template <typename T> static inline T sqrt_pi_by_3() { return std::sqrt(utils::pi<T>()/T(3)); }

    template <typename T>
    GauntT<T>::GauntT(int Lmax_, int lmax_, int lpmax_)
      : Lmax(Lmax_), lmax(lmax_), lpmax(lpmax_) {
      // (l, m) packs to l*(l+1) + m; range [0, (lmax+1)^2 - 1]. Same for (L, M).
      // mp is implicit from the m-sum rule (mp = M - m), so we omit an mp axis.
      lm_stride = static_cast<std::size_t>(lpmax) + 1;
      Lm_stride = static_cast<std::size_t>(lmax + 1) * (lmax + 1) * lm_stride;
      const std::size_t total = static_cast<std::size_t>(Lmax + 1) * (Lmax + 1) * Lm_stride;
      table.assign(total, T(0));

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(dynamic)
#endif
      for(int L = 0; L <= Lmax; ++L) {
        for(int l = 0; l <= lmax; ++l) {
          const int lp_lo = std::abs(L - l);
          const int lp_hi = std::min(lpmax, L + l);
          for(int lp = lp_lo; lp <= lp_hi; ++lp) {
            if(!gaunt_triple_nonzero(L, l, lp)) continue;
            for(int M = -L; M <= L; ++M) {
              const T signM = (M & 1) ? T(-1) : T(1);
              const int m_lo = std::max(-l, M - lp);
              const int m_hi = std::min( l, M + lp);
              for(int m = m_lo; m <= m_hi; ++m) {
                const int mp = M - m;
                const T g = wigner_gaunt<T>::eval(2*L, -2*M, 2*l, 2*m,
                                                  2*lp, 2*mp);
                table[flat_index(L, M, l, m, lp)] = signM * g;
              }
            }
          }
        }
      }
    }

    template <typename T>
    T GauntT<T>::coeff(int L, int M, int l, int m, int lp) const {
      // Out-of-range probes return 0 (callers iterate over wide ranges).
      if(L < 0 || L > Lmax) return T(0);
      if(l < 0 || l > lmax) return T(0);
      if(lp < 0 || lp > lpmax) return T(0);
      if(std::abs(M) > L) return T(0);
      if(std::abs(m) > l) return T(0);
      // mp is implicit (= M - m); a stored cell with |M - m| > lp is zero.
      return table[flat_index(L, M, l, m, lp)];
    }

    template <typename T>
    T GauntT<T>::mod_coeff(int lj, int mj, int L, int M, int li, int mi) const {
      // mod_coeff = integral Y_lj^mj* (cos^2 th) Y_L^M Y_li^mi: m-sum forces
      // mj = M + mi for any term to be nonzero.
      if(mj != M + mi) return T(0);

      static const T const0(T(2)/T(3)*sqrt_pi<T>());
      static const T const2(T(4)/T(15)*sqrt_5pi<T>());

      // Coupling through Y_0^0
      const T cpl0 = coeff(L,M,0,0,L) * coeff(lj,mj,li,mi,L);

      // Coupling through Y_2^0
      T cpl2 = T(0);
      for(int Lp = std::max(std::max(L-2, 0), std::abs(M)); Lp <= L+2; ++Lp)
        cpl2 += coeff(Lp,M,2,0,L) * coeff(lj,mj,li,mi,Lp);

      return const0*cpl0 + const2*cpl2;
    }

    // The cos^n / sin^n couplings expand the angular factor in Y_n^0; the m-sum
    // forces mi = mj — otherwise the integral vanishes. Guard explicitly so we
    // don't index into an unrelated cell of the implicit-mp table.
    template <typename T>
    T GauntT<T>::cosine_coupling(int lj, int mj, int li, int mi) const {
      if(mi != mj) return T(0);
      static const T const1(T(2)*sqrt_pi_by_3<T>());
      return const1*coeff(lj,mj,1,0,li);
    }

    template <typename T>
    T GauntT<T>::cosine2_coupling(int lj, int mj, int li, int mi) const {
      if(mi != mj) return T(0);
      static const T const0(T(2)/T(3)*sqrt_pi<T>());
      static const T const2(T(4)/T(15)*sqrt_5pi<T>());
      return const0*coeff(lj,mj,0,0,li) + const2*coeff(lj,mj,2,0,li);
    }

    template <typename T>
    T GauntT<T>::cosine3_coupling(int lj, int mj, int li, int mi) const {
      if(mi != mj) return T(0);
      static const T const1(T(2)/T(5)*sqrt_3pi<T>());
      static const T const3(T(4)/T(35)*sqrt_7pi<T>());
      return const1*coeff(lj,mj,1,0,li) + const3*coeff(lj,mj,3,0,li);
    }

    template <typename T>
    T GauntT<T>::cosine4_coupling(int lj, int mj, int li, int mi) const {
      if(mi != mj) return T(0);
      static const T const0(T(2)/T(5)*sqrt_pi<T>());
      static const T const2(T(8)/T(35)*sqrt_5pi<T>());
      static const T const4(T(16)/T(105)*sqrt_pi<T>());
      return const0*coeff(lj,mj,0,0,li) + const2*coeff(lj,mj,2,0,li) + const4*coeff(lj,mj,4,0,li);
    }

    template <typename T>
    T GauntT<T>::cosine5_coupling(int lj, int mj, int li, int mi) const {
      if(mi != mj) return T(0);
      static const T const1(T(2)/T(7)*sqrt_3pi<T>());
      static const T const3(T(8)/T(63)*sqrt_7pi<T>());
      static const T const5(T(16)/T(693)*sqrt_11pi<T>());
      return const1*coeff(lj,mj,1,0,li) + const3*coeff(lj,mj,3,0,li) + const5*coeff(lj,mj,5,0,li);
    }

    template <typename T>
    T GauntT<T>::sine2_coupling(int lj, int mj, int li, int mi) const {
      if(mi != mj) return T(0);
      static const T const0(T(4)/T(3)*sqrt_pi<T>());
      static const T const2(T(-4)/T(15)*sqrt_5pi<T>());
      return const0*coeff(lj,mj,0,0,li) + const2*coeff(lj,mj,2,0,li);
    }

    template <typename T>
    T GauntT<T>::cosine2_sine2_coupling(int lj, int mj, int li, int mi) const {
      if(mi != mj) return T(0);
      static const T const0(T(4)/T(15)*sqrt_pi<T>());
      static const T const2(T(4)/T(105)*sqrt_5pi<T>());
      static const T const4(T(-16)/T(105)*sqrt_pi<T>());
      return const0*coeff(lj,mj,0,0,li) + const2*coeff(lj,mj,2,0,li) + const4*coeff(lj,mj,4,0,li);
    }

    template double      gaunt_coefficient_T<double>     (int, int, int, int, int, int);
    template long double gaunt_coefficient_T<long double>(int, int, int, int, int, int);

    template class GauntT<double>;
    template class GauntT<long double>;

#ifdef HELFEM_HAVE_FLOAT128
    template _Float128 gaunt_coefficient_T<_Float128>(int, int, int, int, int, int);
    template class GauntT<_Float128>;
#endif
  }
}
