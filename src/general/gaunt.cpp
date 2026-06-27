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

#include "gaunt.h"
#include "wignernj.hpp"

namespace helfem {
  namespace gaunt {

    double gaunt_coefficient(int L, int M, int l, int m, int lp, int mp) {
      // gaunt_coefficient(L,M,l,m,lp,mp) = integral Y_L^M* Y_l^m Y_lp^mp dOmega.
      // Y_L^M* = (-1)^M Y_L^{-M}, so this equals
      //   (-1)^M * integral Y_L^{-M} Y_l^m Y_lp^mp dOmega = (-1)^M * wignernj::gaunt(L,-M,l,m,lp,mp).
      const double sign = (M & 1) ? -1.0 : 1.0;
      return sign * wignernj::gaunt<double>(2*L, -2*M, 2*l, 2*m, 2*lp, 2*mp);
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

    Gaunt::Gaunt(int Lmax_, int lmax_, int lpmax_)
      : Lmax(Lmax_), lmax(lmax_), lpmax(lpmax_) {
      // (l, m) packs to l*(l+1) + m; range [0, (lmax+1)^2 - 1]. Same for (L, M).
      // mp is implicit from the m-sum rule (mp = M - m), so we omit an mp axis.
      lm_stride = static_cast<std::size_t>(lpmax) + 1;
      Lm_stride = static_cast<std::size_t>(lmax + 1) * (lmax + 1) * lm_stride;
      const std::size_t total = static_cast<std::size_t>(Lmax + 1) * (Lmax + 1) * Lm_stride;
      table.assign(total, 0.0);

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
              const double signM = (M & 1) ? -1.0 : 1.0;
              const int m_lo = std::max(-l, M - lp);
              const int m_hi = std::min( l, M + lp);
              for(int m = m_lo; m <= m_hi; ++m) {
                const int mp = M - m;
                const double g = wignernj::gaunt<double>(2*L, -2*M, 2*l, 2*m,
                                                         2*lp, 2*mp);
                table[flat_index(L, M, l, m, lp)] = signM * g;
              }
            }
          }
        }
      }
    }

    double Gaunt::coeff(int L, int M, int l, int m, int lp) const {
      // Out-of-range probes return 0 (callers iterate over wide ranges).
      if(L < 0 || L > Lmax) return 0.0;
      if(l < 0 || l > lmax) return 0.0;
      if(lp < 0 || lp > lpmax) return 0.0;
      if(std::abs(M) > L) return 0.0;
      if(std::abs(m) > l) return 0.0;
      // mp is implicit (= M - m); a stored cell with |M - m| > lp is zero.
      return table[flat_index(L, M, l, m, lp)];
    }

    double Gaunt::mod_coeff(int lj, int mj, int L, int M, int li, int mi) const {
      // mod_coeff = integral Y_lj^mj* (cos^2 th) Y_L^M Y_li^mi: m-sum forces
      // mj = M + mi for any term to be nonzero.
      if(mj != M + mi) return 0.0;

      static const double const0(2.0/3.0*sqrt(M_PI));
      static const double const2(4.0/15.0*sqrt(5.0*M_PI));

      // Coupling through Y_0^0
      const double cpl0 = coeff(L,M,0,0,L) * coeff(lj,mj,li,mi,L);

      // Coupling through Y_2^0
      double cpl2 = 0.0;
      for(int Lp = std::max(std::max(L-2, 0), std::abs(M)); Lp <= L+2; ++Lp)
        cpl2 += coeff(Lp,M,2,0,L) * coeff(lj,mj,li,mi,Lp);

      return const0*cpl0 + const2*cpl2;
    }

    // The cos^n / sin^n couplings expand the angular factor in Y_n^0; the m-sum
    // forces mi = mj — otherwise the integral vanishes. Guard explicitly so we
    // don't index into an unrelated cell of the implicit-mp table.
    double Gaunt::cosine_coupling(int lj, int mj, int li, int mi) const {
      if(mi != mj) return 0.0;
      static const double const1(2.0*sqrt(M_PI/3.0));
      return const1*coeff(lj,mj,1,0,li);
    }

    double Gaunt::cosine2_coupling(int lj, int mj, int li, int mi) const {
      if(mi != mj) return 0.0;
      static const double const0(2.0/3.0*sqrt(M_PI));
      static const double const2(4.0/15.0*sqrt(5.0*M_PI));
      return const0*coeff(lj,mj,0,0,li) + const2*coeff(lj,mj,2,0,li);
    }

    double Gaunt::cosine3_coupling(int lj, int mj, int li, int mi) const {
      if(mi != mj) return 0.0;
      static const double const1(2.0/5.0*sqrt(3.0*M_PI));
      static const double const3(4.0/35.0*sqrt(7.0*M_PI));
      return const1*coeff(lj,mj,1,0,li) + const3*coeff(lj,mj,3,0,li);
    }

    double Gaunt::cosine4_coupling(int lj, int mj, int li, int mi) const {
      if(mi != mj) return 0.0;
      static const double const0(2.0/5.0*sqrt(M_PI));
      static const double const2(8.0/35.0*sqrt(5.0*M_PI));
      static const double const4(16.0/105.0*sqrt(M_PI));
      return const0*coeff(lj,mj,0,0,li) + const2*coeff(lj,mj,2,0,li) + const4*coeff(lj,mj,4,0,li);
    }

    double Gaunt::cosine5_coupling(int lj, int mj, int li, int mi) const {
      if(mi != mj) return 0.0;
      static const double const1(2.0/7.0*sqrt(3.0*M_PI));
      static const double const3(8.0/63.0*sqrt(7.0*M_PI));
      static const double const5(16.0/693.0*sqrt(11.0*M_PI));
      return const1*coeff(lj,mj,1,0,li) + const3*coeff(lj,mj,3,0,li) + const5*coeff(lj,mj,5,0,li);
    }

    double Gaunt::sine2_coupling(int lj, int mj, int li, int mi) const {
      if(mi != mj) return 0.0;
      static const double const0(4.0/3.0*sqrt(M_PI));
      static const double const2(-4.0/15.0*sqrt(5.0*M_PI));
      return const0*coeff(lj,mj,0,0,li) + const2*coeff(lj,mj,2,0,li);
    }

    double Gaunt::cosine2_sine2_coupling(int lj, int mj, int li, int mi) const {
      if(mi != mj) return 0.0;
      static const double const0(4.0/15.0*sqrt(M_PI));
      static const double const2(4.0/105.0*sqrt(5.0*M_PI));
      static const double const4(-16.0/105.0*sqrt(M_PI));
      return const0*coeff(lj,mj,0,0,li) + const2*coeff(lj,mj,2,0,li) + const4*coeff(lj,mj,4,0,li);
    }
  }
}
