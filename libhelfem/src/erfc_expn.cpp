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

#include "erfc_expn.h"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <stdexcept>
#include <sstream>

// The Ángyán-Gerber-Marsman erfc/Legendre expansion, templated on the
// scalar type T and explicitly instantiated for double, long double and
// (under HELFEM_HAVE_FLOAT128) _Float128 at the bottom of the namespace.
//
// The algorithm is byte-for-byte the same one as the original double-only
// implementation: the Fn recursion (eq. 22), the Hn tail (eq. 24), the
// general expansion (eq. 21), the Dnk damping coefficients (eqs. 28/29) and
// the short-range Taylor series (eq. 30). Precision-capping constants have
// been replaced by their T-aware equivalents:
//   * M_PI            -> helfem::utils::pi<T>() (arithmetic order preserved,
//                        so the double instantiation is bit-identical: the
//                        long-double pi literal rounds to exactly M_PI at
//                        T = double, and 2/sqrt(pi) is still spelled as the
//                        same T(2)/std::sqrt(pi<T>()) division as before).
//   * DBL_EPSILON     -> std::numeric_limits<T>::epsilon(), so the short-range
//                        Taylor series runs to T's precision, not double's.
// The short-range convergence loop cap was raised from 30 to 200 so that the
// series still reaches the (now much tighter) eps(T) tolerance at long double
// and _Float128. At T = double the eps break fires well before k = 30, so the
// double results are unchanged.

namespace helfem {
  namespace atomic {
    namespace erfc_expn {
      template <typename T> static T double_factorial(unsigned int n) {
        T v = T(1);
        for(unsigned int k=n; k>=2; k-=2)
          v *= T(k);
        return v;
      }

      template <typename T> static T factorial(unsigned int n) {
        T v = T(1);
        for(unsigned int k=2; k<=n; ++k)
          v *= T(k);
        return v;
      }

      template <typename T> static T choose(int n, int m) {
        // Special cases
        if(n==-1)
          // choose(-1,m) = (-1)^m
          return std::pow(T(-1),m);
        if(n==0)
          // choose(0,m) = 0 except for choose(0,0) = 1
          return m==0 ? T(1) : T(0);
        if(m==0)
          // choose(n,0) = 1 for all n
          return T(1);
        if(m==1)
          // choose(n,1) = n for all n
          return T(n);
        if(n>0 && m>0 && m>n)
          // choose(n,m) = 0 for m>n positive
          return T(0);

        // Negative binomials
        if(n<0) {
          return choose<T>(n+m-1,m)*std::pow(T(-1),m);
        } else {
          // Multiplicative formula keeps the running value as a T
          // and avoids any factorial overflow for large n.
          T v = T(1);
          const int k = std::min(m, n - m);
          for(int i=0; i<k; ++i)
            v *= T(n - i) / T(i + 1);
          return v;
        }
      }

      // Angyan et al, equation (22)
      template <typename T> static T Fn(unsigned int n, T Xi, T xi) {
        // Exponential factors
        T explus(std::exp(-std::pow(Xi+xi,2)));
        T exminus(std::exp(-std::pow(Xi-xi,2)));

        // Prefactor
        T prefac(T(-1)/(T(4)*Xi*xi));

        T F=T(0);
        // Looks like there's a typo in the equation: I can't make the
        // function match the equations in the appendix unless the
        // lower limit is 0 instead of 1.
        for(unsigned int p=0;p<=n;p++) {
          F += std::pow(prefac,(int)(p+1)) * (factorial<T>(n+p)/(factorial<T>(p)*factorial<T>(n-p))) * (std::pow(T(-1),(int)(n-p)) * explus - exminus);
        }
        // Apply prefactor
        return T(2)/std::sqrt(utils::pi<T>())*F;
      }

      // Angyan et al, equation (24)
      template <typename T> static T Hn(unsigned int n, T Xi, T xi) {
        if(Xi<xi)
          throw std::logic_error("Xi < xi");

        T Xi2np1=std::pow(Xi,(int)(2*n+1));
        T xi2np1=std::pow(xi,(int)(2*n+1));

        T Hn = (Xi2np1+xi2np1)*std::erfc(Xi+xi) - (Xi2np1-xi2np1)*std::erfc(Xi-xi);
        return Hn/(T(2)*std::pow(xi*Xi,(int)(n+1)));
      }

      // Angyan et al, equation (21)
      template <typename T> static T Phi_general(unsigned int n, T Xi, T xi) {
        // Make sure arguments are in the correct order
        if(Xi < xi)
          std::swap(Xi,xi);

        std::vector<T> Fnarr(n);
        for(unsigned int i=0;i<n;i++)
          Fnarr[i]=Fn<T>(i,Xi,xi);

        T sum = T(0);
        for(unsigned int m=1;m<=n;m++) {
          T Xim(std::pow(Xi,(int)m));
          T xim(std::pow(xi,(int)m));
          sum += Fnarr[n-m]*((Xim*Xim + xim*xim)/(Xim*xim));
        }

        return Fn<T>(n,Xi,xi) + sum + Hn<T>(n,Xi,xi);
      }

      // Angyan et al, equations 28 and 29
      template <typename T> static T Dnk(int n, int k, T Xi) {
        // Prefactor
        T prefac = std::exp(-std::pow(Xi,2))/std::sqrt(utils::pi<T>())*std::pow(T(2),n+1)*std::pow(Xi,2*n+1);

        T D = T(0);
        if(k==0) {
          // Compute the sum
          T sum = T(0);
          for(int m=1;m<=n;m++)
            sum += T(1)/(double_factorial<T>(2*(n-m)+1)*std::pow(T(2)*Xi*Xi,m));

          D = std::erfc(Xi) + prefac*sum;
        } else {
          // Compute the sum
          T sum = T(0);
          for(int m=1;m<=k;m++)
            sum += choose<T>(m-k-1,m-1)*std::pow(T(2)*Xi*Xi,k-m)/double_factorial<T>(2*(n+k-m)+1);

          D = prefac * (T(2)*T(n)+T(1))/(factorial<T>(k)*(T(2)*T(n+k)+T(1))) * sum;
        }

        return D;
      }

      // Angyan et al, equation (30), evaluated to full numerical precision
      template <typename T> static T Phi_short(unsigned int n, T Xi, T xi) {
        // Make sure arguments are in the correct order
        if(Xi < xi)
          std::swap(Xi,xi);
        // this is a power series in xi
        if(xi == T(0) && n>0)
          return T(0);
        else if(n == 0 && xi == T(0) && Xi == T(0))
          // This is an edge case and I think this should be the right value
          return T(1);

        T Phi = T(0);
        T dPhi = T(0);
        unsigned int k;
        const T tol = std::numeric_limits<T>::epsilon();
        // Convergence is tested against max(|Phi|, 1.0): the relative test
        // alone fails when alternating-sign cancellation drives |Phi| toward
        // zero, even though the increment is genuinely below numerical noise.
        // The cap is 200 (was 30): at long double / _Float128 the eps(T)
        // tolerance is far tighter, so the series needs more terms before the
        // increment falls under noise. At T = double the eps break still fires
        // by k ~ 30, leaving the double results untouched.
        bool converged = false;
        for(k=0; k<=200; k+=2) {
          // Unroll odd values so that we don't truncate too soon by accident
          dPhi = Dnk<T>(n,k  ,Xi)*std::pow(xi,(int)(n+2*k))
            +    Dnk<T>(n,k+1,Xi)*std::pow(xi,(int)(n+2*(k+1)));
          Phi += dPhi;
          if(std::abs(dPhi) < tol*std::max(std::abs(Phi), T(1))) {
            converged = true;
            break;
          }
        }
        if(!converged) {
          std::ostringstream oss;
          oss << "Phi_short Taylor series failed to converge: n=" << n
              << " Xi=" << static_cast<double>(Xi) << " xi=" << static_cast<double>(xi)
              << " dPhi=" << static_cast<double>(dPhi) << " Phi=" << static_cast<double>(Phi) << "\n";
          throw std::runtime_error(oss.str());
        }

        return Phi/std::pow(Xi,(int)(n+1));
      }

      // Wrapper
      template <typename T> T Phi(unsigned int n, T Xi, T xi) {
        // Make sure arguments are in the correct order
        if(Xi < xi)
          std::swap(Xi,xi);

        // See text on top of page 8624 of Angyan et al
        if(xi < T(0.4) || (Xi < T(0.5) && xi < T(2)*Xi)) {
          // Short-range Taylor polynomial
          return Phi_short<T>(n,Xi,xi);
        } else {
          // General expansion, susceptible to numerical noise for
          // small arguments
          return Phi_general<T>(n,Xi,xi);
        }
      }

      // Explicit instantiations.
      template double      Phi<double>     (unsigned int, double, double);
      template long double Phi<long double>(unsigned int, long double, long double);
#ifdef HELFEM_HAVE_FLOAT128
      template _Float128   Phi<_Float128>  (unsigned int, _Float128, _Float128);
#endif
    }
  }
}
