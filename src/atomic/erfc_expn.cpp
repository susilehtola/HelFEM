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
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "erfc_expn.h"
#include <cmath>
#include <stdexcept>

// For factorials
extern "C" {
#include <gsl/gsl_sf_gamma.h>
}

namespace helfem {
  namespace atomic {
    namespace erfc_expn {
      static double double_factorial(unsigned int n) {
        if(n==0)
          return 1.0;

        return gsl_sf_doublefact(n);
      }

      static double factorial(unsigned int n) {
        if(n==0)
          return 1.0;

        return gsl_sf_fact(n);
      }

      static double choose(int n, int m) {
        // Special cases
        if(n==-1)
          return std::pow(-1.0,m);
        if(m==0)
          return 1.0;

        // Negative binomials
        if(n<0) {
          return gsl_sf_choose(n+m-1,m)*std::pow(-1,m);
        } else {
          return gsl_sf_choose(n,m);
        }
      }

      static inline double Fncoeff(int n, int p) {
        return factorial(n+p)/(factorial(p)*factorial(std::abs(n-p)));
      }

      // Angyan et al, equation (22)
      double Fn(unsigned int n, double Xi, double xi) {
        // Exponential factors
        double explus(std::exp(-std::pow(Xi+xi,2)));
        double exminus(std::exp(-std::pow(Xi-xi,2)));

        // Prefactor
        double prefac(-1.0/(4.0*Xi*xi));

        double F=0.0;
        for(unsigned int p=1;p<=n;p++) {
          F += std::pow(prefac,p+1) * Fncoeff(n,p) * (std::pow(-1,n-p) * explus - exminus);
        }
        // Apply prefactor
        return 2.0/sqrt(M_PI)*F;
      }

      // Angyan et al, equation (24)
      double Hn(unsigned int n, double Xi, double xi) {
        if(Xi<xi)
          throw std::logic_error("Xi < xi");

        double Xi2np1=std::pow(Xi,2*n+1);
        double xi2np1=std::pow(xi,2*n+1);

        double Hn = (Xi2np1+xi2np1)*std::erfc(Xi+xi) - (Xi2np1-xi2np1)*std::erfc(Xi-xi);
        return Hn/(2.0*std::pow(xi*Xi,n+1));
      }

      // Angyan et al, equation (21)
      double Phi_general(unsigned int n, double Xi, double xi) {
        double Fnarr[n+1];
        for(unsigned int i=0;i<=n;i++)
          Fnarr[i]=Fn(i,Xi,xi);

        double Phi = Fnarr[n] + Hn(n,Xi,xi);
        for(unsigned int m=1;m<=n;m++) {
          double Xim(std::pow(Xi,m));
          double xim(std::pow(xi,m));
          Phi += Fnarr[n-m]*((Xim*Xim + xim*xim)/(Xim*xim));
        }

        return Phi;
      }

      // Angyan et al, equations 28 and 29
      double Dnk(int n, int k, double Xi) {
        // Prefactor
        double prefac = std::exp(-std::pow(Xi,2))/sqrt(M_PI)*std::pow(2,n+1)*std::pow(Xi,2*n+1);

        double D = 0.0;
        if(k==0) {
          // Compute the sum
          double sum = 0.0;
          for(int m=1;m<=n;m++)
            sum += 1.0/(double_factorial(2*(n-m)+1)*std::pow(2*Xi*Xi,m));

          D = std::erfc(Xi) + prefac*sum;
        } else {
          // Compute the sum
          double sum = 0.0;
          for(int m=1;m<=k;m++)
            sum += choose(m-k-1,m-1)*std::pow(2*Xi*Xi,k-m)/double_factorial(2*(n+k-m)+1);

          D = prefac * (2*n+1.0)/(2*n + 2*k + 1.0) * sum;
        }

        return D;
      }

      // Angyan et al, equation 30
      double Phi_short(unsigned int n, double Xi, double xi) {
        double Phi = 0.0;
        for(unsigned int k=0; k<=4; k++) {
          double D = Dnk(n,k,Xi);
          double term = D*std::pow(xi,n+2*k);
          Phi += term;
        }
        return Phi/std::pow(Xi,n+1);
      }

      // Wrapper
      double Phi(unsigned int n, double Xi, double xi) {
        // Make sure arguments are in the correct order
        if(Xi < xi)
          std::swap(Xi,xi);

        // See text on top of page 8624 of Angyan et al
        if(xi < 0.4 || (Xi < 0.5 && xi < 2*Xi)) {
          // Short-range Taylor polynomial
          return Phi_short(n,Xi,xi);
        } else {
          // General expansion, susceptible to numerical noise for small
          // arguments
          return Phi_general(n,Xi,xi);
        }
      }
    }
  }
}
