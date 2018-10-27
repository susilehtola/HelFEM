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
#include <cstdio>
#include <cmath>
#include "gaunt.h"

extern "C" {
  // 3j symbols
#include <gsl/gsl_sf_coupling.h>
}

/// Index of (l,m) in tables: l^2 + l + m
#define genind(l,m) ( ((size_t) (l))*(size_t (l)) + (size_t) (l) + (size_t) (m))
/// Index of (l,m) in m limited table
#define LMind(L,M) ( ((size_t) (L))*(size_t (2*Mmax+1)) + (size_t) (Mmax) + (size_t) (M))
#define lmind(L,M) ( ((size_t) (L))*(size_t (2*mmax+1)) + (size_t) (mmax) + (size_t) (M))
#define lpmpind(L,M) ( ((size_t) (L))*(size_t (2*mpmax+1)) + (size_t) (mpmax) + (size_t) (M))

namespace helfem {
  namespace gaunt {

    double gaunt_coefficient(int L, int M, int l, int m, int lp, int mp) {
      if(M != m+mp)
        // Coefficient vanishes if l values don't match
        return 0.0;
      if(L < abs(l-lp) || L > l+lp)
        // Can't couple
        return 0.0;

      // First, compute square root factor
      double res=sqrt((2*L+1)*(2*l+1)*(2*lp+1)/(4.0*M_PI));
      // Plug in (l1 l2 l3 | 0 0 0), GSL uses half units
      res*=gsl_sf_coupling_3j(2*L,2*l,2*lp,0,0,0);
      // Plug in (l1 l2 l3 | m1 m2 m3)
      res*=gsl_sf_coupling_3j(2*L,2*l,2*lp,-2*M,2*m,2*mp);
      // and the phase factor
      res*=pow(-1.0,M);

      return res;
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

    Gaunt::Gaunt() {
    }

    Gaunt::Gaunt(int Lmax, int lmax, int lpmax) {
      // Allocate storage
      mlimit=false;
      table=arma::cube(genind(Lmax,Lmax)+1,genind(lmax,lmax)+1,genind(lpmax,lpmax)+1);

      // Compute coefficients
#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
      for(int L=0;L<=Lmax;L++)
	  for(int l=0;l<=lmax;l++)
            for(int lp=0;lp<=lpmax;lp++)
              for(int M=-L;M<=L;M++)
                for(int m=-l;m<=l;m++)
                  for(int mp=-lp;mp<=lp;mp++)
                    table(genind(L,M),genind(l,m),genind(lp,mp))=gaunt_coefficient(L,M,l,m,lp,mp);
    }

    Gaunt::Gaunt(int Lmax, int Mmax_, int lmax, int mmax_, int lpmax, int mpmax_) : Mmax(Mmax_), mmax(mmax_), mpmax(mpmax_) {
      // Allocate storage
      mlimit=true;
      table=arma::cube(LMind(Lmax,Mmax)+1,lmind(lmax,mmax)+1,lpmpind(lpmax,mpmax)+1);

      // Compute coefficients
#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
      for(int L=0;L<=Lmax;L++)
	  for(int l=0;l<=lmax;l++)
            for(int lp=0;lp<=lpmax;lp++)
              for(int M=-Mmax;M<=Mmax;M++)
                for(int m=-mmax;m<=mmax;m++)
                  for(int mp=-mpmax;mp<=mpmax;mp++)
                    table(LMind(L,M),lmind(l,m),lpmpind(lp,mp))=gaunt_coefficient(L,M,l,m,lp,mp);
    }

    Gaunt::~Gaunt() {
    }

    double Gaunt::coeff(int L, int M, int l, int m, int lp, int mp) const {
      if(std::abs(M)>L) return 0.0;
      if(std::abs(m)>l) return 0.0;
      if(std::abs(mp)>lp) return 0.0;

      size_t irow, icol, islice;
      if(mlimit) {
        irow=LMind(L,M);
        icol=lmind(l,m);
        islice=lpmpind(lp,mp);
      } else {
        irow=genind(L,M);
        icol=genind(l,m);
        islice=genind(lp,mp);
      }

#ifndef ARMA_NO_DEBUG
      if(irow>table.n_rows) {
        std::ostringstream oss;
        oss << "Row index overflow for coeff(" << L << "," << M << "," << l << "," << m << "," << lp << "," << mp << ")!\n";
        oss << "Wanted element at (" << irow << "," << icol << "," << islice << ") but table is " << table.n_rows << " x " << table.n_cols << " x " << table.n_slices << "\n";
        throw std::logic_error(oss.str());
      }

      if(icol>table.n_cols) {
        std::ostringstream oss;
        oss << "Column index overflow for coeff(" << L << "," << M << "," << l << "," << m << "," << lp << "," << mp << ")!\n";
        oss << "Wanted element at (" << irow << "," << icol << "," << islice << ") but table is " << table.n_rows << " x " << table.n_cols << " x " << table.n_slices << "\n";
        throw std::logic_error(oss.str());
      }

      if(islice>table.n_slices) {
        std::ostringstream oss;
        oss << "Slice index overflow for coeff(" << L << "," << M << "," << l << "," << m << "," << lp << "," << mp << ")!\n";
        oss << "Wanted element at (" << irow << "," << icol << "," << islice << ") but table is " << table.n_rows << " x " << table.n_cols << " x " << table.n_slices << "\n";
        throw std::logic_error(oss.str());
      }
#endif

      return table(irow,icol,islice);
    }

    double Gaunt::cosine_coupling(int lj, int mj, int li, int mi) const {
      // cos th = const1 * Y_1^0
      static const double const1(2.0*sqrt(M_PI/3.0));
      return const1*coeff(lj,mj,1,0,li,mi);
    }

    double Gaunt::cosine2_coupling(int lj, int mj, int li, int mi) const {
      // cos^2 th = const0 * Y_0^0 + const2 * Y_2^0
      static const double const0(2.0/3.0*sqrt(M_PI));
      static const double const2(4.0/15.0*sqrt(5.0*M_PI));
      return const0*coeff(lj,mj,0,0,li,mi) + const2*coeff(lj,mj,2,0,li,mi);
    }

    double Gaunt::mod_coeff(int lj, int mj, int L, int M, int li, int mi) const {
      static const double const0(2.0/3.0*sqrt(M_PI));
      static const double const2(4.0/15.0*sqrt(5.0*M_PI));

      // Coupling through Y_0^0
      double cpl0(coeff(L,M,0,0,L,M)*coeff(lj,mj,li,mi,L,M));

      // Coupling through Y_2^0
      double cpl2=0.0;
      for(int Lp=std::max(std::max(L-2,0),std::abs(M));Lp<=L+2;Lp++)
        cpl2+=coeff(Lp,M,2,0,L,M)*coeff(lj,mj,li,mi,Lp,M);

      return const0*cpl0+const2*cpl2;
    }

    double Gaunt::cosine3_coupling(int lj, int mj, int li, int mi) const {
      // cos^3 th = const1 * Y_1^0 + const3 * Y_3^0
      static const double const1(2.0/5.0*sqrt(3.0*M_PI));
      static const double const3(4.0/35.0*sqrt(7.0*M_PI));
      return const1*coeff(lj,mj,1,0,li,mi) + const3*coeff(lj,mj,3,0,li,mi);
    }

    double Gaunt::cosine4_coupling(int lj, int mj, int li, int mi) const {
      // cos^4 th = const0 * Y_0^0 + const2 * Y_2^0 + const4 * Y_4^0
      static const double const0(2.0/5.0*sqrt(M_PI));
      static const double const2(8.0/35.0*sqrt(5.0*M_PI));
      static const double const4(16.0/105.0*sqrt(M_PI));
      return const0*coeff(lj,mj,0,0,li,mi) + const2*coeff(lj,mj,2,0,li,mi) + const4*coeff(lj,mj,4,0,li,mi);
    }

    double Gaunt::cosine5_coupling(int lj, int mj, int li, int mi) const {
      // cos^5 th = const1 * Y_1^0 + const3 * Y_3^0 + const5 * Y_5^0
      static const double const1(2.0/7.0*sqrt(3.0*M_PI));
      static const double const3(8.0/63.0*sqrt(7.0*M_PI));
      static const double const5(16.0/693.0*sqrt(11.0*M_PI));
      return const1*coeff(lj,mj,1,0,li,mi) + const3*coeff(lj,mj,3,0,li,mi) + const5*coeff(lj,mj,5,0,li,mi);
    }

    double Gaunt::sine2_coupling(int lj, int mj, int li, int mi) const {
      // sin^2 th = const0 * Y_0^0 + const2 * Y_2^0
      static const double const0(4.0/3.0*sqrt(M_PI));
      static const double const2(-4.0/15.0*sqrt(5.0*M_PI));
      return const0*coeff(lj,mj,0,0,li,mi) + const2*coeff(lj,mj,2,0,li,mi);
    }

    double Gaunt::cosine2_sine2_coupling(int lj, int mj, int li, int mi) const {
      // cos^2 th sin^2 th = const0 * Y_0^0 + const2 * Y_2^0 + const4 * Y_4^0
      static const double const0(4.0/15.0*sqrt(M_PI));
      static const double const2(4.0/105.0*sqrt(5.0*M_PI));
      static const double const4(-16.0/105.0*sqrt(M_PI));
      return const0*coeff(lj,mj,0,0,li,mi) + const2*coeff(lj,mj,2,0,li,mi) + const4*coeff(lj,mj,4,0,li,mi);
    }
  }
}
