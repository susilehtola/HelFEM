#include <cstdio>
#include <cmath>
#include "gaunt.h"

extern "C" {
  // 3j symbols
#include <gsl/gsl_sf_coupling.h>
}

/// Index of (l,m) in tables: l^2 + l + m
#define lmind(l,m) ( ((size_t) (l))*(size_t (l)) + (size_t) (l) + (size_t) (m))

namespace helfem {
  namespace gaunt {

    double gaunt_coefficient(int L, int M, int l, int m, int lp, int mp) {
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
      table=arma::cube(lmind(Lmax,Lmax)+1,lmind(lmax,lmax)+1,lmind(lpmax,lpmax)+1);

      // Compute coefficients
      for(int L=0;L<=Lmax;L++)
	for(int M=-L;M<=L;M++)

	  for(int l=0;l<=lmax;l++)
	    for(int m=-l;m<=l;m++)

	      for(int lp=0;lp<=lpmax;lp++)
		for(int mp=-lp;mp<=lp;mp++)
		  table(lmind(L,M),lmind(l,m),lmind(lp,mp))=gaunt_coefficient(L,M,l,m,lp,mp);
    }

    Gaunt::~Gaunt() {
    }

    double Gaunt::coeff(int L, int M, int l, int m, int lp, int mp) const {
      if(std::abs(M)>L) return 0.0;
      if(std::abs(m)>l) return 0.0;
      if(std::abs(mp)>lp) return 0.0;

      size_t irow(lmind(L,M));
      size_t icol(lmind(l,m));
      size_t islice(lmind(lp,mp));
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
  }
}
