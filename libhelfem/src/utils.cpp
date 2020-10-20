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
#include "utils.h"
#include <cmath>

extern "C" {
#include <gsl/gsl_sf_bessel.h>
}

namespace helfem {
  namespace utils {
    double arcosh(double x) {
      return log(x+sqrt(x*x-1.0));
    }

    arma::vec arcosh(const arma::vec & x) {
      arma::vec y(x);
      for(size_t i=0;i<x.n_elem;i++)
	y(i)=arcosh(x(i));
      return y;
    }

    double arsinh(double x) {
      return log(x+sqrt(x*x+1.0));
    }

    arma::vec arsinh(const arma::vec & x) {
      arma::vec y(x);
      for(size_t i=0;i<x.n_elem;i++)
	y(i)=arsinh(x(i));
      return y;
    }

    arma::vec bessel_il(const arma::vec & x, int L) {
      arma::vec y(x);
      for(size_t i=0;i<x.n_elem;i++)
        // GSL calculates exp(-|x|)k_l(x)
	y(i)=exp(std::abs(x(i)))*gsl_sf_bessel_il_scaled(L, x(i));
      return y;
    }

    arma::vec bessel_kl(const arma::vec & x, int L) {
      arma::vec y(x);
      for(size_t i=0;i<x.n_elem;i++)
        // GSL calculates exp(x)k_l(x)
	y(i)=exp(-x(i))*gsl_sf_bessel_kl_scaled(L, x(i));

      // The definition in GSL is \sqrt(\pi/(2x)), not \sqrt(2/(\pi x))
      y /= M_PI_2;

      return y;
    }

    arma::mat product_tei(const arma::mat & ijint, const arma::mat & klint) {
      const size_t Ni(ijint.n_rows);
      const size_t Nj(ijint.n_cols);
      const size_t Nk(klint.n_rows);
      const size_t Nl(klint.n_cols);

      arma::mat teiblock(Ni*Nj,Nk*Nl);
      teiblock.zeros();

      // Form block
      for(size_t fk=0;fk<Nk;fk++)
        for(size_t fl=0;fl<Nl;fl++) {
          // Use temp variable
          double kl(klint(fk,fl));

          for(size_t fi=0;fi<Ni;fi++)
            for(size_t fj=0;fj<Nj;fj++)
              // (ij|kl) in Armadillo compatible indexing
              teiblock(fj*Ni+fi,fl*Nk+fk)=kl*ijint(fi,fj);
        }

      return teiblock;
    }

    void check_tei_symmetry(const arma::mat & tei, size_t Ni, size_t Nj, size_t Nk, size_t Nl) {
      arma::mat teiwrk(tei);

      // (ij|kl) = (ji|kl)
      for(size_t ii=0;ii<Ni;ii++)
        for(size_t jj=0;jj<Nj;jj++)
          for(size_t kk=0;kk<Nk;kk++)
            for(size_t ll=0;ll<Nl;ll++)
              teiwrk(ii*Nj+jj,ll*Nk+kk)=tei(jj*Ni+ii,ll*Nk+kk);
      teiwrk-=tei;
      double jinorm(arma::norm(teiwrk,"fro"));

      // (ij|kl) = (ij|lk)
      for(size_t ii=0;ii<Ni;ii++)
        for(size_t jj=0;jj<Nj;jj++)
          for(size_t kk=0;kk<Nk;kk++)
            for(size_t ll=0;ll<Nl;ll++)
              teiwrk(jj*Ni+ii,kk*Nl+ll)=tei(jj*Ni+ii,ll*Nk+kk);
      teiwrk-=tei;
      double lknorm(arma::norm(teiwrk,"fro"));

      // (ij|kl) = (ji|lk)
      arma::mat tei_jilk(tei);
      for(size_t ii=0;ii<Ni;ii++)
        for(size_t jj=0;jj<Nj;jj++)
          for(size_t kk=0;kk<Nk;kk++)
            for(size_t ll=0;ll<Nl;ll++)
              teiwrk(ii*Nj+jj,kk*Nl+ll)=tei(jj*Ni+ii,ll*Nk+kk);
      teiwrk-=tei;
      double jilknorm(arma::norm(teiwrk,"fro"));

      printf("%e %e %e\n",jinorm,lknorm,jilknorm);
    }

    arma::mat exchange_tei(const arma::mat & tei, size_t Ni, size_t Nj, size_t Nk, size_t Nl) {
#ifndef ARMA_NO_DEBUG
      if(tei.n_rows != Ni*Nj) {
        std::ostringstream oss;
        oss << "Invalid input tei: was supposed to get " << Ni*Nj << " rows but got " << tei.n_rows << "!\n";
        throw std::logic_error(oss.str());
      }
      if(tei.n_cols != Nk*Nl) {
        std::ostringstream oss;
        oss << "Invalid input tei: was supposed to get " << Nk*Nl << " cols but got " << tei.n_cols << "!\n";
        throw std::logic_error(oss.str());
      }
#endif

      arma::mat ktei(Nj*Nk,Ni*Nl);
      ktei.zeros();
      for(size_t ii=0;ii<Ni;ii++)
        for(size_t jj=0;jj<Nj;jj++)
          for(size_t kk=0;kk<Nk;kk++)
            for(size_t ll=0;ll<Nl;ll++)
              // (ik|jl) in Armadillo compatible indexing
              ktei(kk*Nj+jj,ll*Ni+ii)=tei(jj*Ni+ii,ll*Nk+kk);

      return ktei;
    }

    int stricmp(const std::string & str1, const std::string & str2) {
      return strcasecmp(str1.c_str(),str2.c_str());
    }

    arma::mat invh(arma::mat S, bool chol) {
      // Get the basis function norms
      arma::vec bfnormlz(arma::pow(arma::diagvec(S),-0.5));
      // Go to normalized basis
      S=arma::diagmat(bfnormlz)*S*arma::diagmat(bfnormlz);

      // Half-inverse is
      arma::mat Sinvh;
      if(chol) {
        Sinvh = arma::inv(arma::chol(S));
      } else {
        arma::vec Sval;
        arma::mat Svec;
        if(!arma::eig_sym(Sval,Svec,S)) {
          throw std::logic_error("Diagonalization of overlap matrix failed\n");
        }
        printf("Smallest eigenvalue of overlap matrix is % e, condition number %e\n",Sval(0),Sval(Sval.n_elem-1)/Sval(0));

        Sinvh=Svec*arma::diagmat(arma::pow(Sval,-0.5))*arma::trans(Svec);
      }

      Sinvh=arma::diagmat(bfnormlz)*Sinvh;
      return Sinvh;
    }
  }
}
