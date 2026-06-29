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
#include "utils.h"
#include <lib1dfem/math.h>
#include <cmath>
#include <cstring>

namespace helfem {
  namespace utils {
    // v2 refactor (Phase 1): the pure-math helpers (arcosh, arsinh,
    // bessel_il, bessel_kl) now live in lib1dfem and are templated on the
    // scalar type. The functions below are thin double-only compatibility
    // shims that forward to the templated implementations.

    double arcosh(double x) {
      return helfem::lib1dfem::math::arcosh<double>(x);
    }

    // Phase 5.1: lib1dfem math helpers are Eigen-typed. The arma::vec
    // shims convert one direction at the boundary.
    arma::vec arcosh(const arma::vec & x) {
      Eigen::Matrix<double, Eigen::Dynamic, 1> xe(x.n_elem);
      std::memcpy(xe.data(), x.memptr(), sizeof(double) * x.n_elem);
      auto ye = helfem::lib1dfem::math::arcosh<double>(xe);
      arma::vec y(ye.size());
      std::memcpy(y.memptr(), ye.data(), sizeof(double) * (size_t) ye.size());
      return y;
    }

    double arsinh(double x) {
      return helfem::lib1dfem::math::arsinh<double>(x);
    }

    arma::vec arsinh(const arma::vec & x) {
      Eigen::Matrix<double, Eigen::Dynamic, 1> xe(x.n_elem);
      std::memcpy(xe.data(), x.memptr(), sizeof(double) * x.n_elem);
      auto ye = helfem::lib1dfem::math::arsinh<double>(xe);
      arma::vec y(ye.size());
      std::memcpy(y.memptr(), ye.data(), sizeof(double) * (size_t) ye.size());
      return y;
    }

    double bessel_il(double r, int L) {
      return helfem::lib1dfem::math::bessel_il<double>(r, L);
    }

    arma::vec bessel_il(const arma::vec & r, int L) {
      Eigen::Matrix<double, Eigen::Dynamic, 1> re(r.n_elem);
      std::memcpy(re.data(), r.memptr(), sizeof(double) * r.n_elem);
      auto ye = helfem::lib1dfem::math::bessel_il<double>(re, L);
      arma::vec y(ye.size());
      std::memcpy(y.memptr(), ye.data(), sizeof(double) * (size_t) ye.size());
      return y;
    }

    double bessel_kl(double r, int L) {
      return helfem::lib1dfem::math::bessel_kl<double>(r, L);
    }

    arma::vec bessel_kl(const arma::vec & r, int L) {
      Eigen::Matrix<double, Eigen::Dynamic, 1> re(r.n_elem);
      std::memcpy(re.data(), r.memptr(), sizeof(double) * r.n_elem);
      auto ye = helfem::lib1dfem::math::bessel_kl<double>(re, L);
      arma::vec y(ye.size());
      std::memcpy(y.memptr(), ye.data(), sizeof(double) * (size_t) ye.size());
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

    helfem::Matrix exchange_tei(const helfem::Matrix & tei,
                                 size_t Ni, size_t Nj, size_t Nk, size_t Nl) {
      // Eigen overload of the same scalar-by-scalar (ij|kl) -> (jk|il)
      // permutation. Both arma and Eigen are column-major so the
      // packed-pair index layout (a-fast, b-slow) is identical.
      if (static_cast<size_t>(tei.rows()) != Ni*Nj) {
        std::ostringstream oss;
        oss << "Invalid input tei: was supposed to get " << Ni*Nj
            << " rows but got " << tei.rows() << "!\n";
        throw std::logic_error(oss.str());
      }
      if (static_cast<size_t>(tei.cols()) != Nk*Nl) {
        std::ostringstream oss;
        oss << "Invalid input tei: was supposed to get " << Nk*Nl
            << " cols but got " << tei.cols() << "!\n";
        throw std::logic_error(oss.str());
      }
      helfem::Matrix ktei = helfem::Matrix::Zero(Nj*Nk, Ni*Nl);
      for (size_t ii = 0; ii < Ni; ++ii)
        for (size_t jj = 0; jj < Nj; ++jj)
          for (size_t kk = 0; kk < Nk; ++kk)
            for (size_t ll = 0; ll < Nl; ++ll)
              ktei(kk*Nj+jj, ll*Ni+ii) = tei(jj*Ni+ii, ll*Nk+kk);
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
