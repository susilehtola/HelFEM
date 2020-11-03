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
#include "lobatto.h"
#include "polynomial_basis.h"
#include "polynomial.h"
#include "orthpoly.h"
#include <cfloat>

// Legendre polynomials
extern "C" {
#include <gsl/gsl_sf_legendre.h>
}

namespace helfem {
  namespace polynomial_basis {
    void drop_first(arma::uvec & idx) {
      // This is simple - just drop first function
      idx=idx.subvec(1,idx.n_elem-1);
    }

    void drop_last(arma::uvec & idx, int noverlap) {
      // This is simple too - drop the last noverlap functions to force function and its derivatives to zero
      idx=idx.subvec(0,idx.n_elem-noverlap-1);
    }

    arma::uvec primitive_indices(int nprim, int noverlap, bool first, bool last) {
      arma::uvec idx(arma::linspace<arma::uvec>(0,nprim-1,nprim));
      if(first)
	drop_first(idx);
      if(last)
	drop_last(idx, noverlap);
      return idx;
    }

    PolynomialBasis * get_basis(int primbas, int Nnodes) {
      if(Nnodes<2)
        throw std::logic_error("Can't have finite element basis with less than two nodes per element.\n");

      // Primitive basis
      polynomial_basis::PolynomialBasis * poly;
      switch(primbas) {
      case(0):
      case(1):
      case(2):
        poly=new HermiteBasis(Nnodes,primbas);
      printf("Basis set composed of %i nodes with %i:th derivative continuity.\n",Nnodes,primbas);
      printf("This means using primitive polynomials of order %i.\n",Nnodes*(primbas+1)-1);
      break;

      case(3):
        poly=new polynomial_basis::LegendreBasis(Nnodes,primbas);
        printf("Basis set composed of %i-node spectral elements.\n",Nnodes);
        break;

      case(4):
        {
          arma::vec x, w;
          ::lobatto_compute(Nnodes,x,w);
          poly=new polynomial_basis::LIPBasis(x,primbas);
          printf("Basis set composed of %i-node LIPs with Gauss-Lobatto nodes.\n",Nnodes);
          break;
        }

      default:
        throw std::logic_error("Unsupported primitive basis.\n");
      }

      // Print out
      //poly->print();

      return poly;
    }

    PolynomialBasis::PolynomialBasis() {
    }

    PolynomialBasis::~PolynomialBasis() {
    }

    int PolynomialBasis::get_nbf() const {
      return nbf;
    }

    int PolynomialBasis::get_noverlap() const {
      return noverlap;
    }

    int PolynomialBasis::get_id() const {
      return id;
    }

    int PolynomialBasis::get_order() const {
      return order;
    }

    void PolynomialBasis::print(const std::string & str) const {
      arma::vec x(arma::linspace<arma::vec>(-1.0,1.0,1001));
      arma::mat bf, df;
      eval(x,bf,df);

      bf.insert_cols(0,x);
      df.insert_cols(0,x);

      std::string fname("bf" + str + ".dat");
      std::string dname("df" + str + ".dat");
      bf.save(fname,arma::raw_ascii);
      df.save(dname,arma::raw_ascii);
    }

    void PolynomialBasis::eval_lapl(const arma::vec & x, arma::mat & lf) const {
      throw std::logic_error("Laplacians haven't been implemented for the used family of basis polynomials.\n");
    }

    HermiteBasis::HermiteBasis(int n_nodes, int der_order) {
      bf_C=polynomial::hermite_coeffs(n_nodes, der_order);
      df_C=polynomial::derivative_coeffs(bf_C, 1);

      // Number of basis functions is
      nbf=bf_C.n_cols;
      // Number of overlapping functions is
      noverlap=der_order+1;

      /// Identifier is
      id=der_order;
      /// Order is
      order=n_nodes;
    }

    HermiteBasis::~HermiteBasis() {
    }

    HermiteBasis * HermiteBasis::copy() const {
      return new HermiteBasis(*this);
    }

    arma::mat HermiteBasis::eval(const arma::vec & x) const {
      return polynomial::polyval(bf_C,x);
    }

    void HermiteBasis::eval(const arma::vec & x, arma::mat & f, arma::mat & df) const {
      f=polynomial::polyval(bf_C,x);
      df=polynomial::polyval(df_C,x);
    }

    void HermiteBasis::eval_lapl(const arma::vec & x, arma::mat & lf) const {
      lf=polynomial::polyval(polynomial::derivative_coeffs(bf_C, 2), x);
    }

    void HermiteBasis::drop_first() {
      // Only drop function value, not derivatives
      arma::uvec idx(arma::linspace<arma::uvec>(0,bf_C.n_cols-1,bf_C.n_cols));
      polynomial_basis::drop_first(idx);

      bf_C=bf_C.cols(idx);
      df_C=df_C.cols(idx);
      nbf=bf_C.n_cols;
    }

    void HermiteBasis::drop_last() {
      // Only drop function value, not derivatives
      arma::uvec idx(arma::linspace<arma::uvec>(0,bf_C.n_cols-1,bf_C.n_cols));
      polynomial_basis::drop_last(idx, noverlap);

      bf_C=bf_C.cols(idx);
      df_C=df_C.cols(idx);
      nbf=bf_C.n_cols;
    }

    LegendreBasis::LegendreBasis(int n_nodes, int id_) {
      lmax=n_nodes-1;

      // Transformation matrix
      T.zeros(lmax+1,lmax+1);

      // First function is (P0-P1)/2
      T(0,0)=0.5;
      T(1,0)=-0.5;
      // Last function is (P0+P1)/2
      T(0,lmax)=0.5;
      T(1,lmax)=0.5;

      // Shape functions [Flores, Clementi, Sonnad, Chem. Phys. Lett. 163, 198 (1989)]
      for(int j=1;j<lmax;j++) {
        double sqfac(1.0/sqrt(4.0*j+2.0));
        T(j+1,j)=sqfac;
        T(j-1,j)=-sqfac;
      }

      noverlap=1;
      nbf=T.n_cols;

      /// Identifier is
      id=id_;
      /// Order is
      order=n_nodes;
    }

    LegendreBasis::~LegendreBasis() {
    }

    LegendreBasis * LegendreBasis::copy() const {
      return new LegendreBasis(*this);
    }

    inline static double sanitize_x(double x) {
        if(x<-1.0) x=-1.0;
        if(x>1.0) x=1.0;
        return x;
    }

    arma::mat LegendreBasis::f_eval(const arma::vec & x) const {
      // Memory for values
      arma::mat ft(x.n_elem,lmax+1);
      // Fill in array
      for(int l=0;l<=lmax;l++)
        for(size_t i=0;i<x.n_elem;i++)
          ft(i,l) = oomph::Orthpoly::legendre(l, x(i));
      return ft;
    }

    arma::mat LegendreBasis::df_eval(const arma::vec & x) const {
      // Memory for values
      arma::mat dt(x.n_elem,lmax+1);
      // Fill in array
      for(int l=0;l<=lmax;l++)
        for(size_t i=0;i<x.n_elem;i++)
          dt(i,l) = oomph::Orthpoly::dlegendre(l, x(i));
      return dt;
    }

    arma::mat LegendreBasis::lf_eval(const arma::vec & x) const {
      // Memory for values
      arma::mat lt(x.n_elem,lmax+1);
      // Fill in array
      for(int l=0;l<=lmax;l++)
        for(size_t i=0;i<x.n_elem;i++)
          lt(i,l) = oomph::Orthpoly::ddlegendre(l, x(i));
      return lt;
    }

    arma::mat LegendreBasis::eval(const arma::vec & x) const {
      return f_eval(x)*T;
    }

    void LegendreBasis::eval(const arma::vec & x, arma::mat & f, arma::mat & df) const {
      f=f_eval(x)*T;
      df=df_eval(x)*T;
    }

    void LegendreBasis::eval_lapl(const arma::vec & x, arma::mat & lf) const {
      lf=lf_eval(x)*T;
    }

    void LegendreBasis::drop_first() {
      T=T.cols(1,T.n_cols-1);
      nbf=T.n_cols;
    }

    void LegendreBasis::drop_last() {
      T=T.cols(0,T.n_cols-2);
      nbf=T.n_cols;
    }

    LIPBasis::LIPBasis(const arma::vec & x, int id_) {
      // Make sure nodes are in order
      x0=arma::sort(x,"ascend");

      // Sanity check
      if(std::abs(x(0)+1)>=sqrt(DBL_EPSILON))
        throw std::logic_error("LIP leftmost node is not at -1!\n");
      if(std::abs(x(x.n_elem-1)-1)>=sqrt(DBL_EPSILON))
        throw std::logic_error("LIP rightmost node is not at -1!\n");

      // One overlapping function
      noverlap=1;
      nbf=x0.n_elem;
      // All functions are enabled
      enabled=arma::linspace<arma::uvec>(0,x0.n_elem-1,x0.n_elem);

      /// Identifier is
      id=id_;
      /// Order is
      order=x.n_elem;
    }

    LIPBasis::~LIPBasis() {
    }

    LIPBasis * LIPBasis::copy() const {
      return new LIPBasis(*this);
    }

    arma::mat LIPBasis::eval(const arma::vec & x) const {
      // Memory for values
      arma::mat bf(x.n_elem,x0.n_elem);

      // Fill in array
      for(size_t ix=0;ix<x.n_elem;ix++) {
        // Loop over polynomials: x_i term excluded
        for(size_t fi=0;fi<x0.n_elem;fi++) {
          // Evaluate the l_i polynomial
          double fval=1.0;
          for(size_t fj=0;fj<x0.n_elem;fj++) {
            // Term not included
            if(fi==fj)
              continue;
            // Compute ratio
            fval *= (x(ix)-x0(fj))/(x0(fi)-x0(fj));
          }
          // Store value
          bf(ix,fi)=fval;
        }
      }

      bf=bf.cols(enabled);

      return bf;
    }

    void LIPBasis::eval(const arma::vec & x, arma::mat & f, arma::mat & df) const {
      // Function values
      f=eval(x);

      // Derivative
      df.zeros(x.n_elem,x0.n_elem);
      for(size_t ix=0;ix<x.n_elem;ix++) {
        // Loop over polynomials
        for(size_t fi=0;fi<x0.n_elem;fi++) {
          // Derivative yields a sum over one of the indices
          for(size_t fj=0;fj<x0.n_elem;fj++) {
            if(fi==fj)
              continue;

            double fval=1.0;
            for(size_t fk=0;fk<x0.n_elem;fk++) {
              // Term not included
              if(fi==fk)
                continue;
              if(fj==fk)
                continue;
              // Compute ratio
              fval *= (x(ix)-x0(fk))/(x0(fi)-x0(fk));
            }
            // Increment derivative
            df(ix,fi)+=fval/(x0(fi)-x0(fj));
          }
        }
      }
      df=df.cols(enabled);
    }

    void LIPBasis::eval_lapl(const arma::vec & x, arma::mat & lf) const {
      // Second derivative
      lf.zeros(x.n_elem,x0.n_elem);
      for(size_t ix=0;ix<x.n_elem;ix++) {
        // Loop over polynomials
        for(size_t fi=0;fi<x0.n_elem;fi++) {
          // Derivative yields a sum over one of the indices
          for(size_t fj=0;fj<x0.n_elem;fj++) {
            if(fi==fj)
              continue;
            // Second derivative yields another sum over the indices
            for(size_t fk=0;fk<x0.n_elem;fk++) {
              if(fi==fk)
                continue;
              if(fj==fk)
                continue;

              double fval=1.0;
              for(size_t fl=0;fl<x0.n_elem;fl++) {
                // Term not included
                if(fi==fl)
                  continue;
                if(fj==fl)
                  continue;
                if(fk==fl)
                  continue;
                // Compute ratio
                fval *= (x(ix)-x0(fl))/(x0(fi)-x0(fl));
              }
              // Increment second derivative
              lf(ix,fi)+=fval/((x0(fi)-x0(fj))*(x0(fi)-x0(fk)));
            }
          }
        }
      }
      lf=lf.cols(enabled);
    }

    void LIPBasis::drop_first() {
      enabled=enabled.subvec(1,enabled.n_elem-1);
      nbf=enabled.n_elem;
    }

    void LIPBasis::drop_last() {
      enabled=enabled.subvec(0,enabled.n_elem-2);
      nbf=enabled.n_elem;
    }
  }
}
