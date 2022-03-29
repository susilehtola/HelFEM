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
        throw std::runtime_error("Deprecated primitive basis, use 3, 4, or 5.\n");
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

      case(5):
        {
          arma::vec x, w;
          ::lobatto_compute(Nnodes,x,w);
          poly=new polynomial_basis::HIPBasis(x,primbas);
          printf("Basis set composed of %i-node HIPs with Gauss-Lobatto nodes.\n",Nnodes);
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
      printf("In PolynomialBasis::PolynomialBasis\n");
    }

    PolynomialBasis::~PolynomialBasis() {
    }

    int PolynomialBasis::get_nprim() const {
      return nprim;
    }

    int PolynomialBasis::get_nbf() const {
      return enabled.n_elem;
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
      eval_f(x,bf,1.0);
      eval_df(x,df,1.0);

      bf.insert_cols(0,x);
      df.insert_cols(0,x);

      std::string fname("bf" + str + ".dat");
      std::string dname("df" + str + ".dat");
      bf.save(fname,arma::raw_ascii);
      df.save(dname,arma::raw_ascii);
    }

    arma::mat PolynomialBasis::eval_f(const arma::vec & x, double element_length) const {
      arma::mat f;
      eval_f(x, f, element_length);
      return f;
    }

    arma::mat PolynomialBasis::eval_df(const arma::vec & x, double element_length) const {
      arma::mat df;
      eval_df(x, df, element_length);
      return df;
    }

    arma::mat PolynomialBasis::eval_d2f(const arma::vec & x, double element_length) const {
      arma::mat d2f;
      eval_d2f(x, d2f, element_length);
      return d2f;
    }

    arma::mat PolynomialBasis::eval_d3f(const arma::vec & x, double element_length) const {
      arma::mat d3f;
      eval_d3f(x, d3f, element_length);
      return d3f;
    }

    void PolynomialBasis::eval_prim_f(const arma::vec & x, arma::mat & f, double element_length) const {
      (void) x;
      (void) f;
      (void) element_length;
      throw std::logic_error("Values haven't been implemented for the used family of basis polynomials.\n");
    }

    void PolynomialBasis::eval_prim_df(const arma::vec & x, arma::mat & df, double element_length) const {
      (void) x;
      (void) df;
      (void) element_length;
      throw std::logic_error("Derivatives haven't been implemented for the used family of basis polynomials.\n");
    }

    void PolynomialBasis::eval_prim_d2f(const arma::vec & x, arma::mat & d2f, double element_length) const {
      (void) x;
      (void) d2f;
      (void) element_length;
      throw std::logic_error("Second derivatives haven't been implemented for the used family of basis polynomials.\n");
    }

    void PolynomialBasis::eval_prim_d3f(const arma::vec & x, arma::mat & d3f, double element_length) const {
      (void) x;
      (void) d3f;
      (void) element_length;
      throw std::logic_error("Third derivatives haven't been implemented for the used family of basis polynomials.\n");
    }

    void PolynomialBasis::eval_f(const arma::vec & x, arma::mat & f, double element_length) const {
      eval_prim_f(x, f, element_length);
      // No scaling needed here
      f = f.cols(enabled);
    }

    void PolynomialBasis::eval_df(const arma::vec & x, arma::mat & df, double element_length) const {
      eval_prim_df(x, df, element_length);
      // Derivative is scaled by element length
      df = df.cols(enabled) / element_length;
    }

    void PolynomialBasis::eval_d2f(const arma::vec & x, arma::mat & d2f, double element_length) const {
      eval_prim_d2f(x, d2f, element_length);
      // Second derivative is scaled by element length squared
      d2f = d2f.cols(enabled) / std::pow(element_length, 2);
    }

    void PolynomialBasis::eval_d3f(const arma::vec & x, arma::mat & d3f, double element_length) const {
      eval_prim_d3f(x, d3f, element_length);
      // Third derivative is scaled by element length cubed
      d3f = d3f.cols(enabled) / std::pow(element_length, 3);
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
      nprim=T.n_cols;
      enabled=arma::linspace<arma::uvec>(0,T.n_cols-1,T.n_cols);

      /// Identifier is
      id=id_;
      /// Order is
      order=enabled.n_elem;
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

    arma::mat LegendreBasis::d2f_eval(const arma::vec & x) const {
      // Memory for values
      arma::mat lt(x.n_elem,lmax+1);
      // Fill in array
      for(int l=0;l<=lmax;l++)
        for(size_t i=0;i<x.n_elem;i++)
          lt(i,l) = oomph::Orthpoly::ddlegendre(l, x(i));
      return lt;
    }

    void LegendreBasis::eval_prim_f(const arma::vec & x, arma::mat & f, double element_length) const {
      (void) element_length;
      f=f_eval(x)*T;
    }

    void LegendreBasis::eval_prim_df(const arma::vec & x, arma::mat & df, double element_length) const {
      (void) element_length;
      df=df_eval(x)*T;
    }

    void LegendreBasis::eval_prim_d2f(const arma::vec & x, arma::mat & d2f, double element_length) const {
      (void) element_length;
      d2f=d2f_eval(x)*T;
    }

    void LegendreBasis::drop_first(bool deriv) {
      (void) deriv;
      enabled=enabled(1,T.n_cols-1);
    }

    void LegendreBasis::drop_last(bool deriv) {
      (void) deriv;
      enabled=enabled(0,T.n_cols-2);
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
      nprim=x0.n_elem;
      // All functions are enabled
      enabled=arma::linspace<arma::uvec>(0,x0.n_elem-1,x0.n_elem);

      /// Identifier is
      id=id_;
      /// Order is
      order=enabled.n_elem;
    }

    LIPBasis::~LIPBasis() {
    }

    LIPBasis * LIPBasis::copy() const {
      return new LIPBasis(*this);
    }

    void LIPBasis::eval_f_raw(const arma::vec & x, arma::mat & bf) const {
      // Memory for values
      bf.zeros(x.n_elem,x0.n_elem);

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
    }

    void LIPBasis::eval_df_raw(const arma::vec & x, arma::mat & df) const {
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
    }

    void LIPBasis::eval_d2f_raw(const arma::vec & x, arma::mat & d2f) const {
      // Second derivative
      d2f.zeros(x.n_elem,x0.n_elem);
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
              d2f(ix,fi)+=fval/((x0(fi)-x0(fj))*(x0(fi)-x0(fk)));
            }
          }
        }
      }
    }

    void LIPBasis::eval_d3f_raw(const arma::vec & x, arma::mat & d3f) const {
      // Third derivative
      d3f.zeros(x.n_elem,x0.n_elem);
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
              // Third derivative yields yet another sum over the indices
              for(size_t fl=0;fl<x0.n_elem;fl++) {
                if(fi==fl)
                  continue;
                if(fj==fl)
                  continue;
                if(fk==fl)
                  continue;

                double fval=1.0;
                for(size_t fm=0;fm<x0.n_elem;fm++) {
                  // Term not included
                  if(fi==fm)
                    continue;
                  if(fj==fm)
                    continue;
                  if(fk==fm)
                    continue;
                  if(fl==fm)
                    continue;
                  // Compute ratio
                  fval *= (x(ix)-x0(fm))/(x0(fi)-x0(fm));
                }
                // Increment third derivative
                d3f(ix,fi)+=fval/((x0(fi)-x0(fj))*(x0(fi)-x0(fk))*(x0(fi)-x0(fl)));
              }
            }
          }
        }
      }
    }

    void LIPBasis::eval_prim_f(const arma::vec & x, arma::mat & f, double element_length) const {
      (void) element_length;
      eval_f_raw(x, f);
    }

    void LIPBasis::eval_prim_df(const arma::vec & x, arma::mat & df, double element_length) const {
      (void) element_length;
      eval_df_raw(x, df);
    }

    void LIPBasis::eval_prim_d2f(const arma::vec & x, arma::mat & d2f, double element_length) const {
      (void) element_length;
      eval_d2f_raw(x, d2f);
    }

    void LIPBasis::eval_prim_d3f(const arma::vec & x, arma::mat & d3f, double element_length) const {
      (void) element_length;
      eval_d3f_raw(x, d3f);
    }

    void LIPBasis::drop_first(bool deriv) {
      (void) deriv;
      enabled=enabled.subvec(1,enabled.n_elem-1);
    }

    void LIPBasis::drop_last(bool deriv) {
      (void) deriv;
      enabled=enabled.subvec(0,enabled.n_elem-2);
    }

    HIPBasis::HIPBasis(const arma::vec & x, int id_) : LIPBasis(x, id_) {
      // Two overlapping functions
      noverlap=2;
      nprim=2*x0.n_elem;
      // All functions are enabled
      enabled=arma::linspace<arma::uvec>(0,2*x0.n_elem-1,2*x0.n_elem);
      /// Order is
      order=enabled.n_elem;

      // Evaluate derivatives at nodes
      arma::mat dlip;
      LIPBasis::eval_df_raw(x, dlip);
      lipxi = arma::diagvec(dlip);
    }

    HIPBasis::~HIPBasis() {
    }

    HIPBasis * HIPBasis::copy() const {
      return new HIPBasis(*this);
    }

    void HIPBasis::eval_prim_f(const arma::vec & x, arma::mat & f, double element_length) const {
      // Evaluate LIP basis
      arma::mat lip, dlip;
      LIPBasis::eval_f_raw(x, lip);
      LIPBasis::eval_df_raw(x, dlip);

      // Basis function values
      f.zeros(x.n_elem, 2*x0.n_elem);
      for(size_t ix=0;ix<x.n_elem;ix++) {
        for(size_t fi=0;fi<x0.n_elem;fi++) {
          double dx = x(ix)-x0(fi);
          f(ix,2*fi)   = (1.0 - 2.0*dx*lipxi(fi)) * std::pow(lip(ix,fi),2);
          f(ix,2*fi+1) = dx * std::pow(lip(ix,fi),2) / element_length;
        }
      }
    }

    void HIPBasis::eval_prim_df(const arma::vec & x, arma::mat & df, double element_length) const {
      // Evaluate LIP basis
      arma::mat lip, dlip;
      LIPBasis::eval_f_raw(x, lip);
      LIPBasis::eval_df_raw(x, dlip);

      df.zeros(x.n_elem, 2*x0.n_elem);
      for(size_t ix=0;ix<x.n_elem;ix++) {
        for(size_t fi=0;fi<x0.n_elem;fi++) {
          df(ix,2*fi)   = 2.0*dlip(ix,fi)*lip(ix,fi)*(1.0 - 2.0*(x(ix)-x0(fi))*lipxi(fi)) - 2.0*lipxi(fi)*std::pow(lip(ix,fi),2);
          df(ix,2*fi+1) = (std::pow(lip(ix,fi),2) + 2.0*(x(ix)-x0(fi))*lip(ix,fi)*dlip(ix,fi)) / element_length;
        }
      }
    }

    void HIPBasis::eval_prim_d2f(const arma::vec & x, arma::mat & d2f, double element_length) const {
      // Evaluate LIP basis
      arma::mat lip, dlip, d2lip, d3lip;
      LIPBasis::eval_f_raw(x, lip);
      LIPBasis::eval_df_raw(x, dlip);
      LIPBasis::eval_d2f_raw(x, d2lip);
      LIPBasis::eval_d3f_raw(x, d3lip);

      d2f.zeros(x.n_elem, 2*x0.n_elem);
      for(size_t ix=0;ix<x.n_elem;ix++) {
        for(size_t fi=0;fi<x0.n_elem;fi++) {
          d2f(ix,2*fi)   = 2.0*(d2lip(ix,fi)*lip(ix,fi) + std::pow(dlip(ix,fi),2))*(1.0 - 2.0*(x(ix)-x0(fi))*lipxi(fi)) - 8.0*lip(ix,fi)*dlip(ix,fi)*lipxi(fi);
          d2f(ix,2*fi+1) = (4.0*lip(ix,fi)*dlip(ix,fi) + 2.0*(x(ix)-x0(fi))*(d2lip(ix,fi)*lip(ix,fi) + std::pow(dlip(ix,fi),2))) / element_length;
        }
      }
    }

    void HIPBasis::drop_first(bool deriv) {
      if(deriv) {
        enabled=enabled.subvec(1,enabled.n_elem-1);
      } else {
        enabled=enabled.subvec(2,enabled.n_elem-1);
      }
    }

    void HIPBasis::drop_last(bool deriv) {
      if(deriv) {
        enabled=enabled.subvec(0,enabled.n_elem-3);
      } else {
        arma::uvec new_enabled(enabled.n_elem-2);
        new_enabled.subvec(0,enabled.n_elem-3) = enabled.subvec(0,enabled.n_elem-3);
        new_enabled(enabled.n_elem-2) = enabled(enabled.n_elem-1);
        enabled = new_enabled;
      }
    }
  }
}
