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

#include "helfem/PolynomialBasis.h"
#include "lobatto.h"
#include "LIPBasis.h"
#include "HIPBasis.h"
#include "GeneralHIPBasis.h"
#include "LegendreBasis.h"

namespace helfem {
  namespace polynomial_basis {
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

      case(6):
      case(7):
      case(8):
      case(9):
        {
          arma::vec x, w;
          ::lobatto_compute(Nnodes,x,w);
          int nder=primbas-6;
          poly=new polynomial_basis::GeneralHIPBasis(x,primbas,nder);
          printf("Basis set composed of %i-node %i:th order HIPs with Gauss-Lobatto nodes.\n",Nnodes,nder);
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

    int PolynomialBasis::get_nnodes() const {
      return nnodes;
    }

    arma::uvec PolynomialBasis::get_enabled() const {
      return enabled;
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
  }
}
