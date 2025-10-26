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

#include "PolynomialBasis.h"
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

      case(100):
        {
          arma::vec ang(Nnodes);
          for(int i=0;i<Nnodes;i++)
            ang(i) = M_PI*(Nnodes-1-i)/(Nnodes-1);
          arma::vec x=arma::cos(ang);
          poly=new polynomial_basis::LIPBasis(x,4);
          printf("Basis set composed of %i-node LIPs with Chebyshev nodes.\n",Nnodes);
          break;
        }

      case(101):
        {
          arma::vec ang(Nnodes);
          for(int i=0;i<Nnodes;i++)
            ang(i) = M_PI*(Nnodes-1-i)/(Nnodes-1);
          arma::vec x=arma::cos(ang);
          poly=new polynomial_basis::HIPBasis(x,5);
          printf("Basis set composed of %i-node HIPs with Chebyshev nodes.\n",Nnodes);
          break;
        }

      case(6):
      case(7):
      case(8):
      case(9):
      case(10):
      case(11):
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

    arma::vec PolynomialBasis::get_nodes() const {
      arma::vec n(2);
      n(0)=-1.0;
      n(1)=1.0;
      return n;
    }

    arma::uvec PolynomialBasis::get_enabled() const {
      return enabled;
    }

    void PolynomialBasis::print(const std::string & str) const {
      arma::vec x(arma::linspace<arma::vec>(-1.0,1.0,1001));
      arma::mat bf, df;
      eval_dnf(x,bf,0,1.0);
      eval_dnf(x,df,1,1.0);

      bf.insert_cols(0,x);
      df.insert_cols(0,x);

      std::string fname("bf" + str + ".dat");
      std::string dname("df" + str + ".dat");
      bf.save(fname,arma::raw_ascii);
      df.save(dname,arma::raw_ascii);
    }

    arma::mat PolynomialBasis::eval_dnf(const arma::vec & x, int n, double element_length) const {
      arma::mat dnf;
      eval_dnf(x, dnf, n, element_length);
      return dnf;
    }

    void PolynomialBasis::eval_prim_dnf(const arma::vec & x, arma::mat & dnf, int n, double element_length) const {
      (void) x;
      (void) dnf;
      (void) n;
      (void) element_length;
      throw std::logic_error("Values haven't been implemented for the used family of basis polynomials.\n");
    }

    void PolynomialBasis::eval_dnf(const arma::vec & x, arma::mat & dnf, int n, double element_length) const {
      eval_prim_dnf(x, dnf, n, element_length);
      // Apply scaling
      dnf = dnf.cols(enabled) / std::pow(element_length, n);
    }
  }
}
