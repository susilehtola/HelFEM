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

// v2 refactor (Phase 1): the abstract PolynomialBasis base class definitions
// have moved into lib1dfem/include/lib1dfem/PolynomialBasis.h (templated,
// header-only). This translation unit now only holds the factory that
// constructs the concrete libhelfem-side polynomial bases by primitive ID.

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

      return poly;
    }
  }
}
