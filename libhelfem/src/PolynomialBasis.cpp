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
#include "LegendreBasis.h"
#include <lib1dfem/HIP2Basis.h>  // analytic HIP order 2 (primbas=8 since v2)
#include <lib1dfem/HIP3Basis.h>  // analytic HIP order 3 (primbas=9 since v2)
// (GeneralHIPBasis was removed when its callers were either rerouted to
// analytic HIP2/HIP3 (primbas=8, 9) or deprecated (primbas=6, 7, 10, 11).)

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

      case(8):
        {
          // Analytic HIP2 (closed-form Hermite, second-order). Replaces the
          // runtime-inverted GeneralHIPBasis(nder=2) path.
          arma::vec x, w;
          ::lobatto_compute(Nnodes, x, w);
          poly = new lib1dfem::polynomial_basis::HIP2Basis<double>(x, primbas);
          printf("Basis set composed of %i-node 2nd order analytic HIPs with Gauss-Lobatto nodes.\n", Nnodes);
          break;
        }

      case(9):
        {
          // Analytic HIP3 (closed-form Hermite, third-order). Replaces the
          // runtime-inverted GeneralHIPBasis(nder=3) path.
          arma::vec x, w;
          ::lobatto_compute(Nnodes, x, w);
          poly = new lib1dfem::polynomial_basis::HIP3Basis<double>(x, primbas);
          printf("Basis set composed of %i-node 3rd order analytic HIPs with Gauss-Lobatto nodes.\n", Nnodes);
          break;
        }

      case(6):
        throw std::runtime_error(
            "primbas=6 (generic HIP with nder=0) is deprecated and no longer "
            "supported; use primbas=4 (LIP) which provides the same "
            "function-interpolating basis.\n");

      case(7):
        throw std::runtime_error(
            "primbas=7 (generic HIP with nder=1) is deprecated and no longer "
            "supported; use primbas=5 (HIP) which provides the same first-"
            "order Hermite basis.\n");

      case(10):
      case(11):
        throw std::runtime_error(
            "primbas=10 / 11 (generic HIP with nder=4 / 5) is deprecated and "
            "no longer supported; only analytic HIP2 (primbas=8) and HIP3 "
            "(primbas=9) are exposed in v2.\n");

      default:
        throw std::logic_error("Unsupported primitive basis.\n");
      }

      return poly;
    }
  }
}
