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
#ifndef POLYNOMIAL_BASIS_HIPBASIS_H
#define POLYNOMIAL_BASIS_HIPBASIS_H

#include "PolynomialBasis.h"
#include "LIPBasis.h"
#include <armadillo>

namespace helfem {
  namespace polynomial_basis {
    /// Hermite interpolating polynomials
    class HIPBasis: public LIPBasis {
    protected:
      /// LIP derivatives at nodes
      arma::vec lipxi;
    public:
      /// Constructor
      HIPBasis(const arma::vec & x0, int id=5);
      /// Destructor
      ~HIPBasis();
      /// Get a copy
      HIPBasis * copy() const override;

      /// Drop first function
      void drop_first(bool func, bool deriv) override;
      /// Drop last function
      void drop_last(bool func, bool deriv) override;

      /// Evaluate the basis functions
      void eval_prim_dnf(const arma::vec & x, arma::mat & dnf, int n, double element_length) const override;
    };
  }
}
#endif
