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
#ifndef POLYNOMIAL_BASIS_LIPBASIS_H
#define POLYNOMIAL_BASIS_LIPBASIS_H

#include "PolynomialBasis.h"
#include <armadillo>

namespace helfem {
  namespace polynomial_basis {
    /// Lagrange interpolating polynomials
    class LIPBasis: public PolynomialBasis {
    protected:
      /// Control nodes
      arma::vec x0;
    public:
      /// Dummy constructor
      LIPBasis();
      /// Constructor
      LIPBasis(const arma::vec & x0, int id=4);
      /// Destructor
      ~LIPBasis();
      /// Get a copy
      LIPBasis * copy() const override;

      /// Drop first function
      void drop_first(bool func, bool deriv) override;
      /// Drop last function
      void drop_last(bool func, bool deriv) override;

      /// Evaluate polynomials at given points
      void eval_prim_dnf(const arma::vec & x, arma::mat & dnf, int n, double element_length) const override;

      /// Return nodes
      arma::vec get_nodes() const override;
    };
  }
}
#endif
