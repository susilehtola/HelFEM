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
#ifndef POLYNOMIAL_BASIS_LEGENDREBASIS_H
#define POLYNOMIAL_BASIS_LEGENDREBASIS_H

#include "PolynomialBasis.h"
#include <armadillo>

namespace helfem {
  namespace polynomial_basis {/// Legendre functions
    class LegendreBasis: public PolynomialBasis {
    protected:
      /// Maximum order
      int lmax;
      /// Transformation matrix
      arma::mat T;

      /// Evaluate Legendre polynomials
      arma::mat f_eval(const arma::vec & x) const;
      /// Evaluate Legendre polynomials' derivatives
      arma::mat df_eval(const arma::vec & x) const;
      /// Evaluate Legendre polynomials' second derivatives
      arma::mat d2f_eval(const arma::vec & x) const;
    public:
      /// Constructor
      LegendreBasis(int nfuncs, int id=3);
      /// Destructor
      ~LegendreBasis();
      /// Get a copy
      LegendreBasis * copy() const override;

      /// Drop first function
      void drop_first(bool func, bool deriv) override;
      /// Drop last function
      void drop_last(bool func, bool deriv) override;

      /// Evaluate polynomials at given points
      void eval_prim_dnf(const arma::vec & x, arma::mat & f, int n, double element_length) const override;
    };
  }
}
#endif
