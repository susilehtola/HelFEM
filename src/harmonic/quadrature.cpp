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
#include "quadrature.h"
#include "chebyshev.h"

namespace helfem {
  namespace quadrature {
    arma::mat radial_integral(const helfem::polynomial_basis::FiniteElementBasis & fem, size_t iel, int n, const arma::vec & xq, const arma::vec & wxq) {
      // x values
      arma::vec x(fem.eval_coord(xq,iel));
      // Calculate total weight per point
      arma::vec wp(wxq*fem.scaling_factor(iel));
      if(n!=0) {
	if(n==2)
	  wp%=arma::square(x);
	else
	  throw std::logic_error("Case not implemented.\n");
      }

      // Basis functions
      arma::mat bf(fem.eval_f(xq, iel));

      // Put in weight
      arma::mat wbf(bf);
      for(size_t i=0;i<bf.n_cols;i++)
	wbf.col(i)%=wp;

      // Matrix elements are then
      return arma::trans(wbf)*bf;
    }

    arma::mat derivative_integral(const helfem::polynomial_basis::FiniteElementBasis & fem, size_t iel, const arma::vec & xq, const arma::vec & wxq) {
      // Calculate total weight per point
      arma::vec wp(wxq*fem.scaling_factor(iel));
      // Derivatives
      arma::mat dbf(fem.eval_df(xq, iel));
      // Put in weight
      arma::mat wdbf(dbf);
      for(size_t i=0;i<dbf.n_cols;i++)
	wdbf.col(i)%=wp;

      // Integral is
      return arma::trans(wdbf)*dbf;
    }
  }
}
