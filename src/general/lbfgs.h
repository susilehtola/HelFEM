/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Copyright © 2015 The Regents of the University of California
 * All Rights Reserved
 *
 * Written by Susi Lehtola, Lawrence Berkeley National Laboratory
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#ifndef ERKALE_LBFGS
#define ERKALE_LBFGS

#include <armadillo>

class LBFGS {
 protected:
  /// Maximum number of matrices
  size_t nmax;

  /// Coordinates x_k
  std::vector<arma::vec> xk;
  /// Gradients g_k
  std::vector<arma::vec> gk;

  /// Apply diagonal Hessian: r = H_0 q
  virtual arma::vec apply_diagonal_hessian(const arma::vec & q) const;

 public:
  /// Constructor
  LBFGS(size_t nmax=10);
  /// Destructor
  virtual ~LBFGS();

  /// Update
  void update(const arma::vec & x, const arma::vec & g);
  /// Solve for new search direction
  arma::vec solve() const;
  /// Clear stack
  void clear();
};

#endif
