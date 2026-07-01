/*
 *                This source code is part of
 *
 *                          HelFEM
 *                             -
 * Finite element methods for electronic structure calculations on small systems
 *
 * Written by Susi Lehtola, 2015-
 * Copyright (c) 2015- Susi Lehtola
 *
 * Originally part of the ERKALE package.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * See the LICENSE file at the root of this source distribution
 * for the full license text.
 */

#ifndef ERKALE_LBFGS
#define ERKALE_LBFGS

// Phase 5.17: LBFGS migrated to Eigen.
#include <Matrix.h>
#include <vector>

class LBFGS {
 protected:
  /// Maximum number of matrices
  size_t nmax;

  /// Coordinates x_k
  std::vector<helfem::Vector> xk;
  /// Gradients g_k
  std::vector<helfem::Vector> gk;

  /// Apply diagonal Hessian: r = H_0 q
  virtual helfem::Vector apply_diagonal_hessian(const helfem::Vector & q) const;

 public:
  /// Constructor
  LBFGS(size_t nmax=10);
  /// Destructor
  virtual ~LBFGS();

  /// Update
  void update(const helfem::Vector & x, const helfem::Vector & g);
  /// Solve for new search direction
  helfem::Vector solve() const;
  /// Clear stack
  void clear();
};

#endif
