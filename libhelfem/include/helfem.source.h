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
#ifndef __HELFEM__
#define __HELFEM__

#include <Matrix.h>
#include <string>

#define __HELFEM_VERSION__ "${LIBHELFEM_VERSION}"

namespace helfem {
  /**
   * Global boolean indicating whether the library is in verbose or non-verbose
   * mode. * By default, the library is in the non-verbose mode.
   *
   * This variable is not part of the public API. The verbosity should be
   * enabled or disabled using helfem::set_verbosity.
   */
  extern bool verbose;

  /**
   * Set the global verbosity flag for the libhelfem library.
   *
   * @param verbose whether to enable or disable verbosity.
   */
  void set_verbosity(bool verbosity);

  std::string version();

  // Utilities
  namespace utils {
    /**
     * Form radial grid for a calculation, ranging from r=0 to r=rmax.
     *
     * igrid: 0 for linear grid
     *        1 for quadratic grid
     *        2 for generalized polynomial grid with exponent zexp
     *        3 for generalized exponential grid with parameter zexp
     */
    ::helfem::Vector get_grid(double rmax, int num_el, int igrid, double zexp);

    /**
     * Calculates the half-inverse of a matrix.
     * Phase 5.10: Eigen-typed.
     */
    helfem::Matrix invh(helfem::Matrix S, bool chol);
  } // namespace utils
} // namespace helfem

#include "GaussianNucleus.h"
#include "HollowNucleus.h"
#include "ModelPotential.h"
#include "PointNucleus.h"
#include "PolynomialBasis.h"
#include "RadialBasis.h"
#include "SphericalNucleus.h"

namespace helfem {
  namespace polynomial_basis {
    /// Get the wanted basis
    PolynomialBasis *get_basis(int primbas, int Nnodes);
  } // namespace polynomial_basis
} // namespace helfem

#endif
