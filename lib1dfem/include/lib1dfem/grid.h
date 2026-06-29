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
#ifndef LIB1DFEM_GRID_H
#define LIB1DFEM_GRID_H

#include <lib1dfem/types.h>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace helfem {
namespace lib1dfem {
namespace grid {

/// Form a 1D grid of element boundaries on [0, rmax] with `num_el`
/// elements. The shape is selected by `igrid`:
///
///   1: linear
///   2: quadratic (Schweizer et al. 1999)
///   3: generalised polynomial   r_i = rmax * (i/num_el)^zexp
///   4: generalised exponential  built from exp(linspace^zexp) - 1
///   5: geometric (Cancès & Mourad 2018; 0 < zexp < 1)
///
/// Endpoints are snapped to exactly 0 and rmax to avoid floating-point
/// roundoff. Templated on the scalar type T.
template <typename T>
Vec<T> get_grid(T rmax, int num_el, int igrid, T zexp,
                bool verbose = false) {
  Vec<T> bval;

  switch (igrid) {
    case 1: { // linear
      if (verbose) std::printf("Using linear grid\n");
      bval = Vec<T>::LinSpaced(num_el + 1, T(0), rmax);
      break;
    }
    case 2: { // quadratic
      if (verbose) std::printf("Using quadratic grid\n");
      bval = Vec<T>::Zero(num_el + 1);
      const T inv_n2 = T(1) / (T(num_el) * T(num_el));
      for (int i = 0; i <= num_el; ++i)
        bval(i) = T(i) * T(i) * rmax * inv_n2;
      break;
    }
    case 3: { // generalised polynomial
      if (verbose)
        std::printf("Using generalized polynomial grid, zexp = %e\n",
                    static_cast<double>(zexp));
      bval = Vec<T>::Zero(num_el + 1);
      const T inv_n = T(1) / T(num_el);
      for (int i = 0; i <= num_el; ++i)
        bval(i) = rmax * std::pow(T(i) * inv_n, zexp);
      break;
    }
    case 4: { // generalised exponential
      if (verbose)
        std::printf("Using generalized exponential grid, zexp = %e\n",
                    static_cast<double>(zexp));
      const T upper = std::pow(std::log(rmax + T(1)), T(1) / zexp);
      Vec<T> t = Vec<T>::LinSpaced(num_el + 1, T(0), upper);
      bval = Vec<T>(num_el + 1);
      for (int i = 0; i <= num_el; ++i)
        bval(i) = std::exp(std::pow(t(i), zexp)) - T(1);
      break;
    }
    case 5: { // geometric
      if (verbose)
        std::printf("Using geometric grid of doi:10.2140/camcos.2018.13.139, "
                    "s = %e\n", static_cast<double>(zexp));
      if (zexp <= T(0) || zexp >= T(1))
        throw std::logic_error("Invalid value for s parameter!\n");
      // h_k from p.158 of the Cancès-Mourad paper:
      Vec<T> hk(num_el);
      hk(num_el - 1) = (T(1) - zexp) / (T(1) - std::pow(zexp, num_el)) * rmax;
      for (int iel = num_el - 2; iel >= 0; --iel)
        hk(iel) = zexp * hk(iel + 1);
      bval = Vec<T>::Zero(num_el + 1);
      for (int iel = 0; iel < num_el; ++iel)
        bval(iel + 1) = bval(iel) + hk(iel);
      break;
    }
    default:
      throw std::logic_error("Invalid choice for grid\n");
  }

  // Make endpoints numerically exact.
  bval(0) = T(0);
  bval(bval.size() - 1) = rmax;
  return bval;
}

} // namespace grid
} // namespace lib1dfem
} // namespace helfem

#endif
