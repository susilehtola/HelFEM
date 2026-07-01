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
#include "lcao.h"
#include <cmath>

namespace helfem {
  namespace lcao {
    static double double_factorial(unsigned int n) {
      double v = 1.0;
      for(unsigned int k=n; k>=2; k-=2)
        v *= static_cast<double>(k);
      return v;
    }
    static double factorial(unsigned int n) {
      double v = 1.0;
      for(unsigned int k=2; k<=n; ++k)
        v *= static_cast<double>(k);
      return v;
    }

    /// Evaluate radial GTO
    double radial_GTO(double r, int l, double alpha) {
      return std::pow(2,l+2) * std::pow(alpha,(2*l+3)/4.0) * std::pow(r,l) * exp(-alpha*r*r) / ( std::pow(2.0*M_PI,0.25) * sqrt(double_factorial(2*l+1)));
    }

    helfem::Matrix radial_GTO(const helfem::Vector & r, int l, const helfem::Vector & alpha) {
      helfem::Matrix gto(r.size(), alpha.size());
      for (Eigen::Index j = 0; j < alpha.size(); ++j)
        for (Eigen::Index i = 0; i < r.size(); ++i)
          gto(i, j) = radial_GTO(r(i), l, alpha(j));
      return gto;
    }

    /// Evaluate radial STO
    double radial_STO(double r, int l, double zeta) {
      return std::pow(2*zeta,l+1.5)/sqrt(factorial(2*l+2)) * std::pow(r,l) * exp(-zeta*r);
    }

    helfem::Matrix radial_STO(const helfem::Vector & r, int l, const helfem::Vector & alpha) {
      helfem::Matrix gto(r.size(), alpha.size());
      for (Eigen::Index j = 0; j < alpha.size(); ++j)
        for (Eigen::Index i = 0; i < r.size(); ++i)
          gto(i, j) = radial_STO(r(i), l, alpha(j));
      return gto;
    }
  }
}
