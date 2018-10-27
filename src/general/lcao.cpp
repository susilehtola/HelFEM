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
#include "lcao.h"
#include <cmath>

// For factorials
extern "C" {
#include <gsl/gsl_sf_gamma.h>
}

namespace helfem {
  namespace lcao {
    static double double_factorial(unsigned int n) {
      return gsl_sf_doublefact(n);
    }
    static double factorial(unsigned int n) {
      return gsl_sf_fact(n);
    }
    
    /// Evaluate radial GTO
    double radial_GTO(double r, int l, double alpha) {
      return std::pow(2,l+2) * std::pow(alpha,(2*l+3)/4.0) * std::pow(r,l) * exp(-alpha*r*r) / sqrt( std::pow(2.0*M_PI,0.25) * double_factorial(2*l+1));
    }
    /// Evaluate radial STO
    double radial_STO(double r, int l, double zeta) {
      return std::pow(2*zeta,l+1.5)/sqrt(factorial(2*l+2)) * std::pow(r,l) * exp(-zeta*r);
    }
  }
}
