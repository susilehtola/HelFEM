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
