#include "RadialPotential.h"

namespace helfem {
  namespace modelpotential {
    RadialPotential::RadialPotential(int n_) : n(n_) {
    }

    RadialPotential::~RadialPotential() {
    }

    double RadialPotential::V(double R) const {
      return std::pow(R,n);
    }
  }
}
