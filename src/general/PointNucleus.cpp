#include "PointNucleus.h"

namespace helfem {
  namespace modelpotential {
    PointNucleus::PointNucleus(int Z_) : Z(Z_) {
    }

    PointNucleus::~PointNucleus() {
    }

    double PointNucleus::V(double R) const {
      return -Z/R;
    }
  }
}
