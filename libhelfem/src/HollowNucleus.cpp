#include "helfem/HollowNucleus.h"

namespace helfem {
  namespace modelpotential {
    HollowNucleus::HollowNucleus(int Z_, double R_) : Z(Z_), R(R_) {
    }

    HollowNucleus::~HollowNucleus() {
    }

    double HollowNucleus::V(double r) const {
      if(r>=R) {
        return -Z/r;
      } else {
        return -Z/R;
      }
    }

    double HollowNucleus::get_R() const {
      return R;
    }

    void HollowNucleus::set_R(double R_) {
      R=R_;
    }
  }
}
