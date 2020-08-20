#include "ModelPotential.h"

namespace helfem {
  namespace modelpotential {
    ModelPotential::ModelPotential() {
    }

    ModelPotential::~ModelPotential() {
    }

    arma::vec ModelPotential::V(const arma::vec & r) const {
      arma::vec pot(r.n_elem);
      for(size_t i=0;i<r.n_elem;i++)
        pot(i)=V(r(i));
      return pot;
    }
  }
}
