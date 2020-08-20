#ifndef MODELPOTENTIAL_MODELPOTENTIAL_H
#define MODELPOTENTIAL_MODELPOTENTIAL_H

#include <armadillo>

namespace helfem {
  namespace modelpotential {
    /// Model potential
    class ModelPotential {
    public:
      /// Constructor
      ModelPotential();
      /// Destructor
      virtual ~ModelPotential();

      /// Potential
      virtual double V(double r) const=0;
      /// Potential
      arma::vec V(const arma::vec & r) const;
    };
  }
}

#endif
