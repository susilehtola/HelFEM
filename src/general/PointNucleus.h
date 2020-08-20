#ifndef MODELPOTENTIAL_POINTNUCLEUS_H
#define MODELPOTENTIAL_POINTNUCLEUS_H

#include "ModelPotential.h"

namespace helfem {
  namespace modelpotential {
    /// Point nucleus
    class PointNucleus : public ModelPotential {
      /// Charge
      int Z;
    public:
      /// Constructor
      PointNucleus(int Z);
      /// Destructor
      ~PointNucleus();
      /// Potential
      double V(double r) const override;
    };
  }
}

#endif
