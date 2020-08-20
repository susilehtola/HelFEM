#ifndef MODELPOTENTIAL_RADIALPOTENTIAL_H
#define MODELPOTENTIAL_RADIALPOTENTIAL_H

#include "ModelPotential.h"

namespace helfem {
  namespace modelpotential {
    /// Simple r^n radial potential
    class RadialPotential : public ModelPotential {
      /// Exponent
      int n;
    public:
      /// Constructor
      RadialPotential(int n);
      /// Destructor
      ~RadialPotential();
      /// Potential
      double V(double r) const override;
    };
  }
}

#endif
