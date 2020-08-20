#ifndef MODELPOTENTIAL_HOLLOWNUCLEUS_H
#define MODELPOTENTIAL_HOLLOWNUCLEUS_H

#include "ModelPotential.h"

namespace helfem {
  namespace modelpotential {
    /// Thin hollow nucleus
    class HollowNucleus : public ModelPotential {
      /// Charge
      int Z;
      /// Size
      double R;
    public:
      /// Constructor
      HollowNucleus(int Z, double R);
      /// Destructor
      ~HollowNucleus();
      /// Potential
      double V(double r) const override;
      /// Get R
      double get_R() const;
      /// Set R
      void set_R(double R);
    };
  }
}

#endif
