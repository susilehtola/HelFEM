#ifndef MODELPOTENTIAL_SPHERICALNUCLEUS_H
#define MODELPOTENTIAL_SPHERICALNUCLEUS_H

#include "ModelPotential.h"

namespace helfem {
  namespace modelpotential {
    /// Uniformly charged spherical nucleus
    class SphericalNucleus : public ModelPotential {
      /// Charge
      int Z;
      /// Size
      double R0;
    public:
      /// Constructor
      SphericalNucleus(int Z, double Rrms);
      /// Destructor
      ~SphericalNucleus();
      /// Potential
      double V(double r) const override;
      /// Get R0
      double get_R0() const;
      /// Set R0
      void set_R0(double R0);
    };
  }
}

#endif
