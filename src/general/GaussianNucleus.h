#ifndef MODELPOTENTIAL_GAUSSIANNUCLEUS_H
#define MODELPOTENTIAL_GAUSSIANNUCLEUS_H

#include <helfem/ModelPotential.h>

namespace helfem {
  namespace modelpotential {
    /// Gaussian nucleus
    class GaussianNucleus : public ModelPotential {
      /// Charge
      int Z;
      /// Size
      double mu;

      /// Cutoff for Taylor series
      double Rcut;
    public:
      /// Constructor
      GaussianNucleus(int Z, double Rrms);
      /// Destructor
      ~GaussianNucleus();
      /// Potential
      double V(double r) const override;
      /// Get mu
      double get_mu() const;
      /// Set mu
      void set_mu(double mu);
    };
  }
}

#endif
