#ifndef NUCLEAR_MODEL_H
#define NUCLEAR_MODEL_H

#include <armadillo>
#include "ModelPotential.h"
#include "RadialPotential.h"
#include "GaussianNucleus.h"

namespace helfem {
  namespace modelpotential {
    /// Nuclear model
    typedef enum {
          POINT_NUCLEUS,
          GAUSSIAN_NUCLEUS,
          SPHERICAL_NUCLEUS,
          HOLLOW_NUCLEUS,
          NOSUCH_NUCLEUS
    } nuclear_model_t;
    /// Get nuclear model
    ModelPotential * get_nuclear_model(nuclear_model_t model, int Z, double Rrms);

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

    /// Thomas-Fermi atom
    class TFAtom : public ModelPotential {
      /// Charge
      int Z;
    public:
      /// Constructor
      TFAtom(int Z);
      /// Constructor
      TFAtom(int Z, double dz, double Hz);
      /// Destructor
      ~TFAtom();
      /// Potential
      double V(double r) const override;
    };

    /// Green-Sellin-Zachor atom
    class GSZAtom : public ModelPotential {
      /// Charge
      int Z;
      /// GSZ parameters
      double dz, Hz;
    public:
      /// Constructor
      GSZAtom(int Z);
      /// Constructor
      GSZAtom(int Z, double dz, double Hz);
      /// Destructor
      ~GSZAtom();
      /// Potential
      double V(double r) const override;
    };

    /// Superposition of atomic potentials
    class SAPAtom : public ModelPotential {
      /// Charge
      int Z;
    public:
      /// Constructor
      SAPAtom(int Z);
      /// Destructor
      ~SAPAtom();
      /// Potential
      double V(double r) const override;
    };
  }
}

#endif
