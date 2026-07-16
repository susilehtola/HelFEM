#ifndef NUCLEAR_MODEL_H
#define NUCLEAR_MODEL_H

#include <ModelPotential.h>
#include <PointNucleus.h>
#include <HollowNucleus.h>
#include <SphericalNucleus.h>
#include <GaussianNucleus.h>
#include <RegularizedNucleus.h>
#include "RadialPotential.h"

namespace helfem {
  namespace modelpotential {
    /// Nuclear model
    typedef enum {
          POINT_NUCLEUS,
          GAUSSIAN_NUCLEUS,
          SPHERICAL_NUCLEUS,
          HOLLOW_NUCLEUS,
          REGULARIZED_NUCLEUS,
          NOSUCH_NUCLEUS
    } nuclear_model_t;
    /// Get nuclear model.
    ///
    /// Templated on the scalar type: every nucleus model below it
    /// (PointNucleusT, GaussianNucleusT, ...) already follows T, so the
    /// factory must too, otherwise a FEMRadialBasisT<long double> would
    /// evaluate a double-precision potential inside its quadrature. T is
    /// deduced from Rrms, so every existing double caller is unchanged
    /// (and gets back a ModelPotentialT<double>* == ModelPotential*).
    /// Instantiated for double, long double and (under
    /// HELFEM_HAVE_FLOAT128) _Float128.
    template <typename T>
    ModelPotentialT<T> * get_nuclear_model(nuclear_model_t model, int Z, T Rrms);

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
