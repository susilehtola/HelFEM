#ifndef NUCLEAR_MODEL_H
#define NUCLEAR_MODEL_H

#include <ModelPotential.h>
#include <PointNucleus.h>
#include <HollowNucleus.h>
#include <SphericalNucleus.h>
#include <GaussianNucleus.h>
#include <RegularizedNucleus.h>
#include "RadialPotential.h"
#include "atomdb.h"

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
    ModelPotentialT<T> * nuclear_model(nuclear_model_t model, int Z, T Rrms);

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

    /// Superposition of atomic potentials, from the tabulated effective
    /// charge of sap.cpp. Zeff is interpolated between tabulation knots,
    /// so the potential is only piecewise smooth on the tabulation grid.
    /// Superposition of atomic potentials, from the tabulated table in
    /// sap.cpp. The table is interpolated LINEARLY between its knots, so
    /// the potential is only C0: it has a kink at every knot, and
    /// breakpoints() reports the ones inside the element so the
    /// quadrature can split there. Without that the refinement converges
    /// only algebraically (measured O(n^-2), stalling at ~2e-8) and
    /// grinds to the order cap on every run that uses this guess.
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
      /// Knots of the interpolation table inside (a, b).
      std::vector<double> breakpoints(double a, double b) const override;
    };

    /// Superposition of atomic potentials, evaluated on the fly from the
    /// tabulated atomic wave function (see src/general/atomdb.h) instead
    /// of from an interpolated table of the effective charge.
    ///
    /// The potential this builds is the same object SAPAtom approximates,
    /// but it is exact at every r: the density comes from the stored
    /// orbitals, the Hartree screening from an exact partial-element
    /// integration of that density, and the exchange screening from
    /// libxc. Its only kinks are the element boundaries of the wave
    /// function's own grid, which breakpoints() reports so the quadrature
    /// can split there.
    class SAPFEAtom : public ModelPotential {
      /// The tabulated atom. Empty for Z=0: the wave function database
      /// covers 1..118 and rightly refuses Z=0, but a zero-charge dummy
      /// centre is a legitimate thing to place in a diatomic calculation.
      /// It has no electrons, hence no density and no screening, so the
      /// potential is identically zero and no record is needed.
      std::unique_ptr<atomdb::Atom> atom;
      /// Exchange and correlation functional ids used for the screening
      int x_func, c_func;
    public:
      /// Constructor, with the LDA exchange-only screening the SAP
      /// potential is defined with.
      SAPFEAtom(int Z);
      /// Constructor with an explicit libxc screening functional. LDAs
      /// only: the screening is evaluated from the density alone.
      SAPFEAtom(int Z, int x_func, int c_func);
      /// Destructor
      ~SAPFEAtom();
      /// Potential
      double V(double r) const override;
      /// The element boundaries of the wave function's radial grid
      std::vector<double> breakpoints(double a, double b) const override;
    };
  }
}

#endif
