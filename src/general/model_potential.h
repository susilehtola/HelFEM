#ifndef NUCLEAR_MODEL_H
#define NUCLEAR_MODEL_H

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
