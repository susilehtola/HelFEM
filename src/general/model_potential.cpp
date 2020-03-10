#include "model_potential.h"
#include "sap.h"
#include "gsz.h"
#include <cfloat>

namespace helfem {
  namespace modelpotential {
    ModelPotential::ModelPotential() {
    }

    ModelPotential::~ModelPotential() {
    }

    arma::vec ModelPotential::V(const arma::vec & r) const {
      arma::vec pot(r.n_elem);
      for(size_t i=0;i<r.n_elem;i++)
        pot(i)=V(r(i));
      return pot;
    }

    RadialPotential::RadialPotential(int n_) : n(n_) {
    }

    RadialPotential::~RadialPotential() {
    }

    double RadialPotential::V(double R) const {
      return std::pow(R,n);
    }

    PointNucleus::PointNucleus(int Z_) : Z(Z_) {
    }

    PointNucleus::~PointNucleus() {
    }

    double PointNucleus::V(double R) const {
      return -Z/R;
    }

    GaussianNucleus::GaussianNucleus(int Z_, double Rrms) : Z(Z_) {
      // Eqn (11) in Visscher-Dyall 1997
      mu = sqrt(3.0/2.0)/Rrms;

      // Taylor series cutoff: sixth-order term is epsilon
      Rcut = std::pow(42.0*DBL_EPSILON, 1.0/6.0)/mu;
    }

    GaussianNucleus::~GaussianNucleus() {
    }

    double GaussianNucleus::V(double R) const {
      // Taylor series for erf(mu*r)/r from Maple
      if(R <= Rcut) {
        double mur2 = std::pow(mu*R,2);
        return -Z*M_2_SQRTPI*mu*( 1.0 + (-1.0/3.0 + (1.0/10.0 - 1.0/42.0*mur2)*mur2)*mur2);
          } else {
        return -Z*erf(mu*R)/R;
      }
    }

    SphericalNucleus::SphericalNucleus(int Z_, double Rrms) : Z(Z_) {
      // Eqn (4) in Visscher-Dyall 1997
      R0 = sqrt(5.0/3.0)*Rrms;
    }

    SphericalNucleus::~SphericalNucleus() {
    }

    double SphericalNucleus::V(double r) const {
      if(r>=R0) {
        // See full charge, eqn (7a) in Visscher-Dyall 1997
        return -Z/r;
      } else {
        // See only charge inside, eqn (7b) in Visscher-Dyall 1997
        return -Z/(2.0*R0)*(3.0-std::pow(r/R0,2));
      }
    }

    TFAtom::TFAtom(int Z_) : Z(Z_) {
    }

    TFAtom::~TFAtom() {
    }

    double TFAtom::V(double r) const {
      return -GSZ::Z_thomasfermi(r,Z)/r;
    }

    GSZAtom::GSZAtom(int Z_) : Z(Z_) {
      GSZ::GSZ_parameters(Z,dz,Hz);
    }

    GSZAtom::GSZAtom(int Z_, double dz_, double Hz_) : Z(Z_), dz(dz_), Hz(Hz_) {
    }

    GSZAtom::~GSZAtom() {
    }

    double GSZAtom::V(double r) const {
      return -GSZ::Z_GSZ(r,Z,dz,Hz)/r;
    }

    SAPAtom::SAPAtom(int Z_) : Z(Z_) {
    }

    SAPAtom::~SAPAtom() {
    }

    double SAPAtom::V(double r) const {
      return -::sap_effective_charge(Z,r)/r;
    }
  }
}
