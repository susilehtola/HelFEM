#include "model_potential.h"
#include "sap.h"
#include "gsz.h"
#include <cfloat>

namespace helfem {
  namespace modelpotential {
    ModelPotential * get_nuclear_model(nuclear_model_t model, int Z, double Rrms) {
      switch(model) {
      case(POINT_NUCLEUS):
        printf("Getting point nucleus with Z=%i\n",Z);
        return new PointNucleus(Z);
      case(GAUSSIAN_NUCLEUS):
        printf("Getting Gaussian nucleus with Z=%i Rrms=%e\n",Z,Rrms);
        return new GaussianNucleus(Z,Rrms);
      case(HOLLOW_NUCLEUS):
        printf("Getting hollow spherical nucleus with Z=%i Rrms=%e\n",Z,Rrms);
        return new HollowNucleus(Z,Rrms);
      case(SPHERICAL_NUCLEUS):
        printf("Getting uniformly charged spherical nucleus with Z=%i Rrms=%e\n",Z,Rrms);
        return new SphericalNucleus(Z,Rrms);
      case(NOSUCH_NUCLEUS):
        throw std::logic_error("No such nucleus!\n");
      }

      throw std::logic_error("Unrecognized model\n");
    }

    PointNucleus::PointNucleus(int Z_) : Z(Z_) {
    }

    PointNucleus::~PointNucleus() {
    }

    double PointNucleus::V(double R) const {
      return -Z/R;
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

    double SphericalNucleus::get_R0() const {
      return R0;
    }

    void SphericalNucleus::set_R0(double R0_) {
      R0=R0_;
    }

    HollowNucleus::HollowNucleus(int Z_, double R_) : Z(Z_), R(R_) {
    }

    HollowNucleus::~HollowNucleus() {
    }

    double HollowNucleus::V(double r) const {
      if(r>=R) {
        return -Z/r;
      } else {
        return -Z/R;
      }
    }

    double HollowNucleus::get_R() const {
      return R;
    }

    void HollowNucleus::set_R(double R_) {
      R=R_;
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
