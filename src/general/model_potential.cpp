#include "model_potential.h"
#include "sap.h"
#include "gsz.h"
#include <cfloat>

namespace helfem {
  namespace modelpotential {
    // printf has no conversion for a general T, so the diagnostics go through
    // double at the I/O boundary only; the potential itself is built in T.
    template <typename T>
    ModelPotentialT<T> * get_nuclear_model(nuclear_model_t model, int Z, T Rrms) {
      switch(model) {
      case(POINT_NUCLEUS):
        printf("Getting point nucleus with Z=%i\n",Z);
        return new PointNucleusT<T>(Z);
      case(GAUSSIAN_NUCLEUS):
        printf("Getting Gaussian nucleus with Z=%i Rrms=%e\n",Z,(double) Rrms);
        return new GaussianNucleusT<T>(Z,Rrms);
      case(HOLLOW_NUCLEUS):
        printf("Getting hollow spherical nucleus with Z=%i Rrms=%e\n",Z,(double) Rrms);
        return new HollowNucleusT<T>(Z,Rrms);
      case(SPHERICAL_NUCLEUS):
        printf("Getting uniformly charged spherical nucleus with Z=%i Rrms=%e\n",Z,(double) Rrms);
        return new SphericalNucleusT<T>(Z,Rrms);
      case(REGULARIZED_NUCLEUS):
        printf("Getting regularized nucleus with Z=%i a=%e\n",Z,(double) Rrms);
        return new RegularizedNucleusT<T>(Z,Rrms);
      case(NOSUCH_NUCLEUS):
        throw std::logic_error("No such nucleus!\n");
      }

      throw std::logic_error("Unrecognized model\n");
    }

    template ModelPotentialT<double> *
    get_nuclear_model<double>(nuclear_model_t, int, double);
    template ModelPotentialT<long double> *
    get_nuclear_model<long double>(nuclear_model_t, int, long double);
#ifdef HELFEM_HAVE_FLOAT128
    template ModelPotentialT<_Float128> *
    get_nuclear_model<_Float128>(nuclear_model_t, int, _Float128);
#endif

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
