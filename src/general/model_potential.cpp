#include "model_potential.h"
#include "sap.h"
#include "gsz.h"
#include "eigen_io.h"
#include <xc_funcs.h>
#include <cfloat>

namespace helfem {
  namespace modelpotential {
    // printf has no conversion for a general T, so the T-typed radius is
    // formatted with helfem::io::fmt_sci (scientific, at T's own precision)
    // and spliced in as %s -- no truncation to double at the I/O boundary.
    template <typename T>
    ModelPotentialT<T> * get_nuclear_model(nuclear_model_t model, int Z, T Rrms) {
      switch(model) {
      case(POINT_NUCLEUS):
        printf("Getting point nucleus with Z=%i\n",Z);
        return new PointNucleusT<T>(Z);
      case(GAUSSIAN_NUCLEUS):
        printf("Getting Gaussian nucleus with Z=%i Rrms=%s\n",Z,helfem::io::fmt_sci(Rrms).c_str());
        return new GaussianNucleusT<T>(Z,Rrms);
      case(HOLLOW_NUCLEUS):
        printf("Getting hollow spherical nucleus with Z=%i Rrms=%s\n",Z,helfem::io::fmt_sci(Rrms).c_str());
        return new HollowNucleusT<T>(Z,Rrms);
      case(SPHERICAL_NUCLEUS):
        printf("Getting uniformly charged spherical nucleus with Z=%i Rrms=%s\n",Z,helfem::io::fmt_sci(Rrms).c_str());
        return new SphericalNucleusT<T>(Z,Rrms);
      case(REGULARIZED_NUCLEUS):
        printf("Getting regularized nucleus with Z=%i a=%s\n",Z,helfem::io::fmt_sci(Rrms).c_str());
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

    SAPFEAtom::SAPFEAtom(int Z_) : SAPFEAtom(Z_, XC_LDA_X, 0) {
    }

    SAPFEAtom::SAPFEAtom(int Z_, int x_func_, int c_func_) : x_func(x_func_), c_func(c_func_) {
      // A zero-charge centre carries no electrons, so there is nothing to
      // look up and nothing to screen. Constructing the record would throw:
      // the database covers 1..118.
      if(Z_ != 0)
        atom = std::make_unique<atomdb::Atom>(Z_);
    }

    SAPFEAtom::~SAPFEAtom() {
    }

    double SAPFEAtom::V(double r) const {
      if(!atom) return 0.0;   // dummy centre
      return -atom->effective_charge(r,x_func,c_func)/r;
    }

    std::vector<double> SAPFEAtom::breakpoints(double a, double b) const {
      // The density, and with it the potential, is a different polynomial
      // in each element of the wave function's grid. Only the boundaries
      // strictly inside (a, b) are reported: one coinciding with an
      // element end is already handled by the element decomposition.
      if(!atom) return std::vector<double>();   // dummy centre
      const helfem::Vector bval(atom->element_boundaries());
      std::vector<double> bp;
      for(Eigen::Index i=0;i<bval.size();i++)
        if(bval(i) > a && bval(i) < b)
          bp.push_back(bval(i));
      return bp;
    }
  }
}
