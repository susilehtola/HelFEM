#include "../general/lobatto.h"
#include "polynomial_basis.h"
#include "polynomial.h"

// Legendre polynomials
extern "C" {
#include <gsl/gsl_sf_legendre.h>
}

namespace helfem {
  namespace polynomial_basis {
    PolynomialBasis * get_basis(int primbas, int Nnodes) {
      // Primitive basis
      polynomial_basis::PolynomialBasis * poly;
      switch(primbas) {
      case(0):
      case(1):
      case(2):
        poly=new polynomial_basis::HermiteBasis(Nnodes,primbas);
      printf("Basis set composed of %i nodes with %i:th derivative continuity.\n",Nnodes,primbas);
      printf("This means using primitive polynomials of order %i.\n",Nnodes*(primbas+1)-1);
      break;

      case(3):
        poly=new polynomial_basis::LegendreBasis(Nnodes-1);
        printf("Basis set composed of %i-node spectral elements.\n",Nnodes);
        break;

      case(4):
        poly=new polynomial_basis::LobattoBasis(Nnodes);
        printf("Basis set composed of %i-node LIPs with Lobatto nodes.\n",Nnodes);
        break;

      default:
        throw std::logic_error("Unsupported primitive basis.\n");
      }

      return poly;
    }

    PolynomialBasis::PolynomialBasis() {
    }

    PolynomialBasis::~PolynomialBasis() {
    }

    int PolynomialBasis::get_nbf() const {
      return nbf;
    }

    int PolynomialBasis::get_noverlap() const {
      return noverlap;
    }

    HermiteBasis::HermiteBasis(int n_nodes, int der_order) {
      bf_C=polynomial::hermite_coeffs(n_nodes, der_order);
      df_C=polynomial::derivative_coeffs(bf_C, 1);

      // Number of basis functions is
      nbf=bf_C.n_cols;
      // Number of overlapping functions is
      noverlap=der_order+1;
    }

    HermiteBasis::~HermiteBasis() {
    }

    HermiteBasis * HermiteBasis::copy() const {
      return new HermiteBasis(*this);
    }

    arma::mat HermiteBasis::eval(const arma::vec & x) const {
      return polynomial::polyval(bf_C,x);
    }

    void HermiteBasis::eval(const arma::vec & x, arma::mat & f, arma::mat & df) const {
      f=polynomial::polyval(bf_C,x);
      df=polynomial::polyval(df_C,x);
    }

    void HermiteBasis::drop_first() {
      bf_C=bf_C.cols(noverlap,bf_C.n_cols-1);
      df_C=df_C.cols(noverlap,df_C.n_cols-1);
      nbf=bf_C.n_cols;
    }

    void HermiteBasis::drop_last() {
      bf_C=bf_C.cols(0,bf_C.n_cols-1-noverlap);
      df_C=df_C.cols(0,df_C.n_cols-1-noverlap);
      nbf=bf_C.n_cols;
    }

    LegendreBasis::LegendreBasis(int lmax_) {
      if(lmax_<1)
        throw std::logic_error("Legendre basis requires l>=1.\n");
      lmax=lmax_;

      // Transformation matrix
      T.zeros(lmax+1,lmax+1);

      // First function is (P0-P1)/2
      T(0,0)=0.5;
      T(1,0)=-0.5;
      // Last function is (P0+P1)/2
      T(0,lmax)=0.5;
      T(1,lmax)=0.5;

      // Shape functions [Flores, Clementi, Sonnad, Chem. Phys. Lett. 163, 198 (1989)]
      for(int j=1;j<lmax;j++) {
        double sqfac(1.0/sqrt(4.0*j+2.0));
        T(j+1,j)=sqfac;
        T(j-1,j)=-sqfac;
      }

      noverlap=1;
      nbf=T.n_cols;
    }

    LegendreBasis::~LegendreBasis() {
    }

    LegendreBasis * LegendreBasis::copy() const {
      return new LegendreBasis(*this);
    }

    static double sanitize_x(double x) {
        if(x<-1.0) x=-1.0;
        if(x>1.0) x=1.0;
        return x;
    }

    arma::mat LegendreBasis::eval(const arma::vec & x) const {
      // Memory for values
      arma::mat ft(lmax+1,x.n_elem);
      // Fill in array
      for(size_t i=0;i<x.n_elem;i++) {
        gsl_sf_legendre_Pl_array(lmax,sanitize_x(x(i)),ft.colptr(i));
      }
      return arma::trans(ft)*T;
    }

    void LegendreBasis::eval(const arma::vec & x, arma::mat & f, arma::mat & df) const {
      // Memory for values
      f.zeros(lmax+1,x.n_elem);
      df.zeros(lmax+1,x.n_elem);
      // Fill in array
      for(size_t i=0;i<x.n_elem;i++)
        gsl_sf_legendre_Pl_deriv_array(lmax,sanitize_x(x(i)),f.colptr(i),df.colptr(i));

      f=arma::trans(f)*T;
      df=arma::trans(df)*T;
    }

    void LegendreBasis::drop_first() {
      T=T.cols(1,T.n_cols-1);
      nbf=T.n_cols;
    }

    void LegendreBasis::drop_last() {
      T=T.cols(0,T.n_cols-2);
      nbf=T.n_cols;
    }

    LobattoBasis::LobattoBasis(int nnodes) {
      arma::vec w;
      ::lobatto_compute(nnodes,x0,w);

      // Make sure nodes are in order
      x0=arma::sort(x0,"ascend");
      // and that the number matches
      if(x0.n_elem != (arma::uword) nnodes)
        throw std::logic_error("Wrong number of functions!\n");

      noverlap=1;
      nbf=x0.n_elem;
      // All functions are enabled
      enabled=arma::linspace<arma::uvec>(0,nnodes-1,nnodes);
    }

    LobattoBasis::~LobattoBasis() {
    }

    LobattoBasis * LobattoBasis::copy() const {
      return new LobattoBasis(*this);
    }

    arma::mat LobattoBasis::eval(const arma::vec & x) const {
      // Memory for values
      arma::mat bf(x.n_elem,x0.n_elem);

      // Fill in array
      for(size_t ix=0;ix<x.n_elem;ix++) {
        for(size_t fi=0;fi<x0.n_elem;fi++) {
          // Evaluate
          double fval=1.0;
          for(size_t fj=0;fj<x0.n_elem;fj++) {
            // Term not included
            if(fi==fj)
              continue;
            // Compute ratio
            fval *= (x(ix)-x0(fj))/(x0(fi)-x0(fj));
          }
          bf(ix,fi)=fval;
        }
      }
      bf=bf.cols(enabled);

      return bf;
    }

    void LobattoBasis::eval(const arma::vec & x, arma::mat & f, arma::mat & df) const {
      // Function values
      f=eval(x);

      // Derivative
      df.zeros(x.n_elem,x0.n_elem);
      for(size_t ix=0;ix<x.n_elem;ix++) {
        for(size_t fi=0;fi<x0.n_elem;fi++) {
          // Evaluate
          for(size_t fj=0;fj<x0.n_elem;fj++) {
            if(fi==fj)
              continue;

            double fval=1.0;
            for(size_t fk=0;fk<x0.n_elem;fk++) {
              // Term not included
              if(fi==fk)
                continue;
              if(fj==fk)
                continue;
              // Compute ratio
              fval *= (x(ix)-x0(fk))/(x0(fi)-x0(fk));
            }
            // Increment derivative
            df(ix,fi)+=fval/(x0(fi)-x0(fj));
          }
        }
      }
      df=df.cols(enabled);
    }

    void LobattoBasis::drop_first() {
      enabled=enabled.subvec(1,enabled.n_elem-1);
      nbf=enabled.n_elem;
    }

    void LobattoBasis::drop_last() {
      enabled=enabled.subvec(0,enabled.n_elem-2);
      nbf=enabled.n_elem;
    }
  }
}
