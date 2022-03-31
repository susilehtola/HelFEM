#include "LegendreBasis.h"
#include "orthpoly.h"

// Legendre polynomials
extern "C" {
#include <gsl/gsl_sf_legendre.h>
}

namespace helfem {
  namespace polynomial_basis {
    LegendreBasis::LegendreBasis(int n_nodes, int id_) {
      lmax=n_nodes-1;

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
      nprim=T.n_cols;
      enabled=arma::linspace<arma::uvec>(0,T.n_cols-1,T.n_cols);

      /// Identifier is
      id=id_;
      /// Order is
      order=enabled.n_elem;
    }

    LegendreBasis::~LegendreBasis() {
    }

    LegendreBasis * LegendreBasis::copy() const {
      return new LegendreBasis(*this);
    }

    inline static double sanitize_x(double x) {
      if(x<-1.0) x=-1.0;
      if(x>1.0) x=1.0;
      return x;
    }

    arma::mat LegendreBasis::f_eval(const arma::vec & x) const {
      // Memory for values
      arma::mat ft(x.n_elem,lmax+1);
      // Fill in array
      for(int l=0;l<=lmax;l++)
        for(size_t i=0;i<x.n_elem;i++)
          ft(i,l) = oomph::Orthpoly::legendre(l, x(i));
      return ft;
    }

    arma::mat LegendreBasis::df_eval(const arma::vec & x) const {
      // Memory for values
      arma::mat dt(x.n_elem,lmax+1);
      // Fill in array
      for(int l=0;l<=lmax;l++)
        for(size_t i=0;i<x.n_elem;i++)
          dt(i,l) = oomph::Orthpoly::dlegendre(l, x(i));
      return dt;
    }

    arma::mat LegendreBasis::d2f_eval(const arma::vec & x) const {
      // Memory for values
      arma::mat lt(x.n_elem,lmax+1);
      // Fill in array
      for(int l=0;l<=lmax;l++)
        for(size_t i=0;i<x.n_elem;i++)
          lt(i,l) = oomph::Orthpoly::ddlegendre(l, x(i));
      return lt;
    }

    void LegendreBasis::eval_prim_f(const arma::vec & x, arma::mat & f, double element_length) const {
      (void) element_length;
      f=f_eval(x)*T;
    }

    void LegendreBasis::eval_prim_df(const arma::vec & x, arma::mat & df, double element_length) const {
      (void) element_length;
      df=df_eval(x)*T;
    }

    void LegendreBasis::eval_prim_d2f(const arma::vec & x, arma::mat & d2f, double element_length) const {
      (void) element_length;
      d2f=d2f_eval(x)*T;
    }

    void LegendreBasis::drop_first(bool deriv) {
      (void) deriv;
      enabled=enabled(1,T.n_cols-1);
    }

    void LegendreBasis::drop_last(bool deriv) {
      (void) deriv;
      enabled=enabled(0,T.n_cols-2);
    }
  }
}
