#include "../general/lobatto.h"
#include "polynomial_basis.h"
#include "polynomial.h"
#include <cfloat>

// Legendre polynomials
extern "C" {
#include <gsl/gsl_sf_legendre.h>
}

namespace helfem {
  namespace polynomial_basis {
    void drop_first(arma::uvec & idx) {
      // This is simple - just drop first function
      idx=idx.subvec(1,idx.n_elem-1);
    }

    void drop_last(arma::uvec & idx, int noverlap) {
      // This is simple too - drop the last noverlap functions to force function and its derivatives to zero
      idx=idx.subvec(0,idx.n_elem-noverlap-1);
    }

    arma::uvec primitive_indices(int nprim, int noverlap, bool first, bool last) {
      arma::uvec idx(arma::linspace<arma::uvec>(0,nprim-1,nprim));
      if(first)
	drop_first(idx);
      if(last)
	drop_last(idx, noverlap);
      return idx;
    }

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
        {
          arma::vec x, w;
          ::lobatto_compute(Nnodes,x,w);
          poly=new polynomial_basis::LIPBasis(x);
          printf("Basis set composed of %i-node LIPs with Gauss-Lobatto nodes.\n",Nnodes);
          break;
        }

      default:
        throw std::logic_error("Unsupported primitive basis.\n");
      }

      // Print out
      poly->print();

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

    void PolynomialBasis::print(const std::string & str) const {
      arma::vec x(arma::linspace<arma::vec>(-1.0,1.0,1001));
      arma::mat bf, df;
      eval(x,bf,df);

      bf.insert_cols(0,x);
      df.insert_cols(0,x);

      std::string fname("bf" + str + ".dat");
      std::string dname("df" + str + ".dat");
      bf.save(fname,arma::raw_ascii);
      df.save(dname,arma::raw_ascii);
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
      // Only drop function value, not derivatives
      arma::uvec idx(arma::linspace<arma::uvec>(0,bf_C.n_cols-1,bf_C.n_cols));
      polynomial_basis::drop_first(idx);

      bf_C=bf_C.cols(idx);
      df_C=df_C.cols(idx);
      nbf=bf_C.n_cols;
    }

    void HermiteBasis::drop_last() {
      // Only drop function value, not derivatives
      arma::uvec idx(arma::linspace<arma::uvec>(0,bf_C.n_cols-1,bf_C.n_cols));
      polynomial_basis::drop_last(idx, noverlap);

      bf_C=bf_C.cols(idx);
      df_C=df_C.cols(idx);
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

    LIPBasis::LIPBasis(const arma::vec & x) {
      // Make sure nodes are in order
      x0=arma::sort(x,"ascend");

      // Sanity check
      if(std::abs(x(0)+1)>=sqrt(DBL_EPSILON))
        throw std::logic_error("LIP leftmost node is not at -1!\n");
      if(std::abs(x(x.n_elem-1)-1)>=sqrt(DBL_EPSILON))
        throw std::logic_error("LIP rightmost node is not at -1!\n");

      // One overlapping function
      noverlap=1;
      nbf=x0.n_elem;
      // All functions are enabled
      enabled=arma::linspace<arma::uvec>(0,x0.n_elem-1,x0.n_elem);
    }

    LIPBasis::~LIPBasis() {
    }

    LIPBasis * LIPBasis::copy() const {
      return new LIPBasis(*this);
    }

    arma::mat LIPBasis::eval(const arma::vec & x) const {
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

    void LIPBasis::eval(const arma::vec & x, arma::mat & f, arma::mat & df) const {
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

    void LIPBasis::drop_first() {
      enabled=enabled.subvec(1,enabled.n_elem-1);
      nbf=enabled.n_elem;
    }

    void LIPBasis::drop_last() {
      enabled=enabled.subvec(0,enabled.n_elem-2);
      nbf=enabled.n_elem;
    }
  }
}
