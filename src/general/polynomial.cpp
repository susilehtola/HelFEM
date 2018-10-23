#include "polynomial.h"

namespace helfem {
  namespace polynomial {
    double polyval(const arma::vec & cv, double x) {
      double f=cv(cv.n_elem-1);
      for(size_t ip=cv.n_elem-2;ip<cv.n_elem;ip--)
        f=f*x+cv(ip);
      return f;
    }

    arma::mat polyval(const arma::mat & cv, const arma::vec & xv) {
      arma::mat fv(xv.n_elem,cv.n_cols);
      for(size_t ic=0;ic<cv.n_cols;ic++)
	for(size_t ix=0;ix<xv.n_elem;ix++)
	  fv(ix,ic)=polyval(cv.col(ic),xv(ix));
      return fv;
    }

    static double logprod(const arma::vec & x0, double xv, int & sign) {
      double f=0.0;
      for(size_t ip=1;ip<x0.n_elem;ip++) {
        double dx=xv-x0(ip);
        if(dx<0.0) {
          sign*=-1;
          dx=-dx;
        }
        f+=log(dx);
      }
      return f;
    }

    double lipval(const arma::vec & x0, double x) {
      /*
        Assume we're evaluating the polynomial for x=x0(0). Evaluate
        products using logarithms and exponentials to avoid under- and
        overflows.
      */

      // Overall sign
      int sign=1.0;
      // Compute numerator
      double num(logprod(x0,x,sign));
      // Compute denominator
      double denom(logprod(x0,x0(0),sign));

      return sign*exp(num-denom);
    }

    arma::mat lipval(const arma::mat & x0, const arma::vec & xv) {
      arma::mat fv(xv.n_elem,x0.n_cols);
      for(size_t ic=0;ic<x0.n_cols;ic++)
	for(size_t ix=0;ix<xv.n_elem;ix++)
	  fv(ix,ic)=lipval(x0.col(ic),xv(ix));
      return fv;
    }

    double factorial(int m) {
      double r=1.0;
      for(int p=m;p>0;p--)
        r*=p;
      return r;
    }

    double factorial_ratio(int pmax, int pmin) {
      // Check consistency of arguments
      if(pmax < pmin)
        return 1.0/factorial_ratio(pmin, pmax);

      // Calculate ratio
      double r=1.0;
      for(int p=pmax;p>pmin;p--)
        r*=p;

      return r;
    }

    double choose(int n, int k) {
      double val=factorial_ratio(n,n-k)/factorial(k);
      return val;
    }

    arma::mat derivative_coeffs(const arma::mat & c, int der_order) {
      arma::mat d(c.n_rows-der_order,c.n_cols);

      // Compute factorial ratios
      arma::vec fr(d.n_rows);
      for(size_t ir=0;ir<d.n_rows;ir++)
        fr(ir)=factorial_ratio(ir+der_order,ir);

      for(size_t ic=0;ic<d.n_cols;ic++)
        for(size_t ir=0;ir<d.n_rows;ir++)
          d(ir,ic)=fr(ir)*c(ir+der_order,ic);

      return d;
    }

    arma::mat hermite_coeffs(int n_nodes, int der_order) {
      // Nodes
      arma::vec xi(arma::linspace<arma::vec>(-1.0,1.0,n_nodes));
      //xi.print("xi");

      // We want the functions to be continuous to the kth order, so the
      // polynomials have to be of order p-1 where
      int pmax=(der_order+1)*n_nodes;

      // Construct the coefficient matrix
      arma::mat ximat(pmax,pmax);
      ximat.zeros();

      // Loop over nodes
      for(int inode=0;inode<n_nodes;inode++)
        // Loop over derivative order
        for(int ider=0;ider<=der_order;ider++) {
          // Row index is
          int row=(der_order+1)*inode+ider;
          /*
            The matrix reads
            1  xi_0  xi_0^2 ...      xi_0^{p-1}
            0   1    2 xi_0 ...   (p-1) xi_0^{p-2}
            0   0       2   ... (p-1)(p-2) xi_0^{p-3}
            ...
          */
          for(int col=ider;col<pmax;col++)
            ximat(row,col) = factorial_ratio(col,col-ider)*std::pow(xi(inode),col-ider);
        }
      //ximat.print("ximat");

      // The coefficient matrix is simply
      return arma::inv(ximat);
    }

    arma::mat conversion_matrix(size_t xmax, double rmin, double rmax) {
      // Midpoint of interval
      double rmid((rmax+rmin)/2.0);
      double invrmid((rmax+rmin)/2.0);
      // Inverse length of interval
      double invrlen(2.0/(rmax-rmin));

      arma::mat T(xmax,xmax);
      T.zeros();

      // Loop over functions
      for(size_t m=0;m<xmax;m++)
        for(size_t k=0;k<=m;k++) {
          int mmk=m-k;
          double sign = ((m-k)%2) ? -1.0 : 1.0;
          double rmp = (mmk < 0) ? std::pow(invrmid,-mmk) : std::pow(rmid,mmk);
          T(k,m)+=sign*std::pow(invrlen,m)*choose(m,k)*rmp;
        }

      return T;
    }

    arma::mat convert_coeffs(const arma::mat & C, double rmin, double rmax) {
      return conversion_matrix(C.n_rows,rmin,rmax)*C;
    }
  }
}
