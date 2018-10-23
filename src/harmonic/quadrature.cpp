#include "quadrature.h"
#include "../general/chebyshev.h"

namespace helfem {
  namespace quadrature {
    arma::mat radial_integral(double xmin, double xmax, int n, const arma::vec & xq, const arma::vec & wxq, const arma::mat & bf) {
#ifndef ARMA_NO_DEBUG
      if(xq.n_elem != wxq.n_elem) throw std::logic_error("x and wx not compatible!\n");
      if(xq.n_elem != bf.n_rows) throw std::logic_error("x and bf not compatible!\n");
#endif
      
      // Midpoint is at
      double xmid(0.5*(xmax+xmin));
      // and half-length of interval is
      double xlen(0.5*(xmax-xmin));
      // x values are then
      arma::vec x(xmid*arma::ones<arma::vec>(xq.n_elem)+xlen*xq);
      
      // Calculate total weight per point
      arma::vec wp(wxq*xlen);
      if(n!=0) {
	if(n==2)
	  wp%=arma::square(x);
	else
	  throw std::logic_error("Case not implemented.\n");
      }

      // Put in weight
      arma::mat wbf(bf);
      for(size_t i=0;i<bf.n_cols;i++)
	wbf.col(i)%=wp;

      // Matrix elements are then
      return arma::trans(wbf)*bf;
    }

    arma::mat derivative_integral(double xmin, double xmax, const arma::vec & xq, const arma::vec & wxq, const arma::mat & dbf) {
#ifndef ARMA_NO_DEBUG
      if(xq.n_elem != wxq.n_elem) throw std::logic_error("x and wx not compatible!\n");
      if(xq.n_elem != dbf.n_rows) throw std::logic_error("x and dbf not compatible!\n");
#endif

      // Midpoint is at
      double xmid(0.5*(xmax+xmin));
      // and half-length of interval is
      double xlen(0.5*(xmax-xmin));
      // R values are then
      arma::vec r(xmid*arma::ones<arma::vec>(xq.n_elem)+xlen*xq);

      // Put in weight
      arma::mat wdbf(dbf);
      for(size_t i=0;i<dbf.n_cols;i++)
	// We get +1 rlen from the jacobian, but -2 from the derivatives
	wdbf.col(i)%=wxq/xlen;

      // Integral is
      return arma::trans(wdbf)*dbf;
    }
  }
}
