#include "quadrature.h"
#include "../general/chebyshev.h"

namespace helfem {
  namespace quadrature {
    arma::mat radial_integral(double rmin, double rmax, int n, const arma::vec & x, const arma::vec & wx, const arma::mat & bf) {
#ifndef ARMA_NO_DEBUG
      if(x.n_elem != wx.n_elem) throw std::logic_error("x and wx not compatible!\n");
      if(x.n_elem != bf.n_rows) throw std::logic_error("x and bf not compatible!\n");
#endif

      // Midpoint is at
      double rmid(0.5*(rmax+rmin));
      // and half-length of interval is
      double rlen(0.5*(rmax-rmin));
      // r values are then
      arma::vec r(rmid*arma::ones<arma::vec>(x.n_elem)+rlen*x);

      // Calculate total weight per point
      arma::vec wp(wx*rlen);
      if(n!=0) {
	if(n==1)
	  wp%=r;
	else if(n==2)
	  wp%=arma::square(r);
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

    arma::mat derivative_integral(double rmin, double rmax, const arma::vec & x, const arma::vec & wx, const arma::mat & dbf) {
#ifndef ARMA_NO_DEBUG
      if(x.n_elem != wx.n_elem) throw std::logic_error("x and wx not compatible!\n");
      if(x.n_elem != dbf.n_rows) throw std::logic_error("x and dbf not compatible!\n");
#endif

      // Midpoint is at
      double rmid(0.5*(rmax+rmin));
      // and half-length of interval is
      double rlen(0.5*(rmax-rmin));
      // R values are then
      arma::vec r(rmid*arma::ones<arma::vec>(x.n_elem)+rlen*x);

      // Put in weight
      arma::mat wdbf(dbf);
      for(size_t i=0;i<dbf.n_cols;i++)
	// We get +1 rlen from the jacobian, but -2 from the derivatives
	wdbf.col(i)%=wx%arma::square(r)/rlen;

      // Integral is
      return arma::trans(wdbf)*dbf;
    }
  }
}
