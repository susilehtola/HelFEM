#include "quadrature.h"
#include "../general/chebyshev.h"

namespace helfem {
  namespace quadrature {
    arma::mat radial_integral(double mumin, double mumax, int k, int l, const arma::vec & x, const arma::vec & wx, const arma::mat & bf) {
#ifndef ARMA_NO_DEBUG
      if(x.n_elem != wx.n_elem) throw std::logic_error("x and wx not compatible!\n");
      if(x.n_elem != bf.n_rows) throw std::logic_error("x and bf not compatible!\n");
#endif

      // Midpoint is at
      double mumid(0.5*(mumax+mumin));
      // and half-length of interval is
      double mulen(0.5*(mumax-mumin));
      // Mu values are then
      arma::vec mu(mumid*arma::ones<arma::vec>(x.n_elem)+mulen*x);

      // Calculate total weight per point
      arma::vec wp(wx*mulen);
      if(k!=0) {
	// Sine part
	if(k==1)
	  wp%=arma::sinh(mu);
	else if(k==-1)
	  wp/=arma::sinh(mu);
	else
	  throw std::logic_error("Case not implemented.\n");
      }
      if(l!=0) {
	// Cosine part
	if(l==1)
	  wp%=arma::cosh(mu);
	else if(l==2)
	  wp%=arma::square(arma::cosh(mu));
	else
	  throw std::logic_error("Case not implemented.\n");
      }

      // Put in weight
      arma::mat wbf(bf);
      for(size_t i=0;i<bf.n_cols;i++)
	wbf.col(i)%=wp;

      // Overlap matrix is then
      return arma::trans(wbf)*wbf;
    }

    arma::mat derivative_integral(double mumin, double mumax, const arma::vec & x, const arma::vec & wx, const arma::mat & bf, const arma::mat & dbf, const arma::mat & hbf) {
#ifndef ARMA_NO_DEBUG
      if(x.n_elem != wx.n_elem) throw std::logic_error("x and wx not compatible!\n");
      if(x.n_elem != dbf.n_rows) throw std::logic_error("x and dbf not compatible!\n");
#endif

      // Midpoint is at
      double mumid(0.5*(mumax+mumin));
      // and half-length of interval is
      double mulen(0.5*(mumax-mumin));
      // Mu values are then
      arma::vec mu(mumid*arma::ones<arma::vec>(x.n_elem)+mulen*x);

      // Put in weight
      arma::mat wbf(bf);
      for(size_t i=0;i<bf.n_cols;i++)
	wbf.col(i)%=wx;

      arma::vec shmu(sinh(mu));
      arma::vec chmu(sinh(mu));

      // Construct matrix: one mulen cancels out with quadrature weight
      arma::mat der(bf);
      for(size_t i=0;i<der.n_cols;i++)
	der.col(i)=shmu%hbf.col(i)/mulen + chmu%dbf.col(i);

      // Integral is
      return arma::trans(wbf)*der;
    }
  }
}
