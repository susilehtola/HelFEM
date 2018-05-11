#include "quadrature.h"
#include "../general/chebyshev.h"

namespace helfem {
  namespace quadrature {
    arma::mat radial_integral(double rmin, double rmax, int n, const arma::vec & x, const arma::vec & wx, const arma::mat & bf) {
#ifndef ARMA_NO_DEBUG
      if(x.n_elem != wx.n_elem) {
        std::ostringstream oss;
        oss << "x and wx not compatible: " << x.n_elem << " vs " << wx.n_elem << "!\n";
        throw std::logic_error(oss.str());
      }
      if(x.n_elem != bf.n_rows) {
        std::ostringstream oss;
        oss << "x and bf not compatible: " << x.n_elem << " vs " << bf.n_rows << "!\n";
        throw std::logic_error(oss.str());
      }
#endif

      // Midpoint is at
      double rmid(0.5*(rmax+rmin));
      // and half-length of interval is
      double rlen(0.5*(rmax-rmin));
      // r values are then
      arma::vec r(rmid*arma::ones<arma::vec>(x.n_elem)+rlen*x);

      // Calculate total weight per point
      arma::vec wp(wx*rlen);
      if(n!=0)
        wp%=arma::pow(r,n);

      // Put in weight
      arma::mat wbf(bf);
      for(size_t i=0;i<bf.n_cols;i++)
	wbf.col(i)%=wp;

      // Matrix elements are then
      return arma::trans(wbf)*bf;
    }

    arma::mat derivative_integral(double rmin, double rmax, const arma::vec & x, const arma::vec & wx, const arma::mat & dbf) {
#ifndef ARMA_NO_DEBUG
      if(x.n_elem != wx.n_elem) {
        std::ostringstream oss;
        oss << "x and wx not compatible: " << x.n_elem << " vs " << wx.n_elem << "!\n";
        throw std::logic_error(oss.str());
      }
      if(x.n_elem != dbf.n_rows) {
        std::ostringstream oss;
        oss << "x and dbf not compatible: " << x.n_elem << " vs " << dbf.n_rows << "!\n";
        throw std::logic_error(oss.str());
      }
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

    arma::mat twoe_inner_integral(double rmin, double rmax, const arma::vec & x, const arma::vec & wx, const arma::mat & bf, int L) {
      // Product functions
      arma::mat bfprod(bf.n_rows,bf.n_cols*bf.n_cols);
      for(size_t fi=0;fi<bf.n_cols;fi++)
        for(size_t fj=0;fj<bf.n_cols;fj++)
          bfprod.col(fi*bf.n_cols+fj)=bf.col(fi)%bf.col(fj);

      // Midpoint is at
      double rmid(0.5*(rmax+rmin));
      // and half-length of interval is
      double rlen(0.5*(rmax-rmin));
      // r values are then
      arma::vec r(rmid*arma::ones<arma::vec>(x.n_elem)+rlen*x);

      // Compute the "inner" integrals as function of r.
      arma::mat inner(bfprod.n_rows, bfprod.n_cols);
      inner.zeros();
      // Calculate total weight per point
      arma::vec wp(wx*rlen);
      // Put in r^(2+L)
      wp%=arma::pow(r,2+L);

      // Put in weight
      for(size_t i=0;i<bfprod.n_cols;i++)
        inner.col(i)=bfprod.col(i)%wp;

      // The running integral is then
      for(size_t ic=0;ic<inner.n_cols;ic++)
        for(size_t ir=1;ir<inner.n_rows;ir++)
          inner(ir,ic)+=inner(ir-1,ic);

      r.save("r.dat",arma::raw_ascii);
      bf.save("bf.dat",arma::raw_ascii);
      inner.save("inner.dat",arma::raw_ascii);

      return inner;
    }

    arma::mat twoe_integral(double rmin, double rmax, const arma::vec & x, const arma::vec & wx, const arma::mat & bf, int L) {
#ifndef ARMA_NO_DEBUG
      if(x.n_elem != wx.n_elem) {
        std::ostringstream oss;
        oss << "x and wx not compatible: " << x.n_elem << " vs " << wx.n_elem << "!\n";
        throw std::logic_error(oss.str());
      }
      if(x.n_elem != bf.n_rows) {
        std::ostringstream oss;
        oss << "x and bf not compatible: " << x.n_elem << " vs " << bf.n_rows << "!\n";
        throw std::logic_error(oss.str());
      }
#endif
      // Product functions
      arma::mat bfprod(bf.n_rows,bf.n_cols*bf.n_cols);
      for(size_t fi=0;fi<bf.n_cols;fi++)
        for(size_t fj=0;fj<bf.n_cols;fj++)
          bfprod.col(fi*bf.n_cols+fj)=bf.col(fi)%bf.col(fj);

      // Midpoint is at
      double rmid(0.5*(rmax+rmin));
      // and half-length of interval is
      double rlen(0.5*(rmax-rmin));
      // r values are then
      arma::vec r(rmid*arma::ones<arma::vec>(x.n_elem)+rlen*x);

      // First, compute the "inner" integrals as function of r.
      arma::mat inner(twoe_inner_integral(rmin, rmax, x, wx, bf, L));

      // Next, the outer integrals
      arma::vec wp(wx*rlen);
      // Put in r^(1-L)
      wp%=arma::pow(r,1-L);
      for(size_t i=0;i<bfprod.n_cols;i++)
        bfprod.col(i)%=wp;

      // Integrals are then
      arma::mat ints(4.0*M_PI/(2*L+1)*arma::trans(bfprod)*inner);
      // but we are still missing the second term which can be
      // obtained as simply as
      return ints+arma::trans(ints);
    }
  }
}
