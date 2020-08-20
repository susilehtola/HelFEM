/*
 *                This source code is part of
 *
 *                          HelFEM
 *                             -
 * Finite element methods for electronic structure calculations on small systems
 *
 * Written by Susi Lehtola, 2018-
 * Copyright (c) 2018- Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */
#include "basis.h"
#include "quadrature.h"
#include "../general/polynomial_basis.h"
#include "chebyshev.h"
#include "../general/spherical_harmonics.h"
#include "../general/gaunt.h"
#include "../general/utils.h"
#include "../general/scf_helpers.h"
#include <cassert>
#include <cfloat>
#include <helfem>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace helfem {
  namespace atomic {
    namespace basis {
      RadialBasis::RadialBasis() {
      }

      RadialBasis::RadialBasis(const polynomial_basis::PolynomialBasis * poly_, int n_quad, const arma::vec & bval_) {
	// Polynomial basis
        poly=poly_->copy();

        // Get quadrature rule
        chebyshev::chebyshev(n_quad,xq,wq);
        for(size_t i=0;i<xq.n_elem;i++) {
          if(!std::isfinite(xq[i]))
            printf("xq[%i]=%e\n",(int) i, xq[i]);
          if(!std::isfinite(wq[i]))
            printf("wq[%i]=%e\n",(int) i, wq[i]);
        }

        // Evaluate polynomials at quadrature points
        poly->eval(xq,bf,df);

        // Element boundaries
        bval=bval_;
      }

      RadialBasis::RadialBasis(const RadialBasis & rh) {
        *this = rh;
      }

      RadialBasis & RadialBasis::operator=(const RadialBasis & rh) {
        xq=rh.xq;
        wq=rh.wq;
        poly=rh.poly->copy();
        bf=rh.bf;
        df=rh.df;
        bval=rh.bval;
        return *this;
      }

      RadialBasis::~RadialBasis() {
        delete poly;
      }

      static arma::vec concatenate_grid(const arma::vec & left, const arma::vec & right) {
        if(!left.n_elem)
          return right;
        if(!right.n_elem)
          return left;

        if(left(0) != 0.0)
          throw std::logic_error("left vector doesn't start from zero");
        if(right(0) != 0.0)
          throw std::logic_error("right vector doesn't start from zero");

        // Concatenated vector
        arma::vec ret(left.n_elem + right.n_elem - 1);
        ret.subvec(0,left.n_elem-1)=left;
        ret.subvec(left.n_elem,ret.n_elem-1)=right.subvec(1,right.n_elem-1) + left(left.n_elem-1)*arma::ones<arma::vec>(right.n_elem-1);
        return ret;
      }

      arma::vec normal_grid(int num_el, double rmax, int igrid, double zexp) {
        return utils::get_grid(rmax,num_el,igrid,zexp);
      }

      arma::vec finite_nuclear_grid(int num_el, double rmax, int igrid, double zexp, int num_el_nuc, double rnuc, int igrid_nuc, double zexp_nuc) {
        if(num_el_nuc) {
          // Grid for the finite nucleus
          arma::vec bnuc(utils::get_grid(rnuc,num_el_nuc,igrid_nuc,zexp_nuc));
          // and the one for the electrons
          arma::vec belec(utils::get_grid(rmax-rnuc,num_el,igrid,zexp));

          arma::vec bnucel(concatenate_grid(bnuc,bnuc));
          return concatenate_grid(bnucel,belec);
        } else {
          return utils::get_grid(rmax,num_el,igrid,zexp);
        }
      }

      arma::vec offcenter_nuclear_grid(int num_el0, int Zm, int Zlr, double Rhalf, int num_el, double rmax, int igrid, double zexp) {
        // First boundary at
        int b0used = (Zm != 0);
        double b0=Zm*Rhalf/(Zm+Zlr);
        // Second boundary at
        int b1used = (Zlr != 0);
        double b1=Rhalf;
        // Last boundary at
        double b2=rmax;

        printf("b0 = %e, b0used = %i\n",b0,b0used);
        printf("b1 = %e, b1used = %i\n",b1,b1used);
        printf("b2 = %e\n",b2);

        // Get grids
        arma::vec bval0, bval1;
        if(b0used) {
          // 0 to b0
          bval0=utils::get_grid(b0,num_el0,igrid,zexp);
        }
        if(b1used) {
          // b0 to b1

          // Reverse grid to get tighter spacing around nucleus
          bval1=-arma::reverse(utils::get_grid(b1-b0,num_el0,igrid,zexp));
          bval1+=arma::ones<arma::vec>(bval1.n_elem)*(b1-b0);
          // Assert numerical exactness
          bval1(0)=0.0;
          bval1(bval1.n_elem-1)=b1-b0;
        }
        arma::vec bval2=utils::get_grid(b2-b1,num_el,igrid,zexp);

        arma::vec bval;
        if(b0used && b1used) {
          bval=concatenate_grid(bval0,bval1);
        } else if(b0used) {
          bval=bval0;
        } else if(b1used) {
          bval=bval1;
        }
        if(b0used || b1used) {
          bval=concatenate_grid(bval,bval2);
        } else {
          bval=bval2;
        }

        return bval;
      }

      arma::vec form_grid(modelpotential::nuclear_model_t model, double Rrms, int Nelem, double Rmax, int igrid, double zexp, int Nelem0, int igrid0, double zexp0, int Z, int Zl, int Zr, double Rhalf) {
        // Construct the radial basis
        arma::vec bval;
        if(model != modelpotential::POINT_NUCLEUS) {
          printf("Finite-nucleus grid\n");

          if(Zl != 0 || Zr != 0)
            throw std::logic_error("Off-center nuclei not supported in finite nucleus mode!\n");

          double rnuc;
          if(model == modelpotential::HOLLOW_NUCLEUS)
            rnuc = Rrms;
          else if(model == modelpotential::SPHERICAL_NUCLEUS)
            rnuc = sqrt(5.0/3.0)*Rrms;
          else if(model == modelpotential::GAUSSIAN_NUCLEUS)
            rnuc = 3*Rrms;
          else
            throw std::logic_error("Nuclear grid not handled!\n");

          bval=atomic::basis::finite_nuclear_grid(Nelem,Rmax,igrid,zexp,Nelem0,rnuc,igrid0,zexp0);

        } else if(Zl != 0 || Zr != 0) {
          printf("Off-center grid\n");
          bval=atomic::basis::offcenter_nuclear_grid(Nelem0,Z,std::max(Zl,Zr),Rhalf,Nelem,Rmax,igrid,zexp);
        } else {
          printf("Normal grid\n");
          bval=atomic::basis::normal_grid(Nelem,Rmax,igrid,zexp);
        }

        bval.print("Grid");

        return bval;
      }

      void angular_basis(int lmax, int mmax, arma::ivec & lval, arma::ivec & mval) {
        size_t nang=0;
        for(int l=0;l<=lmax;l++)
          nang+=2*std::min(mmax,l)+1;

        // Allocate memory
        lval.zeros(nang);
        mval.zeros(nang);

        // Store values
        size_t iang=0;
        for(int mabs=0;mabs<=mmax;mabs++)
          for(int l=mabs;l<=lmax;l++) {
            lval(iang)=l;
            mval(iang)=mabs;
            iang++;
            if(mabs>0) {
              lval(iang)=l;
              mval(iang)=-mabs;
              iang++;
            }
          }
        if(iang!=lval.n_elem)
          throw std::logic_error("Error.\n");
      }

      void RadialBasis::add_boundary(double r) {
        // Check that r is not in bval
        bool in_bval=false;
        for(size_t i=0;i<bval.n_elem;i++)
          if(bval(i)==r)
            in_bval=true;

        // Add
        if(!in_bval) {
          arma::vec newbval(bval.n_elem+1);
          newbval.subvec(0,bval.n_elem-1)=bval;
          newbval(bval.n_elem)=r;
          bval = arma::sort(newbval, "ascend");
        }
      }

      /// Get polynomial basis
      polynomial_basis::PolynomialBasis * RadialBasis::get_poly() const {
        return poly->copy();
      }

      int RadialBasis::get_nquad() const {
        return (int) xq.n_elem;
      }

      arma::vec RadialBasis::get_bval() const {
        return bval;
      }

      int RadialBasis::get_poly_id() const {
        return poly->get_id();
      }

      int RadialBasis::get_poly_order() const {
        return poly->get_order();
      }

      arma::uvec RadialBasis::basis_indices(size_t iel) const {
	// Number of overlapping functions
	int noverlap(get_noverlap());
	// Number of primitive functions
	int nprim(bf.n_cols);

	return polynomial_basis::primitive_indices(nprim, noverlap, iel==0, iel==bval.n_elem-2);
      }

      arma::mat RadialBasis::get_basis(const arma::mat & bas, size_t iel) const {
	arma::uvec idx(basis_indices(iel));
	return bas.cols(idx);
      }

      polynomial_basis::PolynomialBasis * RadialBasis::get_basis(const polynomial_basis::PolynomialBasis * polynom, size_t iel) const {
        polynomial_basis::PolynomialBasis * p(polynom->copy());
	if(iel==0)
          p->drop_first();
	if(iel==bval.n_elem-2)
          p->drop_last();

        return p;
      }

      size_t RadialBasis::get_noverlap() const {
        return poly->get_noverlap();
      }

      size_t RadialBasis::Nel() const {
        // Number of elements is
        return bval.n_elem-1;
      }

      size_t RadialBasis::Nbf() const {
        // Number of basis functions is Nprim*Nel - (Nel-1)*noverlap - 1 - noverlap
        return Nel()*(bf.n_cols-poly->get_noverlap())-1;
      }

      size_t RadialBasis::Nprim(size_t iel) const {
	return basis_indices(iel).n_elem;
      }

      size_t RadialBasis::max_Nprim() const {
        return bf.n_cols;
      }

      void RadialBasis::get_idx(size_t iel, size_t & ifirst, size_t & ilast) const {
	// Compute the storage indices of elements.
        // The first function in the element will be
        ifirst=iel*(bf.n_cols - poly->get_noverlap());
        // and the last one will be
        ilast=ifirst+bf.n_cols-1;

        // Account for the function deleted at the origin
        ilast--;
        if(iel>0)
	  // First function in first element is always 0
          ifirst--;

        // Last element does not have trailing functions
        if(iel==bval.n_elem-2)
          ilast-=poly->get_noverlap();
      }

      arma::mat RadialBasis::radial_integral(int Rexp, size_t iel) const {
        return radial_integral(bf,Rexp,iel);
      }

      arma::mat RadialBasis::radial_integral(const arma::mat & funcs, int Rexp, size_t iel) const {
        double Rmin(bval(iel));
        double Rmax(bval(iel+1));

        // Integral by quadrature
        return quadrature::radial_integral(Rmin,Rmax,Rexp,xq,wq,get_basis(funcs,iel));
      }

      arma::mat RadialBasis::bessel_il_integral(int L, double lambda, size_t iel) const {
        double Rmin(bval(iel));
        double Rmax(bval(iel+1));

        // Integral by quadrature
        return quadrature::bessel_il_integral(Rmin,Rmax,L,lambda,xq,wq,get_basis(bf,iel));
      }

      arma::mat RadialBasis::bessel_kl_integral(int L, double lambda, size_t iel) const {
        double Rmin(bval(iel));
        double Rmax(bval(iel+1));

        // Integral by quadrature
        return quadrature::bessel_kl_integral(Rmin,Rmax,L,lambda,xq,wq,get_basis(bf,iel));
      }

      arma::mat RadialBasis::radial_integral(const RadialBasis & rh, int n, bool lhder, bool rhder) const {
        modelpotential::RadialPotential rad(n);
        return model_potential(rh,&rad,lhder,rhder);
      }

      arma::mat RadialBasis::model_potential(const RadialBasis & rh, const modelpotential::ModelPotential * model, bool lhder, bool rhder) const {
	// Use the larger number of quadrature points to assure
	// projection is computed ok
	size_t n_quad(std::max(xq.n_elem,rh.xq.n_elem));

	arma::vec xproj, wproj;
	chebyshev::chebyshev(n_quad,xproj,wproj);

        // Form list of overlapping elements
        std::vector< std::vector<size_t> > overlap(bval.n_elem-1);
        for(size_t iel=0;iel<bval.n_elem-1;iel++) {
          // Range of element i
          double istart(bval(iel));
          double iend(bval(iel+1));

          for(size_t jel=0;jel<rh.bval.n_elem-1;jel++) {
            // Range of element j
            double jstart(rh.bval(jel));
            double jend(rh.bval(jel+1));

            // Is there overlap?
            if((jstart >= istart && jstart<iend) || (istart >= jstart && istart < jend)) {
              overlap[iel].push_back(jel);
	      //printf("New element %i overlaps with old element %i\n",iel,jel);
	    }
          }
        }

        // Form overlap matrix
        arma::mat S(Nbf(),rh.Nbf());
        S.zeros();
        for(size_t iel=0;iel<bval.n_elem-1;iel++) {
          // Loop over overlapping elements
          for(size_t jj=0;jj<overlap[iel].size();jj++) {
            // Index of element is
            size_t jel=overlap[iel][jj];

	    // Because the functions are only defined within a single
	    // element, the product can be very raggedy. However,
	    // since we *know* where the overlap is non-zero, we can
	    // restrict the quadrature to just that zone.

	    // Limits
	    double imin(bval(iel));
	    double imax(bval(iel+1));
            // Range of element
            double jmin(rh.bval(jel));
            double jmax(rh.bval(jel+1));

	    // Range of integral is thus
	    double intstart(std::max(imin,jmin));
	    double intend(std::min(imax,jmax));
	    // Inteval mid-point is at
            double intmid(0.5*(intend+intstart));
            double intlen(0.5*(intend-intstart));

	    // the r values we're going to use are then
	    arma::vec r(intmid*arma::ones<arma::vec>(xproj.n_elem)+intlen*xproj);

	    // Basis function indices
	    arma::uvec iidx(basis_indices(iel));
            arma::uvec jidx(rh.basis_indices(jel));
	    // Where are we in the matrix?
	    size_t ifirst, ilast;
	    get_idx(iel,ifirst,ilast);
            size_t jfirst, jlast;
            rh.get_idx(jel,jfirst,jlast);

	    // Back-transform r values into i:th and j:th elements
	    double imid(0.5*(imax+imin));
	    double ilen(0.5*(imax-imin));
            double jmid(0.5*(jmax+jmin));
            double jlen(0.5*(jmax-jmin));

            // Calculate x values the polynomials should be evaluated at
            arma::vec xi((r-imid*arma::ones<arma::vec>(r.n_elem))/ilen);
	    arma::vec xj((r-jmid*arma::ones<arma::vec>(r.n_elem))/jlen);

	    // Calculate total weight per point
	    arma::vec wtot(wproj*intlen);
	    // Put in the potential
            wtot %= model->V(r);

            // Evaluate radial basis functions
            arma::mat ibf, idf;
            poly->eval(xi, ibf, idf);
            arma::mat jbf, jdf;
            rh.poly->eval(xj, jbf, jdf);

            // Need to divide derivatives by the element size
            idf /= (bval(iel+1)-bval(iel))/2;
            jdf /= (rh.bval(jel+1)-rh.bval(jel))/2;

            const arma::mat & ifunc = lhder ? idf : ibf;
            const arma::mat & jfunc = rhder ? jdf : jbf;

	    // Perform quadrature
            arma::mat s(arma::trans(ifunc)*arma::diagmat(wtot)*jfunc);

            // Increment overlap matrix
            S.submat(ifirst,jfirst,ilast,jlast)+=s(iidx,jidx);
          }
        }

        return S;
      }

      arma::mat RadialBasis::overlap(const RadialBasis & rh) const {
        return radial_integral(rh,0);
      }

      arma::mat RadialBasis::kinetic(size_t iel) const {
        // We get 1/rlen^2 from the derivatives
        double rlen((bval(iel+1)-bval(iel))/2);

        return 0.5*radial_integral(df,0,iel)/(rlen*rlen);
      }

      arma::mat RadialBasis::kinetic_l(size_t iel) const {
        return 0.5*radial_integral(-2,iel);
      }

      arma::mat RadialBasis::nuclear(size_t iel) const {
        return -radial_integral(-1,iel);
      }

      arma::mat RadialBasis::model_potential(const modelpotential::ModelPotential * model, size_t iel) const {
        double Rmin(bval(iel));
        double Rmax(bval(iel+1));

        // Integral by quadrature
        return quadrature::model_potential_integral(Rmin,Rmax,model,xq,wq,get_basis(bf,iel));
      }

      arma::mat RadialBasis::nuclear_offcenter(size_t iel, double Rhalf, int L) const {
        if(bval(iel)>=Rhalf)
          return -sqrt(4.0*M_PI/(2*L+1))*radial_integral(-L-1,iel)*std::pow(Rhalf,L);
        else if(bval(iel+1)<=Rhalf)
          return -sqrt(4.0*M_PI/(2*L+1))*radial_integral(L,iel)*std::pow(Rhalf,-L-1);
        else
          throw std::logic_error("Nucleus placed within element!\n");
      }

      arma::mat RadialBasis::twoe_integral(int L, size_t iel) const {
        double Rmin(bval(iel));
        double Rmax(bval(iel+1));

        // Integral by quadrature
        polynomial_basis::PolynomialBasis * p(get_basis(poly,iel));
        arma::mat tei(quadrature::twoe_integral(Rmin,Rmax,xq,wq,p,L));
        delete p;

        return tei;
      }

      arma::mat RadialBasis::yukawa_integral(int L, double lambda, size_t iel) const {
        double Rmin(bval(iel));
        double Rmax(bval(iel+1));

        // Integral by quadrature
        polynomial_basis::PolynomialBasis * p(get_basis(poly,iel));
        arma::mat tei(quadrature::yukawa_integral(Rmin,Rmax,xq,wq,p,L,lambda));
        delete p;

        return tei;
      }

      arma::mat RadialBasis::erfc_integral(int L, double mu, size_t iel, size_t kel) const {
        double Rmini(bval(iel));
        double Rmaxi(bval(iel+1));
        double Rmink(bval(kel));
        double Rmaxk(bval(kel+1));

        // Number of quadrature points
        size_t Nq = xq.n_elem;
        // Number of subintervals
        size_t Nint;

        if(iel == kel) {
          // Intraelement integral is harder to converge due to the
          // electronic cusp, so let's do the same trick as in the
          // separable case and use a tighter grid for the "inner"
          // integral.
          Nint = Nq;

        } else {
          // A single interval suffices since there's no cusp.
          Nint = 1;
        }

        // Get lh quadrature points
        arma::vec xi, wi;
        chebyshev::chebyshev(Nq,xi,wi);
        // and basis function values
        arma::mat ibf(poly->eval(xi));

        // Rh quadrature points
        arma::vec xk(Nq*Nint);
        arma::vec wk(Nq*Nint);
        for(size_t ii=0;ii<Nint;ii++) {
          // Interval starts at
          double istart = ii*2.0/Nint - 1.0;
          double iend = (ii+1)*2.0/Nint - 1.0;
          // Midpoint and half-length of interval
          double imid = 0.5*(iend+istart);
          double ilen = 0.5*(iend-istart);

          // Place quadrature points at
          xk.subvec(ii*Nq,(ii+1)*Nq-1)=arma::ones<arma::vec>(Nq)*imid + xi*ilen;
          // which have the renormalized weights
          wk.subvec(ii*Nq,(ii+1)*Nq-1)=wi*ilen;
        }
        // and basis function values
        arma::mat kbf(poly->eval(xk));

        // Evaluate integral
        arma::mat tei(quadrature::erfc_integral(Rmini,Rmaxi,get_basis(ibf,iel),xi,wi,Rmink,Rmaxk,get_basis(kbf,kel),xk,wk,L,mu));
        // Symmetrize just to be sure, since quadrature points were
        // different
        if(iel == kel)
          tei=0.5*(tei+tei.t());

        return tei;
      }

      arma::mat RadialBasis::spherical_potential(size_t iel) const {
        double Rmin(bval(iel));
        double Rmax(bval(iel+1));

        // Integral by quadrature
        polynomial_basis::PolynomialBasis * p(get_basis(poly,iel));
        arma::mat pot(quadrature::spherical_potential(Rmin,Rmax,xq,wq,p));
        delete p;

        return pot;
      }

      arma::mat RadialBasis::get_bf(size_t iel) const {
        // Element function values at quadrature points are
        arma::mat val(get_basis(bf,iel));

        // but we also need to put in the 1/r factor
        double rmin(bval(iel));
        double rmax(bval(iel+1));
        double rmid=(rmax+rmin)/2;
        double rlen=(rmax-rmin)/2;
        arma::vec r(rmid*arma::ones<arma::vec>(xq.n_elem)+rlen*xq);
        for(size_t j=0;j<val.n_cols;j++)
          for(size_t i=0;i<val.n_rows;i++)
            val(i,j)/=r(i);

        return val;
      }

      arma::mat RadialBasis::get_df(size_t iel) const {
        // Element function values at quadrature points are
        arma::mat fval(get_basis(bf,iel));
        arma::mat dval(get_basis(df,iel));

        // Calculate r values
        double rmin(bval(iel));
        double rmax(bval(iel+1));
        double rmid=(rmax+rmin)/2;
        double rlen=(rmax-rmin)/2;
        arma::vec r(rmid*arma::ones<arma::vec>(xq.n_elem)+rlen*xq);

        // Derivative is then
        arma::mat der(fval);
        for(size_t j=0;j<fval.n_cols;j++)
          for(size_t i=0;i<fval.n_rows;i++)
            // Get one rlen from derivative
            der(i,j)=dval(i,j)/(rlen*r(i)) - fval(i,j)/(r(i)*r(i));

        return der;
      }

      arma::mat RadialBasis::get_lf(size_t iel) const {
        // Element function values at quadrature points are
        arma::mat fval(get_basis(bf,iel));
        arma::mat dval(get_basis(df,iel));

        arma::mat lf;
        poly->eval_lapl(xq,lf);
        arma::mat lval(get_basis(lf,iel));

        // Calculate r values
        double rmin(bval(iel));
        double rmax(bval(iel+1));
        double rmid=(rmax+rmin)/2;
        double rlen=(rmax-rmin)/2;
        arma::vec r(rmid*arma::ones<arma::vec>(xq.n_elem)+rlen*xq);

        // Laplacian is then
        arma::mat lapl(fval);
        for(size_t j=0;j<fval.n_cols;j++)
          for(size_t i=0;i<fval.n_rows;i++)
            // Get one rlen from each derivative
            lapl(i,j)=lval(i,j)/(rlen*rlen*r(i)) - 2.0*dval(i,j)/(rlen*r(i)*r(i)) + 2.0*fval(i,j)/(r(i)*r(i)*r(i));

        return lapl;
      }

      arma::vec RadialBasis::get_wrad(size_t iel) const {
        // Full radial weight
        double rmin(bval(iel));
        double rmax(bval(iel+1));
        double rlen=(rmax-rmin)/2;

        // This is just the radial rule, no r^2 factor included here
        return rlen*wq;
      }

      arma::vec RadialBasis::get_r(size_t iel) const {
        // Full radial weight
        double rmin(bval(iel));
        double rmax(bval(iel+1));
        double rmid=(rmax+rmin)/2;
        double rlen=(rmax-rmin)/2;

        return rmid*arma::ones<arma::vec>(xq.n_elem)+rlen*xq;
      }

      double RadialBasis::nuclear_density(const arma::mat & Prad) const {
        if(Prad.n_rows != Nbf() || Prad.n_cols != Nbf())
          throw std::logic_error("nuclear_density expects a radial density matrix\n");

        // Nuclear coordinate
        arma::vec x(1);
        // Remember that the primitive basis polynomials belong to [-1,1]
        x(0)=-1.0;

        // Evaluate derivative at nucleus
        double rlen((bval(1)-bval(0))/2);

        arma::mat func, der;
        poly->eval(x,func,der);
        der=(get_basis(der,0)/rlen);

        // Radial functions in element
        size_t ifirst, ilast;
        get_idx(0,ifirst,ilast);
        // Density submatrix
        arma::mat Psub(Prad.submat(ifirst,ifirst,ilast,ilast));
        // P_uv B_u'(0) B_v'(0)
        double den(arma::as_scalar(der*Psub*arma::trans(der)));

        return den;
      }

      double RadialBasis::nuclear_density_gradient(const arma::mat & Prad) const {
        if(Prad.n_rows != Nbf() || Prad.n_cols != Nbf())
          throw std::logic_error("nuclear_density_gradient expects a radial density matrix\n");

        // Nuclear coordinate
        arma::vec x(1);
        // Remember that the primitive basis polynomials belong to [-1,1]
        x(0)=-1.0;

        // Evaluate derivative at nucleus
        double rlen((bval(1)-bval(0))/2);

        arma::mat func, der, lapl;
        poly->eval(x,func,der);
        der=(get_basis(der,0)/rlen);
        poly->eval_lapl(x,lapl);
        lapl=(get_basis(lapl,0)/(rlen*rlen));

        // Radial functions in element
        size_t ifirst, ilast;
        get_idx(0,ifirst,ilast);
        // Density submatrix
        arma::mat Psub(Prad.submat(ifirst,ifirst,ilast,ilast));
        // P_uv B_u'(0) B_v''(0)
        double den(arma::as_scalar(der*Psub*arma::trans(lapl)));

        return den;
      }

      arma::rowvec RadialBasis::nuclear_orbital(const arma::mat & C) const {
        // Nuclear coordinate
        arma::vec x(1);
        // Remember that the primitive basis polynomials belong to [-1,1]
        x(0)=-1.0;

        // Evaluate derivative at nucleus
        double rlen((bval(1)-bval(0))/2);

        arma::mat func, der;
        poly->eval(x,func,der);
        der=(get_basis(der,0)/rlen);

        // Radial functions in element
        size_t ifirst, ilast;
        get_idx(0,ifirst,ilast);
        // Density submatrix
        arma::mat Csub(C.rows(ifirst,ilast));

        // C_ui B_u'(0)
        return der*Csub;
      }
    }
  }
}
