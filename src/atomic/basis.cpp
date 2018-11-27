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
#include "../general/polynomial.h"
#include "../general/chebyshev.h"
#include "../general/spherical_harmonics.h"
#include "../general/gaunt.h"
#include "../general/gsz.h"
#include "../general/utils.h"
#include "../general/scf_helpers.h"
#include <cassert>
#include <cfloat>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace helfem {
  namespace atomic {
    namespace basis {

      static arma::vec get_grid(double rmax, int num_el, int igrid, double zexp) {
        // Boundary values
        arma::vec bval;

        // Get boundary values
        switch(igrid) {
          // linear grid
        case(1):
          printf("Using linear grid\n");
          bval=arma::linspace<arma::vec>(0,rmax,num_el+1);
          break;

          // quadratic grid (Schweizer et al 1999)
        case(2):
          printf("Using quadratic grid\n");
          bval.zeros(num_el+1);
          for(int i=0;i<=num_el;i++)
            bval(i)=i*i*rmax/(num_el*num_el);
          break;

          // generalized polynomial grid, monotonic decrease till zexp~3, after that fails to work
        case(3):
          printf("Using generalized polynomial grid, zexp = %e\n",zexp);
          bval.zeros(num_el+1);
          for(int i=0;i<=num_el;i++)
            bval(i)=rmax*std::pow(i*1.0/num_el,zexp);
          break;

          // generalized logarithmic grid, monotonic decrease till zexp~2, after that fails to work
        case(4):
          printf("Using generalized logarithmic grid, zexp = %e\n",zexp);
          bval=arma::exp(arma::pow(arma::linspace<arma::vec>(0,std::pow(log(rmax+1),1.0/zexp),num_el+1),zexp))-arma::ones<arma::vec>(num_el+1);
          break;

        default:
          throw std::logic_error("Invalid choice for grid\n");
        }

        // Make sure start and end points are numerically exact
        bval(0)=0.0;
        bval(bval.n_elem-1)=rmax;

        return bval;
      }

      RadialBasis::RadialBasis() {
      }

      RadialBasis::RadialBasis(const polynomial_basis::PolynomialBasis * poly_, int n_quad, int num_el, double rmax, int igrid, double zexp) {
        // Polynomial basis
        poly=poly_;

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

        // Get boundary values
        bval=get_grid(rmax,num_el,igrid,zexp);

        //bval.print("Element boundaries");
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

      RadialBasis::RadialBasis(const polynomial_basis::PolynomialBasis * poly_, int n_quad, int num_el0, int Zm, int Zlr, double Rhalf, int num_el, double rmax, int igrid, double zexp) {
	// Polynomial basis
	poly=poly_;

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
          bval0=get_grid(b0,num_el0,igrid,zexp);
        }
        if(b1used) {
          // b0 to b1

          // Reverse grid to get tighter spacing around nucleus
          bval1=-arma::reverse(get_grid(b1-b0,num_el0,igrid,zexp));
          bval1+=arma::ones<arma::vec>(bval1.n_elem)*(b1-b0);
          // Assert numerical exactness
          bval1(0)=0.0;
          bval1(bval1.n_elem-1)=b1-b0;
        }
        arma::vec bval2=get_grid(b2-b1,num_el,igrid,zexp);

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
        bval.print("Element boundaries");
      }

      RadialBasis::~RadialBasis() {
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
        polynomial_basis::PolynomialBasis *p(polynom->copy());
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

      arma::mat RadialBasis::gsz(double Z, double dz, double Hz, size_t iel) const {
        double Rmin(bval(iel));
        double Rmax(bval(iel+1));

        // Integral by quadrature
        return quadrature::gsz_integral(Z,dz,Hz,Rmin,Rmax,xq,wq,get_basis(bf,iel));
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
            der(i,j)=(dval(i,j)/rlen-fval(i,j)/r(i))/r(i);

        return der;
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

      double RadialBasis::nuclear_density(const arma::mat & P) const {
        // Nuclear coordinate
        arma::vec x(1);
        // Remember that the primitive basis polynomials belong to [-1,1]
        x(0)=-1.0;

        // Evaluate derivative at nucleus
        double rlen((bval(1)-bval(0))/2);

        arma::mat func, der;
        poly->eval(x,func,der);
        der=(get_basis(der,0)/rlen);

        // P_uv B_u'(0) B_v'(0)
        double den(arma::as_scalar(der*P*arma::trans(der)));

        return den;
      }

      TwoDBasis::TwoDBasis() {
      }

      TwoDBasis::TwoDBasis(int Z_, const polynomial_basis::PolynomialBasis * poly, int n_quad, int num_el, double rmax, int lmax, int mmax, int igrid, double zexp) {
        // Nuclear charge
        Z=Z_;
        Zl=0;
        Zr=0;
        Rhalf=0.0;

        // Construct radial basis
        radial=RadialBasis(poly, n_quad, num_el, rmax, igrid, zexp);

        // Construct angular basis
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

      TwoDBasis::TwoDBasis(int Z_, const polynomial_basis::PolynomialBasis * poly, int n_quad, int num_el0, int num_el, double rmax, int lmax, int mmax, int igrid, double zexp, int Zl_, int Zr_, double Rhalf_) {
        // Nuclear charge
        Z=Z_;
        Zl=Zl_;
        Zr=Zr_;
        Rhalf=Rhalf_;

        // Construct radial basis
        radial=RadialBasis(poly, n_quad, num_el0, Z_, std::max(Zl_,Zr_), Rhalf, num_el, rmax, igrid, zexp);

        // Construct angular basis
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

      TwoDBasis::~TwoDBasis() {
      }

      size_t TwoDBasis::Ndummy() const {
        return lval.n_elem*radial.Nbf();
      }

      size_t TwoDBasis::Nbf() const {
        return Ndummy();
      }

      size_t TwoDBasis::Nrad() const {
        return radial.Nbf();
      }

      size_t TwoDBasis::Nang() const {
        return lval.n_elem;
      }

      arma::uvec TwoDBasis::pure_indices() const {
        return arma::linspace<arma::uvec>(0,Nbf()-1,Nbf());
      }

      arma::ivec TwoDBasis::get_l() const {
        return lval;
      }

      arma::ivec TwoDBasis::get_m() const {
        return mval;
      }

      arma::uvec TwoDBasis::m_indices(int m) const {
        // Count how many functions
        size_t nm=0;
        for(size_t i=0;i<mval.n_elem;i++) {
          if(mval(i)==m) {
            nm += radial.Nbf();
          }
        }

        // Collect functions
        arma::uvec idx(nm);
        size_t ioff=0;
        size_t ibf=0;
        for(size_t i=0;i<mval.n_elem;i++) {
          // Number of functions on shell is
          size_t nsh=radial.Nbf();
          if(mval(i)==m) {
            idx.subvec(ioff,ioff+nsh-1)=arma::linspace<arma::uvec>(ibf,ibf+nsh-1,nsh);
            ioff+=nsh;
          }
          ibf+=nsh;
        }

        return idx;
      }

      arma::uvec TwoDBasis::lm_indices(int l, int m) const {
        // Count how many functions
        size_t nm=radial.Nbf();

        // Collect functions
        arma::uvec idx(nm);
        size_t ioff=0;
        size_t ibf=0;
        for(size_t i=0;i<mval.n_elem;i++) {
          // Number of functions on shell is
          size_t nsh=radial.Nbf();
          if(mval(i)==m && lval(i)==l) {
            idx.subvec(ioff,ioff+nsh-1)=arma::linspace<arma::uvec>(ibf,ibf+nsh-1,nsh);
            ioff+=nsh;
          }
          ibf+=nsh;
        }

        return idx;
      }

      std::vector<arma::uvec> TwoDBasis::get_sym_idx(int symm) const {
        std::vector<arma::uvec> idx;
        if(symm==0) {
          idx.resize(1);
          idx[0]=arma::linspace<arma::uvec>(0,Nbf()-1,Nbf());
        } else if(symm==1) {
          // Find unique m values
          arma::uvec muni(arma::find_unique(mval));
          arma::ivec mv(mval(muni));

          idx.resize(mv.n_elem);
          for(size_t i=0;i<mv.n_elem;i++)
            idx[i]=m_indices(mv(i));
        } else if(symm==2) {
          idx.resize(mval.n_elem);
          for(size_t i=0;i<mval.n_elem;i++) {
            idx[i]=lm_indices(lval(i),mval(i));
          }
        } else
          throw std::logic_error("Unknown symmetry\n");

        return idx;
      }

      arma::mat TwoDBasis::Shalf(bool chol, int sym) const {
        // Form overlap matrix
        arma::mat S(overlap());

        // Get the basis function norms
        arma::vec bfnormlz(arma::pow(arma::diagvec(S),-0.5));
        arma::vec bfinvnormlz(arma::pow(arma::diagvec(S),0.5));
	printf("Smallest normalization constant % e, largest % e\n",arma::min(bfnormlz),arma::max(bfnormlz));
        // Go to normalized basis
        S=arma::diagmat(bfnormlz)*S*arma::diagmat(bfnormlz);

        if(chol && sym==0) {
          // Half-inverse is
          return arma::diagmat(bfinvnormlz) * arma::chol(S);

        } else {
          arma::vec Sval;
          arma::mat Svec;
          if(sym) {
            // Symmetries
            std::vector<arma::uvec> midx(get_sym_idx(sym));
            scf::eig_sym_sub(Sval,Svec,S,midx);
          } else {
            if(!arma::eig_sym(Sval,Svec,S)) {
              S.save("S.dat",arma::raw_ascii);
              throw std::logic_error("Diagonalization of overlap matrix failed\n");
            }
          }
          printf("Smallest eigenvalue of overlap matrix is % e, condition number %e\n",Sval(0),Sval(Sval.n_elem-1)/Sval(0));

          arma::mat Shalf(Svec*arma::diagmat(arma::pow(Sval,0.5))*arma::trans(Svec));
          Shalf=arma::diagmat(bfinvnormlz)*Shalf;

          return Shalf;
        }
      }

      arma::mat TwoDBasis::Sinvh(bool chol, int sym) const {
        // Form overlap matrix
        arma::mat S(overlap());

        // Get the basis function norms
        arma::vec bfnormlz(arma::pow(arma::diagvec(S),-0.5));
        // Go to normalized basis
        S=arma::diagmat(bfnormlz)*S*arma::diagmat(bfnormlz);

        // Half-inverse is
        if(chol) {
          return arma::diagmat(bfnormlz) * arma::inv(arma::chol(S));
        } else {
          arma::vec Sval;
          arma::mat Svec;
          if(sym) {
            // Symmetries
            std::vector<arma::uvec> midx(get_sym_idx(sym));
            scf::eig_sym_sub(Sval,Svec,S,midx);
          } else {
            if(!arma::eig_sym(Sval,Svec,S)) {
              S.save("S.dat",arma::raw_ascii);
              throw std::logic_error("Diagonalization of overlap matrix failed\n");
            }
          }
          printf("Smallest eigenvalue of overlap matrix is % e, condition number %e\n",Sval(0),Sval(Sval.n_elem-1)/Sval(0));

          arma::mat Sinvh(Svec*arma::diagmat(arma::pow(Sval,-0.5))*arma::trans(Svec));
          Sinvh=arma::diagmat(bfnormlz)*Sinvh;

          return Sinvh;
        }
      }

      void TwoDBasis::set_sub(arma::mat & M, size_t iang, size_t jang, const arma::mat & Mrad) const {
        M.submat(iang*radial.Nbf(),jang*radial.Nbf(),(iang+1)*radial.Nbf()-1,(jang+1)*radial.Nbf()-1)=Mrad;
      }

      void TwoDBasis::add_sub(arma::mat & M, size_t iang, size_t jang, const arma::mat & Mrad) const {
        M.submat(iang*radial.Nbf(),jang*radial.Nbf(),(iang+1)*radial.Nbf()-1,(jang+1)*radial.Nbf()-1)+=Mrad;
      }

      arma::mat TwoDBasis::get_sub(const arma::mat & M, size_t iang, size_t jang) const {
        return M.submat(iang*radial.Nbf(),jang*radial.Nbf(),(iang+1)*radial.Nbf()-1,(jang+1)*radial.Nbf()-1);
      }

      arma::mat TwoDBasis::radial_integral(int Rexp) const {
        // Build radial elements
        size_t Nrad(radial.Nbf());
        arma::mat Orad(Nrad,Nrad);
        Orad.zeros();

        // Loop over elements
        for(size_t iel=0;iel<radial.Nel();iel++) {
          // Where are we in the matrix?
          size_t ifirst, ilast;
          radial.get_idx(iel,ifirst,ilast);
	  Orad.submat(ifirst,ifirst,ilast,ilast)+=radial.radial_integral(Rexp,iel);
        }

        // Full overlap matrix
        arma::mat O(Ndummy(),Ndummy());
        O.zeros();
        // Fill elements
        for(size_t iang=0;iang<lval.n_elem;iang++)
          set_sub(O,iang,iang,Orad);

        return remove_boundaries(O);
      }

      arma::mat TwoDBasis::overlap() const {
        return radial_integral(0);
      }

      arma::mat TwoDBasis::kinetic() const {
        // Build radial kinetic energy matrix
        size_t Nrad(radial.Nbf());
        arma::mat Trad(Nrad,Nrad);
        Trad.zeros();
        arma::mat Trad_l(Nrad,Nrad);
        Trad_l.zeros();

        // Loop over elements
        for(size_t iel=0;iel<radial.Nel();iel++) {
          // Where are we in the matrix?
          size_t ifirst, ilast;
          radial.get_idx(iel,ifirst,ilast);
          Trad.submat(ifirst,ifirst,ilast,ilast)+=radial.kinetic(iel);
          Trad_l.submat(ifirst,ifirst,ilast,ilast)+=radial.kinetic_l(iel);
        }

        // Full kinetic energy matrix
        arma::mat T(Ndummy(),Ndummy());
        T.zeros();
        // Fill elements
        for(size_t iang=0;iang<lval.n_elem;iang++) {
          set_sub(T,iang,iang,Trad);
          if(lval(iang)>0) {
            // We also get the l(l+1) term
            add_sub(T,iang,iang,lval(iang)*(lval(iang)+1)*Trad_l);
          }
        }

        return remove_boundaries(T);
      }

      arma::mat TwoDBasis::nuclear() const {
        // Full nuclear attraction matrix
        arma::mat V(Ndummy(),Ndummy());
        V.zeros();

        if(Z!=0.0) {
          size_t Nrad(radial.Nbf());
          arma::mat Vrad(Nrad,Nrad);
          Vrad.zeros();
          // Loop over elements
          for(size_t iel=0;iel<radial.Nel();iel++) {
            // Where are we in the matrix?
            size_t ifirst, ilast;
            radial.get_idx(iel,ifirst,ilast);
            Vrad.submat(ifirst,ifirst,ilast,ilast)+=radial.radial_integral(-1,iel);
          }
          // Fill elements
          for(size_t iang=0;iang<lval.n_elem;iang++)
            set_sub(V,iang,iang,-Z*Vrad);
        }

        if(Zl != 0.0 || Zr != 0.0) {
          // Auxiliary matrices
          size_t Nrad(radial.Nbf());
          int Lmax(2*arma::max(lval));
          std::vector<arma::mat> Vaux(Lmax+1);
#ifdef _OPENMP
#pragma omp parallel for
#endif
          for(int L=0;L<=Lmax;L++) {
            Vaux[L].zeros(Nrad,Nrad);
            for(size_t iel=0;iel<radial.Nel();iel++) {
              // Where are we in the matrix?
              size_t ifirst, ilast;
              radial.get_idx(iel,ifirst,ilast);
              Vaux[L].submat(ifirst,ifirst,ilast,ilast)+=radial.nuclear_offcenter(iel,Rhalf,L);
            }
          }

          int gmax(std::max(arma::max(lval),arma::max(mval)));
          gaunt::Gaunt gaunt(gmax,2*gmax,gmax);

          /// Loop over basis set
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
          for(size_t iang=0;iang<lval.n_elem;iang++) {
            for(size_t jang=0;jang<lval.n_elem;jang++) {
              int li(lval(iang));
              int mi(mval(iang));

              int lj(lval(jang));
              int mj(mval(jang));

              // Zero contribution
              if(mi!=mj)
                continue;

              // Loop over L
              for(int L=std::abs(li-lj);L<=li+lj;L++) {
                double cpl(gaunt.coeff(li,mi,L,0,lj,mj));
                if(cpl==0.0)
                  continue;

                add_sub(V,iang,jang,cpl*(std::pow(-1.0,L)*Zl + Zr)*Vaux[L]);
              }
            }
          }
        }

        return remove_boundaries(V);
      }

      arma::mat TwoDBasis::gsz(double dz, double Hz) const {
        // Full nuclear attraction matrix
        arma::mat V(Ndummy(),Ndummy());
        V.zeros();

	size_t Nrad(radial.Nbf());
	arma::mat Vrad(Nrad,Nrad);
	Vrad.zeros();
	// Loop over elements
	for(size_t iel=0;iel<radial.Nel();iel++) {
	  // Where are we in the matrix?
	  size_t ifirst, ilast;
	  radial.get_idx(iel,ifirst,ilast);
	  Vrad.submat(ifirst,ifirst,ilast,ilast)+=radial.gsz(Z,dz,Hz,iel);
	}
	// Fill elements
	for(size_t iang=0;iang<lval.n_elem;iang++)
	  set_sub(V,iang,iang,Vrad);

        return remove_boundaries(V);
      }

      arma::mat TwoDBasis::gsz() const {
	// Get default parameters
	double dz, Hz;
	GSZ::GSZ_parameters(Z,dz,Hz);
	return gsz(dz,Hz);
      }

      arma::mat TwoDBasis::dipole_z() const {
        // Build radial elements
        size_t Nrad(radial.Nbf());
        arma::mat Orad(Nrad,Nrad);
        Orad.zeros();

        // Full electric couplings
        arma::mat V(Ndummy(),Ndummy());
        V.zeros();

        // Loop over elements
        for(size_t iel=0;iel<radial.Nel();iel++) {
          // Where are we in the matrix?
          size_t ifirst, ilast;
          radial.get_idx(iel,ifirst,ilast);
          Orad.submat(ifirst,ifirst,ilast,ilast)+=radial.radial_integral(1,iel);
        }

        int gmax(std::max(arma::max(lval),arma::max(mval)));
        gaunt::Gaunt gaunt(gmax,1,gmax);

        // Fill elements
        for(size_t iang=0;iang<lval.n_elem;iang++) {
          int li(lval(iang));
          int mi(mval(iang));

          for(size_t jang=0;jang<lval.n_elem;jang++) {
            int lj(lval(jang));
            int mj(mval(jang));

            // Calculate coupling
            double cpl(gaunt.cosine_coupling(lj,mj,li,mi));
            if(cpl!=0.0)
              set_sub(V,iang,jang,Orad*cpl);
          }
        }

        return remove_boundaries(V);
      }

      arma::mat TwoDBasis::quadrupole_zz() const {
        // Build radial elements
        size_t Nrad(radial.Nbf());
        arma::mat Orad(Nrad,Nrad);
        Orad.zeros();

        // Full electric couplings
        arma::mat V(Ndummy(),Ndummy());
        V.zeros();

        // Loop over elements
        for(size_t iel=0;iel<radial.Nel();iel++) {
          // Where are we in the matrix?
          size_t ifirst, ilast;
          radial.get_idx(iel,ifirst,ilast);
          Orad.submat(ifirst,ifirst,ilast,ilast)+=radial.radial_integral(2,iel);
        }

        int gmax(std::max(arma::max(lval),arma::max(mval)));
        gaunt::Gaunt gaunt(gmax,2,gmax);

        // Fill elements
        for(size_t iang=0;iang<lval.n_elem;iang++) {
          int li(lval(iang));
          int mi(mval(iang));

          for(size_t jang=0;jang<lval.n_elem;jang++) {
            int lj(lval(jang));
            int mj(mval(jang));

            // Calculate coupling
            double cpl(gaunt.coeff(lj,mj,2,0,li,mi));
            if(cpl!=0.0) {
              const double c0(2.0/5.0*sqrt(5.0*M_PI));
              cpl*=c0;
              set_sub(V,iang,jang,Orad*cpl);
            }
          }
        }

        return remove_boundaries(V);
      }

      arma::mat TwoDBasis::Bz_field(double B) const {
        // Build radial elements
        size_t Nrad(radial.Nbf());
        arma::mat O0rad(Nrad,Nrad);
        O0rad.zeros();
        arma::mat O2rad(Nrad,Nrad);
        O2rad.zeros();

        // Loop over elements
        for(size_t iel=0;iel<radial.Nel();iel++) {
          // Where are we in the matrix?
          size_t ifirst, ilast;
          radial.get_idx(iel,ifirst,ilast);
          O0rad.submat(ifirst,ifirst,ilast,ilast)+=radial.radial_integral(0,iel);
          O2rad.submat(ifirst,ifirst,ilast,ilast)+=radial.radial_integral(2,iel);
        }

        // Full coupling
        arma::mat V(Ndummy(),Ndummy());
        V.zeros();

        int gmax(std::max(arma::max(lval),arma::max(mval)));
        gaunt::Gaunt gaunt(gmax,4,gmax);

        // Fill elements
        for(size_t iang=0;iang<lval.n_elem;iang++) {
          int li(lval(iang));
          int mi(mval(iang));

          for(size_t jang=0;jang<lval.n_elem;jang++) {
            int lj(lval(jang));
            int mj(mval(jang));

            // Calculate coupling
            double cpl(gaunt.sine2_coupling(lj,mj,li,mi));
            if(cpl!=0.0) {
              set_sub(V,iang,jang,B*B/8.0*O2rad*cpl);
            }

            if(li==lj && mi==mj) {
              add_sub(V,iang,jang,-0.5*B*mj*O0rad);
            }
          }
        }

        return remove_boundaries(V);
      }

      size_t TwoDBasis::mem_1el() const {
        return Nbf()*Nbf()*sizeof(double);
      }

      size_t TwoDBasis::mem_1el_aux() const {
        size_t Nel(radial.Nel());
        size_t Nprim(radial.max_Nprim());
        size_t N_L(2*arma::max(lval)+1);

        return 2*N_L*Nel*Nprim*Nprim*sizeof(double);
      }

      size_t TwoDBasis::mem_2el_aux() const {
        // Auxiliary integrals required up to
        size_t N_L(2*arma::max(lval)+1);
        // Number of elements
        size_t Nel(radial.Nel());
        // Number of primitive functions per element
        size_t Nprim(radial.max_Nprim());

        // Memory use is thus
        //return 2*N_L*Nel*Nel*Nprim*Nprim*Nprim*Nprim*sizeof(double);
        // No off-diagonal storage
        return 2*N_L*Nel*Nprim*Nprim*Nprim*Nprim*sizeof(double);
      }


      void TwoDBasis::compute_tei(bool exchange) {
        // Number of distinct L values is
        size_t N_L(2*arma::max(lval)+1);
        size_t Nel(radial.Nel());

        // Compute disjoint integrals
        disjoint_L.resize(Nel*N_L);
        disjoint_m1L.resize(Nel*N_L);
        for(size_t L=0;L<N_L;L++)
          for(size_t iel=0;iel<Nel;iel++) {
            disjoint_L[L*Nel+iel]=radial.radial_integral(L,iel);
            disjoint_m1L[L*Nel+iel]=radial.radial_integral(-L-1,iel);
          }

        // Form two-electron integrals
        prim_tei.resize(Nel*Nel*N_L);
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
        for(size_t L=0;L<N_L;L++) {
          for(size_t iel=0;iel<Nel;iel++) {
            // In-element integral
            prim_tei[Nel*Nel*L + iel*Nel + iel]=radial.twoe_integral(L,iel);

            /*
              for(size_t jel=0;jel<Nel;jel++) {
	      // Disjoint integrals. When r(iel)>r(jel), iel gets -1-L, jel gets L.
	      const arma::mat & iint=(iel>jel) ? disjoint_m1L[L*Nel+iel] : disjoint_L[L*Nel+iel];
	      const arma::mat & jint=(iel>jel) ? disjoint_L[L*Nel+jel] : disjoint_m1L[L*Nel+jel];

	      // Store integrals
	      prim_tei[Nel*Nel*L + iel*Nel + jel]=utils::product_tei(iint,jint);
              }
              }
            */
          }
        }

        /*
          The exchange matrix is given by
          K(jk) = (ij|kl) P(il)
          i.e. the complex conjugation hits i and l as
          in the density matrix.

          To get this in the proper order, we permute the integrals
          K(jk) = (jk;il) P(il)
          so we don't have to reform the permutations in the exchange routine.
        */
        if(exchange) {
          prim_ktei.resize(Nel*Nel*N_L);
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
          for(size_t L=0;L<N_L;L++)
            for(size_t iel=0;iel<Nel;iel++) {
              // Diagonal integrals
              size_t Ni(radial.Nprim(iel));
              prim_ktei[Nel*Nel*L + iel*Nel + iel]=utils::exchange_tei(prim_tei[Nel*Nel*L + iel*Nel + iel],Ni,Ni,Ni,Ni);

              // Off-diagonal integrals (not used since faster to contract
              // the integrals in factorized form)
              /*
                for(size_t jel=0;jel<iel;jel++) {
                size_t Nj(radial.Nprim(jel));
                prim_ktei[Nel*Nel*L + iel*Nel + jel]=utils::exchange_tei(prim_tei[Nel*Nel*L + iel*Nel + jel],Ni,Ni,Nj,Nj);
                }
                for(size_t jel=iel+1;jel<Nel;jel++) {
                size_t Nj(radial.Nprim(jel));
                prim_ktei[Nel*Nel*L + iel*Nel + jel]=utils::exchange_tei(prim_tei[Nel*Nel*L + iel*Nel + jel],Ni,Ni,Nj,Nj);
                }
              */
            }
        }
      }

      arma::mat TwoDBasis::coulomb(const arma::mat & P0) const {
        if(!prim_tei.size())
          throw std::logic_error("Primitive teis have not been computed!\n");

        // Extend to boundaries
        arma::mat P(expand_boundaries(P0));

        // Number of radial elements
        size_t Nel(radial.Nel());
        // Number of radial functions
        size_t Nrad(radial.Nbf());
        // Gaunt coefficient table
        int gmax(std::max(arma::max(lval),arma::max(mval)));
        gaunt::Gaunt gaunt(gmax,2*gmax,gmax);

        // maximal M value
        int Mmax=arma::max(mval)-arma::min(mval);

        // Radial helper matrices
        std::vector< std::vector<arma::mat> > Paux(2*arma::max(lval)+1);
        for(int L=0;L<(int) Paux.size();L++) {
          Paux[L].resize(2*Mmax+1);
          for(int M=-std::min(L,Mmax);M<=std::min(L,Mmax);M++) {
            Paux[L][M+Mmax].zeros(Nrad,Nrad);
          }
        }

        // Form radial helpers: contract ket
        for(size_t kang=0;kang<lval.n_elem;kang++) {
          for(size_t lang=0;lang<lval.n_elem;lang++) {
            // l and m values
            int lk(lval(kang));
            int mk(mval(kang));
            int ll(lval(lang));
            int ml(mval(lang));
            // RH m value
            int M(mk-ml);
            // M values match. Loop over possible couplings
            int Lmin=std::max(std::abs(lk-ll),abs(M));
            int Lmax=lk+ll;
            for(int L=Lmin;L<=Lmax;L++) {
              // Calculate coupling coefficient
              double cpl(gaunt.coeff(lk,mk,L,M,ll,ml));
              // Increment
              Paux[L][M+Mmax]+=cpl*P.submat(kang*Nrad,lang*Nrad,(kang+1)*Nrad-1,(lang+1)*Nrad-1);
            }
          }
        }

        // Helper matrices
        std::vector< std::vector<arma::mat> > Jaux(2*arma::max(lval)+1);
        for(int L=0;L<(int) Jaux.size();L++) {
          Jaux[L].resize(2*Mmax+1);
          for(int M=-std::min(L,Mmax);M<=std::min(L,Mmax);M++) {
            Jaux[L][M+Mmax].zeros(Nrad,Nrad);
          }
        }
        // Contract integrals
        for(int L=0;L<(int) Paux.size();L++) {
          const double Lfac=4.0*M_PI/(2*L+1);

          for(int M=-std::min(L,Mmax);M<=std::min(L,Mmax);M++) {
            for(size_t jel=0;jel<Nel;jel++) {
              size_t jfirst, jlast;
              radial.get_idx(jel,jfirst,jlast);
              size_t Nj(jlast-jfirst+1);

              // Get density submatrices
              arma::mat Psub(Paux[L][M+Mmax].submat(jfirst,jfirst,jlast,jlast));

              // Contract integrals
              double jsmall = Lfac*arma::trace(disjoint_L[L*Nel+jel]*Psub);
              double jbig = Lfac*arma::trace(disjoint_m1L[L*Nel+jel]*Psub);

              // Increment J: jel>iel
              for(size_t iel=0;iel<jel;iel++) {
                size_t ifirst, ilast;
                radial.get_idx(iel,ifirst,ilast);

                const arma::mat & iint=disjoint_L[L*Nel+iel];
                Jaux[L][M+Mmax].submat(ifirst,ifirst,ilast,ilast)+=jbig*iint;
              }

              // Increment J: jel<iel
              for(size_t iel=jel+1;iel<Nel;iel++) {
                size_t ifirst, ilast;
                radial.get_idx(iel,ifirst,ilast);

                const arma::mat & iint=disjoint_m1L[L*Nel+iel];
                Jaux[L][M+Mmax].submat(ifirst,ifirst,ilast,ilast)+=jsmall*iint;
              }

              // In-element contribution
              {
                size_t iel=jel;
                size_t ifirst=jfirst;
                size_t ilast=jlast;
                size_t Ni=Nj;

                // Contract integrals
                Psub.reshape(Nj*Nj,1);

                const size_t idx(Nel*Nel*L + iel*Nel + jel);
                arma::mat Jsub(Lfac*(prim_tei[idx]*Psub));
                Jsub.reshape(Ni,Ni);

                Jaux[L][M+Mmax].submat(ifirst,ifirst,ilast,ilast)+=Jsub;
              }
            }
          }
        }

        // Full Coulomb matrix
        arma::mat J(Ndummy(),Ndummy());
        J.zeros();
        for(size_t iang=0;iang<lval.n_elem;iang++) {
          for(size_t jang=0;jang<lval.n_elem;jang++) {
            // l and m values
            int li(lval(iang));
            int mi(mval(iang));
            int lj(lval(jang));
            int mj(mval(jang));
            // LH m value
            int M(mj-mi);

            int Lmin=std::max(std::abs(lj-li),abs(M));
            int Lmax=lj+li;
            for(int L=Lmin;L<=Lmax;L++) {
              // Coupling
              double cpl(gaunt.coeff(lj,mj,L,M,li,mi));
              if(cpl!=0.0) {
                J.submat(iang*Nrad,jang*Nrad,(iang+1)*Nrad-1,(jang+1)*Nrad-1)+=cpl*Jaux[L][M+Mmax];
              }
            }
          }
        }

        return remove_boundaries(J);
      }

      arma::mat TwoDBasis::exchange(const arma::mat & P0) const {
        if(!prim_ktei.size())
          throw std::logic_error("Primitive teis have not been computed!\n");

        // Extend to boundaries
        arma::mat P(expand_boundaries(P0));

        // Gaunt coefficient table
        int gmax(std::max(arma::max(lval),arma::max(mval)));
        gaunt::Gaunt gaunt(gmax,2*gmax,gmax);

        // Number of radial elements
        size_t Nel(radial.Nel());
        // Number of radial basis functions
        size_t Nrad(radial.Nbf());

        // Full exchange matrix
        arma::mat K(Ndummy(),Ndummy());
        K.zeros();

        // Helper memory
#ifdef _OPENMP
        const int nth(omp_get_max_threads());
#else
        const int nth(1);
#endif
        std::vector<arma::vec> mem_Ksub(nth);
        std::vector<arma::vec> mem_Psub(nth);
        std::vector<arma::vec> mem_T(nth);

#ifdef _OPENMP
#pragma omp parallel
#endif
        {
#ifdef _OPENMP
          const int ith(omp_get_thread_num());
#else
          const int ith(0);
#endif
          // These are only small submatrices!
          mem_Psub[ith].zeros(radial.max_Nprim()*radial.max_Nprim());
          mem_Ksub[ith].zeros(radial.max_Nprim()*radial.max_Nprim());
          mem_T[ith].zeros(radial.max_Nprim()*radial.max_Nprim());

          // Increment
#ifdef _OPENMP
#pragma omp for collapse(2)
#endif
          for(size_t jang=0;jang<lval.n_elem;jang++) {
            for(size_t kang=0;kang<lval.n_elem;kang++) {
              int lj(lval(jang));
              int mj(mval(jang));

              int lk(lval(kang));
              int mk(mval(kang));

              // Form radial helpers
              size_t N_L(2*arma::max(lval)+1);
              std::vector<arma::mat> Rmat(N_L);
              for(size_t i=0;i<N_L;i++) {
                Rmat[i].zeros(Nrad,Nrad);
              }
              // Is there a coupling to the channel?
              std::vector<bool> couple(N_L,false);

              // Perform angular sums
              for(size_t iang=0;iang<lval.n_elem;iang++) {
                int li(lval(iang));
                int mi(mval(iang));

                for(size_t lang=0;lang<lval.n_elem;lang++) {
                  int ll(lval(lang));
                  int ml(mval(lang));

                  // LH m value
                  int M(mj-mi);
                  // RH m value
                  int Mp(mk-ml);
                  if(M!=Mp)
                    continue;

                  // Do we have any density in this block?
                  double bdens(arma::norm(P.submat(iang*Nrad,lang*Nrad,(iang+1)*Nrad-1,(lang+1)*Nrad-1),"fro"));
                  //printf("(%i %i) (%i %i) density block norm %e\n",li,mi,ll,ml,bdens);
                  if(bdens<10*DBL_EPSILON)
                    continue;

                  // M values match. Loop over possible couplings
                  int Lmin=std::max(std::max(std::abs(li-lj),std::abs(lk-ll)),abs(M));
                  int Lmax=std::min(li+lj,lk+ll);

                  for(int L=Lmin;L<=Lmax;L++) {
                    // Calculate total coupling coefficient
                    double cpl(gaunt.coeff(lj,mj,L,M,li,mi)*gaunt.coeff(lk,mk,L,M,ll,ml));
                    if(cpl==0.0)
                      continue;

                    // L factor
                    double Lfac=4.0*M_PI/(2*L+1);
                    Rmat[L]+=(Lfac*cpl)*P.submat(iang*Nrad,lang*Nrad,(iang+1)*Nrad-1,(lang+1)*Nrad-1);
                    couple[L]=true;
                  }
                }
              }

              // Loop over elements: output
              for(size_t iel=0;iel<Nel;iel++) {
                size_t ifirst, ilast;
                radial.get_idx(iel,ifirst,ilast);

                // Input
                for(size_t jel=0;jel<Nel;jel++) {
                  size_t jfirst, jlast;
                  radial.get_idx(jel,jfirst,jlast);

                  // Number of functions in the two elements
                  size_t Ni(ilast-ifirst+1);
                  size_t Nj(jlast-jfirst+1);

                  if(iel == jel) {
                    /*
                      The exchange matrix is given by
                      K(jk) = (ij|kl) P(il)
                      i.e. the complex conjugation hits i and l as
                      in the density matrix.

                      To get this in the proper order, we permute the integrals
                      K(jk) = (jk;il) P(il)
                    */

                    // Exchange submatrix
                    arma::mat Ksub(mem_Ksub[ith].memptr(),Ni*Nj,1,false,true);
                    Ksub.zeros();

                    for(size_t L=0;L<N_L;L++) {
                      if(!couple[L])
                        continue;
                      Ksub+=prim_ktei[Nel*Nel*L + iel*Nel + jel]*arma::vectorise(Rmat[L].submat(ifirst,jfirst,ilast,jlast));
                    }
                    Ksub.reshape(Ni,Nj);

                    // Increment global exchange matrix
                    K.submat(jang*Nrad+ifirst,kang*Nrad+jfirst,jang*Nrad+ilast,kang*Nrad+jlast)-=Ksub;

                    //arma::vec Ptgt(arma::vectorise(P.submat(jang*Nrad+ifirst,kang*Nrad+jfirst,jang*Nrad+ilast,kang*Nrad+jlast)));
                    //printf("(%i %i) (%i %i) (%i %i) (%i %i) [%i %i]\n",li,mi,lj,mj,lk,mk,ll,ml,L,M);
                    //printf("Element %i - %i contribution to exchange energy % .10e\n",(int) iel,(int) jel,-0.5*arma::dot(Ksub,Ptgt));

                  } else {
                    // Exchange submatrix
                    arma::mat Ksub(mem_Ksub[ith].memptr(),Ni,Nj,false,true);
                    Ksub.zeros();

                    for(size_t L=0;L<N_L;L++) {
                      if(!couple[L])
                        continue;

                      // Disjoint integrals. When r(iel)>r(jel), iel gets -1-L, jel gets L.
                      const arma::mat & iint=(iel>jel) ? disjoint_m1L[L*Nel+iel] : disjoint_L[L*Nel+iel];
                      const arma::mat & jint=(iel>jel) ? disjoint_L[L*Nel+jel] : disjoint_m1L[L*Nel+jel];

                      // Get density submatrix (Niel x Njel)
                      arma::mat Psub(mem_Psub[ith].memptr(),Ni,Nj,false,true);
                      Psub=Rmat[L].submat(ifirst,jfirst,ilast,jlast);

                      // Calculate helper
                      arma::mat T(mem_T[ith].memptr(),Ni,Nj,false,true);
                      // (Niel x Njel) = (Niel x Njel) x (Njel x Njel)
                      T=Psub*arma::trans(jint);

                      // Increment
                      Ksub+=iint*T;
                    }

                    K.submat(jang*Nrad+ifirst,kang*Nrad+jfirst,jang*Nrad+ilast,kang*Nrad+jlast)-=Ksub;
                  }
                }
              }
            }
          }
        }

        return remove_boundaries(K);
      }

      arma::mat TwoDBasis::remove_boundaries(const arma::mat & Fnob) const {
        if(Fnob.n_rows != Ndummy() || Fnob.n_cols != Ndummy()) {
          std::ostringstream oss;
          oss << "Matrix does not have expected size! Got " << Fnob.n_rows << " x " << Fnob.n_cols << ", expected " << Ndummy() << " x " << Ndummy() << "!\n";
          throw std::logic_error(oss.str());
        }

        // Get indices
        arma::uvec idx(pure_indices());

        // Matrix with the boundary conditions removed
        arma::mat Fpure(Fnob(idx,idx));

        //Fnob.print("Input: w/o built-in boundaries");
        //Fpure.print("Output: w built-in boundaries");

        return Fpure;
      }

      arma::mat TwoDBasis::expand_boundaries(const arma::mat & Ppure) const {
        if(Ppure.n_rows != Nbf() || Ppure.n_cols != Nbf()) {
          std::ostringstream oss;
          oss << "Matrix does not have expected size! Got " << Ppure.n_rows << " x " << Ppure.n_cols << ", expected " << Nbf() << " x " << Nbf() << "!\n";
          throw std::logic_error(oss.str());
        }

        // Get indices
        arma::uvec idx(pure_indices());

        // Matrix with the boundary conditions removed
        arma::mat Pnob(Ndummy(),Ndummy());
        Pnob.zeros();
        Pnob(idx,idx)=Ppure;

        //Ppure.print("Input: w built-in boundaries");
        //Pnob.print("Output: w/o built-in boundaries");

        return Pnob;
      }

      std::vector<arma::mat> TwoDBasis::get_prim_tei() const {
        return prim_tei;
      }

      arma::cx_mat TwoDBasis::eval_bf(size_t iel, double cth, double phi) const {
        // Evaluate spherical harmonics
        arma::cx_vec sph(lval.n_elem);
        for(size_t i=0;i<lval.n_elem;i++)
          sph(i)=::spherical_harmonics(lval(i),mval(i),cth,phi);

        // Evaluate radial functions
        arma::mat rad(radial.get_bf(iel));

        // Form supermatrix
        arma::cx_mat bf(rad.n_rows,lval.n_elem*rad.n_cols);
        for(size_t i=0;i<lval.n_elem;i++)
          bf.cols(i*rad.n_cols,(i+1)*rad.n_cols-1)=sph(i)*rad;

        return bf;
      }

      void TwoDBasis::eval_df(size_t iel, double cth, double phi, arma::cx_mat & dr, arma::cx_mat & dth, arma::cx_mat & dphi) const {
        // Evaluate spherical harmonics
        arma::cx_vec sph(lval.n_elem);
        for(size_t i=0;i<lval.n_elem;i++)
          sph(i)=::spherical_harmonics(lval(i),mval(i),cth,phi);

        // Evaluate radial functions
        arma::mat frad(radial.get_bf(iel));
        arma::mat drad(radial.get_df(iel));

        // Form supermatrices
        dr.zeros(frad.n_rows,lval.n_elem*frad.n_cols);
        dth.zeros(frad.n_rows,lval.n_elem*frad.n_cols);
        dphi.zeros(frad.n_rows,lval.n_elem*frad.n_cols);

        // Radial one is easy
        for(size_t i=0;i<lval.n_elem;i++)
          dr.cols(i*frad.n_cols,(i+1)*frad.n_cols-1)=sph(i)*drad;
        // and so is phi
        for(size_t i=0;i<lval.n_elem;i++)
          dphi.cols(i*frad.n_cols,(i+1)*frad.n_cols-1)=std::complex<double>(0.0,mval(i))*sph(i)*frad;
        // but theta is nastier
        for(size_t i=0;i<lval.n_elem;i++) {
          // cot th = 1/tan th = cos th / sin th
          double cotth=cth/sqrt(1.0-cth*cth);

          int l(lval(i));
          int m(mval(i));

          // Angular factor
          std::complex<double> angfac(m*cotth*sph(i));
          if(mval(i)<lval(i))
            angfac+=sqrt((l-m)*(l+m+1))*std::exp(std::complex<double>(0,-phi))*::spherical_harmonics(lval(i),mval(i)+1,cth,phi);

          dth.cols(i*frad.n_cols,(i+1)*frad.n_cols-1)=angfac*frad;
        }
      }

      arma::uvec TwoDBasis::bf_list(size_t iel) const {
        // Radial functions in element
        size_t ifirst, ilast;
        radial.get_idx(iel,ifirst,ilast);
        // Number of radial functions in element
        size_t Nr(ilast-ifirst+1);

        // Total number of radial functions
        size_t Nrad(radial.Nbf());

        // List of functions in the element
        arma::uvec idx(Nr*lval.n_elem);
        for(size_t iam=0;iam<lval.n_elem;iam++)
          for(size_t j=0;j<Nr;j++)
            idx(iam*Nr+j)=Nrad*iam+ifirst+j;

        //printf("Basis function in element %i\n",(int) iel);
        //idx.print();

        return idx;
      }

      size_t TwoDBasis::get_rad_Nel() const {
        return radial.Nel();
      }

      arma::vec TwoDBasis::get_wrad(size_t iel) const {
        return radial.get_wrad(iel);
      }

      arma::vec TwoDBasis::get_r(size_t iel) const {
        return radial.get_r(iel);
      }

      arma::vec TwoDBasis::nuclear_density(const arma::mat & P0) const {
        // Radial functions in first element
        size_t ifirst, ilast;
        radial.get_idx(0,ifirst,ilast);
        // Total number of radial functions
        size_t Nrad(radial.Nbf());

        // Expand density matrix to boundary conditions
        arma::mat P(expand_boundaries(P0));

        // Loop over angular momentum
        double nucden=0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:nucden)
#endif
        for(size_t iam=0;iam<lval.n_elem;iam++) {
          // Integration over angles yields extra factor 4 pi that must be removed
          nucden+=radial.nuclear_density(P.submat(Nrad*iam,Nrad*iam,Nrad*iam+ilast,Nrad*iam+ilast))/(4.0*M_PI);
        }

        arma::vec den(1);
        den(0)=nucden;

        return den;
      }
    }
  }
}
