#include "basis.h"
#include "quadrature.h"
#include "../general/polynomial.h"
#include "../general/chebyshev.h"
#include "../general/gaunt.h"
#include <cassert>
#include <cfloat>

/// Use quadrature for one-electron integrals?
#define OEI_QUADRATURE
/// Use quadrature for two-electron integrals?
#define TEI_QUADRATURE
/// Use Cholesky for Sinvh?
//#define CHOL_SINVH

namespace helfem {
  namespace basis {
    RadialBasis::RadialBasis() {
    }

    RadialBasis::RadialBasis(int n_nodes, int der_order, int n_quad, int num_el, double rmax, int igrid, double zexp) {
      // Get primitive polynomial representation
      bf_C=polynomial::hermite_coeffs(n_nodes, der_order);
      df_C=polynomial::derivative_coeffs(bf_C, 1);

      // Get quadrature rule
      chebyshev::chebyshev(n_quad,xq,wq);
      for(size_t i=0;i<xq.n_elem;i++) {
        if(!std::isfinite(xq[i]))
          printf("xq[%i]=%e\n",(int) i, xq[i]);
        if(!std::isfinite(wq[i]))
          printf("wq[%i]=%e\n",(int) i, wq[i]);
      }

      // Evaluate polynomials at quadrature points
      bf=polynomial::polyval(bf_C,xq);
      df=polynomial::polyval(df_C,xq);

      // Number of overlapping functions is
      noverlap=der_order+1;

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

      //bval.print("Element boundaries");

#ifdef OEI_QUADRATURE
      printf("One-electron integrals evaluated using quadrature.\n");
#else
      printf("One-electron integrals evaluated analytically.\n");
#endif
#ifdef TEI_QUADRATURE
      printf("Two-electron integrals evaluated using quadrature.\n");
#else
      printf("Two-electron integrals evaluated analytically.\n");
#endif
    }

    RadialBasis::~RadialBasis() {
    }

    arma::mat RadialBasis::get_basis(const arma::mat & bas, size_t iel) const {
      if(iel==0 && iel==bval.n_elem-2)
	// Boundary condition both at r=0 and at r=infinity
	return bas.cols(noverlap,bf.n_cols-1-noverlap);
      else if(iel==0)
	// Boundary condition at r=0
	return bas.cols(noverlap,bf.n_cols-1);
      else if(iel==bval.n_elem-2)
	// Boundary condition at r=infinity
	return bas.cols(0,bf.n_cols-1-noverlap);
      else
	return bas;
    }


    size_t RadialBasis::get_noverlap() const {
      return noverlap;
    }

    size_t RadialBasis::Nel() const {
      // Number of elements is
      return bval.n_elem-1;
    }

    size_t RadialBasis::Nbf() const {
      // The number of basis functions is Nbf*Nel - (Nel-1)*Noverlap -
      // 2*Noverlap or just
      return Nel()*(bf.n_cols-noverlap)-noverlap;
    }

    size_t RadialBasis::Nprim(size_t iel) const {
      if(iel==0 && iel==bval.n_elem-2)
	return bf.n_cols-2*noverlap;
      else if(iel==0)
	return bf.n_cols-noverlap;
      else if(iel==bval.n_elem-2)
	return bf.n_cols-noverlap;
      else
	return bf.n_cols;
    }

    void RadialBasis::get_idx(size_t iel, size_t & ifirst, size_t & ilast) const {
      // The first function in the element will be
      ifirst=iel*(bf.n_cols-noverlap);
      // and the last one will be
      ilast=ifirst+bf.n_cols-1;

      // Account for the functions deleted at the origin
      ilast-=noverlap;
      if(iel>0)
	ifirst-=noverlap;

      // Last element does not have trailing functions
      if(iel==bval.n_elem-2)
        ilast-=noverlap;
    }

    arma::mat RadialBasis::radial_integral(int Rexp, size_t iel) const {
      return radial_integral(bf_C,Rexp,iel);
    }

#ifndef OEI_QUADRATURE
    static double primrad_int(int n, double Rmin, double Rmax) {
      if(n==-1)
        return log(Rmax)-log(Rmin);
      else
        return (std::pow(Rmax,n+1)-std::pow(Rmin,n+1))/(n+1);
    }
#endif

    arma::mat RadialBasis::radial_integral(const arma::mat & bf_cexp, int Rexp, size_t iel) const {
      double Rmin(bval(iel));
      double Rmax(bval(iel+1));

#ifdef OEI_QUADRATURE
      // Integral by quadrature
      return quadrature::radial_integral(Rmin,Rmax,Rexp,xq,wq,get_basis(polynomial::polyval(bf_cexp,xq),iel));
#else
      // Coefficients
      arma::mat C(polynomial::convert_coeffs(get_basis(bf_cexp,iel),Rmin,Rmax));
      size_t Nx(C.n_rows);

      // Primitive integrals
      arma::mat primint(Nx,Nx);
      for(size_t i=0;i<Nx;i++)
        for(size_t j=0;j<Nx;j++)
          primint(i,j)=primrad_int(i+j+Rexp,Rmin,Rmax);

      primint.print("primitive integrals");

      // Set any diverging integrals to zero
      for(size_t i=0;i<Nx;i++)
        for(size_t j=0;j<Nx;j++)
          if(!std::isnormal(primint(i,j)))
            primint(i,j)=0.0;

      // Analytical integrals
      arma::mat anal(C.t()*primint*C);
      anal.print("analytical");

      arma::mat quad(quadrature::radial_integral(Rmin,Rmax,Rexp,xq,wq,get_basis(polynomial::polyval(bf_cexp,xq),iel)));
      quad.print("quadrature");

      arma::mat diff(quad-anal);
      printf("Error in analytical integral for Rexp=%i in element %i is %e\n",Rexp,(int) iel,arma::norm(diff,"fro"));

      return (anal+anal.t())/2.0;
#endif
    }

    arma::mat RadialBasis::kinetic(size_t iel) const {
      // We get 1/rlen^2 from the derivatives
      double rlen((bval(iel+1)-bval(iel))/2);

      return 0.5*radial_integral(df_C,0,iel)/(rlen*rlen);
    }

    arma::mat RadialBasis::kinetic_l(size_t iel) const {
      return 0.5*radial_integral(-2,iel);
    }

    arma::mat RadialBasis::nuclear(size_t iel) const {
      return -radial_integral(-1,iel);
    }

#ifndef TEI_QUADRATURE
    // Two-electron primitive integral
    static double primitive_tei(double rmin, double rmax, int k, int l) {
      assert(l!=-1);
      if(k==-1)
        return 1.0/(l+1)*( 1.0/(l+1)*(std::pow(rmax,l+1)-std::pow(rmin,l+1)) - std::pow(rmin,l+1)*log(rmax/rmin));
      else {
        assert(k+l+1!=-1);
        return 1.0/(l+1)*( 1.0/(k+l+2)*(std::pow(rmax,k+l+2)-std::pow(rmin,k+l+2)) - 1.0/(k+1)*std::pow(rmin,l+1)*(std::pow(rmax,k+1)-std::pow(rmin,k+1)));
      }
    }
#endif

    arma::mat RadialBasis::twoe_integral(int L, size_t iel) const {
      double Rmin(bval(iel));
      double Rmax(bval(iel+1));

#ifdef TEI_QUADRATURE
      // Integral by quadrature
      return quadrature::twoe_integral(Rmin,Rmax,xq,wq,get_basis(bf_C,iel),L);
#else
      arma::mat quad(quadrature::twoe_integral(Rmin,Rmax,xq,wq,get_basis(bf_C,iel),L));

      // Coefficients
      arma::mat C(polynomial::convert_coeffs(get_basis(bf_C,iel),Rmin,Rmax));
      size_t Nx(C.n_rows);
      size_t N(C.n_cols);

      // Primitive integrals
      arma::mat primint(2*Nx,2*Nx);
      for(size_t ij=0;ij<2*Nx;ij++)
        for(size_t kl=0;kl<2*Nx;kl++)
          primint(ij,kl)=primitive_tei(Rmin,Rmax,ij-L-1,kl+L);

      // Set any diverging primitive integrals to zero
      for(size_t ij=0;ij<2*Nx;ij++)
        for(size_t kl=0;kl<2*Nx;kl++)
          if(!std::isnormal(primint(ij,kl)))
            primint(ij,kl)=0.0;

      // and the factor
      primint*=4.0*M_PI/(2*L+1);

      // Half-transformed eris
      arma::mat heri(N*N,2*Nx);
      for(size_t fi=0;fi<N;fi++)
        for(size_t fj=0;fj<N;fj++)
          for(size_t kl=0;kl<2*Nx;kl++)
              {
                double el=0.0;
                for(size_t pi=0;pi<Nx;pi++)
                  for(size_t pj=0;pj<Nx;pj++)
                    el+=primint(pi+pj,kl)*C(pi,fi)*C(pj,fj);
                heri(fj*N+fi,kl)=el;
              }

      // Full transform
      arma::mat eri(N*N,N*N);
      for(size_t fk=0;fk<N;fk++)
        for(size_t fl=0;fl<N;fl++)
          for(size_t fi=0;fi<N;fi++)
            for(size_t fj=0;fj<N;fj++)
              {
                double el=0.0;
                for(size_t pk=0;pk<Nx;pk++)
                  for(size_t pl=0;pl<Nx;pl++)
                    el+=heri(fj*N+fi,pk+pl)*C(pk,fk)*C(pl,fl);
                eri(fj*N+fi,fl*N+fk)=el;
              }

      // Add in other half
      eri+=eri.t();

      eri.print("Analytical integrals");
      quad.print("Quadrature integrals");

      return eri;
#endif
    }

    TwoDBasis::TwoDBasis(int Z_, int n_nodes, int der_order, int n_quad, int num_el, double rmax, int lmax, int mmax, int igrid, double zexp) {
      // Nuclear charge
      Z=Z_;
      // Construct radial basis
      radial=RadialBasis(n_nodes, der_order, n_quad, num_el, rmax, igrid, zexp);

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

    size_t TwoDBasis::Nbf() const {
      return lval.n_elem*radial.Nbf();
    }

    size_t TwoDBasis::angular_nbf(size_t iam) const {
      /*
      if(lval(iam)==0)
	return radial.Nbf();
      else
	// Boundary condition with l>0: remove first noverlap functions!
	return radial.Nbf()-radial.get_noverlap();
      */
      return radial.Nbf();
    }

    size_t TwoDBasis::angular_offset(size_t iam) const {
      size_t ioff=0;
      for(size_t i=0;i<iam;i++)
	ioff+=angular_nbf(i);
      return ioff;
    }

    arma::mat TwoDBasis::Sinvh() const {
      // Form overlap matrix
      arma::mat S(overlap());

      // Get the basis function norms
      arma::vec bfnormlz(arma::pow(arma::diagvec(S),-0.5));
      // Go to normalized basis
      S=arma::diagmat(bfnormlz)*S*arma::diagmat(bfnormlz);

#ifdef CHOL_SINVH
      // Half-inverse is
      return arma::diagmat(bfnormlz) * arma::inv(arma::chol(S));
#else
      arma::vec Sval;
      arma::mat Svec;
      if(!arma::eig_sym(Sval,Svec,S)) {
        S.save("S.dat",arma::raw_ascii);
        throw std::logic_error("Diagonalization of overlap matrix failed\n");
      }
      printf("Smallest eigenvalue of overlap matrix is % e, condition number %e\n",Sval(0),Sval(Sval.n_elem-1)/Sval(0));

      arma::mat Sinvh(Svec*arma::diagmat(arma::pow(Sval,-0.5))*arma::trans(Svec));
      Sinvh=arma::diagmat(bfnormlz)*Sinvh;

      return Sinvh;
#endif
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
      arma::mat O(Nbf(),Nbf());
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
      arma::mat T(Nbf(),Nbf());
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
      return -Z*radial_integral(-1);
    }

    void TwoDBasis::compute_tei() {
      // Number of distinct L values is
      size_t N_L(2*arma::max(lval)+1);
      size_t Nel(radial.Nel());

      // Compute disjoint integrals
      std::vector<arma::mat> disjoint_L, disjoint_m1L;
      disjoint_L.resize(Nel*N_L);
      disjoint_m1L.resize(Nel*N_L);
      for(size_t L=0;L<N_L;L++)
        for(size_t iel=0;iel<Nel;iel++) {
          disjoint_L[L*Nel+iel]=radial.radial_integral(L,iel);
          disjoint_m1L[L*Nel+iel]=radial.radial_integral(-1-L,iel);
        }

      // Form two-electron integrals
      prim_tei.resize(Nel*Nel*N_L);
      for(size_t L=0;L<N_L;L++) {
        // Normalization factor
        double Lfac=4.0*M_PI/(2*L+1);

        for(size_t iel=0;iel<Nel;iel++) {
          for(size_t jel=0;jel<Nel;jel++) {
	    if(iel==jel) {
	      // In-element integral
	      prim_tei[Nel*Nel*L + iel*Nel + iel]=radial.twoe_integral(L,iel);
	    } else {
	      // Disjoint integrals
	      size_t Ni(radial.Nprim(iel));
	      size_t Nj(radial.Nprim(jel));

	      arma::mat teiblock(Ni*Ni,Nj*Nj);
	      teiblock.zeros();

	      // when r(iel)>r(jel), iel gets -1-L, jel gets L.
	      const arma::mat & iint=(iel>jel) ? disjoint_m1L[L*Nel+iel] : disjoint_L[L*Nel+iel];
	      const arma::mat & jint=(iel>jel) ? disjoint_L[L*Nel+jel] : disjoint_m1L[L*Nel+jel];

	      // Form block
	      for(size_t fk=0;fk<Nj;fk++)
		for(size_t fl=0;fl<Nj;fl++) {
		  // Collect integral in temp variable and put the weight in
		  double klint(Lfac*jint(fk,fl));

		  for(size_t fi=0;fi<Ni;fi++)
		    for(size_t fj=0;fj<Ni;fj++)
		      // (ij|kl) in Armadillo compatible indexing
		      teiblock(fj*Ni+fi,fl*Nj+fk)=klint*iint(fi,fj);
		}

	      // Store integrals
	      prim_tei[Nel*Nel*L + iel*Nel + jel]=teiblock;
	    }
          }
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
      prim_ktei.resize(Nel*Nel*N_L);
      for(size_t L=0;L<N_L;L++)
        for(size_t iel=0;iel<Nel;iel++)
          for(size_t jel=0;jel<Nel;jel++) {
            size_t Ni(radial.Nprim(iel));
            size_t Nj(radial.Nprim(jel));

	    arma::mat tei(Ni*Nj,Ni*Nj);
	    tei.zeros();
	    const arma::mat & ptei(prim_tei[Nel*Nel*L + iel*Nel+jel]);
	    for(size_t ii=0;ii<Ni;ii++)
	      for(size_t jj=0;jj<Ni;jj++)
		for(size_t kk=0;kk<Nj;kk++)
		  for(size_t ll=0;ll<Nj;ll++)
		    // (ik|jl) in Armadillo compatible indexing
		    tei(kk*Ni+jj,ll*Ni+ii)=ptei(jj*Ni+ii,ll*Nj+kk);

	    prim_ktei[Nel*Nel*L + iel*Nel + jel]=tei;
	    // For some reason this gives the wrong answer
	    //prim_ktei[Nel*Nel*L + jel*Nel + iel]=arma::trans(tei);
	  }
    }

    arma::mat TwoDBasis::coulomb(const arma::mat & P0) const {
      if(!prim_tei.size())
        throw std::logic_error("Primitive teis have not been computed!\n");

      // Extend to boundaries
      arma::mat P(expand_boundaries(P0));
      if(P.n_rows != Nbf())
	throw std::logic_error("Density matrix has incorrect size!\n");
      if(P.n_cols != Nbf())
	throw std::logic_error("Density matrix has incorrect size!\n");

      // Gaunt coefficient table
      int lmax(arma::max(lval));
      gaunt::Gaunt gaunt(lmax,2*lmax,lmax);

      // Number of radial elements
      size_t Nel(radial.Nel());
      // Number of radial functions
      size_t Nrad(radial.Nbf());

      // Full Coulomb matrix
      arma::mat J(Nbf(),Nbf());
      J.zeros();

      // Increment
      for(size_t iang=0;iang<lval.n_elem;iang++) {
        int li(lval(iang));
        int mi(mval(iang));

        for(size_t jang=0;jang<lval.n_elem;jang++) {
          int lj(lval(jang));
          int mj(mval(jang));
          // LH m value
          int M(mj-mi);

          for(size_t kang=0;kang<lval.n_elem;kang++) {
            int lk(lval(kang));
            int mk(mval(kang));

            for(size_t lang=0;lang<lval.n_elem;lang++) {
              int ll(lval(lang));
              int ml(mval(lang));

              // RH m value
              int Mp(mk-ml);
              if(M!=Mp)
                continue;

              // M values match. Loop over possible couplings
              int Lmin=std::max(std::max(std::abs(li-lj),std::abs(lk-ll)),abs(M));
              int Lmax=std::min(li+lj,lk+ll);

              for(int L=Lmin;L<=Lmax;L++) {
                // Calculate total coupling coefficient
                double cpl(gaunt.coeff(lj,mj,L,M,li,mi)*gaunt.coeff(lk,mk,L,M,ll,ml));

                if(cpl!=0.0) {
                  // Loop over elements: output
                  for(size_t iel=0;iel<Nel;iel++) {
                    size_t ifirst, ilast;
		    radial.get_idx(iel,ifirst,ilast);

                    // Input
                    for(size_t jel=0;jel<Nel;jel++) {
                      size_t jfirst, jlast;
                      radial.get_idx(jel,jfirst,jlast);

		      // Get density submatrix
		      arma::vec Psub(arma::vectorise(P.submat(kang*Nrad+jfirst,lang*Nrad+jfirst,kang*Nrad+jlast,lang*Nrad+jlast)));

		      // Contract integrals
		      arma::vec Jsub(cpl*(prim_tei[Nel*Nel*L + iel*Nel + jel]*Psub));

		      // Increment global Coulomb matrix
		      J.submat(iang*Nrad+ifirst,jang*Nrad+ifirst,iang*Nrad+ilast,jang*Nrad+ilast)+=arma::reshape(Jsub,ilast-ifirst+1,ilast-ifirst+1);

		      //arma::vec Ptgt(arma::vectorise(P.submat(iang*Nrad+ifirst,jang*Nrad+ifirst,iang*Nrad+ilast,jang*Nrad+ilast)));
		      //printf("(%i %i) (%i %i) (%i %i) (%i %i) [%i %i]\n",li,mi,lj,mj,lk,mk,ll,ml,L,M);
		      //printf("Element %i - %i contribution to Coulomb energy % .10e\n",(int) iel,(int) jel,0.5*arma::dot(Jsub,Ptgt));
                    }
                  }
                }
              }
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

      if(P.n_rows != Nbf())
	throw std::logic_error("Density matrix has incorrect size!\n");
      if(P.n_cols != Nbf())
	throw std::logic_error("Density matrix has incorrect size!\n");

      // Gaunt coefficient table
      int lmax(arma::max(lval));
      gaunt::Gaunt gaunt(lmax,2*lmax,lmax);

      // Number of radial elements
      size_t Nel(radial.Nel());
      // Number of radial basis functions
      size_t Nrad(radial.Nbf());

      // Full exchange matrix
      arma::mat K(Nbf(),Nbf());
      K.zeros();

      // Increment
      for(size_t iang=0;iang<lval.n_elem;iang++) {
        int li(lval(iang));
        int mi(mval(iang));

        for(size_t jang=0;jang<lval.n_elem;jang++) {
          int lj(lval(jang));
          int mj(mval(jang));
          // LH m value
          int M(mj-mi);

          for(size_t kang=0;kang<lval.n_elem;kang++) {
            int lk(lval(kang));
            int mk(mval(kang));

            for(size_t lang=0;lang<lval.n_elem;lang++) {
              int ll(lval(lang));
              int ml(mval(lang));

              // RH m value
              int Mp(mk-ml);
              if(M!=Mp)
                continue;

              // M values match. Loop over possible couplings
              int Lmin=std::max(std::max(std::abs(li-lj),std::abs(lk-ll)),abs(M));
              int Lmax=std::min(li+lj,lk+ll);

              for(int L=Lmin;L<=Lmax;L++) {
                // Calculate total coupling coefficient
                double cpl(gaunt.coeff(lj,mj,L,M,li,mi)*gaunt.coeff(lk,mk,L,M,ll,ml));

                if(cpl!=0.0) {
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

		      /*
			The exchange matrix is given by
			K(jk) = (ij|kl) P(il)
			i.e. the complex conjugation hits i and l as
			in the density matrix.

			To get this in the proper order, we permute the integrals
			K(jk) = (jk;il) P(il)
		      */

                      // Get density submatrix
                      arma::vec Psub(arma::vectorise(P.submat(iang*Nrad+ifirst,lang*Nrad+jfirst,iang*Nrad+ilast,lang*Nrad+jlast)));

		      arma::mat tei(Ni*Nj,Ni*Nj);
		      tei.zeros();
		      const arma::mat & ptei(prim_tei[Nel*Nel*L + iel*Nel+jel]);
		      for(size_t ii=0;ii<Ni;ii++)
			for(size_t jj=0;jj<Ni;jj++)
			  for(size_t kk=0;kk<Nj;kk++)
			    for(size_t ll=0;ll<Nj;ll++)
			      // (ik|jl) in Armadillo compatible indexing
			      tei(kk*Ni+jj,ll*Ni+ii)=ptei(jj*Ni+ii,ll*Nj+kk);

		      // Exchange submatrix
                      arma::vec Ksub(cpl*(prim_ktei[Nel*Nel*L + iel*Nel + jel]*Psub));

                      // Increment global exchange matrix
                      K.submat(jang*Nrad+ifirst,kang*Nrad+jfirst,jang*Nrad+ilast,kang*Nrad+jlast)-=arma::reshape(Ksub,Ni,Nj);

		      //arma::vec Ptgt(arma::vectorise(P.submat(jang*Nrad+ifirst,kang*Nrad+jfirst,jang*Nrad+ilast,kang*Nrad+jlast)));
		      //printf("(%i %i) (%i %i) (%i %i) (%i %i) [%i %i]\n",li,mi,lj,mj,lk,mk,ll,ml,L,M);
		      //printf("Element %i - %i contribution to exchange energy % .10e\n",(int) iel,(int) jel,-0.5*arma::dot(Ksub,Ptgt));
                    }
                  }
                }
              }
            }
          }
        }
      }

      return remove_boundaries(K);
    }

    arma::mat TwoDBasis::remove_boundaries(const arma::mat & Fnob) const {
      // Determine how many functions we need
      size_t Npure=angular_offset(lval.n_elem);
      // Number of functions in radial basis
      size_t Nrad=radial.Nbf();
      if(Fnob.n_rows != Nrad*lval.n_elem || Fnob.n_cols != Nrad*lval.n_elem) {
	std::ostringstream oss;
	oss << "Matrix does not have expected size! Got " << Fnob.n_rows << " x " << Fnob.n_cols << ", expected " << Nrad*lval.n_elem << " x " << Nrad*lval.n_elem << "!\n";
	throw std::logic_error(oss.str());
      }

      // Matrix with the boundary conditions removed
      arma::mat Fpure(Npure,Npure);
      Fpure.zeros();

      // Fill matrix
      for(size_t iang=0;iang<lval.n_elem;iang++)
	for(size_t jang=0;jang<lval.n_elem;jang++) {
	  // Offset
	  size_t ioff=angular_offset(iang);
	  size_t joff=angular_offset(jang);

	  // Number of radial functions on the shells
	  size_t ni=angular_nbf(iang);
	  size_t nj=angular_nbf(jang);

	  // Sanity check for trivial case
	  if(!ni) continue;
	  if(!nj) continue;

	  Fpure.submat(ioff,joff,ioff+ni-1,joff+nj-1)=Fnob.submat((iang+1)*Nrad-ni,(jang+1)*Nrad-nj,(iang+1)*Nrad-1,(jang+1)*Nrad-1);
	}

      //Fnob.print("Input: w/o built-in boundaries");
      //Fpure.print("Output: w built-in boundaries");

      return Fpure;
    }

    arma::mat TwoDBasis::expand_boundaries(const arma::mat & Ppure) const {
      // Determine how many functions we need
      size_t Npure=angular_offset(lval.n_elem);
      // Number of functions in radial basis
      size_t Nrad=radial.Nbf();

      if(Ppure.n_rows != Npure || Ppure.n_cols != Npure) {
	std::ostringstream oss;
	oss << "Matrix does not have expected size! Got " << Ppure.n_rows << " x " << Ppure.n_cols << ", expected " << Npure << " x " << Npure << "!\n";
	throw std::logic_error(oss.str());
      }

      // Matrix with the boundary conditions removed
      arma::mat Pnob(Nrad*lval.n_elem,Nrad*lval.n_elem);
      Pnob.zeros();

      // Fill matrix
      for(size_t iang=0;iang<lval.n_elem;iang++)
	for(size_t jang=0;jang<lval.n_elem;jang++) {
	  // Offset
	  size_t ioff=angular_offset(iang);
	  size_t joff=angular_offset(jang);

	  // Number of radial functions on the shells
	  size_t ni=angular_nbf(iang);
	  size_t nj=angular_nbf(jang);

	  // Sanity check for trivial case
	  if(!ni) continue;
	  if(!nj) continue;

	  Pnob.submat((iang+1)*Nrad-ni,(jang+1)*Nrad-nj,(iang+1)*Nrad-1,(jang+1)*Nrad-1)=Ppure.submat(ioff,joff,ioff+ni-1,joff+nj-1);
	}

      //Ppure.print("Input: w built-in boundaries");
      //Pnob.print("Output: w/o built-in boundaries");

      return Pnob;
    }

    std::vector<arma::mat> TwoDBasis::get_prim_tei() const {
      return prim_tei;
    }
  }
}
