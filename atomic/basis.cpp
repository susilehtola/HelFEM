#include "basis.h"
#include "quadrature.h"
#include "../general/polynomial.h"
#include "../general/chebyshev.h"
#include "../general/spherical_harmonics.h"
#include "../general/gaunt.h"
#include "../general/utils.h"
#include <cassert>
#include <cfloat>

#ifdef _OPENMP
#include <omp.h>
#endif

/// Use quadrature for one-electron integrals?
#define OEI_QUADRATURE
/// Use quadrature for two-electron integrals?
#define TEI_QUADRATURE

namespace helfem {
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
      bval=get_grid(rmax,num_el,igrid,zexp);

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

    RadialBasis::RadialBasis(int n_nodes, int der_order, int n_quad, int num_el0, int Zm, int Zlr, double Rhalf, int num_el, double rmax, int igrid, double zexp) {
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

    size_t RadialBasis::max_Nprim() const {
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

    arma::mat RadialBasis::nuclear_offcenter(size_t iel, double Rhalf, int L) const {
      if(bval(iel)>=Rhalf)
        return -sqrt(4.0*M_PI/(2*L+1))*radial_integral(-L-1,iel)*std::pow(Rhalf,L);
      else if(bval(iel+1)<=Rhalf)
        return -sqrt(4.0*M_PI/(2*L+1))*radial_integral(L,iel)*std::pow(Rhalf,-L-1);
      else
        throw std::logic_error("Nucleus placed within element!\n");
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
      double rmid=(rmax+rmin)/2;
      double rlen=(rmax-rmin)/2;
      arma::vec r(rmid*arma::ones<arma::vec>(xq.n_elem)+rlen*xq);
      return rlen*wq%arma::square(r);
    }

    arma::vec RadialBasis::get_r(size_t iel) const {
      // Full radial weight
      double rmin(bval(iel));
      double rmax(bval(iel+1));
      double rmid=(rmax+rmin)/2;
      double rlen=(rmax-rmin)/2;

      return rmid*arma::ones<arma::vec>(xq.n_elem)+rlen*xq;
    }

    TwoDBasis::TwoDBasis() {
    }

    TwoDBasis::TwoDBasis(int Z_, int n_nodes, int der_order, int n_quad, int num_el, double rmax, int lmax, int mmax, int igrid, double zexp) {
      // Nuclear charge
      Z=Z_;
      Zl=0;
      Zr=0;
      Rhalf=0.0;

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

    TwoDBasis::TwoDBasis(int Z_, int n_nodes, int der_order, int n_quad, int num_el0, int num_el, double rmax, int lmax, int mmax, int igrid, double zexp, int Zl_, int Zr_, double Rhalf_) {
      // Nuclear charge
      Z=Z_;
      Zl=Zl_;
      Zr=Zr_;
      Rhalf=Rhalf_;

      // Construct radial basis
      radial=RadialBasis(n_nodes, der_order, n_quad, num_el0, Z_, std::max(Zl_,Zr_), Rhalf, num_el, rmax, igrid, zexp);

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

    size_t TwoDBasis::Nrad() const {
      return radial.Nbf();
    }

    size_t TwoDBasis::Nang() const {
      return lval.n_elem;
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

    arma::mat TwoDBasis::Sinvh(bool chol) const {
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
        if(!arma::eig_sym(Sval,Svec,S)) {
          S.save("S.dat",arma::raw_ascii);
          throw std::logic_error("Diagonalization of overlap matrix failed\n");
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
      // Full nuclear attraction matrix
      arma::mat V(Nbf(),Nbf());
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

    arma::mat TwoDBasis::dipole_z() const {
      // Build radial elements
      size_t Nrad(radial.Nbf());
      arma::mat Orad(Nrad,Nrad);
      Orad.zeros();

      // Full electric couplings
      arma::mat V(Nbf(),Nbf());
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
      arma::mat V(Nbf(),Nbf());
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


    void TwoDBasis::compute_tei() {
      // Number of distinct L values is
      size_t N_L(2*arma::max(lval)+1);
      size_t Nel(radial.Nel());

      // Compute disjoint integrals
      disjoint_L.resize(Nel*N_L);
      disjoint_m1L.resize(Nel*N_L);
      for(size_t L=0;L<N_L;L++)
        for(size_t iel=0;iel<Nel;iel++) {
          disjoint_L[L*Nel+iel]=radial.radial_integral(L,iel);
          disjoint_m1L[L*Nel+iel]=radial.radial_integral(-1-L,iel);
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
	      // Normalization factor
	      double Lfac=4.0*M_PI/(2*L+1);

	      // Disjoint integrals. When r(iel)>r(jel), iel gets -1-L, jel gets L.
	      const arma::mat & iint=(iel>jel) ? disjoint_m1L[L*Nel+iel] : disjoint_L[L*Nel+iel];
	      const arma::mat & jint=(iel>jel) ? disjoint_L[L*Nel+jel] : disjoint_m1L[L*Nel+jel];

	      // Store integrals
	      prim_tei[Nel*Nel*L + iel*Nel + jel]=utils::product_tei(Lfac*iint,jint);
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
      int gmax(std::max(arma::max(lval),arma::max(mval)));
      gaunt::Gaunt gaunt(gmax,2*gmax,gmax);

      // Number of radial elements
      size_t Nel(radial.Nel());
      // Number of radial functions
      size_t Nrad(radial.Nbf());

      // Full Coulomb matrix
      arma::mat J(Nbf(),Nbf());
      J.zeros();

      // Helper memory
#ifdef _OPENMP
      const int nth(omp_get_max_threads());
#else
      const int nth(1);
#endif
      std::vector<arma::vec> mem_Jsub(nth);
      std::vector<arma::vec> mem_Psub(nth);

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
        mem_Jsub[ith].zeros(radial.max_Nprim()*radial.max_Nprim());

        // Increment
#ifdef _OPENMP
#pragma omp for collapse(2)
#endif
        for(size_t iang=0;iang<lval.n_elem;iang++) {
          for(size_t jang=0;jang<lval.n_elem;jang++) {
            int li(lval(iang));
            int mi(mval(iang));

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

                // Do we have any density in this block?
                double bdens(arma::norm(P.submat(kang*Nrad,lang*Nrad,(kang+1)*Nrad-1,(lang+1)*Nrad-1),"fro"));
                //printf("(%i %i) (%i %i) density block norm %e\n",lk,mk,ll,ml,bdens);
                if(bdens<10*DBL_EPSILON)
                  continue;

                // M values match. Loop over possible couplings
                int Lmin=std::max(std::max(std::abs(li-lj),std::abs(lk-ll)),abs(M));
                int Lmax=std::min(li+lj,lk+ll);

                for(int L=Lmin;L<=Lmax;L++) {
                  // Calculate total coupling coefficient
                  double cpl(gaunt.coeff(lj,mj,L,M,li,mi)*gaunt.coeff(lk,mk,L,M,ll,ml));

                  if(cpl!=0.0) {
		    // Loop over input elements
		    for(size_t jel=0;jel<Nel;jel++) {
		      size_t jfirst, jlast;
		      radial.get_idx(jel,jfirst,jlast);

		      size_t Nj(jlast-jfirst+1);
		      double Lfac=4.0*M_PI/(2*L+1);

		      // Get density submatrix
		      arma::mat Psub(mem_Psub[ith].memptr(),Nj,Nj,false,true);
		      Psub=P.submat(kang*Nrad+jfirst,lang*Nrad+jfirst,kang*Nrad+jlast,lang*Nrad+jlast);
		      // Don't calculate zeros
		      if(arma::norm(Psub,2)==0.0)
			continue;

		      // Contract integrals
		      double jsmall(cpl*Lfac*arma::trace(disjoint_L[L*Nel+jel]*Psub));
		      double jbig(cpl*Lfac*arma::trace(disjoint_m1L[L*Nel+jel]*Psub));

		      // Increment J: jel>iel
		      for(size_t iel=0;iel<jel;iel++) {
			size_t ifirst, ilast;
			radial.get_idx(iel,ifirst,ilast);

			const arma::mat & iint=disjoint_L[L*Nel+iel];
			J.submat(iang*Nrad+ifirst,jang*Nrad+ifirst,iang*Nrad+ilast,jang*Nrad+ilast)+=jbig*iint;
		      }

		      // Increment J: jel<iel
		      for(size_t iel=jel+1;iel<Nel;iel++) {
			size_t ifirst, ilast;
			radial.get_idx(iel,ifirst,ilast);

			const arma::mat & iint=disjoint_m1L[L*Nel+iel];
			J.submat(iang*Nrad+ifirst,jang*Nrad+ifirst,iang*Nrad+ilast,jang*Nrad+ilast)+=jsmall*iint;
		      }

		      // In-element contribution
		      {
			size_t iel=jel;
			size_t ifirst=jfirst;
			size_t ilast=jlast;
			size_t Ni=Nj;

			// Contract integrals
			arma::mat Jsub(mem_Jsub[ith].memptr(),Ni*Ni,1,false,true);
			Psub.reshape(Nj*Nj,1);
			Jsub=cpl*(prim_tei[Nel*Nel*L + iel*Nel + jel]*Psub);
			Jsub.reshape(Ni,Ni);

			// Increment global Coulomb matrix
			J.submat(iang*Nrad+ifirst,jang*Nrad+ifirst,iang*Nrad+ilast,jang*Nrad+ilast)+=Jsub;
		      }
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
      int gmax(std::max(arma::max(lval),arma::max(mval)));
      gaunt::Gaunt gaunt(gmax,2*gmax,gmax);

      // Number of radial elements
      size_t Nel(radial.Nel());
      // Number of radial basis functions
      size_t Nrad(radial.Nbf());

      // Full exchange matrix
      arma::mat K(Nbf(),Nbf());
      K.zeros();

      // Helper memory
#ifdef _OPENMP
      const int nth(omp_get_max_threads());
#else
      const int nth(1);
#endif
      std::vector<arma::vec> mem_Ksub(nth);
      std::vector<arma::vec> mem_Psub(nth);

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

			if(iel == jel) {
			  /*
			    The exchange matrix is given by
			    K(jk) = (ij|kl) P(il)
			    i.e. the complex conjugation hits i and l as
			    in the density matrix.

			    To get this in the proper order, we permute the integrals
			    K(jk) = (jk;il) P(il)
			  */

			  // Get density submatrix
			  arma::vec Psub(mem_Psub[ith].memptr(),Ni*Nj,false,true);
			  Psub=arma::vectorise(P.submat(iang*Nrad+ifirst,lang*Nrad+jfirst,iang*Nrad+ilast,lang*Nrad+jlast));

			  // Don't calculate zeros
			  if(arma::norm(Psub,2)==0.0)
			    continue;

			  // Exchange submatrix
			  arma::mat Ksub(mem_Ksub[ith].memptr(),Ni*Nj,1,false,true);
			  Ksub=cpl*(prim_ktei[Nel*Nel*L + iel*Nel + jel]*Psub);
			  Ksub.reshape(Ni,Nj);

			  // Increment global exchange matrix
			  K.submat(jang*Nrad+ifirst,kang*Nrad+jfirst,jang*Nrad+ilast,kang*Nrad+jlast)-=Ksub;

			  //arma::vec Ptgt(arma::vectorise(P.submat(jang*Nrad+ifirst,kang*Nrad+jfirst,jang*Nrad+ilast,kang*Nrad+jlast)));
			  //printf("(%i %i) (%i %i) (%i %i) (%i %i) [%i %i]\n",li,mi,lj,mj,lk,mk,ll,ml,L,M);
			  //printf("Element %i - %i contribution to exchange energy % .10e\n",(int) iel,(int) jel,-0.5*arma::dot(Ksub,Ptgt));

			} else {
			  // Disjoint integrals. When r(iel)>r(jel), iel gets -1-L, jel gets L.
			  const arma::mat & iint=(iel>jel) ? disjoint_m1L[L*Nel+iel] : disjoint_L[L*Nel+iel];
			  const arma::mat & jint=(iel>jel) ? disjoint_L[L*Nel+jel] : disjoint_m1L[L*Nel+jel];

			  double Lfac=4.0*M_PI/(2*L+1);

			  // Get density submatrix (Niel x Njel)
			  arma::mat Psub(mem_Psub[ith].memptr(),Ni,Nj,false,true);
			  Psub=P.submat(iang*Nrad+ifirst,lang*Nrad+jfirst,iang*Nrad+ilast,lang*Nrad+jlast);

			  // Calculate helper
			  arma::mat T(mem_Ksub[ith].memptr(),Ni*Nj,1,false,true);
			  // (Niel x Njel) = (Niel x Njel) x (Njel x Njel)
			  T=Psub*arma::trans(jint);

			  // Increment global exchange matrix
			  K.submat(jang*Nrad+ifirst,kang*Nrad+jfirst,jang*Nrad+ilast,kang*Nrad+jlast)-=cpl*Lfac*iint*T;
			}
		      }
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

    arma::mat TwoDBasis::expand_boundaries_C(const arma::mat & Cpure) const {
      // Determine how many functions we need
      size_t Npure=angular_offset(lval.n_elem);
      // Number of functions in radial basis
      size_t Nrad=radial.Nbf();

      if(Cpure.n_rows != Npure) {
	std::ostringstream oss;
	oss << "Matrix does not have expected size! Got " << Cpure.n_rows << " rows, expeted " << Npure << " rows!\n";
	throw std::logic_error(oss.str());
      }

      // Matrix with the boundary conditions removed
      arma::mat Cnob(Nrad*lval.n_elem,Cpure.n_cols);
      Cnob.zeros();

      // Fill matrix
      for(size_t iang=0;iang<lval.n_elem;iang++) {
	  // Offset
	  size_t ioff=angular_offset(iang);
	  // Number of radial functions on the shell
	  size_t ni=angular_nbf(iang);
	  // Sanity check for trivial case
	  if(!ni) continue;

	  Cnob.rows((iang+1)*Nrad-ni,(iang+1)*Nrad-1)=Cpure.rows(ioff,ioff+ni-1);
	}

      return Cnob;
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
  }
}
