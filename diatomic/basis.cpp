#include "basis.h"
#include "quadrature.h"
#include "../general/polynomial.h"
#include "../general/chebyshev.h"
#include "../general/gaunt.h"
#include "../general/utils.h"
#include <algorithm>
#include <cassert>
#include <cfloat>

/// Use Cholesky for Sinvh?
//#define CHOL_SINVH

#define SET_ZERO(x) {static bool prt_x=false; if(!prt_x) {printf("Setting %s to zero\n",#x); prt_x=true;} x=0.0;}

namespace helfem {
  namespace diatomic {
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
      }

      RadialBasis::~RadialBasis() {
      }

      arma::mat RadialBasis::get_basis(const arma::mat & bas, size_t iel) const {
        if(iel==bval.n_elem-2)
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
        // The number of basis functions is Nbf*Nel - (Nel-1)*Noverlap
        // - Noverlap or just
        return Nel()*(bf.n_cols-noverlap);
      }

      size_t RadialBasis::Nprim(size_t iel) const {
        if(iel==bval.n_elem-2)
          return bf.n_cols-noverlap;
        else
          return bf.n_cols;
      }

      void RadialBasis::get_idx(size_t iel, size_t & ifirst, size_t & ilast) const {
        // The first function in the element will be
        ifirst=iel*(bf.n_cols-noverlap);
        // and the last one will be
        ilast=ifirst+bf.n_cols-1;
        // Last element does not have trailing functions
        if(iel==bval.n_elem-2)
          ilast-=noverlap;
      }

      arma::mat RadialBasis::radial_integral(int m, int n, size_t iel) const {
        return radial_integral(bf,m,n,iel);
      }

      arma::mat RadialBasis::radial_integral(const arma::mat & bas, int m, int n, size_t iel) const {
        double mumin(bval(iel));
        double mumax(bval(iel+1));

        // Integral by quadrature
        return diatomic::quadrature::radial_integral(mumin,mumax,m,n,xq,wq,get_basis(bas,iel));
      }

      arma::mat RadialBasis::radial_integral(int m, int n) const {
        size_t Nrad(Nbf());
        arma::mat R(Nrad,Nrad);
        R.zeros();

        // Loop over elements
        for(size_t iel=0;iel<Nel();iel++) {
          // Where are we in the matrix?
          size_t ifirst, ilast;
          get_idx(iel,ifirst,ilast);
          R.submat(ifirst,ifirst,ilast,ilast)+=radial_integral(m,n,iel);
        }

        return R;
      }

      arma::mat RadialBasis::Plm_integral(int k, size_t iel, int L, int M) const {
        double mumin(bval(iel));
        double mumax(bval(iel+1));

        // Integral by quadrature
        return diatomic::quadrature::Plm_radial_integral(mumin,mumax,k,xq,wq,get_basis(bf,iel),L,M);
      }

      arma::mat RadialBasis::Qlm_integral(int k, size_t iel, int L, int M) const {
        double mumin(bval(iel));
        double mumax(bval(iel+1));

        // Integral by quadrature
        return diatomic::quadrature::Qlm_radial_integral(mumin,mumax,k,xq,wq,get_basis(bf,iel),L,M);
      }

      arma::mat RadialBasis::kinetic(size_t iel) const {
        // We get 1/rlen^2 from the derivatives
        double rlen((bval(iel+1)-bval(iel))/2);

        return radial_integral(df,1,0,iel)/(rlen*rlen);
      }

      arma::mat RadialBasis::kinetic() const {
        size_t Nrad(Nbf());
        arma::mat T(Nrad,Nrad);
        T.zeros();

        // Loop over elements
        for(size_t iel=0;iel<Nel();iel++) {
          // Where are we in the matrix?
          size_t ifirst, ilast;
          get_idx(iel,ifirst,ilast);
          T.submat(ifirst,ifirst,ilast,ilast)+=kinetic(iel);
        }

        return T;
      }

      arma::mat RadialBasis::twoe_integral(int alpha, int beta, size_t iel, int L, int M) const {
        double mumin(bval(iel));
        double mumax(bval(iel+1));

        // Integral by quadrature
        return quadrature::twoe_integral(mumin,mumax,alpha,beta,xq,wq,get_basis(bf_C,iel),L,M);
      }

      TwoDBasis::TwoDBasis(int Z1_, int Z2_, double Rbond, int n_nodes, int der_order, int n_quad, int num_el, double rmax, int lmax, int mmax, int igrid, double zexp) {
        // Nuclear charge
        Z1=Z1_;
        Z2=Z2_;
        Rhalf=Rbond/2.0;

        // Compute max mu value
        double mumax=utils::arcosh(rmax/Rhalf);

        printf("rmax = %e yields mumax = %e\n",rmax,mumax);

        // Construct radial basis
        radial=RadialBasis(n_nodes, der_order, n_quad, num_el, mumax, igrid, zexp);

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
          }
        for(int mabs=1;mabs<=mmax;mabs++)
          for(int l=mabs;l<=lmax;l++) {
            lval(iang)=l;
            mval(iang)=-mabs;
            iang++;
          }
        if(iang!=lval.n_elem)
          throw std::logic_error("Error.\n");

        arma::imat ang(lval.n_elem,2);
        ang.col(0)=lval;
        ang.col(1)=mval;
        ang.print("Angular basis");
      }

      TwoDBasis::~TwoDBasis() {
      }

      size_t TwoDBasis::Ndummy() const {
        return lval.n_elem*radial.Nbf();
      }

      size_t TwoDBasis::Nbf() const {
	return angular_offset(lval.n_elem);
      }

      size_t TwoDBasis::angular_nbf(size_t iam) const {
        if(mval(iam)==0)
          return radial.Nbf();
        else
          // Boundary condition with m>0: remove first noverlap functions!
          return radial.Nbf()-radial.get_noverlap();
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
	printf("Smallest normalization constant % e, largest % e\n",arma::min(bfnormlz),arma::max(bfnormlz));
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
        // Full overlap matrix
        arma::mat O(Ndummy(),Ndummy());
        O.zeros();

        throw std::logic_error("not implemented.!\n");

        return remove_boundaries(O);
      }

      arma::mat TwoDBasis::overlap() const {
        // Build radial matrix elements
        arma::mat I10(radial.radial_integral(1,0));
        arma::mat I12(radial.radial_integral(1,2));

        // Gaunt coefficients
        int gmax(std::max(arma::max(lval),arma::max(mval)));
        gaunt::Gaunt gaunt(gmax,2,gmax);

        // Full overlap matrix
        arma::mat S(Ndummy(),Ndummy());
        S.zeros();
        // Fill elements
        for(size_t iang=0;iang<lval.n_elem;iang++) {
          int li(lval(iang));
          int mi(mval(iang));

          for(size_t jang=0;jang<lval.n_elem;jang++) {
            int lj(lval(jang));
            int mj(mval(jang));

            // Calculate coupling
            if(mi==mj) {
              if(li==lj)
                set_sub(S,iang,jang,I12);

              // We can also couple through the cos^2 term
              double cpl(gaunt.cosine2_coupling(lj,mj,li,mi));
              if(cpl!=0.0)
                add_sub(S,iang,jang,-I10*cpl);
            }
          }
        }

        // Plug in prefactor
        S*=std::pow(Rhalf,3);

        return remove_boundaries(S);
      }

      arma::mat TwoDBasis::kinetic() const {
        // Build radial kinetic energy matrix
        arma::mat Trad(radial.kinetic());
        arma::mat Ip1(radial.radial_integral(1,0));
        arma::mat Im1(radial.radial_integral(-1,0));

        // Full kinetic energy matrix
        arma::mat T(Ndummy(),Ndummy());
        T.zeros();
        // Fill elements
        for(size_t iang=0;iang<lval.n_elem;iang++) {
          set_sub(T,iang,iang,Trad);
          if(lval(iang)>0) {
            // We also get the l(l+1) term
            add_sub(T,iang,iang,lval(iang)*(lval(iang)+1)*Ip1);
          }
          if(mval(iang)>0) {
            // We also get the m^2 term
            add_sub(T,iang,iang,mval(iang)*mval(iang)*Im1);
          }
        }

        // Plug in prefactor
        T*=Rhalf/2.0;

        return remove_boundaries(T);
      }

      arma::mat TwoDBasis::nuclear() const {
        // Build radial matrices
        arma::mat I10(radial.radial_integral(1,0));
        arma::mat I11(radial.radial_integral(1,1));

        // Full nuclear attraction matrix
        arma::mat V(Ndummy(),Ndummy());
        V.zeros();

        // Gaunt coefficients
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
            if(mi==mj) {
              if(li==lj)
                set_sub(V,iang,jang,(Z1+Z2)*I11);

              // We can also couple through the cos term
	      if(Z1!=Z2) {
		double cpl(gaunt.cosine_coupling(lj,mj,li,mi));
		if(cpl!=0.0)
		  add_sub(V,iang,jang,(Z2-Z1)*I10*cpl);
	      }
            }
          }
        }

        // Plug in prefactor
        V*=-std::pow(Rhalf,2);

        return remove_boundaries(V);
      }

      arma::mat TwoDBasis::electric(double Ez) const {
        // Full electric couplings
        arma::mat V(Ndummy(),Ndummy());
        V.zeros();

        if(Ez!=0.0) {
          // Build radial matrix elements
          arma::mat I11(radial.radial_integral(1,1));
          arma::mat I13(radial.radial_integral(1,3));

          // Gaunt coefficients
          int gmax(std::max(arma::max(lval),arma::max(mval)));
          gaunt::Gaunt gaunt(gmax,3,gmax);

          // Fill elements
          for(size_t iang=0;iang<lval.n_elem;iang++) {
            int li(lval(iang));
            int mi(mval(iang));

            for(size_t jang=0;jang<lval.n_elem;jang++) {
              int lj(lval(jang));
              int mj(mval(jang));

              // Calculate coupling
              if(mi==mj) {
                // Coupling through the cos term
                double cpl1(gaunt.cosine_coupling(lj,mj,li,mi));
                if(cpl1!=0.0)
                  add_sub(V,iang,jang,cpl1*I13);

                // We can also couple through the cos^3 term
                double cpl3(gaunt.cosine3_coupling(lj,mj,li,mi));
		if(cpl3!=0.0)
                  add_sub(V,iang,jang,-cpl3*I11);
              }
            }
          }

          // Plug in prefactors
          V*=Ez*std::pow(Rhalf,4);
        }

        return remove_boundaries(V);
      }

      bool operator<(const lmidx_t & lh, const lmidx_t & rh) {
        if(lh.first < rh.first)
          return true;
        if(lh.first > rh.first)
          return false;

        if(lh.second < rh.second)
          return true;
        if(lh.second > rh.second)
          return false;

        return false;
      }

      bool operator==(const lmidx_t & lh, const lmidx_t & rh) {
        return (lh.first == rh.first) && (lh.second == rh.second);
      }

      void TwoDBasis::compute_tei() {
        // Maximum l value is
        int lmax=arma::max(lval);
        // Maximum m value is
        int mmax=arma::max(mval);

        // Form lm map
        lm_map.clear();
        for(int L=0;L<=2*lmax;L++) // Coupling: l=lmax and L=2*lmax can still couple to l'=lmax (cos^2 term doesn't show up here explicitly)
          for(int M=0;M<=std::min(2*mmax,L);M++) { // m=-mmax and M=2*mmax can still couple to m'=mmax
            lmidx_t p;
            p.first=L;
            p.second=M;

            if(!lm_map.size())
              lm_map.push_back(p);
            else
              lm_map.insert(lm_map.begin()+lmind(L,M,false),p);
          }

        // Number of distinct L values is
        size_t Nel(radial.Nel());

        // Compute disjoint integrals
        std::vector<arma::mat> disjoint_P0(Nel*lm_map.size());
        std::vector<arma::mat> disjoint_P2(Nel*lm_map.size());
        std::vector<arma::mat> disjoint_Q0(Nel*lm_map.size());
        std::vector<arma::mat> disjoint_Q2(Nel*lm_map.size());
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(size_t ilm=0;ilm<lm_map.size();ilm++) {
          int L(lm_map[ilm].first);
          int M(lm_map[ilm].second);
          for(size_t iel=0;iel<Nel;iel++) {
            disjoint_P0[ilm*Nel+iel]=radial.Plm_integral(0,iel,L,M);
            disjoint_P2[ilm*Nel+iel]=radial.Plm_integral(2,iel,L,M);
            disjoint_Q0[ilm*Nel+iel]=radial.Qlm_integral(0,iel,L,M);
            disjoint_Q2[ilm*Nel+iel]=radial.Qlm_integral(2,iel,L,M);
          }
        }

        // Form primitive two-electron integrals
        prim_tei00.resize(Nel*Nel*lm_map.size());
        prim_tei02.resize(Nel*Nel*lm_map.size());
        prim_tei20.resize(Nel*Nel*lm_map.size());
        prim_tei22.resize(Nel*Nel*lm_map.size());

        for(size_t ilm=0;ilm<lm_map.size();ilm++) {
          int L(lm_map[ilm].first);
          int M(lm_map[ilm].second);
	  const double LMfac(std::pow(-1.0,M)/polynomial::factorial_ratio(L+M,L-M));
	  printf("L=%i M=%i coeff = % e\n",L,M,LMfac);

          for(size_t iel=0;iel<Nel;iel++) {
            for(size_t jel=0;jel<Nel;jel++) {
              // Index in array
              const size_t idx(Nel*Nel*ilm + iel*Nel + jel);

              if(iel==jel) {
                // In-element integrals
                prim_tei00[idx]=radial.twoe_integral(0,0,iel,L,M)*LMfac;
                prim_tei02[idx]=radial.twoe_integral(0,2,iel,L,M)*LMfac;
                prim_tei20[idx]=radial.twoe_integral(2,0,iel,L,M)*LMfac;
                prim_tei22[idx]=radial.twoe_integral(2,2,iel,L,M)*LMfac;

              } else {
		// Index in disjoint array
		const size_t iidx(ilm*Nel+iel);
		const size_t jidx(ilm*Nel+jel);

                // when r(iel)>r(jel), iel gets Q
                const arma::mat & i0=(iel>jel) ? disjoint_Q0[iidx] : disjoint_P0[iidx];
                const arma::mat & i2=(iel>jel) ? disjoint_Q2[iidx] : disjoint_P2[iidx];
                // and jel gets P
                const arma::mat & j0=(iel>jel) ? disjoint_P0[jidx] : disjoint_Q0[jidx];
                const arma::mat & j2=(iel>jel) ? disjoint_P2[jidx] : disjoint_Q2[jidx];

                // Store integrals
                prim_tei00[idx]=utils::product_tei(LMfac*i0,j0);
                prim_tei02[idx]=utils::product_tei(LMfac*i0,j2);
                prim_tei20[idx]=utils::product_tei(LMfac*i2,j0);
                prim_tei22[idx]=utils::product_tei(LMfac*i2,j2);
              }
            }
          }
        }

        // Plug in prefactor
        {
          const double teipre(4.0*M_PI*std::pow(Rhalf,5));
          for(size_t i=0;i<prim_tei00.size();i++)
            prim_tei00[i]*=teipre;
          for(size_t i=0;i<prim_tei02.size();i++)
            prim_tei02[i]*=teipre;
          for(size_t i=0;i<prim_tei20.size();i++)
            prim_tei20[i]*=teipre;
          for(size_t i=0;i<prim_tei22.size();i++)
            prim_tei22[i]*=teipre;
        }

        // Make sure teis are symmetric
        /*
        for(size_t ilm=0;ilm<lm_map.size();ilm++)
          for(size_t iel=0;iel<Nel;iel++)
            for(size_t jel=0;jel<Nel;jel++) {
              size_t idx=Nel*Nel*ilm + iel*Nel + jel;
              size_t Ni(radial.Nprim(iel));
              size_t Nj(radial.Nprim(jel));
              printf("ilm = %i, iel = %i, jel = %i\n",(int) ilm, (int) iel, (int) jel);
              utils::check_tei_symmetry(prim_tei00[idx],Ni,Ni,Nj,Nj);
              utils::check_tei_symmetry(prim_tei02[idx],Ni,Ni,Nj,Nj);
              utils::check_tei_symmetry(prim_tei20[idx],Ni,Ni,Nj,Nj);
              utils::check_tei_symmetry(prim_tei22[idx],Ni,Ni,Nj,Nj);
            }
        */

        /*
          The exchange matrix is given by
          K(jk) = (ij|kl) P(il)
          i.e. the complex conjugation hits i and l as
          in the density matrix.

          To get this in the proper order, we permute the integrals
          K(jk) = (jk;il) P(il)
          so we don't have to reform the permutations in the exchange routine.
        */
        prim_ktei00.resize(prim_tei00.size());
        prim_ktei02.resize(prim_tei02.size());
        prim_ktei20.resize(prim_tei20.size());
        prim_ktei22.resize(prim_tei22.size());
        for(size_t ilm=0;ilm<lm_map.size();ilm++)
          for(size_t iel=0;iel<Nel;iel++)
            for(size_t jel=0;jel<Nel;jel++) {
              size_t idx=Nel*Nel*ilm + iel*Nel + jel;
              size_t Ni(radial.Nprim(iel));
              size_t Nj(radial.Nprim(jel));
              prim_ktei00[idx]=utils::exchange_tei(prim_tei00[idx],Ni,Ni,Nj,Nj);
              prim_ktei02[idx]=utils::exchange_tei(prim_tei02[idx],Ni,Ni,Nj,Nj);
              prim_ktei20[idx]=utils::exchange_tei(prim_tei20[idx],Ni,Ni,Nj,Nj);
              prim_ktei22[idx]=utils::exchange_tei(prim_tei22[idx],Ni,Ni,Nj,Nj);
            }
      }

      size_t TwoDBasis::lmind(int L, int M, bool check) const {
        // Switch to |M|
        M=std::abs(M);
        // Find index in the L,|M| table
        lmidx_t p(L,M);
        std::vector<lmidx_t>::const_iterator low(std::lower_bound(lm_map.begin(),lm_map.end(),p));
        if(check && low == lm_map.end()) {
          std::ostringstream oss;
          oss << "Could not find L=" << p.first << ", |M|= " << p.second << " on the list!\n";
          throw std::logic_error(oss.str());
        }
        // Index is
        size_t idx(low-lm_map.begin());
        if(check && (lm_map[idx].first != L || lm_map[idx].second != M)) {
          std::ostringstream oss;
          oss << "Map error: tried to get L = " << L << ", M = " << M << " but got instead L = " << lm_map[idx].first << ", M = " << lm_map[idx].second << "!\n";
          throw std::logic_error(oss.str());
        }

        return idx;
      }

      arma::mat TwoDBasis::coulomb(const arma::mat & P0) const {
        if(!prim_tei00.size())
          throw std::logic_error("Primitive teis have not been computed!\n");

        // Extend to boundaries
        if(P0.n_rows != Nbf())
          throw std::logic_error("Density matrix has incorrect size!\n");
        if(P0.n_cols != Nbf())
          throw std::logic_error("Density matrix has incorrect size!\n");
        arma::mat P(expand_boundaries(P0));

        // Gaunt coefficients
        int gmax(2*arma::max(lval));
        gaunt::Gaunt gaunt(gmax+2,std::max(gmax,2),gmax+2);

        // Number of radial elements
        size_t Nel(radial.Nel());
        // Number of radial functions
        size_t Nrad(radial.Nbf());

        // Full Coulomb matrix
        arma::mat J(Ndummy(),Ndummy());
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
                  double cpl00(gaunt.mod_coeff(lj,mj,L,M,li,mi)*gaunt.mod_coeff(lk,mk,L,M,ll,ml));
                  double cpl02(-gaunt.mod_coeff(lj,mj,L,M,li,mi)*gaunt.coeff(lk,mk,L,M,ll,ml));
                  double cpl20(-gaunt.coeff(lj,mj,L,M,li,mi)*gaunt.mod_coeff(lk,mk,L,M,ll,ml));
                  double cpl22(gaunt.coeff(lj,mj,L,M,li,mi)*gaunt.coeff(lk,mk,L,M,ll,ml));
                  // Index in the L,|M| table
                  const size_t ilm(lmind(L,M));

                  // Debug
                  //SET_ZERO(cpl00);
                  //SET_ZERO(cpl02);
                  //SET_ZERO(cpl20);
                  //SET_ZERO(cpl22);

                  if(cpl00!=0.0 || cpl02!=0.0 || cpl20!=0.0 || cpl22!=0.0) {
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
			arma::vec Pdum(arma::vectorise(P.submat(iang*Nrad+ifirst,jang*Nrad+ifirst,iang*Nrad+ilast,jang*Nrad+ilast)));
                        // Don't calculate zeros
                        if(arma::norm(Psub,2)==0.0)
                          continue;

                        // Allocate helper
                        size_t Ni(ilast-ifirst+1);
                        arma::vec Jsub(Ni*Ni);
                        Jsub.zeros();

                        // Contract integrals
                        const size_t idx(Nel*Nel*ilm + iel*Nel + jel);
                        if(cpl00!=0.0)
                          Jsub+=cpl00*(prim_tei00[idx]*Psub);
                        if(cpl02!=0.0)
                          Jsub+=cpl02*(prim_tei02[idx]*Psub);
                        if(cpl20!=0.0)
                          Jsub+=cpl20*(prim_tei20[idx]*Psub);
                        if(cpl22!=0.0)
			  Jsub+=cpl22*(prim_tei22[idx]*Psub);

                        // Increment global Coulomb matrix
                        J.submat(iang*Nrad+ifirst,jang*Nrad+ifirst,iang*Nrad+ilast,jang*Nrad+ilast)+=arma::reshape(Jsub,Ni,Ni);
                      }
                    }
                  }
                }
              }
            }
          }
        }

	arma::mat Sh(Sinvh());
	arma::mat Jmo(Sh.t()*remove_boundaries(J)*Sh);
        printf("J asymmetricity %e (%e)\n",arma::norm(J-J.t(),"fro"),arma::norm(J-J.t(),"fro")/arma::norm(J,"fro"));
	printf("J_MO asymmetricity %e (%e)\n",arma::norm(Jmo-Jmo.t(),"fro"),arma::norm(Jmo-Jmo.t(),"fro")/arma::norm(Jmo,"fro"));

	//J=0.5*(J+J.t());

        return remove_boundaries(J);
      }

      arma::mat TwoDBasis::exchange(const arma::mat & P0) const {
        if(!prim_ktei00.size())
          throw std::logic_error("Primitive teis have not been computed!\n");

        // Extend to boundaries
        if(P0.n_rows != Nbf())
          throw std::logic_error("Density matrix has incorrect size!\n");
        if(P0.n_cols != Nbf())
          throw std::logic_error("Density matrix has incorrect size!\n");
        arma::mat P(expand_boundaries(P0));

        // Gaunt coefficient table
        int gmax(2*arma::max(lval));
        gaunt::Gaunt gaunt(gmax+2,std::max(gmax,2),gmax+2);

        // Number of radial elements
        size_t Nel(radial.Nel());
        // Number of radial basis functions
        size_t Nrad(radial.Nbf());

        // Full exchange matrix
        arma::mat K(Ndummy(),Ndummy());
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
                  double cpl00(gaunt.mod_coeff(lj,mj,L,M,li,mi)*gaunt.mod_coeff(lk,mk,L,M,ll,ml));
                  double cpl02(-gaunt.mod_coeff(lj,mj,L,M,li,mi)*gaunt.coeff(lk,mk,L,M,ll,ml));
                  double cpl20(-gaunt.coeff(lj,mj,L,M,li,mi)*gaunt.mod_coeff(lk,mk,L,M,ll,ml));
                  double cpl22(gaunt.coeff(lj,mj,L,M,li,mi)*gaunt.coeff(lk,mk,L,M,ll,ml));
                  // Index in the L,|M| table
                  const size_t ilm(lmind(L,M));

                  // Debug
                  SET_ZERO(cpl00);
                  SET_ZERO(cpl02);
                  SET_ZERO(cpl20);
                  //SET_ZERO(cpl22);

                  if(cpl00!=0.0 || cpl02!=0.0 || cpl20!=0.0 || cpl22!=0.0) {
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

			arma::vec Pdum(arma::vectorise(P.submat(jang*Nrad+ifirst,kang*Nrad+jfirst,jang*Nrad+ilast,kang*Nrad+jlast)));
                        // Don't calculate zeros
                        if(arma::norm(Psub,2)==0.0)
                          continue;

                        // Allocate helper
                        arma::vec Ksub(Ni*Nj);
                        Ksub.zeros();

                        // Contract integrals
                        const size_t idx(Nel*Nel*ilm + iel*Nel + jel);
                        if(cpl00!=0.0)
                          Ksub+=cpl00*(prim_ktei00[idx]*Psub);
                        if(cpl02!=0.0)
                          Ksub+=cpl02*(prim_ktei02[idx]*Psub);
                        if(cpl20!=0.0)
                          Ksub+=cpl20*(prim_ktei20[idx]*Psub);
                        if(cpl22!=0.0)
                          Ksub+=cpl22*(prim_ktei22[idx]*Psub);

                        // Increment global exchange matrix
                        K.submat(jang*Nrad+ifirst,kang*Nrad+jfirst,jang*Nrad+ilast,kang*Nrad+jlast)-=arma::reshape(Ksub,Ni,Nj);
                      }
                    }
                  }
                }
              }
            }
          }
        }

        // Analyze asymmetricity
	arma::mat Sh(Sinvh());
	arma::mat Kmo(Sh.t()*remove_boundaries(K)*Sh);
        printf("K asymmetricity %e (%e)\n",arma::norm(K-K.t(),"fro"),arma::norm(K-K.t(),"fro")/arma::norm(K,"fro"));
	printf("K_MO asymmetricity %e (%e)\n",arma::norm(Kmo-Kmo.t(),"fro"),arma::norm(Kmo-Kmo.t(),"fro")/arma::norm(Kmo,"fro"));

	//K=0.5*(K+K.t());

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
    }
  }
}
