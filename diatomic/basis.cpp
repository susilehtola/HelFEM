#include "basis.h"
#include "quadrature.h"
#include "../general/polynomial.h"
#include "../general/chebyshev.h"
#include "../general/gaunt.h"
#include "../general/utils.h"
#include "../general/timer.h"
#include <algorithm>
#include <cassert>
#include <cfloat>

#ifdef _OPENMP
#include <omp.h>
#endif

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

      size_t RadialBasis::max_Nprim() const {
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

      arma::mat RadialBasis::Plm_integral(int k, size_t iel, int L, int M, const legendretable::LegendreTable & legtab) const {
        double mumin(bval(iel));
        double mumax(bval(iel+1));

        // Integral by quadrature
        return diatomic::quadrature::Plm_radial_integral(mumin,mumax,k,xq,wq,get_basis(bf,iel),L,M,legtab);
      }

      arma::mat RadialBasis::Qlm_integral(int k, size_t iel, int L, int M, const legendretable::LegendreTable & legtab) const {
        double mumin(bval(iel));
        double mumax(bval(iel+1));

        // Integral by quadrature
        return diatomic::quadrature::Qlm_radial_integral(mumin,mumax,k,xq,wq,get_basis(bf,iel),L,M,legtab);
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

      arma::mat RadialBasis::twoe_integral(int alpha, int beta, size_t iel, int L, int M, const legendretable::LegendreTable & legtab) const {
        double mumin(bval(iel));
        double mumax(bval(iel+1));

        // Integral by quadrature
        return quadrature::twoe_integral(mumin,mumax,alpha,beta,xq,wq,get_basis(bf_C,iel),L,M,legtab);
      }

      arma::vec RadialBasis::get_chmu_quad() const {
        // Quadrature points for normal integrals
        arma::vec muq((bval.n_elem-1)*xq.n_elem*(xq.n_elem+1));
        size_t ioff=0;

        for(size_t iel=0;iel<bval.n_elem-1;iel++) {
          // Element ranges from
          double mumin0=bval(iel);
          double mumax0=bval(iel+1);

          // Midpoint is at
          double mumid0(0.5*(mumax0+mumin0));
          // and half-length of interval is
          double mulen0(0.5*(mumax0-mumin0));
          // mu values are then
          arma::vec mu0(mumid0*arma::ones<arma::vec>(xq.n_elem)+mulen0*xq);

          // Store values
          muq.subvec(ioff,ioff+mu0.n_elem-1)=mu0;
          ioff+=mu0.n_elem;

          // Subintervals for in-element two-electron integrals
          for(size_t isub=0;isub<xq.n_elem;isub++) {
            double mumin = (isub==0) ? mumin0 : mu0(isub-1);
            double mumax = mu0(isub);

            double mumid(0.5*(mumax+mumin));
            double mulen(0.5*(mumax-mumin));
            arma::vec mu(mumid*arma::ones<arma::vec>(xq.n_elem)+mulen*xq);
            muq.subvec(ioff,ioff+mu.n_elem-1)=mu;
            ioff+=mu.n_elem;
          }
        }

        return arma::cosh(arma::sort(muq,"ascend"));
      }

      TwoDBasis::TwoDBasis(int Z1_, int Z2_, double Rbond, int n_nodes, int der_order, int n_quad, int num_el, double rmax, int lmax, int mmax, int igrid, double zexp, int lpad) {
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

        arma::imat bang(lval.n_elem,2);
        bang.col(0)=lval;
        bang.col(1)=mval;
        bang.print("Angular basis");

        // Gaunt coefficients
        int gmax(std::max(lmax,mmax));
        int Lmax(L_max());
        int Mmax(M_max());

        // One-electron matrices need gmax,3,gmax
        // Two-electron matrices need Lmax+2,Lmax,Lmax+2
        int lrval(std::max(Lmax+2,gmax));
        int mval(std::max(Lmax,3));

        Timer t;
        printf("Computing Gaunt coefficients ... ");
        fflush(stdout);
        gaunt=gaunt::Gaunt(lrval,Mmax,mval,Mmax,lrval,Mmax);
        printf("done (% .3f s)\n",t.get());
        fflush(stdout);

        // Legendre function values
        t.set();
        printf("Computing Legendre function values ... ");
        fflush(stdout);

        // Fill table with necessary values
        legtab=legendretable::LegendreTable(L_max()+lpad,L_max(),M_max());
        arma::vec chmu(radial.get_chmu_quad());
        for(size_t i=0;i<chmu.n_elem;i++)
          legtab.compute(chmu(i));
        printf("done (% .3f s)\n",t.get());
        fflush(stdout);

        // Form lm map
        lm_map.clear();
        for(int L=0;L<=L_max();L++)
          for(int M=0;M<=std::min(M_max(),L);M++) { // m=-mmax and M=2*mmax can still couple to m'=mmax
            lmidx_t p;
            p.first=L;
            p.second=M;

            if(!lm_map.size())
              lm_map.push_back(p);
            else
              // Insert at lower bound
              lm_map.insert(lm_map.begin()+lmind(L,M,false),p);
          }

        // Form LM map
        LM_map.clear();
        for(int L=0;L<=L_max();L++)
          for(int M=-std::min(M_max(),L);M<=std::min(M_max(),L);M++) {
            lmidx_t p;
            p.first=L;
            p.second=M;
            
            if(!LM_map.size())
              LM_map.push_back(p);
            else
              // Insert at lower bound
              LM_map.insert(LM_map.begin()+LMind(L,M,false),p);
          }
      }

      TwoDBasis::~TwoDBasis() {
      }

      size_t TwoDBasis::Ndummy() const {
        return lval.n_elem*radial.Nbf();
      }

      size_t TwoDBasis::Nbf() const {
        // Count total number of basis functions
        size_t nbf=0;
        for(size_t i=0;i<mval.n_elem;i++) {
          nbf+=radial.Nbf();
          if(mval(i)!=0)
            // Remove first noverlap functions
            nbf-=radial.get_noverlap();
        }

        return nbf;
      }

      size_t TwoDBasis::Nrad() const {
        return radial.Nbf();
      }

      size_t TwoDBasis::Nang() const {
        return lval.n_elem;
      }

      arma::uvec TwoDBasis::pure_indices() const {
        // Indices of the pure functions
        arma::uvec idx(Nbf());

        size_t ioff=0;
        for(size_t i=0;i<mval.n_elem;i++) {
          if(mval(i)==0) {
            idx.subvec(ioff,ioff+radial.Nbf()-1)=arma::linspace<arma::uvec>(i*radial.Nbf(),(i+1)*radial.Nbf()-1,radial.Nbf());
            ioff+=radial.Nbf();
          } else {
            idx.subvec(ioff,ioff+radial.Nbf()-radial.get_noverlap()-1)=arma::linspace<arma::uvec>(i*radial.Nbf()+radial.get_noverlap(),(i+1)*radial.Nbf()-1,radial.Nbf()-radial.get_noverlap());
            ioff+=radial.Nbf()-radial.get_noverlap();
          }
        }

        return idx;
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
            nm += (m==0) ? radial.Nbf() : radial.Nbf()-radial.get_noverlap();
          }
        }

        // Collect functions
        arma::uvec idx(nm);
        size_t ioff=0;
        size_t ibf=0;
        for(size_t i=0;i<mval.n_elem;i++) {
          // Number of functions on shell is
          size_t nsh=(mval(i)==0) ? radial.Nbf() : radial.Nbf()-radial.get_noverlap();
          if(mval(i)==m) {
            idx.subvec(ioff,ioff+nsh-1)=arma::linspace<arma::uvec>(ibf,ibf+nsh-1,nsh);
            ioff+=nsh;
          }
          ibf+=nsh;
        }

        return idx;
      }

      arma::mat TwoDBasis::Sinvh(bool chol) const {
        // Form overlap matrix
        arma::mat S(overlap());

        // Get the basis function norms
        arma::vec bfnormlz(arma::pow(arma::diagvec(S),-0.5));
	printf("Smallest normalization constant % e, largest % e\n",arma::min(bfnormlz),arma::max(bfnormlz));
        // Go to normalized basis
        S=arma::diagmat(bfnormlz)*S*arma::diagmat(bfnormlz);

        if(chol) {
          // Half-inverse is
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
          if(lval(iang)!=0) {
            // We also get the l(l+1) term
            add_sub(T,iang,iang,lval(iang)*(lval(iang)+1)*Ip1);
          }
          if(mval(iang)!=0) {
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

      arma::mat TwoDBasis::dipole_z() const {
        // Full electric couplings
        arma::mat V(Ndummy(),Ndummy());
        V.zeros();

        // Build radial matrix elements
        arma::mat I11(radial.radial_integral(1,1));
        arma::mat I13(radial.radial_integral(1,3));

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
        V*=std::pow(Rhalf,4);

        return remove_boundaries(V);
      }

      arma::mat TwoDBasis::quadrupole_zz() const {
        return arma::zeros<arma::mat>(Nbf(),Nbf());
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

      int TwoDBasis::L_max() const {
        // l=lmax and L=2*lmax+2 can still couple to l'=lmax through the cos^2 term
        return 2*arma::max(lval)+2;
      }

      int TwoDBasis::M_max() const {
        // Maximum M value is
        return arma::max(mval)-arma::min(mval);
      }

      size_t TwoDBasis::mem_1el() const {
        return Nbf()*Nbf()*sizeof(double);
      }

      size_t TwoDBasis::mem_1el_aux() const {
        size_t Nel(radial.Nel());
        size_t Nprim(radial.max_Nprim());
        size_t N_LM(lm_map.size());

        return 4*N_LM*Nel*Nprim*Nprim*sizeof(double);
      }

      size_t TwoDBasis::mem_2el_aux() const {
        // Auxiliary integrals required up to
        size_t N_LM(lm_map.size());
        // Number of elements
        size_t Nel(radial.Nel());
        // Number of primitive functions per element
        size_t Nprim(radial.max_Nprim());

        // No off-diagonal storage
        return 4*N_LM*Nel*Nprim*Nprim*Nprim*Nprim*sizeof(double);
      }


      void TwoDBasis::compute_tei() {
        // Number of distinct L values is
        size_t Nel(radial.Nel());

        // Compute disjoint integrals
        disjoint_P0.resize(Nel*lm_map.size());
        disjoint_P2.resize(Nel*lm_map.size());
        disjoint_Q0.resize(Nel*lm_map.size());
        disjoint_Q2.resize(Nel*lm_map.size());
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(size_t ilm=0;ilm<lm_map.size();ilm++) {
          int L(lm_map[ilm].first);
          int M(lm_map[ilm].second);
          for(size_t iel=0;iel<Nel;iel++) {
            disjoint_P0[ilm*Nel+iel]=radial.Plm_integral(0,iel,L,M,legtab);
            disjoint_P2[ilm*Nel+iel]=radial.Plm_integral(2,iel,L,M,legtab);
            disjoint_Q0[ilm*Nel+iel]=radial.Qlm_integral(0,iel,L,M,legtab);
            disjoint_Q2[ilm*Nel+iel]=radial.Qlm_integral(2,iel,L,M,legtab);
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
          // Prefactor for integrals
	  const double LMfac(4.0*M_PI*std::pow(Rhalf,5)*std::pow(-1.0,M)/polynomial::factorial_ratio(L+std::abs(M),L-std::abs(M)));

          for(size_t iel=0;iel<Nel;iel++) {
            // Index in array
            {
              const size_t idx(Nel*Nel*ilm + iel*Nel + iel);

              // In-element integrals
              prim_tei00[idx]=radial.twoe_integral(0,0,iel,L,M,legtab)*LMfac;
              prim_tei02[idx]=radial.twoe_integral(0,2,iel,L,M,legtab)*LMfac;
              prim_tei20[idx]=radial.twoe_integral(2,0,iel,L,M,legtab)*LMfac;
              prim_tei22[idx]=radial.twoe_integral(2,2,iel,L,M,legtab)*LMfac;
            }

            /*
            for(size_t jel=0;jel<Nel;jel++) {
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
            */
          }
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
          for(size_t iel=0;iel<Nel;iel++) {
            // Diagonal integrals
            {
              size_t idx=Nel*Nel*ilm + iel*Nel + iel;
              size_t Ni(radial.Nprim(iel));
              size_t Nj(radial.Nprim(iel));
              prim_ktei00[idx]=utils::exchange_tei(prim_tei00[idx],Ni,Ni,Nj,Nj);
              prim_ktei02[idx]=utils::exchange_tei(prim_tei02[idx],Ni,Ni,Nj,Nj);
              prim_ktei20[idx]=utils::exchange_tei(prim_tei20[idx],Ni,Ni,Nj,Nj);
              prim_ktei22[idx]=utils::exchange_tei(prim_tei22[idx],Ni,Ni,Nj,Nj);
            }

            // Off-diagonal integrals (not used since faster to
            // contract the integrals in factorized form)
            /*
            for(size_t jel=0;jel<iel;jel++) {
              size_t idx=Nel*Nel*ilm + iel*Nel + jel;
              size_t Ni(radial.Nprim(iel));
              size_t Nj(radial.Nprim(jel));
              prim_ktei00[idx]=utils::exchange_tei(prim_tei00[idx],Ni,Ni,Nj,Nj);
              prim_ktei02[idx]=utils::exchange_tei(prim_tei02[idx],Ni,Ni,Nj,Nj);
              prim_ktei20[idx]=utils::exchange_tei(prim_tei20[idx],Ni,Ni,Nj,Nj);
              prim_ktei22[idx]=utils::exchange_tei(prim_tei22[idx],Ni,Ni,Nj,Nj);
            }
            for(size_t jel=iel+1;jel<Nel;jel++) {
              size_t idx=Nel*Nel*ilm + iel*Nel + jel;
              size_t Ni(radial.Nprim(iel));
              size_t Nj(radial.Nprim(jel));
              prim_ktei00[idx]=utils::exchange_tei(prim_tei00[idx],Ni,Ni,Nj,Nj);
              prim_ktei02[idx]=utils::exchange_tei(prim_tei02[idx],Ni,Ni,Nj,Nj);
              prim_ktei20[idx]=utils::exchange_tei(prim_tei20[idx],Ni,Ni,Nj,Nj);
              prim_ktei22[idx]=utils::exchange_tei(prim_tei22[idx],Ni,Ni,Nj,Nj);
            }
            */
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

      size_t TwoDBasis::LMind(int L, int M, bool check) const {
        // Find index in the L,M table
        lmidx_t p(L,M);
        std::vector<lmidx_t>::const_iterator low(std::lower_bound(LM_map.begin(),LM_map.end(),p));
        if(check && low == LM_map.end()) {
          std::ostringstream oss;
          oss << "Could not find L=" << p.first << ", M= " << p.second << " on the list!\n";
          throw std::logic_error(oss.str());
        }
        // Index is
        size_t idx(low-LM_map.begin());
        if(check && (LM_map[idx].first != L || LM_map[idx].second != M)) {
          std::ostringstream oss;
          oss << "Map error: tried to get L = " << L << ", M = " << M << " but got instead L = " << LM_map[idx].first << ", M = " << LM_map[idx].second << "!\n";
          throw std::logic_error(oss.str());
        }

        return idx;
      }

      arma::mat TwoDBasis::coulomb(const arma::mat & P0) const {
        if(!prim_tei00.size())
          throw std::logic_error("Primitive teis have not been computed!\n");

        // Extend to boundaries
        arma::mat P(expand_boundaries(P0));

        // Number of radial elements
        size_t Nel(radial.Nel());
        // Number of radial functions
        size_t Nrad(radial.Nbf());

        // Radial helper matrices
        std::vector<arma::mat> Paux0(LM_map.size());
        std::vector<arma::mat> Paux2(LM_map.size());
        for(size_t i=0;i<Paux0.size();i++) {
          Paux0[i].zeros(Nrad,Nrad);
          Paux2[i].zeros(Nrad,Nrad);
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
            int Lmin=std::max(std::abs(lk-ll)-2,abs(M));
            int Lmax=lk+ll+2;
            for(int L=Lmin;L<=Lmax;L++) {
              const size_t iLM(LMind(L,M));
              // Calculate coupling coefficients
              double cpl0(gaunt.mod_coeff(lk,mk,L,M,ll,ml));
              double cpl2(gaunt.coeff(lk,mk,L,M,ll,ml));
              // Increment
              arma::mat Prad(P.submat(kang*Nrad,lang*Nrad,(kang+1)*Nrad-1,(lang+1)*Nrad-1));
              Paux0[iLM]+=cpl0*Prad;
              Paux2[iLM]+=cpl2*Prad;
            }
          }
        }

        // Full Coulomb matrix
        arma::mat J(Ndummy(),Ndummy());
        J.zeros();

        // Helper memory
#ifdef _OPENMP
        const int nth(omp_get_max_threads());
#else
        const int nth(1);
#endif
        std::vector<arma::vec> mem_Jsub(nth);
        std::vector<arma::vec> mem_Psub0(nth);
        std::vector<arma::vec> mem_Psub2(nth);

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
          mem_Psub0[ith].zeros(radial.max_Nprim()*radial.max_Nprim());
          mem_Psub2[ith].zeros(radial.max_Nprim()*radial.max_Nprim());
          mem_Jsub[ith].zeros(radial.max_Nprim()*radial.max_Nprim());

          // Increment
#ifdef _OPENMP
#pragma omp for collapse(2)
#endif
          for(size_t iang=0;iang<lval.n_elem;iang++) {
            for(size_t jang=0;jang<lval.n_elem;jang++) {
              // l and m values
              int li(lval(iang));
              int mi(mval(iang));
              int lj(lval(jang));
              int mj(mval(jang));
              // LH m value
              int M(mj-mi);
              
              int Lmin=std::max(std::abs(lj-li)-2,abs(M));
              int Lmax=lj+li+2;
              for(int L=Lmin;L<=Lmax;L++) {
                const size_t iLM(LMind(L,M));
                const size_t ilm(lmind(L,M));
                const double LMfac(4.0*M_PI*std::pow(Rhalf,5)*std::pow(-1.0,M)/polynomial::factorial_ratio(L+std::abs(M),L-std::abs(M)));

                // Couplings
                double cpl0(gaunt.mod_coeff(lj,mj,L,M,li,mi));
                double cpl2(gaunt.coeff(lj,mj,L,M,li,mi));
                
                if(cpl0!=0.0 || cpl2!=0.0) {
                  // Loop over input elements
                  for(size_t jel=0;jel<Nel;jel++) {
                    size_t jfirst, jlast;
                    radial.get_idx(jel,jfirst,jlast);
                    size_t Nj(jlast-jfirst+1);
                    
                    // Get density submatrices
                    arma::mat Psub0(mem_Psub0[ith].memptr(),Nj,Nj,false,true);
                    Psub0=Paux0[iLM].submat(jfirst,jfirst,jlast,jlast);
                    arma::mat Psub2(mem_Psub2[ith].memptr(),Nj,Nj,false,true);
                    Psub2=Paux2[iLM].submat(jfirst,jfirst,jlast,jlast);
                    
                    // Contract integrals
                    double jsmall0=0.0, jsmall2=0.0, jbig0=0.0, jbig2=0.0;
                    if(cpl0!=0.0 || cpl2!=0.0) {
                      jsmall0 = LMfac*arma::trace(disjoint_P0[ilm*Nel+jel]*Psub0);
                      jbig0 = LMfac*arma::trace(disjoint_Q0[ilm*Nel+jel]*Psub0);
                    }
                    if(cpl0!=0.0 || cpl2!=0.0) {
                      jsmall2 = LMfac*arma::trace(disjoint_P2[ilm*Nel+jel]*Psub2);
                      jbig2 = LMfac*arma::trace(disjoint_Q2[ilm*Nel+jel]*Psub2);
                    }

                    // Increment J: jel>iel
                    double ifac0(jbig0*cpl0 - jbig2*cpl0);
                    double ifac2(-jbig0*cpl2 + jbig2*cpl2);
                    for(size_t iel=0;iel<jel;iel++) {
                      size_t ifirst, ilast;
                      radial.get_idx(iel,ifirst,ilast);

                      const arma::mat & iint0=disjoint_P0[ilm*Nel+iel];
                      const arma::mat & iint2=disjoint_P2[ilm*Nel+iel];
                      J.submat(iang*Nrad+ifirst,jang*Nrad+ifirst,iang*Nrad+ilast,jang*Nrad+ilast)+=iint0*ifac0 + iint2*ifac2;
                    }

                    // Increment J: jel<iel
                    ifac0=jsmall0*cpl0 - jsmall2*cpl0;
                    ifac2=-cpl2*jsmall0 + cpl2*jsmall2;
                    for(size_t iel=jel+1;iel<Nel;iel++) {
                      size_t ifirst, ilast;
                      radial.get_idx(iel,ifirst,ilast);

                      const arma::mat & iint0=disjoint_Q0[ilm*Nel+iel];
                      const arma::mat & iint2=disjoint_Q2[ilm*Nel+iel];
                      J.submat(iang*Nrad+ifirst,jang*Nrad+ifirst,iang*Nrad+ilast,jang*Nrad+ilast)+=iint0*ifac0 + iint2*ifac2;
                    }

                    // In-element contribution
                    {
                      size_t iel=jel;
                      size_t ifirst=jfirst;
                      size_t ilast=jlast;
                      size_t Ni=Nj;

                      // Contract integrals
                      arma::mat Jsub(mem_Jsub[ith].memptr(),Ni*Ni,1,false,true);
                      Jsub.zeros();

                      Psub0.reshape(Nj*Nj,1);
                      Psub2.reshape(Nj*Nj,1);

                      const size_t idx(Nel*Nel*ilm + iel*Nel + jel);
                      if(cpl0!=0.0) {
                        Jsub+=cpl0*(prim_tei00[idx]*Psub0);
                        Jsub-=cpl0*(prim_tei02[idx]*Psub2);
                      }
                      if(cpl2!=0.0) {
                        Jsub-=cpl2*(prim_tei20[idx]*Psub0);
                        Jsub+=cpl2*(prim_tei22[idx]*Psub2);
                      }
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

        return remove_boundaries(J);
      }

      arma::mat TwoDBasis::exchange(const arma::mat & P0) const {
        if(!prim_ktei00.size())
          throw std::logic_error("Primitive teis have not been computed!\n");

        // Extend to boundaries
        arma::mat P(expand_boundaries(P0));

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
        std::vector<arma::vec> mem_Krad(nth);
        std::vector<arma::vec> mem_Ksub(nth);
        std::vector<arma::vec> mem_T(nth);
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
          mem_Krad[ith].zeros(radial.Nbf()*radial.Nbf());
          mem_Ksub[ith].zeros(radial.max_Nprim()*radial.max_Nprim());
          mem_T[ith].zeros(radial.max_Nprim()*radial.max_Nprim());
          mem_Psub[ith].zeros(radial.max_Nprim()*radial.max_Nprim());

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
                  int Lmin=std::max(std::max(std::abs(li-lj),std::abs(lk-ll))-2,abs(M));
                  int Lmax=std::min(li+lj,lk+ll)+2;

                  for(int L=Lmin;L<=Lmax;L++) {
                    // Calculate total coupling coefficient
                    double cpl00(gaunt.mod_coeff(lj,mj,L,M,li,mi)*gaunt.mod_coeff(lk,mk,L,M,ll,ml));
                    double cpl02(-gaunt.mod_coeff(lj,mj,L,M,li,mi)*gaunt.coeff(lk,mk,L,M,ll,ml));
                    double cpl20(-gaunt.coeff(lj,mj,L,M,li,mi)*gaunt.mod_coeff(lk,mk,L,M,ll,ml));
                    double cpl22(gaunt.coeff(lj,mj,L,M,li,mi)*gaunt.coeff(lk,mk,L,M,ll,ml));
                    // Index in the L,|M| table
                    const size_t ilm(lmind(L,M));
                    const double LMfac(4.0*M_PI*std::pow(Rhalf,5)*std::pow(-1.0,M)/polynomial::factorial_ratio(L+std::abs(M),L-std::abs(M)));

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

                            // Index in tei array
                            size_t idx=Nel*Nel*ilm + iel*Nel + jel;

                            // Exchange submatrix
                            arma::mat Ksub(mem_Ksub[ith].memptr(),Ni*Nj,1,false,true);
                            Ksub.zeros();
                            if(cpl00!=0.0)
                              Ksub+=cpl00*(prim_ktei00[idx]*Psub);
                            if(cpl02!=0.0)
                              Ksub+=cpl02*(prim_ktei02[idx]*Psub);
                            if(cpl20!=0.0)
                              Ksub+=cpl20*(prim_ktei20[idx]*Psub);
                            if(cpl22!=0.0)
                              Ksub+=cpl22*(prim_ktei22[idx]*Psub);
                            Ksub.reshape(Ni,Nj);

                            // Increment global exchange matrix
                            K.submat(jang*Nrad+ifirst,kang*Nrad+jfirst,jang*Nrad+ilast,kang*Nrad+jlast)-=Ksub;

                            //arma::vec Ptgt(arma::vectorise(P.submat(jang*Nrad+ifirst,kang*Nrad+jfirst,jang*Nrad+ilast,kang*Nrad+jlast)));
                            //printf("(%i %i) (%i %i) (%i %i) (%i %i) [%i %i]\n",li,mi,lj,mj,lk,mk,ll,ml,L,M);
                            //printf("Element %i - %i contribution to exchange energy % .10e\n",(int) iel,(int) jel,-0.5*arma::dot(Ksub,Ptgt));

                          } else {
                            // Disjoint integrals. When r(iel)>r(jel), iel gets Q, jel gets P.
                            const arma::mat & iint0=(iel>jel) ? disjoint_Q0[ilm*Nel+iel] : disjoint_P0[ilm*Nel+iel];
                            const arma::mat & iint2=(iel>jel) ? disjoint_Q2[ilm*Nel+iel] : disjoint_P2[ilm*Nel+iel];
                            const arma::mat & jint0=(iel>jel) ? disjoint_P0[ilm*Nel+jel] : disjoint_Q0[ilm*Nel+jel];
                            const arma::mat & jint2=(iel>jel) ? disjoint_P2[ilm*Nel+jel] : disjoint_Q2[ilm*Nel+jel];

                            // Get density submatrix (Niel x Njel)
                            arma::mat Psub(mem_Psub[ith].memptr(),Ni,Nj,false,true);
                            Psub=P.submat(iang*Nrad+ifirst,lang*Nrad+jfirst,iang*Nrad+ilast,lang*Nrad+jlast);

                            // Calculate helper
                            arma::mat T(mem_T[ith].memptr(),Ni*Nj,1,false,true);
                            arma::mat Ksub(mem_Ksub[ith].memptr(),Ni,Nj,false,true);
                            Ksub.zeros();
                            // (Niel x Njel) = (Niel x Njel) x (Njel x Njel)
                            if(cpl00!=0.0 || cpl20!=0.0) {
                              T=Psub*arma::trans(jint0);
                              if(cpl00!=0.0)
                                Ksub-=(cpl00*LMfac)*iint0*T;
                              if(cpl20!=0.0)
                                Ksub-=(cpl20*LMfac)*iint2*T;
                            }
                            if(cpl02!=0.0 || cpl22!=0.0) {
                              T=Psub*arma::trans(jint2);
                              if(cpl02!=0.0)
                                Ksub-=(cpl02*LMfac)*iint0*T;
                              if(cpl22!=0.0)
                                Ksub-=(cpl22*LMfac)*iint2*T;
                            }

                            // Increment global exchange matrix
                            K.submat(jang*Nrad+ifirst,kang*Nrad+jfirst,jang*Nrad+ilast,kang*Nrad+jlast)+=Ksub;
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

      void TwoDBasis::set_zero(int lmax, arma::mat & M) const {
        if(M.n_rows != Nbf())
          throw std::logic_error("Matrix has incorrect size!\n");
        if(M.n_cols != Nbf())
          throw std::logic_error("Matrix has incorrect size!\n");
        M=expand_boundaries(M);

        // Number of functions in radial basis
        size_t Nrad=radial.Nbf();

        for(size_t iang=0;iang<lval.n_elem;iang++)
          for(size_t jang=0;jang<lval.n_elem;jang++)
            if(lval(iang)>lmax || lval(jang)>lmax)
              M.submat(iang*Nrad,jang*Nrad,(iang+1)*Nrad-1,(jang+1)*Nrad-1).zeros();

        M=remove_boundaries(M);
      }
    }
  }
}
