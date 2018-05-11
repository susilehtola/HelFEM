#include "basis.h"
#include "quadrature.h"
#include "../general/polynomial.h"
#include "../general/chebyshev.h"
#include "../general/gaunt.h"

namespace helfem {
  namespace basis {
    RadialBasis::RadialBasis() {
    }

    RadialBasis::RadialBasis(int n_nodes, int der_order, int n_quad, int num_el, double rmax) {
      // Get primitive polynomial representation
      arma::mat bf_C=polynomial::hermite_coeffs(n_nodes, der_order);
      arma::mat df_C=polynomial::derivative_coeffs(bf_C, 1);

      // Get quadrature rule
      chebyshev::chebyshev(n_quad,xq,wq);
      // Evaluate polynomials at quadrature points
      bf=polynomial::polyval(bf_C,xq);
      df=polynomial::polyval(df_C,xq);

      // Number of overlapping functions is
      noverlap=der_order+1;
      // Get boundary values, use exponential grid
      bval=arma::exp(arma::linspace<arma::vec>(0,log(rmax),num_el+1))-arma::ones<arma::vec>(num_el+1);
      //bval=arma::linspace<arma::vec>(0,rmax,num_el+1);

      bval.print("Boundary values");
    }

    RadialBasis::~RadialBasis() {
    }

    size_t RadialBasis::Nel() const {
      // Number of elements is
      return bval.n_elem-1;
    }

    size_t RadialBasis::Nbf() const {
      // The number of basis functions is Nbf*Nel - (Nel-1)*Noverlap -
      // Noverlap or just
      return Nel()*(bf.n_cols-noverlap);
    }

    size_t RadialBasis::Nprim(size_t iel) const {
      return (iel==bval.n_elem-2) ? (bf.n_cols-noverlap) : bf.n_cols;
    }

    void RadialBasis::get_idx(size_t iel, size_t & ifirst, size_t & ilast) const {
      // The first function will be
      ifirst=iel*(bf.n_cols-noverlap);
      // and the last one
      ilast=ifirst+bf.n_cols-1;

      // Last element does not have trailing functions
      if(iel==bval.n_elem-2)
        ilast-=noverlap;
    }

    arma::mat RadialBasis::radial_integral(int Rexp, size_t iel) const {
      if(iel==bval.n_elem-2)
        return quadrature::radial_integral(bval(iel),bval(iel+1),Rexp,xq,wq,bf.cols(0,bf.n_cols-1-noverlap));
      else
        return quadrature::radial_integral(bval(iel),bval(iel+1),Rexp,xq,wq,bf);
    }

    arma::mat RadialBasis::overlap(size_t iel) const {
      return radial_integral(2,iel);
    }

    arma::mat RadialBasis::kinetic(size_t iel) const {
      if(iel==bval.n_elem-2)
        return 0.5*quadrature::derivative_integral(bval(iel),bval(iel+1),xq,wq,df.cols(0,df.n_cols-1-noverlap));
      else
        return 0.5*quadrature::derivative_integral(bval(iel),bval(iel+1),xq,wq,df);
    }

    arma::mat RadialBasis::kinetic_l(size_t iel) const {
      return 0.5*radial_integral(0,iel);
    }

    arma::mat RadialBasis::nuclear(size_t iel) const {
      return -radial_integral(1,iel);
    }

    arma::mat RadialBasis::twoe_integral(int L, size_t iel) const {
      if(iel==bval.n_elem-2)
        return quadrature::twoe_integral(bval(iel),bval(iel+1),xq,wq,bf.cols(0,bf.n_cols-1-noverlap),L);
      else
        return quadrature::twoe_integral(bval(iel),bval(iel+1),xq,wq,bf,L);
    }

    TwoDBasis::TwoDBasis(int Z_, int n_nodes, int der_order, int n_quad, int num_el, double rmax, int lmax, int mmax) {
      // Nuclear charge
      Z=Z_;
      // Construct radial basis
      radial=RadialBasis(n_nodes, der_order, n_quad, num_el, rmax);

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

    arma::mat TwoDBasis::Sinvh() const {
      // Form overlap matrix
      arma::mat S(overlap());

      // Get the basis function norms
      arma::vec bfnormlz(arma::pow(arma::diagvec(S),-0.5));
      // Go to normalized basis
      S=arma::diagmat(bfnormlz)*S*arma::diagmat(bfnormlz);

      // Half-inverse is
      return arma::diagmat(bfnormlz) * arma::inv(arma::chol(S));
    }

    arma::mat TwoDBasis::overlap() const {
      // Build radial overlap matrix
      size_t Nrad(radial.Nbf());
      arma::mat Srad(Nrad,Nrad);
      Srad.zeros();

      // Loop over elements
      for(size_t iel=0;iel<radial.Nel();iel++) {
        // Where are we in the matrix?
        size_t ifirst, ilast;
        radial.get_idx(iel,ifirst,ilast);
        Srad.submat(ifirst,ifirst,ilast,ilast)+=radial.overlap(iel);
      }

      // Full overlap matrix
      arma::mat S(Nbf(),Nbf());
      S.zeros();
      // Fill elements
      for(size_t iang=0;iang<lval.n_elem;iang++)
        S.submat(iang*Nrad,iang*Nrad,(iang+1)*Nrad-1,(iang+1)*Nrad-1)=Srad;

      return S;
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
        if(lval(iang)==0)
          T.submat(iang*Nrad,iang*Nrad,(iang+1)*Nrad-1,(iang+1)*Nrad-1)=Trad;
        else
          T.submat(iang*Nrad,iang*Nrad,(iang+1)*Nrad-1,(iang+1)*Nrad-1)=Trad+lval(iang)*(lval(iang)+1)*Trad_l;
      }

      return T;
    }

    arma::mat TwoDBasis::nuclear() const {
      // Build radial nuclear matrix
      size_t Nrad(radial.Nbf());
      arma::mat Vrad(Nrad,Nrad);
      Vrad.zeros();

      // Loop over elements
      for(size_t iel=0;iel<radial.Nel();iel++) {
        // Where are we in the matrix?
        size_t ifirst, ilast;
        radial.get_idx(iel,ifirst,ilast);
        Vrad.submat(ifirst,ifirst,ilast,ilast)+=radial.nuclear(iel);
      }

      // Full nuclear attraction matrix
      arma::mat V(Nbf(),Nbf());
      V.zeros();
      // Fill elements
      for(size_t iang=0;iang<lval.n_elem;iang++)
        V.submat(iang*Nrad,iang*Nrad,(iang+1)*Nrad-1,(iang+1)*Nrad-1)=Z*Vrad;

      return V;
    }

    void TwoDBasis::compute_tei() {
      // Number of distinct L values is
      size_t N_L(2*arma::max(lval)+1);
      size_t Nel(radial.Nel());

      // Compute disjoint integrals
      std::vector<arma::mat> disjoint_2pL, disjoint_1mL;
      disjoint_2pL.resize(Nel*N_L);
      disjoint_1mL.resize(Nel*N_L);
      for(size_t L=0;L<N_L;L++)
        for(size_t iel=0;iel<Nel;iel++) {
          disjoint_2pL[L*Nel+iel]=radial.radial_integral(2+L,iel);
          disjoint_1mL[L*Nel+iel]=radial.radial_integral(1-L,iel);
        }

      // Form two-electron integrals
      prim_tei.resize(Nel*Nel*N_L);
      for(size_t L=0;L<N_L;L++) {
        // Normalization factor
        double Lfac=4.0*M_PI/(2*L+1);

        for(size_t iel=0;iel<Nel;iel++) {
          // Disjoint integrals
          for(size_t jel=0;jel<iel;jel++) {
            size_t Ni(radial.Nprim(iel));
            size_t Nj(radial.Nprim(jel));

            arma::mat teiblock(Ni*Ni,Nj*Nj);
            teiblock.zeros();

            // r(iel)>r(jel) so iel gets 1-L, jel gets 2+L.
            const arma::mat & iint(disjoint_1mL[L*Nel+iel]);
            const arma::mat & jint(disjoint_2pL[L*Nel+jel]);

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
            prim_tei[Nel*Nel*L + iel*Nel+jel]=teiblock;
            prim_tei[Nel*Nel*L + jel*Nel+iel]=arma::trans(teiblock);
          }

          // In-element integral
          prim_tei[Nel*Nel*L + iel*Nel+iel]=radial.twoe_integral(L,iel);
        }
      }
    }

    arma::mat TwoDBasis::coulomb(const arma::mat & P) const {
      if(!prim_tei.size())
        throw std::logic_error("Primitive teis have not been computed!\n");

      // Gaunt coefficient table
      int lmax(arma::max(lval));
      gaunt::Gaunt gaunt(lmax,2*lmax,lmax);

      // Number of radial basis functions
      size_t Nrad(radial.Nbf());
      // Number of radial elements
      size_t Nel(radial.Nel());

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
              int Lmin=std::max(std::abs(li-lj),std::abs(lk-ll));
              // We must also have L>=|M|
              Lmin=std::max(Lmin,abs(M));

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
                      arma::vec Jsub(prim_tei[Nel*Nel*L + iel*Nel + jel]*Psub);

                      // Increment global Coulomb matrix
                      J.submat(iang*Nrad+ifirst,iang*Nrad+ifirst,iang*Nrad+ilast,iang*Nrad+ilast)+=arma::reshape(Jsub,ilast-ifirst+1,ilast-ifirst+1);

                      //arma::mat Ptgt(P.submat(iang*Nrad+ifirst,iang*Nrad+ifirst,iang*Nrad+ilast,iang*Nrad+ilast));
                      //printf("Coulomb contribution %e from elements %i-%i\n",arma::dot(arma::vectorise(Ptgt),Jsub),(int) iel,(int) jel);
                    }
                  }
                }
              }
            }
          }
        }
      }

      return J;
    }

    arma::mat TwoDBasis::exchange(const arma::mat & P) const {
      if(!prim_tei.size())
        throw std::logic_error("Primitive teis have not been computed!\n");

      // Gaunt coefficient table
      int lmax(arma::max(lval));
      gaunt::Gaunt gaunt(lmax,2*lmax,lmax);

      // Number of radial basis functions
      size_t Nrad(radial.Nbf());
      // Number of radial elements
      size_t Nel(radial.Nel());

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
              int Lmin=std::max(std::abs(li-lj),std::abs(lk-ll));
              // We must also have L>=|M|
              Lmin=std::max(Lmin,abs(M));

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
                      arma::vec Psub(arma::vectorise(P.submat(jang*Nrad+ifirst,lang*Nrad+jfirst,jang*Nrad+ilast,lang*Nrad+jlast)));

                      // Number of functions in the two elements
                      size_t Ni(ilast-ifirst+1);
                      size_t Nj(jlast-jfirst+1);

                      // Integrals
                      arma::mat tei(Ni*Nj,Ni*Nj);

                      // Permute integrals to wanted order
                      const arma::mat & ptei(prim_tei[Nel*Nel*L + iel*Nel+jel]);
                      for(size_t ii=0;ii<Ni;ii++)
                        for(size_t jj=0;jj<Ni;jj++)
                          for(size_t kk=0;kk<Nj;kk++)
                            for(size_t ll=0;ll<Nj;ll++)
                              // (ik|jl) in Armadillo compatible indexing
                              tei(kk*Ni+ii,ll*Ni+jj)=ptei(jj*Ni+ii,ll*Nj+kk);

                      // Contract integrals
                      arma::vec Ksub(tei*Psub);

                      // Increment global exchange matrix
                      K.submat(iang*Nrad+ifirst,jang*Nrad+jfirst,iang*Nrad+ilast,jang*Nrad+jlast)+=arma::reshape(Ksub,Ni,Nj);
                    }
                  }
                }
              }
            }
          }
        }
      }

      return K;
    }
  }
}
