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
#include "helfem/PolynomialBasis.h"
#include "chebyshev.h"
#include "../general/spherical_harmonics.h"
#include "../general/gaunt.h"
#include "../general/gsz.h"
#include "utils.h"
#include "../general/timer.h"
#include "../general/scf_helpers.h"
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

      RadialBasis::RadialBasis(const polynomial_basis::FiniteElementBasis & fem_, int n_quad) : fem(fem_) {
        // Get quadrature rule
        chebyshev::chebyshev(n_quad,xq,wq);
        for(size_t i=0;i<xq.n_elem;i++) {
          if(!std::isfinite(xq[i]))
            printf("xq[%i]=%e\n",(int) i, xq[i]);
          if(!std::isfinite(wq[i]))
            printf("wq[%i]=%e\n",(int) i, wq[i]);
        }
      }

      RadialBasis::~RadialBasis() {
      }

      int RadialBasis::get_nquad() const {
        return (int) xq.n_elem;
      }

      arma::vec RadialBasis::get_bval() const {
        return fem.get_bval();
      }

      int RadialBasis::get_poly_id() const {
        return fem.get_poly_id();
      }

      int RadialBasis::get_poly_nnodes() const {
        return fem.get_poly_nnodes();
      }

      size_t RadialBasis::Nel() const {
        // Number of elements is
        return fem.get_nelem();
      }

      size_t RadialBasis::Nbf() const {
        // Number of basis functions is Nbf*Nel - (Nel-1)*Noverlap - Noverlap
        return fem.get_nbf();
      }

      size_t RadialBasis::Nprim(size_t iel) const {
	return fem.get_nprim(iel);
      }

      size_t RadialBasis::max_Nprim() const {
        return fem.get_max_nprim();
      }

      void RadialBasis::get_idx(size_t iel, size_t & ifirst, size_t & ilast) const {
        fem.get_idx(iel, ifirst, ilast);
      }

      arma::mat RadialBasis::radial_integral(int m, int n) const {
        std::function<double(double)> chsh;
        if(m!=0 && n!=0) {
          chsh = [m, n](double mu) { return std::pow(std::sinh(mu), m)*std::pow(std::cosh(mu), n); };
        } else if(m!=0 && n==0) {
          chsh = [m](double mu) { return std::pow(std::sinh(mu), m); };
        } else if(m==0 && n!=0) {
          chsh = [n](double mu) { return std::pow(std::cosh(mu), n); };
        }
        return fem.matrix_element(false, false, xq, wq, chsh);
      }

      arma::mat RadialBasis::overlap(const RadialBasis & rh, int n) const {
	// Use the larger number of quadrature points to assure
	// projection is computed ok
	size_t n_quad(std::max(xq.n_elem,rh.xq.n_elem));

	arma::vec xproj, wproj;
	chebyshev::chebyshev(n_quad,xproj,wproj);

        // Form list of overlapping elements
        std::vector< std::vector<size_t> > overlap(fem.get_nelem());
        for(size_t iel=0;iel<fem.get_nelem();iel++) {
          // Range of element i
          double istart(fem.element_begin(iel));
          double iend(fem.element_end(iel));

          for(size_t jel=0;jel<rh.fem.get_nelem();jel++) {
            // Range of element j
            double jstart(rh.fem.element_begin(jel));
            double jend(rh.fem.element_end(jel));

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
        for(size_t iel=0;iel<fem.get_nelem();iel++) {
          // Loop over overlapping elements
          for(size_t jj=0;jj<overlap[iel].size();jj++) {
            // Index of element is
            size_t jel=overlap[iel][jj];

	    // Because the functions are only defined within a single
	    // element, the product can be very raggedy. However,
	    // since we *know* where the overlap is non-zero, we can
	    // restrict the quadrature to just that zone.

	    // Limits
	    double imin(fem.element_begin(iel));
	    double imax(fem.element_end(iel));
            // Range of element
            double jmin(rh.fem.element_begin(jel));
            double jmax(rh.fem.element_end(jel));

	    // Range of integral is thus
	    double intstart(std::max(imin,jmin));
	    double intend(std::min(imax,jmax));
	    // Inteval mid-point is at
            double intmid(0.5*(intend+intstart));
            double intlen(0.5*(intend-intstart));

	    // mu values we're going to use are then
	    arma::vec mu(intmid*arma::ones<arma::vec>(xproj.n_elem)+intlen*xproj);

            // Calculate x values the polynomials should be evaluated at
            arma::vec xi(fem.eval_prim(mu, iel));
	    arma::vec xj(rh.fem.eval_prim(mu, jel));

	    // Where are we in the matrix?
	    size_t ifirst, ilast;
	    get_idx(iel,ifirst,ilast);
            size_t jfirst, jlast;
            rh.get_idx(jel,jfirst,jlast);

	    // Calculate total weight per point
	    arma::vec wtot(wproj*intlen);
	    wtot%=arma::sinh(mu);
	    if(n!=0)
	      wtot%=arma::pow(arma::cosh(mu),n);
	    // Put in weight
	    arma::mat ibf(fem.eval_f(xi, iel));
	    arma::mat jbf(rh.fem.eval_f(xj, jel));

	    // Perform quadrature
            arma::mat s(arma::trans(ibf)*arma::diagmat(wtot)*jbf);

            // Increment overlap matrix
            S.submat(ifirst,jfirst,ilast,jlast) += s;
          }
        }

        return S;
      }

      arma::mat RadialBasis::Plm_integral(int k, size_t iel, int L, int M, const legendretable::LegendreTable & legtab) const {
        std::function<double(double)> Plm;
        if(k!=0) {
          Plm = [legtab, k, L, M](double mu) { return std::sinh(mu)*std::pow(std::cosh(mu), k)*legtab.get_Plm(L,M,cosh(mu)); };
        } else {
          Plm = [legtab, L, M](double mu) { return std::sinh(mu)*legtab.get_Plm(L,M,cosh(mu)); };
        }
        return fem.matrix_element(iel, false, false, xq, wq, Plm);
      }

      arma::mat RadialBasis::Qlm_integral(int k, size_t iel, int L, int M, const legendretable::LegendreTable & legtab) const {
        std::function<double(double)> Qlm;
        if(k!=0) {
          Qlm = [legtab, k, L, M](double mu) { return std::sinh(mu)*std::pow(std::cosh(mu), k)*legtab.get_Qlm(L,M,cosh(mu)); };
        } else {
          Qlm = [legtab, L, M](double mu) { return std::sinh(mu)*legtab.get_Qlm(L,M,cosh(mu)); };
        }
        return fem.matrix_element(iel, false, false, xq, wq, Qlm);
      }

      arma::mat RadialBasis::kinetic() const {
        std::function<double(double)> sinhmu = [](double mu) {return std::sinh(mu);};
        return fem.matrix_element(true, true, xq, wq, sinhmu);
      }

      arma::mat RadialBasis::twoe_integral(int alpha, int beta, size_t iel, int L, int M, const legendretable::LegendreTable & legtab) const {
        double mumin=fem.element_begin(iel);
        double mumax=fem.element_end(iel);

        // Integral by quadrature
        std::shared_ptr<const helfem::polynomial_basis::PolynomialBasis> p(fem.get_basis(iel));
        arma::mat tei(quadrature::twoe_integral(mumin,mumax,alpha,beta,xq,wq,p,L,M,legtab));

        return tei;
      }

      arma::vec RadialBasis::get_chmu_quad() const {
        // Quadrature points for normal integrals
        arma::vec muq(fem.get_nelem()*xq.n_elem*(xq.n_elem+1));
        size_t ioff=0;

        for(size_t iel=0;iel<fem.get_nelem();iel++) {
          // Element ranges from
          double mumin0=fem.element_begin(iel);
          double mumax0=fem.element_end(iel);

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

      arma::mat RadialBasis::get_bf(size_t iel) const {
        return get_bf(iel, xq);
      }

      arma::mat RadialBasis::get_bf(size_t iel, const arma::vec & x) const {
        return fem.eval_f(x, iel);
      }

      arma::mat RadialBasis::get_df(size_t iel) const {
        return fem.eval_df(xq, iel);
      }

      arma::vec RadialBasis::get_wrad(size_t iel) const {
        // This is just the radial rule, no r^2 factor included here
        return fem.scaling_factor(iel)*wq;
      }

      arma::vec RadialBasis::get_r(size_t iel) const {
        return fem.eval_coord(xq, iel);
      }

      void lm_to_l_m(const arma::ivec & lmax, arma::ivec & lval, arma::ivec & mval) {
        {
          std::vector<arma::sword> lv, mv;
          for(size_t mabs=0;mabs<lmax.n_elem;mabs++)
            for(arma::sword l=mabs;l<=lmax(mabs);l++) {
              lv.push_back(l);
              mv.push_back(mabs);
              if(mabs>0) {
                lv.push_back(l);
                mv.push_back(-mabs);
              }
            }
          lval=arma::conv_to<arma::ivec>::from(lv);
          mval=arma::conv_to<arma::ivec>::from(mv);
        }
      }

      TwoDBasis::TwoDBasis() {
      }

      TwoDBasis::TwoDBasis(int Z1_, int Z2_, double Rhalf_, const std::shared_ptr<const polynomial_basis::PolynomialBasis> &poly, int n_quad, const arma::vec & bval, const arma::ivec & lval_, const arma::ivec & mval_, int lpad, bool legendre) {
        // Nuclear charge
        Z1=Z1_;
        Z2=Z2_;
        Rhalf=Rhalf_;

        // Construct radial basis
        bool zero_func_left=false; // sigma orbitals are allowed to reach the nucleus; this is cleaned up for non-sigma orbitals elsewhere in the code
        bool zero_deriv_left=false;
        bool zero_func_right=true;
        bool zero_deriv_right=true;
        polynomial_basis::FiniteElementBasis fem(poly, bval, zero_func_left, zero_deriv_left, zero_func_right, zero_deriv_right);
        radial=RadialBasis(fem, n_quad);
        // Angular basis
        lval=lval_;
        mval=mval_;

        // Gaunt coefficients
        int gmax(arma::max(lval)+2);

        // Legendre function values
        if(legendre) {
          int Lmax=0;
          int Mmax=0;

          // Form L|M| and LM maps
          lm_map.clear();
          LM_map.clear();
          for(size_t iang=0;iang<lval.n_elem;iang++) {
            for(size_t jang=0;jang<lval.n_elem;jang++) {
              // l and m values
              int li(lval(iang));
              int mi(mval(iang));
              int lj(lval(jang));
              int mj(mval(jang));
              // LH m value
              int M(mj-mi);

              int Lstart=std::max(std::abs(lj-li)-2,abs(M));
              int Lend=lj+li+2;
              for(int L=Lstart;L<=Lend;L++) {
                lmidx_t p;
                p.first=L;

                // Check maxima
                Lmax=std::max(Lmax,L);
                Mmax=std::max(Mmax,std::abs(M));

                // L|M|
                p.second=std::abs(M);
                if(!lm_map.size())
                  lm_map.push_back(p);
                else {
                  size_t idx=lmind(L,M,false);
                  if(!(lm_map[idx]==p))
                    // Insert at lower bound
                    lm_map.insert(lm_map.begin()+idx,p);
                }

                // LM
                p.second=M;
                if(!LM_map.size())
                  LM_map.push_back(p);
                else {
                  size_t idx=LMind(L,M,false);
                  if(!(LM_map[idx]==p))
                    // Insert at lower bound
                    LM_map.insert(LM_map.begin()+idx,p);
                }
              }
            }
          }

          // One-electron matrices need gmax,5,gmax
          // Two-electron matrices need Lmax+2,Lmax,Lmax+2
          int lrval(std::max(Lmax+2,gmax));
          int midval(std::max(Lmax,5));

          Timer t;
          printf("Computing Gaunt coefficients ... ");
          fflush(stdout);
          gaunt=gaunt::Gaunt(lrval,Mmax,midval,Mmax,lrval,Mmax);
          printf("done (% .3f s)\n",t.get());
          fflush(stdout);

          t.set();
          printf("Computing Legendre function values ... ");
          fflush(stdout);

          // Fill table with necessary values
          legtab=legendretable::LegendreTable(Lmax+lpad,Lmax,Mmax);
          arma::vec chmu(radial.get_chmu_quad());
          for(size_t i=0;i<chmu.n_elem;i++)
            legtab.compute(chmu(i));
          printf("done (% .3f s)\n",t.get());
          fflush(stdout);

        } else {
          // One-electron matrices need gmax,5,gmax
          int lrval(gmax);
          int midval(5);
          int Mmax=arma::max(mval)-arma::min(mval);

          gaunt=gaunt::Gaunt(lrval,Mmax,midval,Mmax,lrval,Mmax);
        }
      }

      TwoDBasis::~TwoDBasis() {
      }

      int TwoDBasis::get_Z1() const {
        return Z1;
      }

      int TwoDBasis::get_Z2() const {
        return Z2;
      }

      double TwoDBasis::get_Rhalf() const {
        return Rhalf;
      }

      arma::ivec TwoDBasis::get_lval() const {
        return lval;
      }

      arma::ivec TwoDBasis::get_mval() const {
        return mval;
      }

      int TwoDBasis::get_nquad() const {
        return radial.get_nquad();
      }

      arma::vec TwoDBasis::get_bval() const {
        return radial.get_bval();
      }

      double TwoDBasis::get_mumax() const {
        return radial.get_bval()(radial.get_bval().n_elem-1);
      }

      int TwoDBasis::get_poly_id() const {
        return radial.get_poly_id();
      }

      int TwoDBasis::get_poly_nnodes() const {
        return radial.get_poly_nnodes();
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
            // Remove first function
            nbf--;
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
	    // Just drop the first function
            idx.subvec(ioff,ioff+radial.Nbf()-2)=arma::linspace<arma::uvec>(i*radial.Nbf()+1,(i+1)*radial.Nbf()-1,radial.Nbf()-1);
            ioff+=radial.Nbf()-1;
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
            nm += (m==0) ? radial.Nbf() : radial.Nbf()-1;
          }
        }

        // Collect functions
        arma::uvec idx(nm);
        size_t ioff=0;
        size_t ibf=0;
        for(size_t i=0;i<mval.n_elem;i++) {
          // Number of functions on shell is
          size_t nsh=(mval(i)==0) ? radial.Nbf() : radial.Nbf()-1;
          if(mval(i)==m) {
            idx.subvec(ioff,ioff+nsh-1)=arma::linspace<arma::uvec>(ibf,ibf+nsh-1,nsh);
            ioff+=nsh;
          }
          ibf+=nsh;
        }

        return idx;
      }

      arma::uvec TwoDBasis::m_indices(int m, bool odd) const {
        // Count how many functions
        size_t nm=0;
        for(size_t i=0;i<mval.n_elem;i++) {
          if(mval(i)==m && lval(i)%2==odd) {
            nm += (m==0) ? radial.Nbf() : radial.Nbf()-1;
          }
        }

        // Collect functions
        arma::uvec idx(nm);
        size_t ioff=0;
        size_t ibf=0;
        for(size_t i=0;i<mval.n_elem;i++) {
          // Number of functions on shell is
          size_t nsh=(mval(i)==0) ? radial.Nbf() : radial.Nbf()-1;
          if(mval(i)==m && lval(i)%2==odd) {
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
          arma::ivec mv(arma::sort(mval(muni),"ascend"));

          idx.resize(mv.n_elem);
          for(size_t i=0;i<mv.n_elem;i++)
            idx[i]=m_indices(mv(i));
        } else if(symm==2) {
          // Find unique m values
          arma::uvec muni(arma::find_unique(mval));
          arma::ivec mv(arma::sort(mval(muni),"ascend"));

          idx.resize(2*mv.n_elem);
          for(size_t i=0;i<mv.n_elem;i++) {
            idx[2*i]=m_indices(mv(i),false);
            idx[2*i+1]=m_indices(mv(i),true);
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

        // Half-inverse is
        if(sym==0) {
          return scf::form_Sinvh(S,chol);
        } else {
          // Get basis function indices
          std::vector<arma::uvec> midx(get_sym_idx(sym));
          // Construct Sinvh in each subblock
          arma::mat Sinvh(Nbf(),Nbf(),arma::fill::zeros);
          size_t ioff=0;
          for(size_t i=0;i<midx.size();i++) {
            if(!midx[i].n_elem)
              continue;

            // Column indices
            arma::uvec cidx(arma::linspace<arma::uvec>(ioff,ioff+midx[i].n_elem-1,midx[i].n_elem));
            Sinvh(midx[i],cidx)=scf::form_Sinvh(S(midx[i],midx[i]),chol);
            // Increment offset
            ioff += midx[i].n_elem;
          }
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

        (void) Rexp;
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

      arma::mat TwoDBasis::overlap(const TwoDBasis & rh) const {
        // Build radial matrix elements
        arma::mat I10(radial.overlap(rh.radial,0));
        arma::mat I12(radial.overlap(rh.radial,2));

        // Full overlap matrix
        arma::mat S(Ndummy(),rh.Ndummy());
        S.zeros();
        // Fill elements
        for(size_t iang=0;iang<lval.n_elem;iang++) {
          int li(lval(iang));
          int mi(mval(iang));

          for(size_t jang=0;jang<rh.lval.n_elem;jang++) {
            int lj(rh.lval(jang));
            int mj(rh.mval(jang));

            // Calculate coupling
            if(mi==mj) {
              if(li==lj)
                S.submat(iang*radial.Nbf(),jang*rh.radial.Nbf(),(iang+1)*radial.Nbf()-1,(jang+1)*rh.radial.Nbf()-1)=I12;

              // We can also couple through the cos^2 term
              double cpl(gaunt.cosine2_coupling(lj,mj,li,mi));
              if(cpl!=0.0)
                S.submat(iang*radial.Nbf(),jang*rh.radial.Nbf(),(iang+1)*radial.Nbf()-1,(jang+1)*rh.radial.Nbf()-1)-=I10*cpl;
            }
          }
        }

        // Plug in prefactor
        S*=std::pow(Rhalf,3);

        // Matrix with the boundary conditions removed
        S=S(pure_indices(),rh.pure_indices());

        return S;
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
        // Full electric couplings
        arma::mat V(Ndummy(),Ndummy());
        V.zeros();

        // Build radial matrix elements
        arma::mat I10(radial.radial_integral(1,0));
        arma::mat I12(radial.radial_integral(1,2));
        arma::mat I14(radial.radial_integral(1,4));

        // Fill elements
        for(size_t iang=0;iang<lval.n_elem;iang++) {
          int li(lval(iang));
          int mi(mval(iang));

          for(size_t jang=0;jang<lval.n_elem;jang++) {
            int lj(lval(jang));
            int mj(mval(jang));

            // Calculate coupling
            if(mi==mj) {
              // Coupling through the cos^4 term
              double cpl4(gaunt.cosine4_coupling(lj,mj,li,mi));
              if(cpl4!=0.0)
                add_sub(V,iang,jang,cpl4*(I10-3.0*I12));

              // We can also couple through the cos^2 term
              double cpl2(gaunt.cosine2_coupling(lj,mj,li,mi));
              if(cpl2!=0.0)
                add_sub(V,iang,jang,cpl2*(3.0*I14-I10));

              // or the delta term
              if(li==lj)
                add_sub(V,iang,jang,I12-I14);
            }
          }
        }

        // Plug in prefactors
        V*=std::pow(Rhalf,5)/2;

        return remove_boundaries(V);
      }

      arma::mat TwoDBasis::Bz_field(double B) const {
        // Full couplings
        arma::mat V(Ndummy(),Ndummy());
        V.zeros();

        // Build radial matrix elements
        arma::mat I10(radial.radial_integral(1,0)*std::pow(Rhalf,3));
        arma::mat I12(radial.radial_integral(1,2)*std::pow(Rhalf,3));
        arma::mat I30(radial.radial_integral(3,0)*std::pow(Rhalf,5));
        arma::mat I32(radial.radial_integral(3,2)*std::pow(Rhalf,5));

        // Fill elements
        for(size_t iang=0;iang<lval.n_elem;iang++) {
          int li(lval(iang));
          int mi(mval(iang));

          for(size_t jang=0;jang<lval.n_elem;jang++) {
            int lj(lval(jang));
            int mj(mval(jang));

            // Calculate coupling
            if(mi==mj) {
              // Coupling strength
              double cs(B*B/8.0);

              // Coupling through the sin^2 term
              double cpl2(gaunt.sine2_coupling(lj,mj,li,mi));
              if(cpl2!=0.0)
                add_sub(V,iang,jang,cs*cpl2*I32);

              // We can also couple through the sin^2 cos^2 term
              double cpl22(gaunt.cosine2_sine2_coupling(lj,mj,li,mi));
              if(cpl22!=0.0)
                add_sub(V,iang,jang,-cs*cpl22*I30);

              // m term
              double ds(-0.5*mj*B);
              if(ds!=0.0) {
                if(li==lj)
                  add_sub(V,iang,jang,ds*I12);

                // We can also couple through the cos^2 term
                double cpl(gaunt.cosine2_coupling(lj,mj,li,mi));
                if(cpl!=0.0)
                  add_sub(V,iang,jang,-ds*I10*cpl);
              }
            }
          }
        }

        return remove_boundaries(V);
      }

      arma::mat TwoDBasis::radial_moments(const arma::mat & P0) const {
        // Returned matrix elements
        arma::mat Rmat(4,3); // -1, 1, 2, 3 and lh, cen, rh
        Rmat.zeros();

        enum moment {mone,
                     one,
                     two,
                     three};

        enum center {left,
                     middle,
                     right};

        // Extend to boundaries
        arma::mat P(expand_boundaries(P0));

        // Build radial matrix elements
        arma::mat I10(radial.radial_integral(1,0));
        arma::mat I11(radial.radial_integral(1,1));
        arma::mat I12(radial.radial_integral(1,2));
        arma::mat I13(radial.radial_integral(1,3));
        arma::mat I14(radial.radial_integral(1,4));
        arma::mat I15(radial.radial_integral(1,5));

        // Fill elements
        for(size_t iang=0;iang<lval.n_elem;iang++) {
          int li(lval(iang));
          int mi(mval(iang));

          for(size_t jang=0;jang<lval.n_elem;jang++) {
            int lj(lval(jang));
            int mj(mval(jang));

            // Calculate coupling
            if(mi==mj) {
              // Radial submatrix
              arma::mat Psub(P.submat(iang*radial.Nbf(),jang*radial.Nbf(),(iang+1)*radial.Nbf()-1,(jang+1)*radial.Nbf()-1));

              // <r^2> wrt center
              {
                // Coupling through the cos^2 term
                double cpl2(gaunt.cosine2_coupling(lj,mj,li,mi));
                // Coupling through the cos^4 term
                double cpl4(gaunt.cosine4_coupling(lj,mj,li,mi));
                double cpl24(cpl2-cpl4);
                if(cpl24!=0.0)
                  Rmat(two,middle)+=std::pow(Rhalf,5)*cpl24*arma::trace(Psub*I10);

                // or the delta term
                if(li==lj)
                  Rmat(two,middle)+=std::pow(Rhalf,5)*arma::trace(Psub*(I14-I12));
              }

              // <r^-1>
              {
                if(li==lj) {
                  double tr(arma::trace(Psub*I11));
                  Rmat(mone,left)+=std::pow(Rhalf,2)*tr;
                  Rmat(mone,right)+=std::pow(Rhalf,2)*tr;
                }
                double cpl(gaunt.cosine_coupling(lj,mj,li,mi));
                if(cpl!=0.0) {
                  double tr(arma::trace(Psub*I10));
                  Rmat(mone,left)-=std::pow(Rhalf,2)*cpl*tr;
                  Rmat(mone,right)+=std::pow(Rhalf,2)*cpl*tr;
                }
              }

              // <r>
              {
                if(li==lj) {
                  double tr(arma::trace(Psub*I13));
                  Rmat(one,left)+=std::pow(Rhalf,4)*tr;
                  Rmat(one,right)+=std::pow(Rhalf,4)*tr;
                }
                double cpl(gaunt.cosine_coupling(lj,mj,li,mi));
                if(cpl!=0.0) {
                  double tr(arma::trace(Psub*I12));
                  Rmat(one,left)+=std::pow(Rhalf,4)*cpl*tr;
                  Rmat(one,right)-=std::pow(Rhalf,4)*cpl*tr;
                }
                double cpl2(gaunt.cosine2_coupling(lj,mj,li,mi));
                if(cpl2!=0.0) {
                  double tr(arma::trace(Psub*I11));
                  Rmat(one,left)-=std::pow(Rhalf,4)*cpl2*tr;
                  Rmat(one,right)-=std::pow(Rhalf,4)*cpl2*tr;
                }
                double cpl3(gaunt.cosine3_coupling(lj,mj,li,mi));
                if(cpl3!=0.0) {
                  double tr(arma::trace(Psub*I10));
                  Rmat(one,left)-=std::pow(Rhalf,4)*cpl3*tr;
                  Rmat(one,right)+=std::pow(Rhalf,4)*cpl3*tr;
                }
              }

              // <r^2>
              {
                if(li==lj) {
                  double tr(arma::trace(Psub*I14));
                  Rmat(two,left)+=std::pow(Rhalf,5)*tr;
                  Rmat(two,right)+=std::pow(Rhalf,5)*tr;
                }
                double cpl(gaunt.cosine_coupling(lj,mj,li,mi));
                if(cpl!=0.0) {
                  double tr(arma::trace(Psub*I13));
                  Rmat(two,left)+=2*std::pow(Rhalf,5)*cpl*tr;
                  Rmat(two,right)-=2*std::pow(Rhalf,5)*cpl*tr;
                }
                double cpl3(gaunt.cosine3_coupling(lj,mj,li,mi));
                if(cpl3!=0.0) {
                  double tr(arma::trace(Psub*I11));
                  Rmat(two,left)-=2*std::pow(Rhalf,5)*cpl3*tr;
                  Rmat(two,right)+=2*std::pow(Rhalf,5)*cpl3*tr;
                }
                double cpl4(gaunt.cosine4_coupling(lj,mj,li,mi));
                if(cpl4!=0.0) {
                  double tr(arma::trace(Psub*I10));
                  Rmat(two,left)-=std::pow(Rhalf,5)*cpl4*tr;
                  Rmat(two,right)-=std::pow(Rhalf,5)*cpl4*tr;
                }
              }

              // <r^3>
              {
                if(li==lj) {
                  double tr(arma::trace(Psub*I15));
                  Rmat(three,left)+=std::pow(Rhalf,6)*tr;
                  Rmat(three,right)+=std::pow(Rhalf,6)*tr;
                }
                double cpl(gaunt.cosine_coupling(lj,mj,li,mi));
                if(cpl!=0.0) {
                  double tr(arma::trace(Psub*I14));
                  Rmat(three,left)+=3*std::pow(Rhalf,6)*cpl*tr;
                  Rmat(three,right)-=3*std::pow(Rhalf,6)*cpl*tr;
                }
                double cpl2(gaunt.cosine2_coupling(lj,mj,li,mi));
                if(cpl2!=0.0) {
                  double tr(arma::trace(Psub*I13));
                  Rmat(three,left)+=2*std::pow(Rhalf,6)*cpl2*tr;
                  Rmat(three,right)+=2*std::pow(Rhalf,6)*cpl2*tr;
                }
                double cpl3(gaunt.cosine3_coupling(lj,mj,li,mi));
                if(cpl3!=0.0) {
                  double tr(arma::trace(Psub*I12));
                  Rmat(three,left)-=2*std::pow(Rhalf,6)*cpl3*tr;
                  Rmat(three,right)+=2*std::pow(Rhalf,6)*cpl3*tr;
                }
                double cpl4(gaunt.cosine4_coupling(lj,mj,li,mi));
                if(cpl4!=0.0) {
                  double tr(arma::trace(Psub*I11));
                  Rmat(three,left)-=3*std::pow(Rhalf,6)*cpl4*tr;
                  Rmat(three,right)-=3*std::pow(Rhalf,6)*cpl4*tr;
                }
                double cpl5(gaunt.cosine5_coupling(lj,mj,li,mi));
                if(cpl5!=0.0) {
                  double tr(arma::trace(Psub*I10));
                  Rmat(three,left)-=std::pow(Rhalf,6)*cpl5*tr;
                  Rmat(three,right)+=std::pow(Rhalf,6)*cpl5*tr;
                }
              }
            }
          }
        }

        return Rmat;
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


      void TwoDBasis::compute_tei(bool exchange) {
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

          for(size_t iel=0;iel<Nel;iel++) {
            // Index in array
            {
              const size_t idx(Nel*Nel*ilm + iel*Nel + iel);

              // In-element integrals
              prim_tei00[idx]=radial.twoe_integral(0,0,iel,L,M,legtab);
              prim_tei02[idx]=radial.twoe_integral(0,2,iel,L,M,legtab);
              prim_tei20[idx]=radial.twoe_integral(2,0,iel,L,M,legtab);
              prim_tei22[idx]=radial.twoe_integral(2,2,iel,L,M,legtab);
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
              prim_tei00[idx]=utils::product_tei(i0,j0);
              prim_tei02[idx]=utils::product_tei(i0,j2);
              prim_tei20[idx]=utils::product_tei(i2,j0);
              prim_tei22[idx]=utils::product_tei(i2,j2);
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
        if(exchange) {
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

      static double factorial_ratio(int pmax, int pmin) {
        // Check consistency of arguments
        if(pmax < pmin)
          return 1.0/factorial_ratio(pmin, pmax);

        // Calculate ratio
        double r=1.0;
        for(int p=pmax;p>pmin;p--)
          r*=p;

        return r;
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
              if(cpl0!=0.0)
                Paux0[iLM]+=cpl0*Prad;
              if(cpl2!=0.0)
                Paux2[iLM]+=cpl2*Prad;
            }
          }
        }

        // Coulomb helpers
        std::vector<arma::mat> Jaux0(LM_map.size());
        std::vector<arma::mat> Jaux2(LM_map.size());
        for(size_t i=0;i<Jaux0.size();i++) {
          Jaux0[i].zeros(Nrad,Nrad);
          Jaux2[i].zeros(Nrad,Nrad);
        }
        for(size_t iLM=0;iLM<LM_map.size();iLM++) {
          // Values of L and M
          int L(LM_map[iLM].first);
          int M(LM_map[iLM].second);

          // Helpers
          const size_t ilm(lmind(L,M));
          const double LMfac(4.0*M_PI*std::pow(Rhalf,5)*std::pow(-1.0,M)/factorial_ratio(L+std::abs(M),L-std::abs(M)));

          // Loop over input elements
          for(size_t jel=0;jel<Nel;jel++) {
            size_t jfirst, jlast;
            radial.get_idx(jel,jfirst,jlast);
            size_t Nj(jlast-jfirst+1);

            // Get density submatrices
            arma::mat Psub0(Paux0[iLM].submat(jfirst,jfirst,jlast,jlast));
            arma::mat Psub2(Paux2[iLM].submat(jfirst,jfirst,jlast,jlast));

            // Contract integrals
            double jsmall0 = LMfac*arma::trace(disjoint_P0[ilm*Nel+jel]*Psub0);
            double jbig0 = LMfac*arma::trace(disjoint_Q0[ilm*Nel+jel]*Psub0);
            double jsmall2 = LMfac*arma::trace(disjoint_P2[ilm*Nel+jel]*Psub2);
            double jbig2 = LMfac*arma::trace(disjoint_Q2[ilm*Nel+jel]*Psub2);

            // Increment J: jel>iel
            double ifac0(jbig0 - jbig2);
            double ifac2(-jbig0 + jbig2);
            for(size_t iel=0;iel<jel;iel++) {
              size_t ifirst, ilast;
              radial.get_idx(iel,ifirst,ilast);

              const arma::mat & iint0=disjoint_P0[ilm*Nel+iel];
              const arma::mat & iint2=disjoint_P2[ilm*Nel+iel];
              Jaux0[iLM].submat(ifirst,ifirst,ilast,ilast)+=iint0*ifac0;
              Jaux2[iLM].submat(ifirst,ifirst,ilast,ilast)+=iint2*ifac2;
            }

            // Increment J: jel<iel
            ifac0=jsmall0 - jsmall2;
            ifac2=-jsmall0 + jsmall2;
            for(size_t iel=jel+1;iel<Nel;iel++) {
              size_t ifirst, ilast;
              radial.get_idx(iel,ifirst,ilast);

              const arma::mat & iint0=disjoint_Q0[ilm*Nel+iel];
              const arma::mat & iint2=disjoint_Q2[ilm*Nel+iel];
              Jaux0[iLM].submat(ifirst,ifirst,ilast,ilast)+=iint0*ifac0;
              Jaux2[iLM].submat(ifirst,ifirst,ilast,ilast)+=iint2*ifac2;
            }

            // In-element contribution
            {
              size_t iel=jel;
              size_t ifirst=jfirst;
              size_t ilast=jlast;
              size_t Ni=Nj;

              // Contract integrals
              arma::mat Jsub0(Ni*Ni,1);
              Jsub0.zeros();
              arma::mat Jsub2(Ni*Ni,1);
              Jsub2.zeros();

              Psub0.reshape(Nj*Nj,1);
              Psub2.reshape(Nj*Nj,1);

              const size_t idx(Nel*Nel*ilm + iel*Nel + jel);
              Jsub0+=LMfac*(prim_tei00[idx]*Psub0);
              Jsub0-=LMfac*(prim_tei02[idx]*Psub2);
              Jsub2-=LMfac*(prim_tei20[idx]*Psub0);
              Jsub2+=LMfac*(prim_tei22[idx]*Psub2);

              Jsub0.reshape(Ni,Ni);
              Jsub2.reshape(Ni,Ni);

              // Increment global Coulomb matrix
              Jaux0[iLM].submat(ifirst,ifirst,ilast,ilast)+=Jsub0;
              Jaux2[iLM].submat(ifirst,ifirst,ilast,ilast)+=Jsub2;
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

            int Lmin=std::max(std::abs(lj-li)-2,abs(M));
            int Lmax=lj+li+2;
            for(int L=Lmin;L<=Lmax;L++) {
              const size_t iLM(LMind(L,M));

              // Couplings
              double cpl0(gaunt.mod_coeff(lj,mj,L,M,li,mi));
              if(cpl0!=0.0) {
                J.submat(iang*Nrad,jang*Nrad,(iang+1)*Nrad-1,(jang+1)*Nrad-1)+=cpl0*Jaux0[iLM];
              }

              double cpl2(gaunt.coeff(lj,mj,L,M,li,mi));
              if(cpl2!=0.0) {
                J.submat(iang*Nrad,jang*Nrad,(iang+1)*Nrad-1,(jang+1)*Nrad-1)+=cpl2*Jaux2[iLM];
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
          mem_Psub[ith].zeros(Nrad*Nrad); // Used in agular sum

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
              std::vector<arma::mat> Rmat00(lm_map.size());
              std::vector<arma::mat> Rmat02(lm_map.size());
              std::vector<arma::mat> Rmat20(lm_map.size());
              std::vector<arma::mat> Rmat22(lm_map.size());
              for(size_t i=0;i<lm_map.size();i++) {
                Rmat00[i].zeros(Nrad,Nrad);
                Rmat02[i].zeros(Nrad,Nrad);
                Rmat20[i].zeros(Nrad,Nrad);
                Rmat22[i].zeros(Nrad,Nrad);
              }
              // Is there a coupling to the channel?
              std::vector<bool> couple(lm_map.size(),false);

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
                  int Lmin=std::max(std::max(std::abs(li-lj),std::abs(lk-ll))-2,abs(M));
                  int Lmax=std::min(li+lj,lk+ll)+2;

                  for(int L=Lmin;L<=Lmax;L++) {
                    // Calculate total coupling coefficient
                    double cpl00(gaunt.mod_coeff(lj,mj,L,M,li,mi)*gaunt.mod_coeff(lk,mk,L,M,ll,ml));
                    double cpl02(-gaunt.mod_coeff(lj,mj,L,M,li,mi)*gaunt.coeff(lk,mk,L,M,ll,ml));
                    double cpl20(-gaunt.coeff(lj,mj,L,M,li,mi)*gaunt.mod_coeff(lk,mk,L,M,ll,ml));
                    double cpl22(gaunt.coeff(lj,mj,L,M,li,mi)*gaunt.coeff(lk,mk,L,M,ll,ml));

                    // Is there any coupling?
                    if(cpl00==0.0 && cpl02==0.0 && cpl20==0.0 && cpl22==0.0)
                      continue;

                    // Index in the L,|M| table
                    const size_t ilm(lmind(L,M));
                    const double LMfac(4.0*M_PI*std::pow(Rhalf,5)*std::pow(-1.0,M)/factorial_ratio(L+std::abs(M),L-std::abs(M)));

                    arma::mat Psub(mem_Psub[ith].memptr(),Nrad,Nrad,false,true);
                    Psub=P.submat(iang*Nrad,lang*Nrad,(iang+1)*Nrad-1,(lang+1)*Nrad-1);

                    Rmat00[ilm]+=(LMfac*cpl00)*Psub;
                    Rmat02[ilm]+=(LMfac*cpl02)*Psub;
                    Rmat20[ilm]+=(LMfac*cpl20)*Psub;
                    Rmat22[ilm]+=(LMfac*cpl22)*Psub;
                    couple[ilm]=true;
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

                    for(size_t ilm=0;ilm<lm_map.size();ilm++) {
                      if(!couple[ilm])
                        continue;
                      // Index in tei array
                      size_t idx=Nel*Nel*ilm + iel*Nel + jel;
                      Ksub+=prim_ktei00[idx]*arma::vectorise(Rmat00[ilm].submat(ifirst,jfirst,ilast,jlast));
                      Ksub+=prim_ktei02[idx]*arma::vectorise(Rmat02[ilm].submat(ifirst,jfirst,ilast,jlast));
                      Ksub+=prim_ktei20[idx]*arma::vectorise(Rmat20[ilm].submat(ifirst,jfirst,ilast,jlast));
                      Ksub+=prim_ktei22[idx]*arma::vectorise(Rmat22[ilm].submat(ifirst,jfirst,ilast,jlast));
                    }

                    Ksub.reshape(Ni,Nj);

                    // Increment global exchange matrix
                    K.submat(jang*Nrad+ifirst,kang*Nrad+jfirst,jang*Nrad+ilast,kang*Nrad+jlast)-=Ksub;

                    //arma::vec Ptgt(arma::vectorise(P.submat(jang*Nrad+ifirst,kang*Nrad+jfirst,jang*Nrad+ilast,kang*Nrad+jlast)));
                    //printf("(%i %i) (%i %i) (%i %i) (%i %i) [%i %i]\n",li,mi,lj,mj,lk,mk,ll,ml,L,M);
                    //printf("Element %i - %i contribution to exchange energy % .10e\n",(int) iel,(int) jel,-0.5*arma::dot(Ksub,Ptgt));

                  } else {
                    arma::mat Ksub(mem_Ksub[ith].memptr(),Ni,Nj,false,true);
                    Ksub.zeros();
                    arma::mat T(mem_T[ith].memptr(),Ni*Nj,1,false,true);
                    for(size_t ilm=0;ilm<lm_map.size();ilm++) {
                      if(!couple[ilm])
                        continue;
                      // Disjoint integrals. When r(iel)>r(jel), iel gets Q, jel gets P.
                      const arma::mat & iint0=(iel>jel) ? disjoint_Q0[ilm*Nel+iel] : disjoint_P0[ilm*Nel+iel];
                      const arma::mat & iint2=(iel>jel) ? disjoint_Q2[ilm*Nel+iel] : disjoint_P2[ilm*Nel+iel];
                      const arma::mat & jint0=(iel>jel) ? disjoint_P0[ilm*Nel+jel] : disjoint_Q0[ilm*Nel+jel];
                      const arma::mat & jint2=(iel>jel) ? disjoint_P2[ilm*Nel+jel] : disjoint_Q2[ilm*Nel+jel];

                      // (Niel x Njel) = (Niel x Njel) x (Njel x Njel)
                      T=Rmat00[ilm].submat(ifirst,jfirst,ilast,jlast)*arma::trans(jint0) + Rmat02[ilm].submat(ifirst,jfirst,ilast,jlast)*arma::trans(jint2);
                      Ksub-=iint0*T;

                      T=Rmat20[ilm].submat(ifirst,jfirst,ilast,jlast)*arma::trans(jint0) + Rmat22[ilm].submat(ifirst,jfirst,ilast,jlast)*arma::trans(jint2);
                      Ksub-=iint2*T;
                    }

                    // Increment global exchange matrix
                    K.submat(jang*Nrad+ifirst,kang*Nrad+jfirst,jang*Nrad+ilast,kang*Nrad+jlast)+=Ksub;
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

      arma::cx_mat TwoDBasis::eval_bf(size_t iel, size_t irad, double cth, double phi) const {
        // Evaluate spherical harmonics
        arma::cx_vec sph(lval.n_elem);
        for(size_t i=0;i<lval.n_elem;i++)
          sph(i)=::spherical_harmonics(lval(i),mval(i),cth,phi);

        // Evaluate radial functions
        arma::mat rad(radial.get_bf(iel));
        rad=rad.rows(irad,irad);

        // Form supermatrix
        arma::cx_mat bf(rad.n_rows,lval.n_elem*rad.n_cols);
        for(size_t i=0;i<lval.n_elem;i++)
          bf.cols(i*rad.n_cols,(i+1)*rad.n_cols-1)=sph(i)*rad;

        return bf;
      }

      arma::cx_mat TwoDBasis::eval_bf(size_t iel, const arma::vec & x, double cth, double phi) const {
        // Evaluate spherical harmonics
        arma::cx_vec sph(lval.n_elem);
        for(size_t i=0;i<lval.n_elem;i++)
          sph(i)=::spherical_harmonics(lval(i),mval(i),cth,phi);

        // Evaluate radial functions
        arma::mat rad(radial.get_bf(iel,x));

        // Form supermatrix
        arma::cx_mat bf(rad.n_rows,lval.n_elem*rad.n_cols);
        for(size_t i=0;i<lval.n_elem;i++)
          bf.cols(i*rad.n_cols,(i+1)*rad.n_cols-1)=sph(i)*rad;

        return bf;
      }

      arma::mat TwoDBasis::eval_bf(size_t iel, size_t irad, double cth, int m) const {
        // Figure out list of functions
        std::vector<arma::uword> flist;
        for(size_t i=0;i<mval.n_elem;i++)
          if(mval(i)==m)
            flist.push_back(i);

        // Evaluate spherical harmonics
        arma::vec sph(flist.size());
        for(size_t i=0;i<flist.size();i++)
          sph(i)=std::real(::spherical_harmonics(lval(flist[i]),mval(flist[i]),cth,0.0));

        // Evaluate radial functions
        arma::mat rad(radial.get_bf(iel));
        rad=rad.rows(irad,irad);

        // Form supermatrix
        arma::mat bf(rad.n_rows,flist.size()*rad.n_cols);
        for(size_t i=0;i<flist.size();i++)
          bf.cols(i*rad.n_cols,(i+1)*rad.n_cols-1)=sph(i)*rad;

        return bf;
      }

      arma::cx_vec TwoDBasis::eval_bf(double mu, double cth, double phi) const {
	// Find out which element mu belongs to
	arma::vec bval(radial.get_bval());
	size_t iel;
	for(iel=0;iel<bval.n_elem-1;iel++)
	  if(bval(iel)<=mu && mu<=bval(iel+1))
	    break;
	if(iel==bval.n_elem-1) {
	  std::ostringstream oss;
	  oss << "mu value " << mu << " not found!\n";
	  throw std::logic_error(oss.str());
	}

	// x value is then
	arma::vec x(1);
	x(0)=2.0*(mu-bval(iel))/(bval(iel+1)-bval(iel)) - 1.0;

	// Evaluate spherical harmonics
        arma::cx_vec sph(lval.n_elem);
        for(size_t i=0;i<lval.n_elem;i++)
          sph(i)=::spherical_harmonics(lval(i),mval(i),cth,phi);
        // Evaluate radial functions
        arma::mat rad(radial.get_bf(iel,x));

	// Get indices of radial functions
	size_t ifirst, ilast;
        radial.get_idx(iel,ifirst,ilast);

        // Form supermatrix
	arma::cx_vec bf;
	bf.zeros(Ndummy());
	for(size_t i=0;i<lval.n_elem;i++) {
	  bf.subvec(i*radial.Nbf()+ifirst,i*radial.Nbf()+ilast)=sph(i)*arma::trans(rad);
	}

	return bf(pure_indices());
      }

      void TwoDBasis::eval_df(size_t iel, size_t irad, double cth, double phi, arma::cx_mat & dr, arma::cx_mat & dth, arma::cx_mat & dphi) const {
        // Evaluate spherical harmonics
        arma::cx_vec sph(lval.n_elem);
        for(size_t i=0;i<lval.n_elem;i++)
          sph(i)=::spherical_harmonics(lval(i),mval(i),cth,phi);

        // Evaluate radial functions
        arma::mat frad(radial.get_bf(iel));
        arma::mat drad(radial.get_df(iel));
        frad=frad.rows(irad,irad);
        drad=drad.rows(irad,irad);

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

      arma::uvec TwoDBasis::dummy_idx_to_real_idx(const arma::uvec & idx) const {
        if(arma::max(idx)>=Ndummy())
          throw std::logic_error("Invalid index vector!\n");

        // idx is a subset of dummy indices, which need to be
        // converted to real indices.  we just need to build the map
        // from dummy to real indices. First, form full list of dummy
        // indices
        arma::uvec dummy_idx(arma::linspace<arma::uvec>(0,Ndummy()-1,Ndummy()));
        // This is the corresponding list of real indices
        arma::uvec real_idx(dummy_idx(pure_indices()));
        // Now the list has all the info needed to construct the
        // mapping between the two
        std::map<arma::uword, arma::uword> mapping;
        for(arma::uword i=0;i<real_idx.n_elem;i++) {
          mapping[real_idx[i]] = i;
        }

        // Mapped indices
        std::vector<arma::uword> mapidx;
        for(size_t i=0;i<idx.n_elem;i++) {
          // Try to find the function
          std::map<arma::uword, arma::uword>::const_iterator pos(mapping.find(idx(i)));
          // Dummy functions are not on the map
          if(pos == mapping.end())
            continue;
          // If we are here, the function is real
          mapidx.push_back(mapping.at(idx(i)));
        }
        return arma::conv_to<arma::uvec>::from(mapidx);
      }

      arma::uvec TwoDBasis::bf_list_dummy(size_t iel) const {
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

      arma::uvec TwoDBasis::bf_list(size_t iel) const {
        return dummy_idx_to_real_idx(bf_list_dummy(iel));
      }

      arma::uvec TwoDBasis::bf_list_dummy(size_t iel, int m) const {
        // Radial functions in element
        size_t ifirst, ilast;
        radial.get_idx(iel,ifirst,ilast);
        // Number of radial functions in element
        size_t Nr(ilast-ifirst+1);

        // Total number of radial functions
        size_t Nrad(radial.Nbf());

        // List of functions in the element
        std::vector<arma::uword> idx;
        for(size_t iam=0;iam<lval.n_elem;iam++)
          if(mval(iam)==m)
            for(size_t j=0;j<Nr;j++)
              idx.push_back(Nrad*iam+ifirst+j);

        return arma::conv_to<arma::uvec>::from(idx);
      }

      size_t TwoDBasis::get_rad_Nel() const {
        return radial.Nel();
      }

      arma::mat TwoDBasis::get_rad_bf(size_t iel) const {
        return radial.get_bf(iel);
      }

      arma::vec TwoDBasis::get_wrad(size_t iel) const {
        return radial.get_wrad(iel);
      }

      arma::vec TwoDBasis::get_r(size_t iel) const {
        return radial.get_r(iel);
      }

      arma::vec TwoDBasis::nuclear_density(const arma::mat & P0) const {
        // List of functions in the first element
        arma::uvec fidx(bf_list(0));

        // Expand density matrix to boundary conditions
        arma::mat P(expand_boundaries(P0));
        // and grab the contribution from the first element
        P=P(fidx,fidx);

        // Nucleus is at -1 on the primitive polynomial interval [-1,1]
        arma::vec x(1);
        x(0)=-1.0;
        // Evaluate first basis functions in first element at both nuclei
        arma::cx_mat bf_one(eval_bf(0,x,1.0,0.0));
        arma::cx_mat bf_none(eval_bf(0,x,-1.0,0.0));

        arma::vec den(2);
        den(0)=arma::as_scalar(arma::real(bf_none*P*arma::trans(bf_none)));
        den(1)=arma::as_scalar(arma::real(bf_one*P*arma::trans(bf_one)));

        return den;
      }
    }
  }
}
