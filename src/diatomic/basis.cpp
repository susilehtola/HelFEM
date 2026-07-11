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
 * SPDX-License-Identifier: BSD-3-Clause
 * See the LICENSE file at the root of this source distribution
 * for the full license text.
 */
#include "basis.h"
#include "quadrature.h"
#include "PolynomialBasis.h"
#include "chebyshev.h"
#include <ArmaEigen.h>
#include "../general/angular_index_helpers.h"
#include <cstring>
#include "../general/spherical_harmonics.h"
#include "../general/gaunt.h"
#include "../general/gsz.h"
#include "tei_utils.h"
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
        lib1dfem::chebyshev::chebyshev<double>(n_quad,xq,wq);
        for(Eigen::Index i=0;i<xq.size();i++) {
          if(!std::isfinite(xq[i]))
            printf("xq[%i]=%e\n",(int) i, xq[i]);
          if(!std::isfinite(wq[i]))
            printf("wq[%i]=%e\n",(int) i, wq[i]);
        }
      }

      RadialBasis::~RadialBasis() {
      }

      int RadialBasis::get_nquad() const {
        return (int) xq.size();
      }

      helfem::Vector RadialBasis::get_bval() const {
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

      helfem::Matrix RadialBasis::radial_integral(int m, int n) const {
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

      helfem::Matrix RadialBasis::overlap(const RadialBasis & rh, int n) const {
        // Use the larger number of quadrature points to make sure the
        // projection is computed correctly.
        size_t n_quad(std::max(xq.size(), rh.xq.size()));

        helfem::Vector xproj, wproj;
        lib1dfem::chebyshev::chebyshev<double>(n_quad, xproj, wproj);

        // Find element pairs that share any mu range.
        std::vector<std::vector<size_t>> overlap(fem.get_nelem());
        for(size_t iel=0;iel<fem.get_nelem();iel++) {
          double istart(fem.element_begin(iel));
          double iend(fem.element_end(iel));
          for(size_t jel=0;jel<rh.fem.get_nelem();jel++) {
            double jstart(rh.fem.element_begin(jel));
            double jend(rh.fem.element_end(jel));
            if((jstart >= istart && jstart<iend) || (istart >= jstart && istart < jend))
              overlap[iel].push_back(jel);
          }
        }

        helfem::Matrix S(helfem::Matrix::Zero(Nbf(), rh.Nbf()));
        for(size_t iel=0;iel<fem.get_nelem();iel++) {
          for(size_t jj=0;jj<overlap[iel].size();jj++) {
            size_t jel=overlap[iel][jj];
            // FE-side element ranges.
            double imin(fem.element_begin(iel));
            double imax(fem.element_end(iel));
            double jmin(rh.fem.element_begin(jel));
            double jmax(rh.fem.element_end(jel));
            // Shared range.
            double intstart(std::max(imin, jmin));
            double intend(std::min(imax, jmax));
            double intmid(0.5*(intend+intstart));
            double intlen(0.5*(intend-intstart));

            helfem::Vector mu(intmid*helfem::Vector::Ones(xproj.size())+intlen*xproj);
            helfem::Vector xi(fem.eval_prim(mu, iel));
            helfem::Vector xj(rh.fem.eval_prim(mu, jel));

            size_t ifirst, ilast;
            get_idx(iel, ifirst, ilast);
            size_t jfirst, jlast;
            rh.get_idx(jel, jfirst, jlast);

            helfem::Vector wtot(wproj*intlen);
            wtot.array() *= mu.array().sinh();
            if(n!=0) wtot.array() *= mu.array().cosh().pow((double) n);
            helfem::Matrix ibf(fem.eval_f(xi, iel));
            helfem::Matrix jbf(rh.fem.eval_f(xj, jel));
            helfem::Matrix s(ibf.transpose()*wtot.asDiagonal()*jbf);
            S.block(ifirst, jfirst, ilast-ifirst+1, jlast-jfirst+1) += s;
          }
        }
        return S;
      }

      helfem::Matrix RadialBasis::Plm_integral(int k, size_t iel, int L, int M, const legendretable::LegendreTable & legtab) const {
        std::function<double(double)> Plm;
        if(k!=0) {
          Plm = [legtab, k, L, M](double mu) { return std::sinh(mu)*std::pow(std::cosh(mu), k)*legtab.get_Plm(L,M,cosh(mu)); };
        } else {
          Plm = [legtab, L, M](double mu) { return std::sinh(mu)*legtab.get_Plm(L,M,cosh(mu)); };
        }
        return fem.matrix_element(iel, false, false, xq, wq, Plm);
      }

      helfem::Matrix RadialBasis::Qlm_integral(int k, size_t iel, int L, int M, const legendretable::LegendreTable & legtab) const {
        std::function<double(double)> Qlm;
        if(k!=0) {
          Qlm = [legtab, k, L, M](double mu) { return std::sinh(mu)*std::pow(std::cosh(mu), k)*legtab.get_Qlm(L,M,cosh(mu)); };
        } else {
          Qlm = [legtab, L, M](double mu) { return std::sinh(mu)*legtab.get_Qlm(L,M,cosh(mu)); };
        }
        return fem.matrix_element(iel, false, false, xq, wq, Qlm);
      }

      helfem::Matrix RadialBasis::kinetic() const {
        std::function<double(double)> sinhmu = [](double mu) {return std::sinh(mu);};
        return fem.matrix_element(true, true, xq, wq, sinhmu);
      }

      helfem::Matrix RadialBasis::twoe_integral(int alpha, int beta, size_t iel, int L, int M, const legendretable::LegendreTable & legtab) const {
        double mumin=fem.element_begin(iel);
        double mumax=fem.element_end(iel);

        // Integral by quadrature
        std::shared_ptr<const helfem::polynomial_basis::PolynomialBasis> p(fem.get_basis(iel));
        helfem::Matrix tei(helfem::to_eigen(quadrature::twoe_integral(mumin,mumax,alpha,beta,helfem::to_arma(xq),helfem::to_arma(wq),p,L,M,legtab)));

        return tei;
      }

      helfem::Vector RadialBasis::get_chmu_quad() const {
        // Quadrature points for normal integrals
        helfem::Vector muq(fem.get_nelem()*xq.size()*(xq.size()+1));
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
          helfem::Vector mu0(mumid0*helfem::Vector::Ones(xq.size())+mulen0*xq);

          // Store values
          muq.segment(ioff,mu0.size())=mu0;
          ioff+=mu0.size();

          // Subintervals for in-element two-electron integrals
          for(Eigen::Index isub=0;isub<xq.size();isub++) {
            double mumin = (isub==0) ? mumin0 : mu0(isub-1);
            double mumax = mu0(isub);

            double mumid(0.5*(mumax+mumin));
            double mulen(0.5*(mumax-mumin));
            helfem::Vector mu(mumid*helfem::Vector::Ones(xq.size())+mulen*xq);
            muq.segment(ioff,mu.size())=mu;
            ioff+=mu.size();
          }
        }

        // Sort ascending, then take cosh
        std::sort(muq.data(), muq.data()+muq.size());
        return muq.array().cosh().matrix();
      }

      helfem::Matrix RadialBasis::get_bf(size_t iel) const {
        return get_bf(iel, xq);
      }

      helfem::Matrix RadialBasis::get_bf(size_t iel, const helfem::Vector & x) const {
        return fem.eval_f(x, iel);
      }

      helfem::Matrix RadialBasis::get_df(size_t iel) const {
        return fem.eval_df(xq, iel);
      }

      helfem::Vector RadialBasis::get_wrad(size_t iel) const {
        // This is just the radial rule, no r^2 factor included here
        return fem.scaling_factor(iel)*wq;
      }

      helfem::Vector RadialBasis::get_r(size_t iel) const {
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

      TwoDBasis::TwoDBasis(int Z1_, int Z2_, double Rhalf_, const std::shared_ptr<const polynomial_basis::PolynomialBasis> &poly, int n_quad, const arma::vec & bval, const arma::ivec & lval_, const arma::ivec & mval_, bool legendre) {
        // Nuclear charge
        Z1=Z1_;
        Z2=Z2_;
        Rhalf=Rhalf_;

        // Construct radial basis
        bool zero_func_left=false; // sigma orbitals are allowed to reach the nucleus; this is cleaned up for non-sigma orbitals elsewhere in the code
        bool zero_deriv_left=false;
        bool zero_func_right=true;
        bool zero_deriv_right=true;
        polynomial_basis::FiniteElementBasis fem(poly, helfem::to_eigen(bval), zero_func_left, zero_deriv_left, zero_func_right, zero_deriv_right);
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

                // L|M|. lmind returns idx == lm_map.size() when the
                // requested (L, |M|) sorts AFTER every existing entry
                // (e.g. right after the first push_back). Do not touch
                // lm_map[idx] in that case -- under GCC 16's hardened
                // std::vector::operator[] that is a hard abort; under
                // older libstdc++ it silently returned garbage that
                // happened to compare unequal to p, so the map was
                // built correctly by luck.
                p.second=std::abs(M);
                if(!lm_map.size()) {
                  lm_map.push_back(p);
                } else {
                  size_t idx=lmind(L,M,false);
                  if (idx == lm_map.size())
                    lm_map.push_back(p);
                  else if (!(lm_map[idx]==p))
                    lm_map.insert(lm_map.begin()+idx,p);
                }

                // LM (same guard as above).
                p.second=M;
                if(!LM_map.size()) {
                  LM_map.push_back(p);
                } else {
                  size_t idx=LMind(L,M,false);
                  if (idx == LM_map.size())
                    LM_map.push_back(p);
                  else if (!(LM_map[idx]==p))
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
          gaunt=gaunt::Gaunt(lrval,midval,lrval);
          printf("done (% .3f s)\n",t.get());
          fflush(stdout);

          t.set();
          printf("Computing Legendre function values ... ");
          fflush(stdout);

          // Fill table with necessary values
          legtab=legendretable::LegendreTable(Lmax,Mmax);
          helfem::Vector chmu(radial.get_chmu_quad());
          for(Eigen::Index i=0;i<chmu.size();i++)
            legtab.compute(chmu(i));
          printf("done (% .3f s)\n",t.get());
          fflush(stdout);

        } else {
          // One-electron matrices need gmax,5,gmax
          int lrval(gmax);
          int midval(5);
          int Mmax=arma::max(mval)-arma::min(mval);

          gaunt=gaunt::Gaunt(lrval,midval,lrval);
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
        return helfem::to_arma(radial.get_bval());
      }

      double TwoDBasis::get_mumax() const {
        helfem::Vector bval(radial.get_bval());
        return bval(bval.size()-1);
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

      arma::uvec TwoDBasis::m_indices(int m) const {
        return helfem::collect_shell_indices(mval.n_elem,
            [&](size_t i) { return (mval(i) == 0) ? radial.Nbf() : radial.Nbf() - 1; },
            [&](size_t i) { return mval(i) == m; });
      }

      arma::uvec TwoDBasis::m_indices(int m, bool odd) const {
        return helfem::collect_shell_indices(mval.n_elem,
            [&](size_t i) { return (mval(i) == 0) ? radial.Nbf() : radial.Nbf() - 1; },
            [&](size_t i) { return mval(i) == m && (lval(i) % 2 == odd); });
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

      helfem::Matrix TwoDBasis::Sinvh(bool chol, int sym) const {
        // Form overlap matrix
        helfem::Matrix S(overlap());

        // Half-inverse is
        if(sym==0) {
          return scf::form_Sinvh(S, chol);
        } else {
          // Get basis function indices
          std::vector<arma::uvec> midx(get_sym_idx(sym));
          // Construct Sinvh in each subblock
          helfem::Matrix Sinvh(helfem::Matrix::Zero(Nbf(),Nbf()));
          size_t ioff=0;
          for(size_t i=0;i<midx.size();i++) {
            if(!midx[i].n_elem)
              continue;

            // Gather the S(midx,midx) subblock
            const size_t n(midx[i].n_elem);
            helfem::Matrix Ssub(n,n);
            for(size_t a=0;a<n;a++)
              for(size_t b=0;b<n;b++)
                Ssub(a,b)=S(midx[i](a), midx[i](b));

            helfem::Matrix block(scf::form_Sinvh(Ssub,chol));

            // Scatter into Sinvh(midx[i], ioff..ioff+n-1)
            for(size_t a=0;a<n;a++)
              for(size_t b=0;b<n;b++)
                Sinvh(midx[i](a), ioff+b)=block(a,b);
            // Increment offset
            ioff += n;
          }
          return Sinvh;
        }
      }

      void TwoDBasis::set_sub(helfem::Matrix & M, size_t iang, size_t jang, const helfem::Matrix & Mrad) const {
        const size_t N(radial.Nbf());
        M.block(iang*N,jang*N,N,N)=Mrad;
      }

      void TwoDBasis::add_sub(helfem::Matrix & M, size_t iang, size_t jang, const helfem::Matrix & Mrad) const {
        const size_t N(radial.Nbf());
        M.block(iang*N,jang*N,N,N)+=Mrad;
      }

      helfem::Matrix TwoDBasis::overlap() const {
        // Build radial matrix elements
        helfem::Matrix I10(radial.radial_integral(1,0));
        helfem::Matrix I12(radial.radial_integral(1,2));

        // Full overlap matrix
        helfem::Matrix S(helfem::Matrix::Zero(Ndummy(),Ndummy()));
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
                add_sub(S,iang,jang,-cpl*I10);
            }
          }
        }

        // Plug in prefactor
        S*=std::pow(Rhalf,3);

        return helfem::to_eigen(remove_boundaries(helfem::to_arma(S)));
      }

      helfem::Matrix TwoDBasis::overlap(const TwoDBasis & rh) const {
        // Cross-basis overlap. Same coupling structure as overlap() but
        // with rh's radial basis on the right.
        helfem::Matrix I10(radial.overlap(rh.radial, 0));
        helfem::Matrix I12(radial.overlap(rh.radial, 2));

        const size_t Ni(radial.Nbf());
        const size_t Nj(rh.radial.Nbf());
        helfem::Matrix S(helfem::Matrix::Zero(Ndummy(), rh.Ndummy()));
        for(size_t iang=0;iang<lval.n_elem;iang++) {
          int li(lval(iang));
          int mi(mval(iang));
          for(size_t jang=0;jang<rh.lval.n_elem;jang++) {
            int lj(rh.lval(jang));
            int mj(rh.mval(jang));
            if(mi==mj) {
              if(li==lj)
                S.block(iang*Ni, jang*Nj, Ni, Nj) = I12;
              double cpl(gaunt.cosine2_coupling(lj, mj, li, mi));
              if(cpl!=0.0)
                S.block(iang*Ni, jang*Nj, Ni, Nj) -= cpl*I10;
            }
          }
        }
        S *= std::pow(Rhalf, 3);
        // Trim boundaries on both sides so shapes align with Sinvh_new /
        // Sinvh_old. pure_indices() (kept arma) gives the gather lists.
        arma::uvec ridx(pure_indices());
        arma::uvec cidx(rh.pure_indices());
        helfem::Matrix Strim(ridx.n_elem, cidx.n_elem);
        for(size_t a=0;a<ridx.n_elem;a++)
          for(size_t b=0;b<cidx.n_elem;b++)
            Strim(a,b)=S(ridx(a), cidx(b));
        return Strim;
      }

      helfem::Matrix TwoDBasis::kinetic() const {
        // Build radial kinetic energy matrix
        helfem::Matrix Trad(radial.kinetic());
        helfem::Matrix Ip1(radial.radial_integral(1,0));
        helfem::Matrix Im1(radial.radial_integral(-1,0));

        // Full kinetic energy matrix
        helfem::Matrix T(helfem::Matrix::Zero(Ndummy(),Ndummy()));
        // Fill elements
        for(size_t iang=0;iang<lval.n_elem;iang++) {
          set_sub(T,iang,iang,Trad);
          if(lval(iang)!=0) {
            // We also get the l(l+1) term
            add_sub(T,iang,iang,(double) (lval(iang)*(lval(iang)+1))*Ip1);
          }
          if(mval(iang)!=0) {
            // We also get the m^2 term
            add_sub(T,iang,iang,(double) (mval(iang)*mval(iang))*Im1);
          }
        }

        // Plug in prefactor
        T*=Rhalf/2.0;

        return helfem::to_eigen(remove_boundaries(helfem::to_arma(T)));
      }

      helfem::Matrix TwoDBasis::nuclear() const {
        // Build radial matrices
        helfem::Matrix I10(radial.radial_integral(1,0));
        helfem::Matrix I11(radial.radial_integral(1,1));

        // Full nuclear attraction matrix
        helfem::Matrix V(helfem::Matrix::Zero(Ndummy(),Ndummy()));

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
                set_sub(V,iang,jang,(double) (Z1+Z2)*I11);

              // We can also couple through the cos term
	      if(Z1!=Z2) {
		double cpl(gaunt.cosine_coupling(lj,mj,li,mi));
		if(cpl!=0.0)
		  add_sub(V,iang,jang,((double) (Z2-Z1)*cpl)*I10);
	      }
            }
          }
        }

        // Plug in prefactor
        V*=-std::pow(Rhalf,2);

        return helfem::to_eigen(remove_boundaries(helfem::to_arma(V)));
      }

      helfem::Matrix TwoDBasis::dipole_z() const {
        // Full electric couplings
        helfem::Matrix V(helfem::Matrix::Zero(Ndummy(),Ndummy()));

        // Build radial matrix elements
        helfem::Matrix I11(radial.radial_integral(1,1));
        helfem::Matrix I13(radial.radial_integral(1,3));

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

        return helfem::to_eigen(remove_boundaries(helfem::to_arma(V)));
      }

      helfem::Matrix TwoDBasis::quadrupole_zz() const {
        // Full electric couplings
        helfem::Matrix V(helfem::Matrix::Zero(Ndummy(),Ndummy()));

        // Build radial matrix elements
        helfem::Matrix I10(radial.radial_integral(1,0));
        helfem::Matrix I12(radial.radial_integral(1,2));
        helfem::Matrix I14(radial.radial_integral(1,4));

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

        return helfem::to_eigen(remove_boundaries(helfem::to_arma(V)));
      }

      helfem::Matrix TwoDBasis::Bz_field(double B) const {
        // Full couplings
        helfem::Matrix V(helfem::Matrix::Zero(Ndummy(),Ndummy()));

        // Build radial matrix elements
        helfem::Matrix I10(radial.radial_integral(1,0)*std::pow(Rhalf,3));
        helfem::Matrix I12(radial.radial_integral(1,2)*std::pow(Rhalf,3));
        helfem::Matrix I30(radial.radial_integral(3,0)*std::pow(Rhalf,5));
        helfem::Matrix I32(radial.radial_integral(3,2)*std::pow(Rhalf,5));

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
                  add_sub(V,iang,jang,(-ds*cpl)*I10);
              }
            }
          }
        }

        return helfem::to_eigen(remove_boundaries(helfem::to_arma(V)));
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
                prim_ktei00[idx]=helfem::to_eigen(utils::exchange_tei(helfem::to_arma(prim_tei00[idx]),Ni,Ni,Nj,Nj));
                prim_ktei02[idx]=helfem::to_eigen(utils::exchange_tei(helfem::to_arma(prim_tei02[idx]),Ni,Ni,Nj,Nj));
                prim_ktei20[idx]=helfem::to_eigen(utils::exchange_tei(helfem::to_arma(prim_tei20[idx]),Ni,Ni,Nj,Nj));
                prim_ktei22[idx]=helfem::to_eigen(utils::exchange_tei(helfem::to_arma(prim_tei22[idx]),Ni,Ni,Nj,Nj));
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

      std::vector<double> TwoDBasis::build_LMfac_abs() const {
        const double Rhalf5_4pi = 4.0 * M_PI * std::pow(Rhalf, 5);
        std::vector<double> tbl(lm_map.size());
        for(size_t i = 0; i < lm_map.size(); ++i) {
          const int L  = lm_map[i].first;
          const int Ma = lm_map[i].second;  // |M|
          double fr = 1.0;
          for(int p = L + Ma; p > L - Ma; --p) fr *= p;
          tbl[i] = Rhalf5_4pi / fr;
        }
        return tbl;
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
        // When check==false callers use idx == lm_map.size() as a
        // "not-found sentinel". Do not touch lm_map[idx] in that path
        // -- with libstdc++'s hardened operator[] (GCC 16 default) an
        // OOB access on operator[] is a hard abort even though the value
        // would just be compared and discarded.
        if (idx == lm_map.size())
          return idx;
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
        // See lmind() above: guard the operator[] access when idx == size.
        if (idx == LM_map.size())
          return idx;
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

      helfem::Matrix TwoDBasis::coulomb(const helfem::Matrix & P_in) const {
        if(!prim_tei00.size())
          throw std::logic_error("Primitive teis have not been computed!\n");

        // Public boundary Eigen. The boundary-expansion trimmer is still
        // arma, so bridge around it: to_arma -> expand_boundaries -> to_eigen.
        // The interior is Eigen-native.
        helfem::Matrix P(helfem::to_eigen(expand_boundaries(helfem::to_arma(P_in))));

        // Number of radial elements
        size_t Nel(radial.Nel());
        // Number of radial functions
        size_t Nrad(radial.Nbf());

        // Radial helper matrices
        std::vector<helfem::Matrix> Paux0(LM_map.size());
        std::vector<helfem::Matrix> Paux2(LM_map.size());
        for(size_t i=0;i<Paux0.size();i++) {
          Paux0[i]=helfem::Matrix::Zero(Nrad,Nrad);
          Paux2[i]=helfem::Matrix::Zero(Nrad,Nrad);
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
              double cpl2(gaunt.coeff(lk,mk,L,M,ll));
              // Increment
              helfem::Matrix Prad(P.block(kang*Nrad,lang*Nrad,Nrad,Nrad));
              if(cpl0!=0.0)
                Paux0[iLM]+=cpl0*Prad;
              if(cpl2!=0.0)
                Paux2[iLM]+=cpl2*Prad;
            }
          }
        }

        // Coulomb helpers
        std::vector<helfem::Matrix> Jaux0(LM_map.size());
        std::vector<helfem::Matrix> Jaux2(LM_map.size());
        for(size_t i=0;i<Jaux0.size();i++) {
          Jaux0[i]=helfem::Matrix::Zero(Nrad,Nrad);
          Jaux2[i]=helfem::Matrix::Zero(Nrad,Nrad);
        }
        // Cache the angular prefactor 4*pi*Rhalf^5 / ((L+|M|)!/(L-|M|)!) for
        // every (L, |M|) in the lm_map. The (-1)^M sign is applied at the
        // lookup site since lm_map is indexed by |M| only. pow(Rhalf,5) and
        // the factorial loop both dominated the inner-loop cost otherwise.
        const std::vector<double> LMfac_abs = build_LMfac_abs();
        for(size_t iLM=0;iLM<LM_map.size();iLM++) {
          // Values of L and M
          int L(LM_map[iLM].first);
          int M(LM_map[iLM].second);

          // Helpers
          const size_t ilm(lmind(L,M));
          const double signM = (M & 1) ? -1.0 : 1.0;
          const double LMfac(signM * LMfac_abs[ilm]);

          // Loop over input elements
          for(size_t jel=0;jel<Nel;jel++) {
            size_t jfirst, jlast;
            radial.get_idx(jel,jfirst,jlast);
            size_t Nj(jlast-jfirst+1);

            // Get density submatrices
            helfem::Matrix Psub0(Paux0[iLM].block(jfirst,jfirst,Nj,Nj));
            helfem::Matrix Psub2(Paux2[iLM].block(jfirst,jfirst,Nj,Nj));

            // Contract integrals
            double jsmall0 = LMfac*(disjoint_P0[ilm*Nel+jel]*Psub0).trace();
            double jbig0 = LMfac*(disjoint_Q0[ilm*Nel+jel]*Psub0).trace();
            double jsmall2 = LMfac*(disjoint_P2[ilm*Nel+jel]*Psub2).trace();
            double jbig2 = LMfac*(disjoint_Q2[ilm*Nel+jel]*Psub2).trace();

            // Increment J: jel>iel
            double ifac0(jbig0 - jbig2);
            double ifac2(-jbig0 + jbig2);
            for(size_t iel=0;iel<jel;iel++) {
              size_t ifirst, ilast;
              radial.get_idx(iel,ifirst,ilast);
              size_t Ni(ilast-ifirst+1);

              const helfem::Matrix & iint0=disjoint_P0[ilm*Nel+iel];
              const helfem::Matrix & iint2=disjoint_P2[ilm*Nel+iel];
              Jaux0[iLM].block(ifirst,ifirst,Ni,Ni)+=ifac0*iint0;
              Jaux2[iLM].block(ifirst,ifirst,Ni,Ni)+=ifac2*iint2;
            }

            // Increment J: jel<iel
            ifac0=jsmall0 - jsmall2;
            ifac2=-jsmall0 + jsmall2;
            for(size_t iel=jel+1;iel<Nel;iel++) {
              size_t ifirst, ilast;
              radial.get_idx(iel,ifirst,ilast);
              size_t Ni(ilast-ifirst+1);

              const helfem::Matrix & iint0=disjoint_Q0[ilm*Nel+iel];
              const helfem::Matrix & iint2=disjoint_Q2[ilm*Nel+iel];
              Jaux0[iLM].block(ifirst,ifirst,Ni,Ni)+=ifac0*iint0;
              Jaux2[iLM].block(ifirst,ifirst,Ni,Ni)+=ifac2*iint2;
            }

            // In-element contribution
            {
              size_t iel=jel;
              size_t ifirst=jfirst;
              size_t Ni=Nj;

              // Column-major vectorisation of the (Nj x Nj) density blocks.
              // Psub0/Psub2 are materialised contiguous matrices, so a Map
              // reproduces arma::vectorise exactly.
              Eigen::Map<const helfem::Vector> pv0(Psub0.data(), Psub0.size());
              Eigen::Map<const helfem::Vector> pv2(Psub2.data(), Psub2.size());

              const size_t idx(Nel*Nel*ilm + iel*Nel + jel);
              helfem::Vector jv0(helfem::Vector::Zero(Ni*Ni));
              helfem::Vector jv2(helfem::Vector::Zero(Ni*Ni));
              jv0+=LMfac*(prim_tei00[idx]*pv0);
              jv0-=LMfac*(prim_tei02[idx]*pv2);
              jv2-=LMfac*(prim_tei20[idx]*pv0);
              jv2+=LMfac*(prim_tei22[idx]*pv2);

              // Reshape back to (Ni x Ni), column-major
              Eigen::Map<const helfem::Matrix> Jsub0(jv0.data(), Ni, Ni);
              Eigen::Map<const helfem::Matrix> Jsub2(jv2.data(), Ni, Ni);

              // Increment global Coulomb matrix
              Jaux0[iLM].block(ifirst,ifirst,Ni,Ni)+=Jsub0;
              Jaux2[iLM].block(ifirst,ifirst,Ni,Ni)+=Jsub2;
            }
          }
        }

        // Full Coulomb matrix
        helfem::Matrix J(helfem::Matrix::Zero(Ndummy(),Ndummy()));
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
                J.block(iang*Nrad,jang*Nrad,Nrad,Nrad)+=cpl0*Jaux0[iLM];
              }

              double cpl2(gaunt.coeff(lj,mj,L,M,li));
              if(cpl2!=0.0) {
                J.block(iang*Nrad,jang*Nrad,Nrad,Nrad)+=cpl2*Jaux2[iLM];
              }
            }
          }
        }

        return helfem::to_eigen(remove_boundaries(helfem::to_arma(J)));
      }

      helfem::Matrix TwoDBasis::exchange(const helfem::Matrix & P_in) const {
        if(!prim_ktei00.size())
          throw std::logic_error("Primitive teis have not been computed!\n");

        // Public boundary Eigen. Bridge around the still-arma boundary
        // expander; the interior is Eigen-native.
        helfem::Matrix P(helfem::to_eigen(expand_boundaries(helfem::to_arma(P_in))));

        // Number of radial elements
        size_t Nel(radial.Nel());
        // Number of radial basis functions
        size_t Nrad(radial.Nbf());

        // Pre-compute the (L, |M|) angular prefactor table once per call so
        // the inner loop reduces to a vector lookup + sign branch.
        const std::vector<double> LMfac_abs = build_LMfac_abs();

        // Full exchange matrix. Each (jang,kang) writes a disjoint block, so
        // the parallel accumulation into K is race-free.
        helfem::Matrix K(helfem::Matrix::Zero(Ndummy(),Ndummy()));

#ifdef _OPENMP
#pragma omp parallel
#endif
        {
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
              std::vector<helfem::Matrix> Rmat00(lm_map.size());
              std::vector<helfem::Matrix> Rmat02(lm_map.size());
              std::vector<helfem::Matrix> Rmat20(lm_map.size());
              std::vector<helfem::Matrix> Rmat22(lm_map.size());
              for(size_t i=0;i<lm_map.size();i++) {
                Rmat00[i]=helfem::Matrix::Zero(Nrad,Nrad);
                Rmat02[i]=helfem::Matrix::Zero(Nrad,Nrad);
                Rmat20[i]=helfem::Matrix::Zero(Nrad,Nrad);
                Rmat22[i]=helfem::Matrix::Zero(Nrad,Nrad);
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
                  double bdens(P.block(iang*Nrad,lang*Nrad,Nrad,Nrad).norm());
                  //printf("(%i %i) (%i %i) density block norm %e\n",li,mi,ll,ml,bdens);
                  if(bdens<10*DBL_EPSILON)
                    continue;

                  // M values match. Loop over possible couplings
                  int Lmin=std::max(std::max(std::abs(li-lj),std::abs(lk-ll))-2,abs(M));
                  int Lmax=std::min(li+lj,lk+ll)+2;

                  for(int L=Lmin;L<=Lmax;L++) {
                    // Calculate total coupling coefficient
                    double cpl00(gaunt.mod_coeff(lj,mj,L,M,li,mi)*gaunt.mod_coeff(lk,mk,L,M,ll,ml));
                    double cpl02(-gaunt.mod_coeff(lj,mj,L,M,li,mi)*gaunt.coeff(lk,mk,L,M,ll));
                    double cpl20(-gaunt.coeff(lj,mj,L,M,li)*gaunt.mod_coeff(lk,mk,L,M,ll,ml));
                    double cpl22(gaunt.coeff(lj,mj,L,M,li)*gaunt.coeff(lk,mk,L,M,ll));

                    // Is there any coupling?
                    if(cpl00==0.0 && cpl02==0.0 && cpl20==0.0 && cpl22==0.0)
                      continue;

                    // Index in the L,|M| table
                    const size_t ilm(lmind(L,M));
                    const double signM = (M & 1) ? -1.0 : 1.0;
                    const double LMfac(signM * LMfac_abs[ilm]);

                    helfem::Matrix Psub(P.block(iang*Nrad,lang*Nrad,Nrad,Nrad));

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

                    // Exchange submatrix (column-major vectorised)
                    helfem::Vector Ksub(helfem::Vector::Zero(Ni*Nj));

                    for(size_t ilm=0;ilm<lm_map.size();ilm++) {
                      if(!couple[ilm])
                        continue;
                      // Index in tei array
                      size_t idx=Nel*Nel*ilm + iel*Nel + jel;
                      // Materialise the (Ni x Nj) Rmat blocks so the Map
                      // reproduces arma::vectorise (column-major) exactly.
                      helfem::Matrix Rb00(Rmat00[ilm].block(ifirst,jfirst,Ni,Nj));
                      helfem::Matrix Rb02(Rmat02[ilm].block(ifirst,jfirst,Ni,Nj));
                      helfem::Matrix Rb20(Rmat20[ilm].block(ifirst,jfirst,Ni,Nj));
                      helfem::Matrix Rb22(Rmat22[ilm].block(ifirst,jfirst,Ni,Nj));
                      Ksub+=prim_ktei00[idx]*Eigen::Map<const helfem::Vector>(Rb00.data(),Rb00.size());
                      Ksub+=prim_ktei02[idx]*Eigen::Map<const helfem::Vector>(Rb02.data(),Rb02.size());
                      Ksub+=prim_ktei20[idx]*Eigen::Map<const helfem::Vector>(Rb20.data(),Rb20.size());
                      Ksub+=prim_ktei22[idx]*Eigen::Map<const helfem::Vector>(Rb22.data(),Rb22.size());
                    }

                    // Reshape to (Ni x Nj), column-major
                    Eigen::Map<const helfem::Matrix> Ksubm(Ksub.data(),Ni,Nj);

                    // Increment global exchange matrix
                    K.block(jang*Nrad+ifirst,kang*Nrad+jfirst,Ni,Nj)-=Ksubm;

                  } else {
                    helfem::Matrix Ksub(helfem::Matrix::Zero(Ni,Nj));
                    for(size_t ilm=0;ilm<lm_map.size();ilm++) {
                      if(!couple[ilm])
                        continue;
                      // Disjoint integrals. When r(iel)>r(jel), iel gets Q, jel gets P.
                      const helfem::Matrix & iint0=(iel>jel) ? disjoint_Q0[ilm*Nel+iel] : disjoint_P0[ilm*Nel+iel];
                      const helfem::Matrix & iint2=(iel>jel) ? disjoint_Q2[ilm*Nel+iel] : disjoint_P2[ilm*Nel+iel];
                      const helfem::Matrix & jint0=(iel>jel) ? disjoint_P0[ilm*Nel+jel] : disjoint_Q0[ilm*Nel+jel];
                      const helfem::Matrix & jint2=(iel>jel) ? disjoint_P2[ilm*Nel+jel] : disjoint_Q2[ilm*Nel+jel];

                      // (Niel x Njel) = (Niel x Njel) x (Njel x Njel)
                      helfem::Matrix T(Rmat00[ilm].block(ifirst,jfirst,Ni,Nj)*jint0.transpose() + Rmat02[ilm].block(ifirst,jfirst,Ni,Nj)*jint2.transpose());
                      Ksub-=iint0*T;

                      T=Rmat20[ilm].block(ifirst,jfirst,Ni,Nj)*jint0.transpose() + Rmat22[ilm].block(ifirst,jfirst,Ni,Nj)*jint2.transpose();
                      Ksub-=iint2*T;
                    }

                    // Increment global exchange matrix
                    K.block(jang*Nrad+ifirst,kang*Nrad+jfirst,Ni,Nj)+=Ksub;
                  }
                }
              }
            }
          }
        }

        return helfem::to_eigen(remove_boundaries(helfem::to_arma(K)));
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

      arma::cx_mat TwoDBasis::eval_bf(size_t iel, size_t irad, double cth, double phi) const {
        // Evaluate spherical harmonics
        arma::cx_vec sph(lval.n_elem);
        for(size_t i=0;i<lval.n_elem;i++)
          sph(i)=::spherical_harmonics(lval(i),mval(i),cth,phi);

        // Evaluate radial functions
        arma::mat rad(helfem::to_arma(radial.get_bf(iel)));
        rad=rad.rows(irad,irad);

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
        arma::mat rad(helfem::to_arma(radial.get_bf(iel)));
        rad=rad.rows(irad,irad);

        // Form supermatrix
        arma::mat bf(rad.n_rows,flist.size()*rad.n_cols);
        for(size_t i=0;i<flist.size();i++)
          bf.cols(i*rad.n_cols,(i+1)*rad.n_cols-1)=sph(i)*rad;

        return bf;
      }

      arma::cx_vec TwoDBasis::eval_bf(double mu, double cth, double phi) const {
	// Find out which element mu belongs to
	arma::vec bval(helfem::to_arma(radial.get_bval()));
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
        arma::mat rad(helfem::to_arma(radial.get_bf(iel,helfem::to_eigen(x))));

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
        arma::mat frad(helfem::to_arma(radial.get_bf(iel)));
        arma::mat drad(helfem::to_arma(radial.get_df(iel)));
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
        // sin^2(theta) = (1 - cth)(1 + cth) avoids the catastrophic
        // cancellation in 1 - cth*cth when cth is close to +/- 1.
        const double sinth = std::sqrt(std::max((1.0-cth)*(1.0+cth), 0.0));
        const double cotth = (sinth > 0.0) ? cth/sinth : 0.0;

        // but theta is nastier
        for(size_t i=0;i<lval.n_elem;i++) {
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
        return helfem::to_arma(radial.get_bf(iel));
      }

      arma::vec TwoDBasis::get_wrad(size_t iel) const {
        return helfem::to_arma(radial.get_wrad(iel));
      }

      arma::vec TwoDBasis::get_r(size_t iel) const {
        return helfem::to_arma(radial.get_r(iel));
      }

    }
  }
}
