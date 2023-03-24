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
#include "chebyshev.h"
#include "../general/spherical_harmonics.h"
#include "../general/gaunt.h"
#include "../general/gsz.h"
#include "utils.h"
#include "../general/scf_helpers.h"
#include "../general/dftfuncs.h"
#include <cassert>
#include <cfloat>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <xc.h>


namespace helfem {
  namespace sadatom {
    namespace basis {
      TwoDBasis::TwoDBasis() {
      }

      TwoDBasis::TwoDBasis(int Z_, modelpotential::nuclear_model_t model_, double Rrms_, const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly, bool zeroder, int n_quad, const arma::vec & bval, int taylor_order, int lmax) {
        // Nuclear charge
        Z=Z_;
        model=model_;
        Rrms=Rrms_;
        // Construct radial basis
        bool zero_func_left=true;
        bool zero_deriv_left=false;
        bool zero_func_right=true;
        polynomial_basis::FiniteElementBasis fem(poly, bval, zero_func_left, zero_deriv_left, zero_func_right, zeroder);
        radial=atomic::basis::RadialBasis(fem, n_quad, taylor_order);
        // Angular basis
        lval=arma::linspace<arma::ivec>(0,lmax,lmax+1);
      }

      TwoDBasis::~TwoDBasis() {
      }

      size_t TwoDBasis::Nbf() const {
        return radial.Nbf();
      }

      int TwoDBasis::charge() const {
        return Z;
      }

      arma::mat TwoDBasis::Sinvh() const {
        // Form overlap matrix
        arma::mat S(overlap());
        return scf::form_Sinvh(S,false);
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

        return Orad;
      }

      arma::mat TwoDBasis::overlap() const {
        return radial_integral(0);
      }

      arma::mat TwoDBasis::kinetic() const {
        // Build radial kinetic energy matrix
        size_t Nrad(radial.Nbf());
        arma::mat Trad(Nrad,Nrad);
        Trad.zeros();

        // Loop over elements
        for(size_t iel=0;iel<radial.Nel();iel++) {
          // Where are we in the matrix?
          size_t ifirst, ilast;
          radial.get_idx(iel,ifirst,ilast);
          Trad.submat(ifirst,ifirst,ilast,ilast)+=radial.kinetic(iel);
        }

        return Trad;
      }

      arma::mat TwoDBasis::kinetic_l() const {
        // Build radial kinetic energy matrix
        size_t Nrad(radial.Nbf());
        arma::mat Trad_l(Nrad,Nrad);
        Trad_l.zeros();

        // Loop over elements
        for(size_t iel=0;iel<radial.Nel();iel++) {
          // Where are we in the matrix?
          size_t ifirst, ilast;
          radial.get_idx(iel,ifirst,ilast);
          Trad_l.submat(ifirst,ifirst,ilast,ilast)+=radial.kinetic_l(iel);
        }

        return Trad_l;
      }

      arma::mat TwoDBasis::nuclear() const {
        if(model != modelpotential::POINT_NUCLEUS) {
          modelpotential::ModelPotential * pot = modelpotential::get_nuclear_model(model,Z,Rrms);
          arma::mat Vrad(model_potential(pot));
          delete pot;
          return Vrad;
        } else {
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

          return -Z*Vrad;
        }
      }

      arma::mat TwoDBasis::model_potential(const modelpotential::ModelPotential * pot) const {
        size_t Nrad(radial.Nbf());
        arma::mat Vrad(Nrad,Nrad);
        Vrad.zeros();
        // Loop over elements
        for(size_t iel=0;iel<radial.Nel();iel++) {
          // Where are we in the matrix?
          size_t ifirst, ilast;
          radial.get_idx(iel,ifirst,ilast);
          Vrad.submat(ifirst,ifirst,ilast,ilast)+=radial.model_potential(pot,iel);
        }
        return Vrad;
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
          }
        }

        // Two-electron exchange integrals
        prim_ktei.resize(Nel*Nel*N_L);
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
        for(size_t L=0;L<N_L;L++)
          for(size_t iel=0;iel<Nel;iel++) {
            // Diagonal integrals
            size_t Ni(radial.Nprim(iel));
            prim_ktei[Nel*Nel*L + iel*Nel + iel]=utils::exchange_tei(prim_tei[Nel*Nel*L + iel*Nel + iel],Ni,Ni,Ni,Ni);
          }
      }

      void TwoDBasis::compute_yukawa(double lambda_) {
        lambda=lambda_;
        yukawa=true;

        // Number of distinct L values is
        size_t N_L(2*arma::max(lval)+1);
        size_t Nel(radial.Nel());

        // Compute disjoint integrals
        disjoint_iL.resize(Nel*N_L);
        disjoint_kL.resize(Nel*N_L);
        for(size_t L=0;L<N_L;L++)
          for(size_t iel=0;iel<Nel;iel++) {
            disjoint_iL[L*Nel+iel]=radial.bessel_il_integral(L,lambda,iel);
            disjoint_kL[L*Nel+iel]=radial.bessel_kl_integral(L,lambda,iel);
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
        rs_ktei.resize(Nel*Nel*N_L);
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
        for(size_t L=0;L<N_L;L++)
          for(size_t iel=0;iel<Nel;iel++) {
            // Diagonal integrals
            size_t Ni(radial.Nprim(iel));
            rs_ktei[Nel*Nel*L + iel*Nel + iel]=utils::exchange_tei(radial.yukawa_integral(L,lambda,iel),Ni,Ni,Ni,Ni);
          }
      }

      void TwoDBasis::compute_erfc(double mu) {
        lambda=mu;
        yukawa=false;

        // Number of distinct L values is
        size_t N_L(2*arma::max(lval)+1);
        size_t Nel(radial.Nel());

        // No disjoint integrals
        disjoint_iL.clear();
        disjoint_kL.clear();

        /*
          The exchange matrix is given by
          K(jk) = (ij|kl) P(il)
          i.e. the complex conjugation hits i and l as
          in the density matrix.

          To get this in the proper order, we permute the integrals
          K(jk) = (jk;il) P(il)
          so we don't have to reform the permutations in the exchange routine.
        */
        rs_ktei.resize(Nel*Nel*N_L);
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
        for(size_t L=0;L<N_L;L++)
          for(size_t iel=0;iel<Nel;iel++) {
            for(size_t kel=0;kel<Nel;kel++) {
              // Diagonal integrals
              size_t Ni(radial.Nprim(iel));
              size_t Nk(radial.Nprim(kel));
              rs_ktei[Nel*Nel*L + iel*Nel + kel]=utils::exchange_tei(radial.erfc_integral(L,lambda,iel,kel),Ni,Ni,Nk,Nk);
            }
          }
      }

      arma::mat TwoDBasis::coulomb(const arma::mat & P) const {
        if(!prim_tei.size())
          throw std::logic_error("Primitive teis have not been computed!\n");

        // Number of radial elements
        size_t Nel(radial.Nel());

        arma::mat J(P);
        J.zeros();
        // Contract integrals
        for(int L=0;L<1;L++) {
          const double Lfac=4.0*M_PI/(2*L+1);

          for(size_t jel=0;jel<Nel;jel++) {
            size_t jfirst, jlast;
            radial.get_idx(jel,jfirst,jlast);
            size_t Nj(jlast-jfirst+1);

            arma::mat Psub(P.submat(jfirst,jfirst,jlast,jlast));

            // Contract integrals
            double jsmall = Lfac*arma::trace(disjoint_L[L*Nel+jel]*Psub);
            double jbig = Lfac*arma::trace(disjoint_m1L[L*Nel+jel]*Psub);

            // Increment J: jel>iel
            for(size_t iel=0;iel<jel;iel++) {
              size_t ifirst, ilast;
              radial.get_idx(iel,ifirst,ilast);

              const arma::mat & iint=disjoint_L[L*Nel+iel];
              J.submat(ifirst,ifirst,ilast,ilast)+=jbig*iint;
            }

            // Increment J: jel<iel
            for(size_t iel=jel+1;iel<Nel;iel++) {
              size_t ifirst, ilast;
              radial.get_idx(iel,ifirst,ilast);

              const arma::mat & iint=disjoint_m1L[L*Nel+iel];
              J.submat(ifirst,ifirst,ilast,ilast)+=jsmall*iint;
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

              J.submat(ifirst,ifirst,ilast,ilast)+=Jsub;
            }
          }
        }

        return J;
      }

      arma::cube TwoDBasis::exchange(const arma::cube & P) const {
        if(!prim_ktei.size())
          throw std::logic_error("Primitive teis have not been computed!\n");

        // Gaunt coefficient table
        int gmax(arma::max(lval));
        gaunt::Gaunt gaunt(gmax,2*gmax,gmax);

        if(P.n_slices != (arma::uword) gmax+1)
          throw std::logic_error("Density matrix am does not match basis set!\n");

        // Number of radial elements
        size_t Nel(radial.Nel());
        // Number of radial basis functions
        size_t Nrad(radial.Nbf());
        if(P.n_rows != Nrad || P.n_cols != Nrad)
          throw std::logic_error("Density matrix does not match basis set!\n");

        // Full exchange matrix
        arma::cube K(Nrad,Nrad,gmax+1);
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
#pragma omp for
#endif
          // Loop over angular momentum of output
          for(int lout=0;lout<=gmax;lout++) {
            // Initialize memory
            arma::cube Prad;
            Prad.zeros(Nrad,Nrad,2*gmax+1);
            // Do we have a coupling
            std::vector<bool> coupling(2*gmax+1,false);

            // Do angular sums: loop over input angular momentum
            for(int lin=0;lin<=gmax;lin++) {
              // Skip if nothing to do
              if(arma::norm(P.slice(lin),2)==0.0)
                continue;

              // Possible couplings (lin,lout) => L
              int Lmin=std::abs(lin-lout);
              int Lmax=lin+lout;

              arma::vec totcoup;
              totcoup.zeros(Lmax+1);
              // Sum over m values: output indices
              for(int mout=-lout;mout<=lout;mout++) {
                // and input indices
                for(int min=-lin;min<=lin;min++) {
                  // LH m value
                  int M(mout-min);
                  for(int L=Lmin;L<=Lmax;L++) {
                    // Calculate total coupling coefficient
                    double cpl(gaunt.coeff(lout,mout,L,M,lin,min)*gaunt.coeff(lout,mout,L,M,lin,min));
                    totcoup(L)+=cpl;
                  }
                }
              }
              // Averaging wrt output
              totcoup /= 2*lout+1;

              // Increment radial density matrix
              for(int L=Lmin;L<=Lmax;L++) {
                // Check if coupling exists
                if(totcoup(L)==0.0)
                  continue;

                // Form density matrix
                double Lfac=4.0*M_PI/(2*L+1);
                Prad.slice(L)+=(Lfac*totcoup(L))*P.slice(lin);
                coupling[L]=true;
              }
            }

            for(size_t L=0;L<coupling.size();L++) {
              if(!coupling[L])
                continue;

              // Radial matrix
              arma::mat P_L(Prad.slice(L));

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
                    Ksub=prim_ktei[Nel*Nel*L + iel*Nel + jel]*arma::vectorise(P_L.submat(ifirst,jfirst,ilast,jlast));
                    Ksub.reshape(Ni,Nj);

                    // Increment global exchange matrix
                    K.slice(lout).submat(ifirst,jfirst,ilast,jlast)-=Ksub;

                  } else {
                    // Disjoint integrals. When r(iel)>r(jel), iel gets -1-L, jel gets L.
                    const arma::mat & iint=(iel>jel) ? disjoint_m1L[L*Nel+iel] : disjoint_L[L*Nel+iel];
                    const arma::mat & jint=(iel>jel) ? disjoint_L[L*Nel+jel] : disjoint_m1L[L*Nel+jel];

                    // Get density submatrix (Niel x Njel)
                    arma::mat Psub(mem_Psub[ith].memptr(),Ni,Nj,false,true);
                    Psub=P_L.submat(ifirst,jfirst,ilast,jlast);

                    // Calculate helper
                    arma::mat T(mem_T[ith].memptr(),Ni,Nj,false,true);
                    // (Niel x Njel) = (Niel x Njel) x (Njel x Njel)
                    T=Psub*arma::trans(jint);
                    // Exchange submatrix
                    arma::mat Ksub(mem_Ksub[ith].memptr(),Ni,Nj,false,true);
                    Ksub=iint*T;

                    // Increment global exchange matrix
                    K.slice(lout).submat(ifirst,jfirst,ilast,jlast)-=Ksub;
                  }
                }
              }
            }
          }
        }

        return K;
      }

      arma::cube TwoDBasis::rs_exchange(const arma::cube & P) const {
        if(!rs_ktei.size())
          throw std::logic_error("Primitive teis have not been computed!\n");

        // Gaunt coefficient table
        int gmax(arma::max(lval));
        gaunt::Gaunt gaunt(gmax,2*gmax,gmax);

        if(P.n_slices != (arma::uword) gmax+1)
          throw std::logic_error("Density matrix am does not match basis set!\n");

        // Number of radial elements
        size_t Nel(radial.Nel());
        // Number of radial basis functions
        size_t Nrad(radial.Nbf());
        if(P.n_rows != Nrad || P.n_cols != Nrad)
          throw std::logic_error("Density matrix does not match basis set!\n");

        // Full exchange matrix
        arma::cube K(Nrad,Nrad,gmax+1);
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
#pragma omp for
#endif
          // Loop over angular momentum of output
          for(int lout=0;lout<=gmax;lout++) {
            // Initialize memory
            arma::cube Prad;
            Prad.zeros(Nrad,Nrad,2*gmax+1);
            // Do we have a coupling
            std::vector<bool> coupling(2*gmax+1,false);

            // Do angular sums: loop over input angular momentum
            for(int lin=0;lin<=gmax;lin++) {
              // Skip if nothing to do
              if(arma::norm(P.slice(lin),2)==0.0)
                continue;

              // Possible couplings (lin,lout) => L
              int Lmin=std::abs(lin-lout);
              int Lmax=lin+lout;

              arma::vec totcoup;
              totcoup.zeros(Lmax+1);
              // Sum over m values: output indices
              for(int mout=-lout;mout<=lout;mout++) {
                // and input indices
                for(int min=-lin;min<=lin;min++) {
                  // LH m value
                  int M(mout-min);
                  for(int L=Lmin;L<=Lmax;L++) {
                    // Calculate total coupling coefficient
                    double cpl(gaunt.coeff(lout,mout,L,M,lin,min)*gaunt.coeff(lout,mout,L,M,lin,min));
                    totcoup(L)+=cpl;
                  }
                }
              }
              // Averaging wrt output
              totcoup /= 2*lout+1;

              // Increment radial density matrix
              for(int L=Lmin;L<=Lmax;L++) {
                // Check if coupling exists
                if(totcoup(L)==0.0)
                  continue;

                // Form density matrix
                double Lfac = yukawa ? 4.0*M_PI*lambda :  4.0*M_PI*lambda/(2*L+1);
                Prad.slice(L)+=(Lfac*totcoup(L))*P.slice(lin);
                coupling[L]=true;
              }
            }

            for(size_t L=0;L<coupling.size();L++) {
              if(!coupling[L])
                continue;

              // Radial matrix
              arma::mat P_L(Prad.slice(L));

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

                  if(!yukawa || iel == jel) {
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
                    Ksub=rs_ktei[Nel*Nel*L + iel*Nel + jel]*arma::vectorise(P_L.submat(ifirst,jfirst,ilast,jlast));
                    Ksub.reshape(Ni,Nj);

                    // Increment global exchange matrix
                    K.slice(lout).submat(ifirst,jfirst,ilast,jlast)-=Ksub;

                  } else {
                    // Disjoint integrals. When r(iel)>r(jel), iel gets -1-L, jel gets L.
                    const arma::mat & iint=(iel>jel) ? disjoint_kL[L*Nel+iel] : disjoint_iL[L*Nel+iel];
                    const arma::mat & jint=(iel>jel) ? disjoint_iL[L*Nel+jel] : disjoint_kL[L*Nel+jel];

                    // Get density submatrix (Niel x Njel)
                    arma::mat Psub(mem_Psub[ith].memptr(),Ni,Nj,false,true);
                    Psub=P_L.submat(ifirst,jfirst,ilast,jlast);

                    // Calculate helper
                    arma::mat T(mem_T[ith].memptr(),Ni,Nj,false,true);
                    // (Niel x Njel) = (Niel x Njel) x (Njel x Njel)
                    T=Psub*arma::trans(jint);
                    // Exchange submatrix
                    arma::mat Ksub(mem_Ksub[ith].memptr(),Ni,Nj,false,true);
                    Ksub=iint*T;

                    // Increment global exchange matrix
                    K.slice(lout).submat(ifirst,jfirst,ilast,jlast)-=Ksub;
                  }
                }
              }
            }
          }
        }

        return K;
      }

      std::vector<arma::mat> TwoDBasis::get_prim_tei() const {
        return prim_tei;
      }

      arma::mat TwoDBasis::eval_bf(size_t iel) const {
        return radial.get_bf(iel);
      }

      arma::mat TwoDBasis::eval_df(size_t iel) const {
        return radial.get_df(iel);
      }

      arma::mat TwoDBasis::eval_lf(size_t iel) const {
        return radial.get_lf(iel);
      }

      arma::uvec TwoDBasis::bf_list(size_t iel) const {
        // Radial functions in element
        size_t ifirst, ilast;
        radial.get_idx(iel,ifirst,ilast);
        // Number of radial functions in element
        size_t Nr(ilast-ifirst+1);

        // List of functions in the element
        arma::uvec idx(Nr);
        for(size_t j=0;j<Nr;j++)
          idx(j)=ifirst+j;

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

      double TwoDBasis::get_small_r_taylor_cutoff() const {
        return radial.get_small_r_taylor_cutoff();
      }

      double TwoDBasis::get_taylor_diff() const {
        return radial.get_taylor_diff();
      }

      double TwoDBasis::nuclear_density(const arma::mat & P) const {
        return radial.nuclear_density(P)/(4.0*M_PI);
      }

      double TwoDBasis::nuclear_density_gradient(const arma::mat & P) const {
        return radial.nuclear_density_gradient(P)/(4.0*M_PI);
      }

      arma::vec TwoDBasis::quadrature_weights() const {
        std::vector<arma::vec> w(radial.Nel());
        size_t ntot=1;
        for(size_t iel=0;iel<radial.Nel();iel++) {
          w[iel]=radial.get_wrad(iel);
          ntot+=w[iel].n_elem;
        }
        arma::vec wt(ntot);
        wt.zeros();
        size_t Npts(w[0].n_elem);
        for(size_t iel=0;iel<radial.Nel();iel++)
          wt.subvec(1+iel*Npts,(iel+1)*Npts)=w[iel];

        return wt;
      }

      arma::vec TwoDBasis::coulomb_screening(const arma::mat & Prad) const {
        std::vector<arma::vec> r(radial.Nel());
        std::vector<arma::vec> V(radial.Nel());

        // Calculate potential due to charge outside the element
        arma::vec zero(radial.Nel());
        arma::vec minusone(radial.Nel());
        for(size_t iel=0;iel<radial.Nel();iel++) {
          // Radial functions in element
          size_t ifirst, ilast;
          radial.get_idx(iel,ifirst,ilast);
          // Density matrix
          arma::mat Psub(Prad.submat(ifirst,ifirst,ilast,ilast));
          arma::mat zm(radial.radial_integral(0,iel));
          arma::mat mo(radial.radial_integral(-1,iel));
          zero(iel)=arma::trace(Psub*zm);
          minusone(iel)=arma::trace(Psub*mo);
        }
        // Sum zero potentials together
        for(size_t iel=1;iel<radial.Nel();iel++)
          zero(iel)+=zero(iel-1);
        // Sum minus one potentials together
        for(size_t iel=radial.Nel()-2;iel<radial.Nel();iel--)
          minusone(iel)+=minusone(iel+1);

        // Form potential
        for(size_t iel=0;iel<radial.Nel();iel++) {
          // Initialize potential
          r[iel]=radial.get_r(iel);
          V[iel].zeros(r[iel].n_elem);

          // Get the density in the element
          size_t ifirst, ilast;
          radial.get_idx(iel,ifirst,ilast);
          arma::vec Pv(arma::vectorise(Prad.submat(ifirst,ifirst,ilast,ilast)));

          // Calculate the in-element potential
          arma::mat pot(radial.spherical_potential(iel));
          V[iel] += pot*Pv;

          // Add in the contributions from the other elements
          if(iel>0)
            for(size_t ip=0;ip<r[iel].n_elem;ip++)
              V[iel](ip) += zero(iel-1)/r[iel](ip);
          if(iel != radial.Nel()-1)
            V[iel] += minusone(iel+1)*arma::ones<arma::vec>(V[iel].n_elem);

          // Multiply by r to convert this into an effective charge
          for(size_t ip=0;ip<r[iel].n_elem;ip++)
            V[iel](ip)*=r[iel](ip);
        }

        // Assemble all of this into an array
        size_t Npts=r[0].n_elem;
        arma::vec Veff(radial.Nel()*Npts+1);
        Veff.zeros();
        for(size_t iel=0;iel<radial.Nel();iel++) {
          Veff.subvec(1+iel*Npts,(iel+1)*Npts)=V[iel];
        }

        return Veff;
      }

      arma::vec TwoDBasis::radii() const {
        std::vector<arma::vec> r(radial.Nel());
        for(size_t iel=0;iel<radial.Nel();iel++) {
          r[iel]=radial.get_r(iel);
        }

        size_t Npts=r[0].n_elem;
        arma::vec rad(radial.Nel()*Npts+1);
        rad.zeros();
        for(size_t iel=0;iel<radial.Nel();iel++) {
          rad.subvec(1+iel*Npts,(iel+1)*Npts)=r[iel];
        }

        return rad;
      }

      arma::mat TwoDBasis::orbitals(const arma::mat & C) const {
        std::vector<arma::mat> c(radial.Nel());
        for(size_t iel=0;iel<radial.Nel();iel++) {
          // Radial functions in element
          size_t ifirst, ilast;
          radial.get_idx(iel,ifirst,ilast);
          // Density matrix
          arma::mat Csub(C.rows(ifirst,ilast));
          arma::mat bf(radial.get_bf(iel));

          c[iel]=bf*Csub;
        }
        size_t Npts=c[0].n_rows;

        arma::mat Cv(radial.Nel()*Npts+1,C.n_cols);
        Cv.zeros();
        // Values at the nucleus
        {
          Cv.row(0)=radial.nuclear_orbital(C);
        }
        // Other points
        for(size_t iel=0;iel<radial.Nel();iel++) {
          Cv.rows(1+iel*Npts,(iel+1)*Npts)=c[iel];
        }

        return Cv;
      }

      arma::vec TwoDBasis::electron_density(const arma::vec & x, size_t iel, const arma::mat & Prad, bool rsqweight) const {
        // Radial functions in element
        size_t ifirst, ilast;
        radial.get_idx(iel,ifirst,ilast);
        // Density matrix
        arma::mat Psub(Prad.submat(ifirst,ifirst,ilast,ilast));
        arma::mat bf(radial.get_bf(x, iel));

        arma::vec density = arma::diagvec(bf*Psub*bf.t());
        if(rsqweight)
          density %= arma::square(radial.get_r(x, iel));
        return density;
      }

      arma::vec TwoDBasis::electron_density(size_t iel, const arma::mat & Prad, bool rsqweight) const {
        return electron_density(radial.get_xq(), iel, Prad, rsqweight);
      }

      arma::vec TwoDBasis::electron_density(const arma::mat & Prad) const {
        std::vector<arma::vec> d(radial.Nel());
        for(size_t iel=0;iel<radial.Nel();iel++) {
          d[iel]=electron_density(iel, Prad);
        }
        size_t Npts=d[0].n_elem;

        arma::vec n(radial.Nel()*Npts+1);
        n.zeros();
        n(0)=4.0*M_PI*nuclear_density(Prad);
        for(size_t iel=0;iel<radial.Nel();iel++) {
          n.subvec(1+iel*Npts,(iel+1)*Npts)=d[iel];
        }

        return n;
      }

      double TwoDBasis::electron_density_maximum(const arma::mat & Prad, double eps) const {
        // Evaluate the density in each quadrature point and take
        // their maximum
        arma::vec den(radial.Nel());
        bool rsqweight = true;

        for(size_t iel=0;iel<radial.Nel();iel++) {
          den(iel)=arma::max(electron_density(iel, Prad, rsqweight));
        }

        // Quadrature points
        arma::vec xq = radial.get_xq();

        // Find the element with the maximum density
        arma::uword iel;
        den.max(iel);

        // Evaluate the density in that element
        arma::vec del = electron_density(xq, iel, Prad, rsqweight);

        // Find the maximum value
        arma::uword imax;
        del.max(imax);

        // Refine the location of the maximum in the element
        double rmax=0.0;
        {
          // Primitive coordinates
          arma::vec a(1), b(1);

          if(imax == 0) {
            a(0) = -1.0;
            b(0) = xq(imax+1);
          } else if(imax == xq.n_elem-1) {
            a(0) = xq(imax-1);
            b(0) = 1.0;
          } else {
            a(0) = xq(imax-1);
            b(0) = xq(imax+1);
          }

          // Golden ratio search
          double golden_ratio = 0.5*(sqrt(5.0)+1.0);
          while(arma::norm(a-b,"inf")>=eps) {
            arma::vec c = b - (b-a)/golden_ratio;
            arma::vec d = a + (b-a)/golden_ratio;
            double density_c = arma::as_scalar(electron_density(c, iel, Prad, rsqweight));
            double density_d = arma::as_scalar(electron_density(d, iel, Prad, rsqweight));
            if(density_c > density_d) {
              b = d;
            } else {
              a = c;
            }
          }
          arma::vec cen=((a+b)/2);
          double dmax = arma::as_scalar(electron_density(cen, iel, Prad, rsqweight));
          if(dmax < del(imax)) {
            std::ostringstream oss;
            oss << "Density maximization failed! Quadrature max " << del(imax) << " optimized max " << dmax << " difference " << dmax-del(imax) << "!\n";
            throw std::logic_error(oss.str());
          }
          // Position of maximum is
          rmax = arma::as_scalar(radial.get_r((a+b)/2,iel));
        }

        return rmax;
      }

      arma::vec TwoDBasis::electron_density_gradient(const arma::mat & Prad) const {
        std::vector<arma::vec> d(radial.Nel());
        for(size_t iel=0;iel<radial.Nel();iel++) {
          // Radial functions in element
          size_t ifirst, ilast;
          radial.get_idx(iel,ifirst,ilast);
          // Density matrix
          arma::mat Psub(Prad.submat(ifirst,ifirst,ilast,ilast));
          arma::mat bf(radial.get_bf(iel));
          arma::mat df(radial.get_df(iel));

          d[iel]=2.0*arma::diagvec(bf*Psub*df.t());
        }

        size_t Npts=d[0].n_elem;
        arma::vec n(radial.Nel()*Npts+1);
        n.zeros();

        // TODO: implement gradient at the nucleus. This requires
        // third derivatives of the basis functions...

        // The other points
        for(size_t iel=0;iel<radial.Nel();iel++) {
          n.subvec(1+iel*Npts,(iel+1)*Npts)=d[iel];
        }

        return n;
      }

      arma::vec TwoDBasis::electron_density_laplacian(const arma::mat & Prad) const {
        std::vector<arma::vec> l(radial.Nel());
        for(size_t iel=0;iel<radial.Nel();iel++) {
          // Radial functions in element
          size_t ifirst, ilast;
          radial.get_idx(iel,ifirst,ilast);
          // Density matrix
          arma::mat Psub(Prad.submat(ifirst,ifirst,ilast,ilast));
          arma::mat bf(radial.get_bf(iel));
          arma::mat df(radial.get_df(iel));
          arma::mat lf(radial.get_lf(iel));
          arma::vec r(radial.get_r(iel));

          // Laplacian is df^2/dr^2 + 2/r df/dr
          l[iel]=2.0*(arma::diagvec(df*Psub*df.t()) + arma::diagvec(bf*Psub*lf.t())) + 4.0*arma::diagvec(bf*Psub*df.t())/r;
        }

        size_t Npts=l[0].n_elem;
        arma::vec n(radial.Nel()*Npts+1);
        n.zeros();

        // Skip the laplacian at the nucleus at least for now...
        for(size_t iel=0;iel<radial.Nel();iel++) {
          n.subvec(1+iel*Npts,(iel+1)*Npts)=l[iel];
        }

        return n;
      }

      arma::vec TwoDBasis::kinetic_energy_density(const arma::cube & Pl0) const {
        // Radial density matrices
        arma::mat P(Pl0.n_rows, Pl0.n_cols, arma::fill::zeros);
        arma::mat Pl(Pl0.n_rows, Pl0.n_cols, arma::fill::zeros);
        for(size_t l=0; l<Pl0.n_slices; l++) {
          P += Pl0.slice(l);
          Pl += l*(l+1)*Pl0.slice(l);
        }

        std::vector<arma::vec> t(radial.Nel());
        for(size_t iel=0;iel<radial.Nel();iel++) {
          // Radial functions in element
          size_t ifirst, ilast;
          radial.get_idx(iel,ifirst,ilast);

          // Radii
          arma::vec r(radial.get_r(iel));

          // Density matrix
          arma::mat Psub(P.submat(ifirst,ifirst,ilast,ilast));
          arma::mat Psubl(Pl.submat(ifirst,ifirst,ilast,ilast));

          // Basis function
          arma::mat bf(radial.get_bf(iel));
          // Basis function derivative
          arma::mat bf_rho(radial.get_df(iel));

          arma::vec term1(arma::diagvec(bf_rho * Psub * bf_rho.t()));
          arma::vec term2(arma::diagvec(bf * Psubl * bf.t())/arma::square(r));
          // The second term is tricky near the nucleus since only s
          // orbitals can contribute to the electron density but their
          // contribution is killed off by the l(l+1) factor
          term2(arma::find(term2<0.0)).zeros();
          t[iel] = 0.5*(term1+term2);
        }

        size_t Npts=t[0].n_elem;
        arma::vec tn(radial.Nel()*Npts+1);
        tn.zeros();

        // Skip the value at the nucleus at least for now...
        for(size_t iel=0;iel<radial.Nel();iel++) {
          tn.subvec(1+iel*Npts,(iel+1)*Npts)=t[iel];
        }

        return tn;
      }

      std::vector< std::pair<int, arma::mat> > TwoDBasis::Rmatrices() const {
        std::vector< std::pair<int, arma::mat> > rmat;
        for(int i=-2;i<=3;i++) {
          if(i==0) continue;
          std::pair<int, arma::mat> p(i, radial_integral(i));
          rmat.push_back(p);
        }
        return rmat;
      }

      arma::vec TwoDBasis::xc_screening(const arma::mat & Prad, int x_func, int c_func) const {
        arma::mat v(xc_screening(Prad/2,Prad/2,x_func,c_func));
        return 0.5*(v.col(0)+v.col(1));
      }

      arma::mat TwoDBasis::xc_screening(const arma::mat & Parad, const arma::mat & Pbrad, int x_func, int c_func) const {
        const double angfac=4.0*M_PI;
        // Get the electron density
        arma::vec rhoa(electron_density(Parad)/angfac);
        arma::vec rhob(electron_density(Pbrad)/angfac);
        // and the density gradient
        arma::vec grada(electron_density_gradient(Parad)/angfac);
        arma::vec gradb(electron_density_gradient(Pbrad)/angfac);
        // and the density Laplacian
        arma::vec lapla(electron_density_laplacian(Parad)/angfac);
        arma::vec laplb(electron_density_laplacian(Pbrad)/angfac);
        // Radial coordinates
        arma::vec r(radii());
        size_t Npoints(r.n_elem);

        // and pack it for libxc
        arma::mat rho_libxc(Npoints,2);
        rho_libxc.col(0)=rhoa;
        rho_libxc.col(1)=rhob;
        // Take transpose so that order is (na0, nb0, na1, nb1, ...)
        rho_libxc=rho_libxc.t();

        // Reduced gradient
        arma::mat sigma_libxc(Npoints,3);
        sigma_libxc.col(0)=grada%grada;
        sigma_libxc.col(1)=grada%gradb;
        sigma_libxc.col(2)=gradb%gradb;
        // Take transpose to get correct order
        sigma_libxc=sigma_libxc.t();

        // Potential
        arma::mat vxc(2,Npoints);
        vxc.zeros();
        arma::mat vsigma(3,Npoints);
        vsigma.zeros();

        // For GGA we also need the second derivative to calculate the
        // correction to the potential
        arma::mat v2rhosigma(6,Npoints);
        v2rhosigma.zeros();
        arma::mat v2sigma2(6,Npoints);
        v2sigma2.zeros();

        // Helper arrays
        arma::mat vxc_wrk(2,Npoints);
        arma::mat vsigma_wrk(3,Npoints);
        arma::mat v2rho2_wrk(3,Npoints);
        arma::mat v2rhosigma_wrk(6,Npoints);
        arma::mat v2sigma2_wrk(6,Npoints);

        bool do_gga=false;

        if(x_func>0) {
          vxc_wrk.zeros();
          vsigma_wrk.zeros();
          v2rhosigma_wrk.zeros();
          v2sigma2_wrk.zeros();

          bool gga, mggat, mggal;
          ::is_gga_mgga(x_func, gga, mggat, mggal);
          if(mggat || mggal)
            throw std::logic_error("Mggas not supported!\n");

          xc_func_type func;
          if(xc_func_init(&func, x_func, XC_POLARIZED) != 0) {
            throw std::logic_error("Error initializing exchange functional!\n");
          }
          if(gga) {
            xc_gga_vxc(&func, rhoa.n_elem, rho_libxc.memptr(), sigma_libxc.memptr(), vxc_wrk.memptr(), vsigma_wrk.memptr());
            xc_gga_fxc(&func, rhoa.n_elem, rho_libxc.memptr(), sigma_libxc.memptr(), v2rho2_wrk.memptr(), v2rhosigma_wrk.memptr(), v2sigma2_wrk.memptr());
            do_gga=true;
            vsigma+=vsigma_wrk;
            v2rhosigma+=v2rhosigma_wrk;
            v2sigma2+=v2sigma2_wrk;
          } else {
            xc_lda_vxc(&func, rhoa.n_elem, rho_libxc.memptr(), vxc_wrk.memptr());
          }
          xc_func_end(&func);

          vxc+=vxc_wrk;
        }
        if(c_func>0) {
          vxc_wrk.zeros();
          vsigma_wrk.zeros();
          v2rhosigma_wrk.zeros();
          v2sigma2_wrk.zeros();

          bool gga, mggat, mggal;
          ::is_gga_mgga(c_func, gga, mggat, mggal);
          if(mggat || mggal)
            throw std::logic_error("Mggas not supported!\n");

          xc_func_type func;
          if(xc_func_init(&func, c_func, XC_POLARIZED) != 0) {
            throw std::logic_error("Error initializing correlation functional!\n");
          }
          if(gga) {
            xc_gga_vxc(&func, rhoa.n_elem, rho_libxc.memptr(), sigma_libxc.memptr(), vxc_wrk.memptr(), vsigma_wrk.memptr());
            xc_gga_fxc(&func, rhoa.n_elem, rho_libxc.memptr(), sigma_libxc.memptr(), v2rho2_wrk.memptr(), v2rhosigma_wrk.memptr(), v2sigma2_wrk.memptr());
            do_gga=true;
            vsigma+=vsigma_wrk;
            v2rhosigma+=v2rhosigma_wrk;
            v2sigma2+=v2sigma2_wrk;
          } else {
            xc_lda_vxc(&func, rhoa.n_elem, rho_libxc.memptr(), vxc_wrk.memptr());
          }

          vxc+=vxc_wrk;
        }

        // Add GGA correction to xc potential.
        if(do_gga) {
          arma::mat corr(2,vxc.n_cols);
          corr.zeros();

          // Loop over points: skip nucleus since there's no laplacian there for now
          for(size_t ip=1;ip<Npoints;ip++) {
            // First term: g(t) ( d^2 E / d n(t) d sigma(ss') ) g(s')

            // (a, aa) + (b, aa)
            corr(0,ip) += 2.0*(grada(ip)*v2rhosigma(0,ip) + gradb(ip)*v2rhosigma(3,ip))*grada(ip);
            // (a, ab) + (b, ab)
            corr(0,ip) += (grada(ip)*v2rhosigma(1,ip) + gradb(ip)*v2rhosigma(4,ip))*gradb(ip);

            // (b, bb) + (a, bb)
            corr(1,ip) += 2.0*(gradb(ip)*v2rhosigma(5,ip) + grada(ip)*v2rhosigma(2,ip))*gradb(ip);
            // (a, ab) + (b, ab)
            corr(1,ip) += (grada(ip)*v2rhosigma(1,ip) + gradb(ip)*v2rhosigma(4,ip))*grada(ip);
          }

          for(size_t ip=1;ip<Npoints;ip++) {
            // Second term: (l(t)g(t') + g(t)l(t')) (d^2 E / d sigma(tt') d sigma(ss')) g(s'). Contract t and t' first, put in factor two later
            double d2Edsaa = lapla(ip)*grada(ip)*v2sigma2(0,ip) + (lapla(ip)*gradb(ip) + grada(ip)*laplb(ip))*v2sigma2(1,ip) + laplb(ip)*gradb(ip)*v2sigma2(2,ip);
            double d2Edsab = lapla(ip)*grada(ip)*v2sigma2(1,ip) + (lapla(ip)*gradb(ip) + grada(ip)*laplb(ip))*v2sigma2(3,ip) + laplb(ip)*gradb(ip)*v2sigma2(4,ip);
            double d2Edsbb = lapla(ip)*grada(ip)*v2sigma2(2,ip) + (lapla(ip)*gradb(ip) + grada(ip)*laplb(ip))*v2sigma2(4,ip) + laplb(ip)*gradb(ip)*v2sigma2(5,ip);
            // and now we can contract this with the gradient
            corr(0,ip) += 4.0*d2Edsaa*grada(ip) + 2.0*d2Edsab*gradb(ip);
            corr(1,ip) += 4.0*d2Edsbb*gradb(ip) + 2.0*d2Edsab*grada(ip);
          }

          for(size_t ip=1;ip<Npoints;ip++) {
            // Third term: dE/dsigma(ss') l(s')
            corr(0,ip) += 2.0*vsigma(0,ip)*lapla(ip) + vsigma(1,ip)*laplb(ip);
            corr(1,ip) += vsigma(1,ip)*lapla(ip) + 2.0*vsigma(2,ip)*laplb(ip);
          }

          for(size_t ip=1;ip<Npoints;ip++) {
            // Second term in the divergence: div A = dA/dr + 2 r A
            corr(0,ip) += 2.0/r(ip)*(2.0*vsigma(0,ip)*grada(ip) + vsigma(1,ip)*gradb(ip));
            corr(1,ip) += 2.0/r(ip)*(vsigma(1,ip)*grada(ip) + 2.0*vsigma(2,ip)*gradb(ip));
          }

          // Perform the correction
          vxc -= corr;
        }

        // Transpose
        vxc = vxc.t();
        // Convert to radial potential (this is how it matches with GPAW)
        for(size_t ic=0;ic<vxc.n_cols;ic++)
          vxc.col(ic)%=r;

        return vxc;
      }
    }
  }
}
