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
  namespace sadatom {
    namespace basis {
      TwoDBasis::TwoDBasis() {
      }

      TwoDBasis::TwoDBasis(int Z_, const polynomial_basis::PolynomialBasis * poly, int n_quad, int num_el, double rmax, int lmax, int igrid, double zexp) {
        // Nuclear charge
        Z=Z_;
        // Construct radial basis
        radial=atomic::basis::RadialBasis(poly, n_quad, num_el, rmax, igrid, zexp);
        // Angular basis
        lval=arma::linspace<arma::ivec>(0,lmax,lmax+1);
      }
      TwoDBasis::~TwoDBasis() {
      }

      size_t TwoDBasis::Nbf() const {
        return radial.Nbf();
      }

      arma::mat TwoDBasis::Sinvh() const {
        // Form overlap matrix
        arma::mat S(overlap());

        // Get the basis function norms
        arma::vec bfnormlz(arma::pow(arma::diagvec(S),-0.5));
        // Go to normalized basis
        S=arma::diagmat(bfnormlz)*S*arma::diagmat(bfnormlz);

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

      void TwoDBasis::compute_tei() {
        // Number of distinct L values is
        size_t N_L(1);
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

      arma::mat TwoDBasis::eval_bf(size_t iel) const {
        // Evaluate radial functions
        arma::mat rad(radial.get_bf(iel));

        // Spherical harmonics contribution
        //rad/=sqrt(4.0*M_PI);

        return rad;
      }

      arma::mat TwoDBasis::eval_df(size_t iel) const {
        // Evaluate radial functions
        arma::mat drad(radial.get_df(iel));

        // Form supermatrices
        //drad/=sqrt(4.0*M_PI);

        return drad;
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

      double TwoDBasis::nuclear_density(const arma::mat & P) const {
        // Radial functions in first element
        size_t ifirst, ilast;
        radial.get_idx(0,ifirst,ilast);

        return radial.nuclear_density(P.submat(ifirst,ifirst,ilast,ilast))/(4.0*M_PI);
      }
    }
  }
}
