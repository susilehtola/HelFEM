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

      arma::mat TwoDBasis::thomasfermi() const {
        size_t Nrad(radial.Nbf());
        arma::mat Vrad(Nrad,Nrad);
        Vrad.zeros();
        // Loop over elements
        for(size_t iel=0;iel<radial.Nel();iel++) {
          // Where are we in the matrix?
          size_t ifirst, ilast;
          radial.get_idx(iel,ifirst,ilast);
          Vrad.submat(ifirst,ifirst,ilast,ilast)+=radial.thomasfermi(Z,iel);
        }

        return Vrad;
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

        Veff.save("vj.dat",arma::raw_ascii);

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

      arma::vec TwoDBasis::electron_density(const arma::mat & Prad) const {
        std::vector<arma::vec> d(radial.Nel());
        for(size_t iel=0;iel<radial.Nel();iel++) {
          // Radial functions in element
          size_t ifirst, ilast;
          radial.get_idx(iel,ifirst,ilast);
          // Density matrix
          arma::mat Psub(Prad.submat(ifirst,ifirst,ilast,ilast));
          arma::mat bf(radial.get_bf(iel));

          d[iel]=arma::diagvec(bf*Psub*bf.t());
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

        // Gradient at the nucleus
        {
          // Get polynomial basis
          polynomial_basis::PolynomialBasis * poly(radial.get_poly());
          arma::vec x(1);
          x(0)=-1;
          arma::mat func, der;
          poly->eval(x,func,der);
          func=radial.get_basis(func,0);

          arma::vec bval=radial.get_bval();
          double rmin(bval(0));
          double rmax(bval(1));
          double rlen=(rmax-rmin)/2;
          der=(radial.get_basis(der,0)/rlen);

          // Radial functions in element
          size_t ifirst, ilast;
          radial.get_idx(0,ifirst,ilast);
          // Density matrix
          arma::mat Psub(Prad.submat(ifirst,ifirst,ilast,ilast));
          n(0)=2.0*arma::as_scalar(func*Psub*der.t());
        }
        // Other points
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

          l[iel]=2.0*(arma::diagvec(df*Psub*df.t()) + arma::diagvec(bf*Psub*lf.t()));
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

        // Energy
        arma::vec exc(Npoints);
        exc.zeros();
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
        arma::vec exc_wrk(Npoints);
        arma::mat vxc_wrk(2,Npoints);
        arma::mat vsigma_wrk(3,Npoints);
        arma::mat v2rho2_wrk(3,Npoints);
        arma::mat v2rhosigma_wrk(6,Npoints);
        arma::mat v2sigma2_wrk(6,Npoints);

        bool do_gga=false;

        if(x_func>0) {
          exc_wrk.zeros();
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
            xc_gga(&func, rhoa.n_elem, rho_libxc.memptr(), sigma_libxc.memptr(), exc_wrk.memptr(), vxc_wrk.memptr(), vsigma_wrk.memptr(), v2rho2_wrk.memptr(), v2rhosigma_wrk.memptr(), v2sigma2_wrk.memptr(), NULL, NULL, NULL, NULL);
            do_gga=true;
            vsigma+=vsigma_wrk;
            v2rhosigma+=v2rhosigma_wrk;
            v2sigma2+=v2sigma2_wrk;
          } else {
            xc_lda_exc_vxc(&func, rhoa.n_elem, rho_libxc.memptr(), exc_wrk.memptr(), vxc_wrk.memptr());
          }
          xc_func_end(&func);

          vxc+=vxc_wrk;
          exc+=exc_wrk;
        }
        if(c_func>0) {
          exc_wrk.zeros();
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
            xc_gga(&func, rhoa.n_elem, rho_libxc.memptr(), sigma_libxc.memptr(), exc_wrk.memptr(), vxc_wrk.memptr(), vsigma_wrk.memptr(), v2rho2_wrk.memptr(), v2rhosigma_wrk.memptr(), v2sigma2_wrk.memptr(), NULL, NULL, NULL, NULL);
            do_gga=true;
            vsigma+=vsigma_wrk;
            v2rhosigma+=v2rhosigma_wrk;
            v2sigma2+=v2sigma2_wrk;
          } else {
            xc_lda_exc_vxc(&func, rhoa.n_elem, rho_libxc.memptr(), exc_wrk.memptr(), vxc_wrk.memptr());
          }

          vxc+=vxc_wrk;
          exc+=exc_wrk;
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
