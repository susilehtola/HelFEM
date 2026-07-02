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
#include <ArmaEigen.h>
#include <CoulombExchangeFE.h>
#include "../general/radial_block_helper.h"
#include "chebyshev.h"
#include "../general/spherical_harmonics.h"
#include "../general/gaunt.h"
#include "../general/gsz.h"
#include "utils.h"
#include "../general/scf_helpers.h"
#include "../general/dftfuncs.h"
#include <cassert>
#include <cfloat>
#include <sstream>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <xc.h>


namespace helfem {
  namespace sadatom {
    namespace basis {
      TwoDBasis::TwoDBasis() {
      }

      TwoDBasis::TwoDBasis(int Z_, modelpotential::nuclear_model_t model_, double Rrms_, const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly, bool zeroder, int n_quad, const arma::vec & bval, int lmax) {
        // Nuclear charge
        Z=Z_;
        model=model_;
        Rrms=Rrms_;
        // Construct radial basis
        bool zero_func_left=true;
        bool zero_deriv_left=false;
        bool zero_func_right=true;
        polynomial_basis::FiniteElementBasis fem(poly, bval, zero_func_left, zero_deriv_left, zero_func_right, zeroder);
        radial=atomic::basis::FEMRadialBasis(fem, n_quad);
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
        arma::mat S(helfem::to_arma(overlap()));
        return helfem::to_arma(scf::form_Sinvh(helfem::to_eigen(S), false));
      }

      arma::mat TwoDBasis::radial_integral(int Rexp) const {
        // Bridge to arma at the public API boundary.
        return helfem::to_arma(helfem::assemble_radial_diagonal(radial,
            [&](size_t iel) { return radial.radial_integral(Rexp, iel); }));
      }

      helfem::Matrix TwoDBasis::overlap() const {
        return helfem::assemble_radial_diagonal(radial,
            [&](size_t iel) { return radial.radial_integral(0, iel); });
      }

      helfem::Matrix TwoDBasis::kinetic() const {
        return helfem::assemble_radial_diagonal(radial,
            [&](size_t iel) { return radial.kinetic(iel); });
      }

      helfem::Matrix TwoDBasis::kinetic_l() const {
        return helfem::assemble_radial_diagonal(radial,
            [&](size_t iel) { return radial.kinetic_l(iel); });
      }

      helfem::Matrix TwoDBasis::nuclear() const {
        if (model != modelpotential::POINT_NUCLEUS) {
          modelpotential::ModelPotential * pot = modelpotential::get_nuclear_model(model, Z, Rrms);
          arma::mat Vrad(model_potential(pot));
          delete pot;
          return helfem::to_eigen(Vrad);
        }
        return -Z * helfem::assemble_radial_diagonal(radial,
            [&](size_t iel) { return radial.radial_integral(-1, iel); });
      }

      arma::mat TwoDBasis::confinement(int N, double r_0, int iconf, double V, double shift_pot) const {
        if (!iconf) {
          const size_t Nrad = radial.Nbf();
          return arma::zeros<arma::mat>(Nrad, Nrad);
        }
        return helfem::to_arma(helfem::assemble_radial_diagonal(radial,
            [&](size_t iel) {
              return radial.confinement_potential(iel, N, r_0, iconf, V, shift_pot);
            }));
      }

      arma::mat TwoDBasis::model_potential(const modelpotential::ModelPotential * pot) const {
        return helfem::to_arma(helfem::assemble_radial_diagonal(radial,
            [&](size_t iel) { return radial.model_potential(pot, iel); }));
      }

      void TwoDBasis::compute_tei() {
        // Delegate to the shared cache builders in CoulombExchangeFE.h
        // -- same path atomic::TwoDBasis uses. Bare Coulomb + in-element
        // prim_tei + exchange-permuted prim_ktei.
        const int N_L = 2 * arma::max(lval) + 1;
        atomic::basis::compute_disjoint_radial_integrals(
            radial, N_L, disjoint_L, disjoint_m1L);
        atomic::basis::compute_in_element_tei(radial, N_L, prim_tei);
        atomic::basis::compute_in_element_ktei_from_tei(
            radial, N_L, prim_tei, prim_ktei);
      }

      void TwoDBasis::compute_yukawa(double lambda_) {
        lambda = lambda_;
        yukawa = true;
        const int N_L = 2 * arma::max(lval) + 1;
        // Yukawa disjoint factors (bessel i / bessel k) + in-element
        // Yukawa 2e -> exchange-permuted form needed by the cached K
        // assembly.
        atomic::basis::compute_disjoint_radial_integrals(
            radial, N_L, disjoint_iL, disjoint_kL, /*yukawa=*/true, lambda);
        std::vector<helfem::Matrix> rs_tei_yk;
        atomic::basis::compute_in_element_tei(
            radial, N_L, rs_tei_yk, /*yukawa=*/true, lambda);
        atomic::basis::compute_in_element_ktei_from_tei(
            radial, N_L, rs_tei_yk, rs_ktei);
      }

      void TwoDBasis::compute_erfc(double mu) {
        lambda = mu;
        yukawa = false;
        // Erfc kernel doesn't factorise -- no disjoint integrals; all
        // (iel, jel) pairs are stored explicitly in rs_ktei.
        disjoint_iL.clear();
        disjoint_kL.clear();
        const int N_L = 2 * arma::max(lval) + 1;
        atomic::basis::compute_erfc_ktei(radial, N_L, lambda, rs_ktei);
      }

      helfem::Matrix TwoDBasis::coulomb(const helfem::Matrix & P_in) const {
        if(!prim_tei.size())
          throw std::logic_error("Primitive teis have not been computed!\n");
        // Phase 3: SCF surface takes Eigen; internal J helper still
        // takes arma -- one conversion at entry, one at exit.
        const arma::mat P = helfem::to_arma(P_in);

        // sadatom is spherically averaged: only the L=0 multipole
        // contributes. Delegate to the shared FE assembly with our
        // SCF-cached per-element integrals.
        const size_t Nel = radial.Nel();
        const int    L   = 0;
        const double Lfac = 4.0 * M_PI / (2 * L + 1);
        auto rs = [&](size_t iel) -> const helfem::Matrix & {
          return disjoint_L[L * Nel + iel];
        };
        auto rb = [&](size_t iel) -> const helfem::Matrix & {
          return disjoint_m1L[L * Nel + iel];
        };
        auto tw = [&](size_t iel) -> const helfem::Matrix & {
          return prim_tei[Nel * Nel * L + iel * Nel + iel];
        };
        return helfem::to_eigen(
            arma::mat(Lfac * atomic::basis::assemble_J_FE_one_multipole_cached(
                radial, rs, rb, tw, P)));
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

        // The per-element K assembly is now delegated to
        // assemble_K_FE_one_multipole_cached -- no per-thread scratch
        // needed here, the helper allocates its own.

#ifdef _OPENMP
#pragma omp parallel
#endif
        {
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
                    double cpl(gaunt.coeff(lout,mout,L,M,lin)*gaunt.coeff(lout,mout,L,M,lin));
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
              // Delegate the per-L K assembly to the shared FE helper.
              // K is accumulated with the standard HF minus sign baked
              // in (matches the pre-refactor convention).
              const int Lint = (int) L;
              auto rs = [&,Lint](size_t iel) -> const helfem::Matrix & {
                return disjoint_L[Lint * Nel + iel];
              };
              auto rb = [&,Lint](size_t iel) -> const helfem::Matrix & {
                return disjoint_m1L[Lint * Nel + iel];
              };
              auto kt = [&,Lint](size_t iel) -> const helfem::Matrix & {
                return prim_ktei[Nel * Nel * Lint + iel * Nel + iel];
              };
              K.slice(lout) -= atomic::basis::assemble_K_FE_one_multipole_cached(
                  radial, rs, rb, kt, Prad.slice(L));
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

        // Per-element K assembly is delegated to the cached helpers in
        // CoulombExchangeFE.h -- no per-thread scratch needed here.
#ifdef _OPENMP
#pragma omp parallel
#endif
        {
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
                    double cpl(gaunt.coeff(lout,mout,L,M,lin)*gaunt.coeff(lout,mout,L,M,lin));
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
              const arma::mat & P_L = Prad.slice(L);
              if (yukawa) {
                // Same FE structure as bare exchange -- just different
                // kernels in the cache. Delegate to the shared helper.
                const size_t Lc = L;
                auto rs = [&,Lc](size_t iel) -> const helfem::Matrix & {
                  return disjoint_iL[Lc*Nel+iel];
                };
                auto rb = [&,Lc](size_t iel) -> const helfem::Matrix & {
                  return disjoint_kL[Lc*Nel+iel];
                };
                auto kt = [&,Lc](size_t iel) -> const helfem::Matrix & {
                  return rs_ktei[Nel*Nel*Lc + iel*Nel + iel];
                };
                K.slice(lout) -=
                  atomic::basis::assemble_K_FE_one_multipole_cached(
                    radial, rs, rb, kt, P_L);
              } else {
                // Erfc: rs_ktei has cross-element entries (iel != jel)
                // because the erfc kernel does not factorise. Delegate
                // to the pairwise cached helper -- same contraction
                // pattern, just no disjoint optimisation.
                const size_t Lc = L;
                auto kt = [&,Lc](size_t iel, size_t jel) -> const helfem::Matrix & {
                  return rs_ktei[Nel*Nel*Lc + iel*Nel + jel];
                };
                K.slice(lout) -=
                  atomic::basis::assemble_K_FE_one_multipole_cached_pairwise(
                    radial, kt, P_L);
              }
            }
          }
        }

        return K;
      }

      std::vector<arma::mat> TwoDBasis::get_prim_tei() const {
        // Phase 2c: prim_tei is std::vector<helfem::Matrix> internally;
        // bridge to arma for the public accessor.
        std::vector<arma::mat> out;
        out.reserve(prim_tei.size());
        for (const auto & m : prim_tei) out.push_back(helfem::to_arma(m));
        return out;
      }

      arma::mat TwoDBasis::eval_bf(size_t iel) const {
        return helfem::to_arma(radial.get_bf(iel));
      }

      arma::mat TwoDBasis::eval_df(size_t iel) const {
        return helfem::to_arma(radial.get_df(iel));
      }

      arma::mat TwoDBasis::eval_lf(size_t iel) const {
        return helfem::to_arma(radial.get_lf(iel));
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

      arma::vec TwoDBasis::eval_orbs(const arma::mat & C, double r) const {
        return helfem::to_arma(radial.eval_orbs(helfem::to_eigen(C), r));
      }

      size_t TwoDBasis::get_rad_Nel() const {
        return radial.Nel();
      }

      arma::vec TwoDBasis::get_wrad(size_t iel) const {
        return helfem::to_arma(radial.get_wrad(iel));
      }

      arma::vec TwoDBasis::get_r(size_t iel) const {
        return helfem::to_arma(radial.get_r(iel));
      }

      double TwoDBasis::nuclear_density(const arma::mat & P) const {
        return radial.nuclear_density(helfem::to_eigen(P))/(4.0*M_PI);
      }

      double TwoDBasis::nuclear_density_gradient(const arma::mat & P) const {
        return radial.nuclear_density_gradient(helfem::to_eigen(P))/(4.0*M_PI);
      }

      arma::vec TwoDBasis::quadrature_weights() const {
        std::vector<arma::vec> w(radial.Nel());
        size_t ntot=1;
        for(size_t iel=0;iel<radial.Nel();iel++) {
          w[iel]=helfem::to_arma(radial.get_wrad(iel));
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
          arma::mat zm(helfem::to_arma(radial.radial_integral(0,iel)));
          arma::mat mo(helfem::to_arma(radial.radial_integral(-1,iel)));
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
          r[iel]=helfem::to_arma(radial.get_r(iel));
          V[iel].zeros(r[iel].n_elem);

          // Get the density in the element
          size_t ifirst, ilast;
          radial.get_idx(iel,ifirst,ilast);
          arma::vec Pv(arma::vectorise(Prad.submat(ifirst,ifirst,ilast,ilast)));

          // Calculate the in-element potential
          arma::mat pot(helfem::to_arma(radial.spherical_potential(iel)));
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
          r[iel]=helfem::to_arma(radial.get_r(iel));
        }

        size_t Npts=r[0].n_elem;
        arma::vec rad(radial.Nel()*Npts+1);
        rad.zeros();
        for(size_t iel=0;iel<radial.Nel();iel++) {
          rad.subvec(1+iel*Npts,(iel+1)*Npts)=r[iel];
        }

        return rad;
      }

      double TwoDBasis::slater_F(int k, const arma::vec & c) const {
        // Bare radial Slater-Condon F^k integral for orbital `c` in the
        // u = r * R basis. Defined as
        //     F^k(c, c) = sum_{ab,cd} P_ab P_cd R^k_FE(ab, cd)
        // with P = c * c.t() and R^k_FE the bare per-multipole 2e integral
        // from CoulombExchangeFE.h (no 4*pi/(2k+1) factor).
        //
        // F^k = trace(P * J^k(P)) where J^k = assemble_J_FE_one_multipole.
        if (c.n_elem != radial.Nbf()) {
          std::ostringstream oss;
          oss << "TwoDBasis::slater_F: orbital length " << c.n_elem
              << " != Nrad " << radial.Nbf() << ".\n";
          throw std::logic_error(oss.str());
        }
        const arma::mat P = c * c.t();
        const arma::mat Jk =
            atomic::basis::assemble_J_FE_one_multipole(radial, k, P);
        return arma::trace(P * Jk);
      }

      arma::mat TwoDBasis::orbitals(const arma::mat & C) const {
        std::vector<arma::mat> c(radial.Nel());
        for(size_t iel=0;iel<radial.Nel();iel++) {
          // Radial functions in element
          size_t ifirst, ilast;
          radial.get_idx(iel,ifirst,ilast);
          // Density matrix
          arma::mat Csub(C.rows(ifirst,ilast));
          arma::mat bf(helfem::to_arma(radial.get_bf(iel)));

          c[iel]=bf*Csub;
        }
        size_t Npts=c[0].n_rows;

        arma::mat Cv(radial.Nel()*Npts+1,C.n_cols);
        Cv.zeros();
        // Values at the nucleus
        {
          {
            const Eigen::RowVectorXd nrow = radial.nuclear_orbital(helfem::to_eigen(C));
            for (Eigen::Index j = 0; j < nrow.size(); ++j)
              Cv(0, j) = nrow(j);
          }
        }
        // Other points
        for(size_t iel=0;iel<radial.Nel();iel++) {
          Cv.rows(1+iel*Npts,(iel+1)*Npts)=c[iel];
        }

        return Cv;
      }

      arma::mat TwoDBasis::orbitals_derivative(const arma::mat & C) const {
        std::vector<arma::mat> c(radial.Nel());
        for(size_t iel=0;iel<radial.Nel();iel++) {
          // Radial functions in element
          size_t ifirst, ilast;
          radial.get_idx(iel,ifirst,ilast);
          // Density matrix
          arma::mat Csub(C.rows(ifirst,ilast));
          arma::mat bf(helfem::to_arma(radial.get_df(iel)));

          c[iel]=bf*Csub;
        }
        size_t Npts=c[0].n_rows;

        arma::mat Cv(radial.Nel()*Npts+1,C.n_cols);
        Cv.zeros();
        // Values at the nucleus
        {
	  // tbd
        }
        // Other points
        for(size_t iel=0;iel<radial.Nel();iel++) {
          Cv.rows(1+iel*Npts,(iel+1)*Npts)=c[iel];
        }

        return Cv;
      }

      arma::mat TwoDBasis::orbitals_second_derivative(const arma::mat & C) const {
        std::vector<arma::mat> c(radial.Nel());
        for(size_t iel=0;iel<radial.Nel();iel++) {
          // Radial functions in element
          size_t ifirst, ilast;
          radial.get_idx(iel,ifirst,ilast);
          // Density matrix
          arma::mat Csub(C.rows(ifirst,ilast));
          arma::mat bf(helfem::to_arma(radial.get_lf(iel)));

          c[iel]=bf*Csub;
        }
        size_t Npts=c[0].n_rows;

        arma::mat Cv(radial.Nel()*Npts+1,C.n_cols);
        Cv.zeros();
        // Values at the nucleus
        {
	  // tbd
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
        arma::mat bf(helfem::to_arma(radial.get_bf(x, iel)));

        arma::vec density = arma::diagvec(bf*Psub*bf.t());
        if(rsqweight)
          density %= arma::square(helfem::to_arma(radial.get_r(x, iel)));
        return density;
      }

      arma::vec TwoDBasis::electron_density(size_t iel, const arma::mat & Prad, bool rsqweight) const {
        return electron_density(helfem::to_arma(radial.get_xq()), iel, Prad, rsqweight);
      }

      arma::vec TwoDBasis::electron_density(const arma::mat & Prad, bool rsqweight) const {
        std::vector<arma::vec> d(radial.Nel());
        for(size_t iel=0;iel<radial.Nel();iel++) {
          d[iel]=electron_density(iel, Prad, rsqweight);
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

      double TwoDBasis::electron_density_maximum_radius(const arma::mat & Prad, bool rsqweight, double eps) const {
        // Evaluate the density in each quadrature point and take
        // their maximum
        arma::vec den(radial.Nel());

        for(size_t iel=0;iel<radial.Nel();iel++) {
          den(iel)=arma::max(electron_density(iel, Prad, rsqweight));
        }

        // Quadrature points
        arma::vec xq = helfem::to_arma(radial.get_xq());

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
          rmax = radial.get_r((a+b)/2, iel)(0);
        }

        return rmax;
      }

      double TwoDBasis::vdw_radius(const arma::mat & Prad, double vdw_threshold, double eps) const {
        // Need to multiply output of electron_density by this factor to get the point-wise density
        double angfac=1.0/(4.0*M_PI);
	bool rsqweight=false;

        // Evaluate the density in each quadrature point and take
        // their maximum
        size_t iel;
        for(iel=radial.Nel()-1;iel<radial.Nel();iel--) {
          arma::vec den(angfac*electron_density(iel, Prad, rsqweight));
          if(arma::max(den)>vdw_threshold) {
            // We found the element
            break;
          }
        }
        if(iel>radial.Nel()) {
          // No point of density is above threshold!
          return 0.0;
        }

        // Now find the position in the element where the density is = vdw_threshold.
        // Evaluate the difference in density from the threshold value.
        // Quadrature points
        arma::vec xq = helfem::to_arma(radial.get_xq());
        arma::vec diff = angfac*electron_density(xq, iel, Prad, rsqweight);
        diff-=vdw_threshold*arma::ones<arma::vec>(diff.n_elem);
        diff=arma::abs(diff);

        // Find the smallest value
        arma::uword imax;
        diff.min(imax);

        // Refine the position.

        double rvdw=0.0;
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

          // Bisection
          size_t ibisect=0;
          while(arma::norm(a-b,"inf")>=eps) {
            arma::vec m = a + (b-a)/2.0;
            double density_m = angfac*arma::as_scalar(electron_density(m, iel, Prad, rsqweight));

            if(density_m < eps) {
              b = m;
            } else if(density_m > eps) {
              a = m;
            } else if(density_m == eps)
              break;

            ibisect++;
            if(ibisect==100)
              throw std::runtime_error("bisection did not converge in 100 iterations\n");
          }
          // Coordinate is
          arma::vec cen=((a+b)/2);
          // Position of maximum is
          rvdw = arma::as_scalar(helfem::to_arma(radial.get_r(cen,iel)));
        }

        return rvdw;
      }

      double TwoDBasis::electron_count_radius(const arma::mat & Prad, const double eps, const double conv_thr) const {
	// Vector with electron density contributions from each element
	std::vector<double> densities(radial.Nel());
	for(size_t iel=0;iel<radial.Nel();iel++) {
	  // Radial functions in element
	  size_t ifirst, ilast;
	  radial.get_idx(iel,ifirst,ilast);
	  // Density matrix
	  arma::mat Psub(Prad.submat(ifirst,ifirst,ilast,ilast));
	  // Overlap matrix (Phase 2a: overlap(iel) returns helfem::Matrix).
	  arma::mat S(helfem::to_arma(radial.overlap(iel)));
	  densities[iel]=arma::trace(Psub*S);
	}

	// Search for the correct element
	double s_left=eps;
	size_t ielement;
	for(ielement=radial.Nel()-1;ielement<radial.Nel();ielement--) {
	  if(s_left>densities[ielement])
	    s_left-=densities[ielement];
	  else
	    break;
	}

	// Radial functions in element
	size_t ifirst, ilast;
	radial.get_idx(ielement,ifirst,ilast);
	// Density matrix within the element
	arma::mat Psub(Prad.submat(ifirst,ifirst,ilast,ilast));

	// Search for the radius within element
	double result;
	// Element limits
	double a=-1.0, b=1.0;
	double m=a+(b-a)/2.0;
	while(b-a>=conv_thr) {
	  m=a+(b-a)/2.0;
	  result=arma::trace(Psub*helfem::to_arma(radial.radial_integral(0,ielement,m,1.0)));
	  if(result<s_left)
	    b=m;
	  else if(result>s_left)
	    a=m;
	  else
	    break;
	}

	return radial.get_r(m, ielement);
      }

      arma::vec TwoDBasis::electron_density_gradient(const arma::mat & Prad) const {
        std::vector<arma::vec> d(radial.Nel());
        for(size_t iel=0;iel<radial.Nel();iel++) {
          // Radial functions in element
          size_t ifirst, ilast;
          radial.get_idx(iel,ifirst,ilast);
          // Density matrix
          arma::mat Psub(Prad.submat(ifirst,ifirst,ilast,ilast));
          arma::mat bf(helfem::to_arma(radial.get_bf(iel)));
          arma::mat df(helfem::to_arma(radial.get_df(iel)));

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
          arma::mat bf(helfem::to_arma(radial.get_bf(iel)));
          arma::mat df(helfem::to_arma(radial.get_df(iel)));
          arma::mat lf(helfem::to_arma(radial.get_lf(iel)));
          arma::vec r(helfem::to_arma(radial.get_r(iel)));

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
          arma::vec r(helfem::to_arma(radial.get_r(iel)));

          // Density matrix
          arma::mat Psub(P.submat(ifirst,ifirst,ilast,ilast));
          arma::mat Psubl(Pl.submat(ifirst,ifirst,ilast,ilast));

          // Basis function
          arma::mat bf(helfem::to_arma(radial.get_bf(iel)));
          // Basis function derivative
          arma::mat bf_rho(helfem::to_arma(radial.get_df(iel)));

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

      std::vector<std::pair<int, atomic::basis::NAORadialBasis>>
      extract_naos_per_l(const TwoDBasis & sad_basis,
                         const arma::cube & Ccube,
                         const std::vector<int> & keep_per_l) {
        const int lmax = static_cast<int>(Ccube.n_slices) - 1;
        if (static_cast<int>(keep_per_l.size()) != lmax + 1) {
          std::ostringstream oss;
          oss << "extract_naos_per_l: keep_per_l has size " << keep_per_l.size()
              << " but Ccube has " << Ccube.n_slices
              << " slices (expected " << lmax + 1 << ").\n";
          throw std::logic_error(oss.str());
        }
        std::vector<std::pair<int, atomic::basis::NAORadialBasis>> out;
        out.reserve(lmax + 1);
        for (int l = 0; l <= lmax; ++l) {
          const int keep = keep_per_l[l];
          if (keep == 0) continue;
          const arma::mat & Cl = Ccube.slice(l);
          if (keep > static_cast<int>(Cl.n_cols)) {
            std::ostringstream oss;
            oss << "extract_naos_per_l: requested " << keep
                << " NAOs for l=" << l << " but only "
                << Cl.n_cols << " orbitals available.\n";
            throw std::logic_error(oss.str());
          }
          const arma::mat Ckeep = (keep < 0) ? Cl
                                             : arma::mat(Cl.cols(0, keep - 1));
          out.emplace_back(l, atomic::basis::NAORadialBasis::from_owned_radial(
              sad_basis.get_radial(), Ckeep));
        }
        return out;
      }
    }
  }
}
