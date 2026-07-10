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
#include "../general/gaunt.h"
#include "utils.h"
#include "../general/scf_helpers.h"
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
        polynomial_basis::FiniteElementBasis fem(poly, helfem::to_eigen(bval), zero_func_left, zero_deriv_left, zero_func_right, zeroder);
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

      helfem::Matrix TwoDBasis::Sinvh() const {
        return scf::form_Sinvh(overlap(), /*chol=*/false);
      }

      helfem::Matrix TwoDBasis::overlap(const TwoDBasis & rh) const {
        return radial.overlap(rh.radial);
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

      helfem::Matrix TwoDBasis::confinement(int N, double r_0, int iconf, double V, double shift_pot) const {
        if (!iconf) {
          const Eigen::Index Nrad = static_cast<Eigen::Index>(radial.Nbf());
          return helfem::Matrix::Zero(Nrad, Nrad);
        }
        return helfem::assemble_radial_diagonal(radial,
            [&](size_t iel) {
              return radial.confinement_potential(iel, N, r_0, iconf, V, shift_pot);
            });
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
        return Lfac * atomic::basis::assemble_J_FE_one_multipole_cached(
            radial, rs, rb, tw, helfem::to_eigen(P));
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
              K.slice(lout) -= helfem::to_arma(
                  atomic::basis::assemble_K_FE_one_multipole_cached(
                      radial, rs, rb, kt, helfem::to_eigen(arma::mat(Prad.slice(L)))));
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
                K.slice(lout) -= helfem::to_arma(
                  atomic::basis::assemble_K_FE_one_multipole_cached(
                    radial, rs, rb, kt, helfem::to_eigen(P_L)));
              } else {
                // Erfc: rs_ktei has cross-element entries (iel != jel)
                // because the erfc kernel does not factorise. Delegate
                // to the pairwise cached helper -- same contraction
                // pattern, just no disjoint optimisation.
                const size_t Lc = L;
                auto kt = [&,Lc](size_t iel, size_t jel) -> const helfem::Matrix & {
                  return rs_ktei[Nel*Nel*Lc + iel*Nel + jel];
                };
                K.slice(lout) -= helfem::to_arma(
                  atomic::basis::assemble_K_FE_one_multipole_cached_pairwise(
                    radial, kt, helfem::to_eigen(P_L)));
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
          // Phase 5.28: NAORadialBasis stores its C matrix as
          // helfem::Matrix; bridge from the caller's arma cube once.
          out.emplace_back(l, atomic::basis::NAORadialBasis::from_owned_radial(
              sad_basis.get_radial(), helfem::to_eigen(Ckeep)));
        }
        return out;
      }
    }
  }
}
