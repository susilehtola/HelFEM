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
#include <CoulombExchangeFE.h>
#include "../general/radial_block_helper.h"
#include "../general/gaunt.h"
#include "utils.h"
#include "../general/scf_helpers.h"
#include "../general/dftfuncs.h"
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

      TwoDBasis::TwoDBasis(int Z_, modelpotential::nuclear_model_t model_, double Rrms_, const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly, bool zeroder, int n_quad, const helfem::Vector & bval, int lmax) {
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
        lval=Eigen::VectorXi::LinSpaced(lmax+1,0,lmax);
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
          helfem::Matrix Vrad = model_potential(pot);
          delete pot;
          return Vrad;
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

      helfem::Matrix TwoDBasis::model_potential(const modelpotential::ModelPotential * pot) const {
        return helfem::assemble_radial_diagonal(radial,
            [&](size_t iel) { return radial.model_potential(pot, iel); });
      }

      void TwoDBasis::compute_tei() {
        // Delegate to the shared cache builders in CoulombExchangeFE.h
        // -- same path atomic::TwoDBasis uses. Bare Coulomb + in-element
        // In-element integrals kept as low-rank Cholesky factors (prim_chol);
        // K comes from RI, so no exchange-ordered tensor is built.
        const int N_L = 2 * lval.maxCoeff() + 1;
        atomic::basis::compute_disjoint_radial_integrals(
            radial, N_L, disjoint_L, disjoint_m1L);
        // In-element integrals in factorized form: T = L L', both J and K
        // assembled from the one J-ordered factor (K via RI). No
        // exchange-ordered tensor is built.
        const size_t Nel = radial.Nel();
        prim_chol.assign((size_t) N_L * Nel, helfem::Matrix());
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
        for (int L = 0; L < N_L; ++L)
          for (size_t iel = 0; iel < Nel; ++iel)
            prim_chol[(size_t) L * Nel + iel] =
                radial.twoe_integral_cholesky(L, iel, chol_tol);
      }

      void TwoDBasis::compute_yukawa(double lambda_) {
        lambda = lambda_;
        yukawa = true;
        const int N_L = 2 * lval.maxCoeff() + 1;
        // Yukawa disjoint factors (bessel i / bessel k) + in-element
        // Yukawa 2e -> exchange-permuted form needed by the cached K
        // assembly.
        atomic::basis::compute_disjoint_radial_integrals(
            radial, N_L, disjoint_iL, disjoint_kL, /*yukawa=*/true, lambda);
        // In-element Yukawa integrals, factorized as above.
        const size_t Nel = radial.Nel();
        rs_chol.assign((size_t) N_L * Nel, helfem::Matrix());
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
        for (int L = 0; L < N_L; ++L)
          for (size_t iel = 0; iel < Nel; ++iel)
            rs_chol[(size_t) L * Nel + iel] =
                radial.yukawa_integral_cholesky(L, lambda, iel, chol_tol);
      }

      void TwoDBasis::compute_erfc(double mu) {
        lambda = mu;
        yukawa = false;
        // Erfc kernel doesn't factorise -- no disjoint integrals; all
        // (iel, jel) pairs are stored explicitly in rs_ktei.
        disjoint_iL.clear();
        disjoint_kL.clear();
        const int N_L = 2 * lval.maxCoeff() + 1;
        atomic::basis::compute_erfc_ktei(radial, N_L, lambda, rs_ktei);
      }

      helfem::Matrix TwoDBasis::coulomb(const helfem::Matrix & P_in) const {
        if(!prim_chol.size())
          throw std::logic_error("Primitive teis have not been computed!\n");

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
          return prim_chol[(size_t) L * Nel + iel];
        };
        return Lfac * atomic::basis::assemble_J_FE_one_multipole_cached_chol(
            radial, rs, rb, tw, P_in);
      }

      helfem::Cube TwoDBasis::exchange(const helfem::Cube & P) const {
        if(!prim_chol.size())
          throw std::logic_error("Primitive teis have not been computed!\n");

        // Gaunt coefficient table
        int gmax(lval.maxCoeff());
        gaunt::Gaunt gaunt(gmax,2*gmax,gmax);

        if((int) P.size() != gmax+1)
          throw std::logic_error("Density matrix am does not match basis set!\n");

        // Number of radial elements
        size_t Nel(radial.Nel());
        // Number of radial basis functions
        size_t Nrad(radial.Nbf());
        if((size_t) P[0].rows() != Nrad || (size_t) P[0].cols() != Nrad)
          throw std::logic_error("Density matrix does not match basis set!\n");

        // Full exchange matrix
        helfem::Cube K(gmax+1, helfem::Matrix::Zero(Nrad,Nrad));

        // Per-element K via the RI contraction on the J-ordered Cholesky
        // factor -- assemble_K_FE_one_multipole_cached_chol. No
        // exchange-ordered tensor, no per-thread scratch.

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
            helfem::Cube Prad(2*gmax+1, helfem::Matrix::Zero(Nrad,Nrad));
            // Do we have a coupling
            std::vector<bool> coupling(2*gmax+1,false);

            // Do angular sums: loop over input angular momentum
            for(int lin=0;lin<=gmax;lin++) {
              // Skip if nothing to do
              if(P[lin].norm()==0.0)
                continue;

              // Possible couplings (lin,lout) => L
              int Lmin=std::abs(lin-lout);
              int Lmax=lin+lout;

              helfem::Vector totcoup = helfem::Vector::Zero(Lmax+1);
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
                Prad[L]+=(Lfac*totcoup(L))*P[lin];
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
                return prim_chol[(size_t) Lint * Nel + iel];
              };
              K[lout] -=
                  atomic::basis::assemble_K_FE_one_multipole_cached_chol(
                      radial, rs, rb, kt, Prad[L]);
            }
          }
        }

        return K;
      }

      helfem::Cube TwoDBasis::rs_exchange(const helfem::Cube & P) const {
        if(!rs_ktei.size() && !rs_chol.size())
          throw std::logic_error("Primitive teis have not been computed!\n");

        // Gaunt coefficient table
        int gmax(lval.maxCoeff());
        gaunt::Gaunt gaunt(gmax,2*gmax,gmax);

        if((int) P.size() != gmax+1)
          throw std::logic_error("Density matrix am does not match basis set!\n");

        // Number of radial elements
        size_t Nel(radial.Nel());
        // Number of radial basis functions
        size_t Nrad(radial.Nbf());
        if((size_t) P[0].rows() != Nrad || (size_t) P[0].cols() != Nrad)
          throw std::logic_error("Density matrix does not match basis set!\n");

        // Full exchange matrix
        helfem::Cube K(gmax+1, helfem::Matrix::Zero(Nrad,Nrad));

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
            helfem::Cube Prad(2*gmax+1, helfem::Matrix::Zero(Nrad,Nrad));
            // Do we have a coupling
            std::vector<bool> coupling(2*gmax+1,false);

            // Do angular sums: loop over input angular momentum
            for(int lin=0;lin<=gmax;lin++) {
              // Skip if nothing to do
              if(P[lin].norm()==0.0)
                continue;

              // Possible couplings (lin,lout) => L
              int Lmin=std::abs(lin-lout);
              int Lmax=lin+lout;

              helfem::Vector totcoup = helfem::Vector::Zero(Lmax+1);
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
                Prad[L]+=(Lfac*totcoup(L))*P[lin];
                coupling[L]=true;
              }
            }

            for(size_t L=0;L<coupling.size();L++) {
              if(!coupling[L])
                continue;
              const helfem::Matrix & P_L = Prad[L];
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
                  return rs_chol[(size_t) Lc * Nel + iel];
                };
                K[lout] -=
                  atomic::basis::assemble_K_FE_one_multipole_cached_chol(
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
                K[lout] -=
                  atomic::basis::assemble_K_FE_one_multipole_cached_pairwise(
                    radial, kt, P_L);
              }
            }
          }
        }

        return K;
      }


      helfem::Matrix TwoDBasis::eval_bf(size_t iel) const {
        return radial.get_bf(iel);
      }

      helfem::Matrix TwoDBasis::eval_df(size_t iel) const {
        return radial.get_df(iel);
      }

      helfem::Matrix TwoDBasis::eval_lf(size_t iel) const {
        return radial.get_lf(iel);
      }

      std::vector<Eigen::Index> TwoDBasis::bf_list(size_t iel) const {
        // Radial functions in element
        size_t ifirst, ilast;
        radial.get_idx(iel,ifirst,ilast);
        // Number of radial functions in element
        size_t Nr(ilast-ifirst+1);

        // List of functions in the element
        std::vector<Eigen::Index> idx(Nr);
        for(size_t j=0;j<Nr;j++)
          idx[j]=(Eigen::Index)(ifirst+j);

        return idx;
      }

      helfem::Vector TwoDBasis::eval_orbs(const helfem::Matrix & C, double r) const {
        return radial.eval_orbs(C, r);
      }

      size_t TwoDBasis::get_rad_Nel() const {
        return radial.Nel();
      }

      helfem::Vector TwoDBasis::get_wrad(size_t iel) const {
        return radial.get_wrad(iel);
      }

      helfem::Vector TwoDBasis::get_r(size_t iel) const {
        return radial.get_r(iel);
      }

      double TwoDBasis::nuclear_density(const helfem::Matrix & P) const {
        return radial.nuclear_density(P)/(4.0*M_PI);
      }
      double TwoDBasis::nuclear_density_gradient(const helfem::Matrix & P) const {
        return radial.nuclear_density_gradient(P)/(4.0*M_PI);
      }

      helfem::Vector TwoDBasis::quadrature_weights() const {
        std::vector<helfem::Vector> w(radial.Nel());
        size_t ntot=1;
        for(size_t iel=0;iel<radial.Nel();iel++) {
          w[iel]=radial.get_wrad(iel);
          ntot+=w[iel].size();
        }
        helfem::Vector wt = helfem::Vector::Zero(ntot);
        size_t Npts(w[0].size());
        for(size_t iel=0;iel<radial.Nel();iel++)
          wt.segment(1+iel*Npts,Npts)=w[iel];

        return wt;
      }

      helfem::Vector TwoDBasis::coulomb_screening(const helfem::Matrix & Prad) const {
        std::vector<helfem::Vector> r(radial.Nel());
        std::vector<helfem::Vector> V(radial.Nel());

        // Calculate potential due to charge outside the element
        helfem::Vector zero(radial.Nel());
        helfem::Vector minusone(radial.Nel());
        for(size_t iel=0;iel<radial.Nel();iel++) {
          // Radial functions in element
          size_t ifirst, ilast;
          radial.get_idx(iel,ifirst,ilast);
          // Density matrix
          helfem::Matrix Psub(Prad.block(ifirst,ifirst,ilast-ifirst+1,ilast-ifirst+1));
          helfem::Matrix zm(radial.radial_integral(0,iel));
          helfem::Matrix mo(radial.radial_integral(-1,iel));
          zero(iel)=(Psub*zm).trace();
          minusone(iel)=(Psub*mo).trace();
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
          V[iel]=helfem::Vector::Zero(r[iel].size());

          // Get the density in the element
          size_t ifirst, ilast;
          radial.get_idx(iel,ifirst,ilast);
          const helfem::Matrix Psub(Prad.block(ifirst,ifirst,ilast-ifirst+1,ilast-ifirst+1));
          const helfem::Vector Pv(Eigen::Map<const helfem::Vector>(Psub.data(), Psub.size()));

          // Calculate the in-element potential
          helfem::Matrix pot(radial.spherical_potential(iel));
          V[iel] += pot*Pv;

          // Add in the contributions from the other elements
          if(iel>0)
            for(Eigen::Index ip=0;ip<r[iel].size();ip++)
              V[iel](ip) += zero(iel-1)/r[iel](ip);
          if(iel != radial.Nel()-1)
            V[iel] += minusone(iel+1)*helfem::Vector::Ones(V[iel].size());

          // Multiply by r to convert this into an effective charge
          for(Eigen::Index ip=0;ip<r[iel].size();ip++)
            V[iel](ip)*=r[iel](ip);
        }

        // Assemble all of this into an array
        size_t Npts=r[0].size();
        helfem::Vector Veff = helfem::Vector::Zero(radial.Nel()*Npts+1);
        for(size_t iel=0;iel<radial.Nel();iel++) {
          Veff.segment(1+iel*Npts,Npts)=V[iel];
        }

        return Veff;
      }

      helfem::Vector TwoDBasis::radii() const {
        std::vector<helfem::Vector> r(radial.Nel());
        for(size_t iel=0;iel<radial.Nel();iel++) {
          r[iel]=radial.get_r(iel);
        }

        size_t Npts=r[0].size();
        helfem::Vector rad = helfem::Vector::Zero(radial.Nel()*Npts+1);
        for(size_t iel=0;iel<radial.Nel();iel++) {
          rad.segment(1+iel*Npts,Npts)=r[iel];
        }

        return rad;
      }

      double TwoDBasis::slater_F(int k, const helfem::Vector & c) const {
        // Bare radial Slater-Condon F^k integral for orbital `c` in the
        // u = r * R basis. Defined as
        //     F^k(c, c) = sum_{ab,cd} P_ab P_cd R^k_FE(ab, cd)
        // with P = c * c.t() and R^k_FE the bare per-multipole 2e integral
        // from CoulombExchangeFE.h (no 4*pi/(2k+1) factor).
        //
        // F^k = trace(P * J^k(P)) where J^k = assemble_J_FE_one_multipole.
        if ((size_t) c.size() != radial.Nbf()) {
          std::ostringstream oss;
          oss << "TwoDBasis::slater_F: orbital length " << c.size()
              << " != Nrad " << radial.Nbf() << ".\n";
          throw std::logic_error(oss.str());
        }
        const helfem::Matrix P = c * c.transpose();
        const helfem::Matrix Jk =
            atomic::basis::assemble_J_FE_one_multipole(radial, k, P);
        return (P * Jk).trace();
      }

      helfem::Matrix TwoDBasis::orbitals(const helfem::Matrix & C) const {
        std::vector<helfem::Matrix> c(radial.Nel());
        for(size_t iel=0;iel<radial.Nel();iel++) {
          // Radial functions in element
          size_t ifirst, ilast;
          radial.get_idx(iel,ifirst,ilast);
          // Density matrix
          helfem::Matrix Csub(C.middleRows(ifirst,ilast-ifirst+1));
          helfem::Matrix bf(radial.get_bf(iel));

          c[iel]=bf*Csub;
        }
        size_t Npts=c[0].rows();

        helfem::Matrix Cv = helfem::Matrix::Zero(radial.Nel()*Npts+1,C.cols());
        // Values at the nucleus
        {
          {
            const Eigen::RowVectorXd nrow = radial.nuclear_orbital(C);
            for (Eigen::Index j = 0; j < nrow.size(); ++j)
              Cv(0, j) = nrow(j);
          }
        }
        // Other points
        for(size_t iel=0;iel<radial.Nel();iel++) {
          Cv.middleRows(1+iel*Npts,Npts)=c[iel];
        }

        return Cv;
      }

      helfem::Matrix TwoDBasis::orbitals_derivative(const helfem::Matrix & C) const {
        std::vector<helfem::Matrix> c(radial.Nel());
        for(size_t iel=0;iel<radial.Nel();iel++) {
          // Radial functions in element
          size_t ifirst, ilast;
          radial.get_idx(iel,ifirst,ilast);
          // Density matrix
          helfem::Matrix Csub(C.middleRows(ifirst,ilast-ifirst+1));
          helfem::Matrix bf(radial.get_df(iel));

          c[iel]=bf*Csub;
        }
        size_t Npts=c[0].rows();

        helfem::Matrix Cv = helfem::Matrix::Zero(radial.Nel()*Npts+1,C.cols());
        // Values at the nucleus
        {
	  // tbd
        }
        // Other points
        for(size_t iel=0;iel<radial.Nel();iel++) {
          Cv.middleRows(1+iel*Npts,Npts)=c[iel];
        }

        return Cv;
      }

      helfem::Matrix TwoDBasis::orbitals_second_derivative(const helfem::Matrix & C) const {
        std::vector<helfem::Matrix> c(radial.Nel());
        for(size_t iel=0;iel<radial.Nel();iel++) {
          // Radial functions in element
          size_t ifirst, ilast;
          radial.get_idx(iel,ifirst,ilast);
          // Density matrix
          helfem::Matrix Csub(C.middleRows(ifirst,ilast-ifirst+1));
          helfem::Matrix bf(radial.get_lf(iel));

          c[iel]=bf*Csub;
        }
        size_t Npts=c[0].rows();

        helfem::Matrix Cv = helfem::Matrix::Zero(radial.Nel()*Npts+1,C.cols());
        // Values at the nucleus
        {
	  // tbd
        }
        // Other points
        for(size_t iel=0;iel<radial.Nel();iel++) {
          Cv.middleRows(1+iel*Npts,Npts)=c[iel];
        }

        return Cv;
      }

      helfem::Vector TwoDBasis::electron_density(const helfem::Vector & x, size_t iel, const helfem::Matrix & Prad, bool rsqweight) const {
        // Radial functions in element
        size_t ifirst, ilast;
        radial.get_idx(iel,ifirst,ilast);
        // Density matrix
        helfem::Matrix Psub(Prad.block(ifirst,ifirst,ilast-ifirst+1,ilast-ifirst+1));
        helfem::Matrix bf(radial.get_bf(x, iel));

        helfem::Vector density = (bf*Psub*bf.transpose()).diagonal();
        if(rsqweight) {
          const helfem::Vector rr = radial.get_r(x, iel);
          density.array() *= rr.array().square();
        }
        return density;
      }

      helfem::Vector TwoDBasis::electron_density(size_t iel, const helfem::Matrix & Prad, bool rsqweight) const {
        return electron_density(radial.get_xq(), iel, Prad, rsqweight);
      }

      helfem::Vector TwoDBasis::electron_density(const helfem::Matrix & Prad, bool rsqweight) const {
        std::vector<helfem::Vector> d(radial.Nel());
        for(size_t iel=0;iel<radial.Nel();iel++) {
          d[iel]=electron_density(iel, Prad, rsqweight);
        }
        size_t Npts=d[0].size();

        helfem::Vector n = helfem::Vector::Zero(radial.Nel()*Npts+1);
        n(0)=4.0*M_PI*nuclear_density(Prad);
        for(size_t iel=0;iel<radial.Nel();iel++) {
          n.segment(1+iel*Npts,Npts)=d[iel];
        }

        return n;
      }

      double TwoDBasis::electron_density_maximum_radius(const helfem::Matrix & Prad, bool rsqweight, double eps) const {
        // Evaluate the density in each quadrature point and take
        // their maximum
        helfem::Vector den(radial.Nel());

        for(size_t iel=0;iel<radial.Nel();iel++) {
          den(iel)=electron_density(iel, Prad, rsqweight).maxCoeff();
        }

        // Quadrature points
        helfem::Vector xq = radial.get_xq();

        // Find the element with the maximum density
        Eigen::Index iel;
        den.maxCoeff(&iel);

        // Evaluate the density in that element
        helfem::Vector del = electron_density(xq, iel, Prad, rsqweight);

        // Find the maximum value
        Eigen::Index imax;
        del.maxCoeff(&imax);

        // Refine the location of the maximum in the element
        double rmax=0.0;
        {
          // Primitive coordinates
          helfem::Vector a(1), b(1);

          if(imax == 0) {
            a(0) = -1.0;
            b(0) = xq(imax+1);
          } else if(imax == xq.size()-1) {
            a(0) = xq(imax-1);
            b(0) = 1.0;
          } else {
            a(0) = xq(imax-1);
            b(0) = xq(imax+1);
          }

          // Golden ratio search
          double golden_ratio = 0.5*(sqrt(5.0)+1.0);
          while((a-b).cwiseAbs().maxCoeff()>=eps) {
            helfem::Vector c = b - (b-a)/golden_ratio;
            helfem::Vector d = a + (b-a)/golden_ratio;
            double density_c = electron_density(c, iel, Prad, rsqweight)(0);
            double density_d = electron_density(d, iel, Prad, rsqweight)(0);
            if(density_c > density_d) {
              b = d;
            } else {
              a = c;
            }
          }
          helfem::Vector cen=((a+b)/2);
          double dmax = electron_density(cen, iel, Prad, rsqweight)(0);
          if(dmax < del(imax)) {
            std::ostringstream oss;
            oss << "Density maximization failed! Quadrature max " << del(imax) << " optimized max " << dmax << " difference " << dmax-del(imax) << "!\n";
            throw std::logic_error(oss.str());
          }
          // Position of maximum is
          rmax = radial.get_r(cen, iel)(0);
        }

        return rmax;
      }

      double TwoDBasis::vdw_radius(const helfem::Matrix & Prad, double vdw_threshold, double eps) const {
        // Need to multiply output of electron_density by this factor to get the point-wise density
        double angfac=1.0/(4.0*M_PI);
	bool rsqweight=false;

        // Evaluate the density in each quadrature point and take
        // their maximum
        size_t iel;
        for(iel=radial.Nel()-1;iel<radial.Nel();iel--) {
          helfem::Vector den(angfac*electron_density(iel, Prad, rsqweight));
          if(den.maxCoeff()>vdw_threshold) {
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
        helfem::Vector xq = radial.get_xq();
        helfem::Vector diff = angfac*electron_density(xq, iel, Prad, rsqweight);
        diff-=vdw_threshold*helfem::Vector::Ones(diff.size());
        diff=diff.cwiseAbs();

        // Find the smallest value
        Eigen::Index imax;
        diff.minCoeff(&imax);

        // Refine the position.

        double rvdw=0.0;
        {
          // Primitive coordinates
          helfem::Vector a(1), b(1);

          if(imax == 0) {
            a(0) = -1.0;
            b(0) = xq(imax+1);
          } else if(imax == xq.size()-1) {
            a(0) = xq(imax-1);
            b(0) = 1.0;
          } else {
            a(0) = xq(imax-1);
            b(0) = xq(imax+1);
          }

          // Bisection
          size_t ibisect=0;
          while((a-b).cwiseAbs().maxCoeff()>=eps) {
            helfem::Vector m = a + (b-a)/2.0;
            double density_m = angfac*electron_density(m, iel, Prad, rsqweight)(0);

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
          helfem::Vector cen=((a+b)/2);
          // Position of maximum is
          rvdw = radial.get_r(cen, iel)(0);
        }

        return rvdw;
      }

      double TwoDBasis::electron_count_radius(const helfem::Matrix & Prad, const double eps, const double conv_thr) const {
	// Vector with electron density contributions from each element
	std::vector<double> densities(radial.Nel());
	for(size_t iel=0;iel<radial.Nel();iel++) {
	  // Radial functions in element
	  size_t ifirst, ilast;
	  radial.get_idx(iel,ifirst,ilast);
	  // Density matrix
	  helfem::Matrix Psub(Prad.block(ifirst,ifirst,ilast-ifirst+1,ilast-ifirst+1));
	  // Overlap matrix (Phase 2a: overlap(iel) returns helfem::Matrix).
	  helfem::Matrix S(radial.overlap(iel));
	  densities[iel]=(Psub*S).trace();
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
	helfem::Matrix Psub(Prad.block(ifirst,ifirst,ilast-ifirst+1,ilast-ifirst+1));

	// Search for the radius within element
	double result;
	// Element limits
	double a=-1.0, b=1.0;
	double m=a+(b-a)/2.0;
	while(b-a>=conv_thr) {
	  m=a+(b-a)/2.0;
	  result=(Psub*radial.radial_integral(0,ielement,m,1.0)).trace();
	  if(result<s_left)
	    b=m;
	  else if(result>s_left)
	    a=m;
	  else
	    break;
	}

	return radial.get_r(m, ielement);
      }

      helfem::Vector TwoDBasis::electron_density_gradient(const helfem::Matrix & Prad) const {
        std::vector<helfem::Vector> d(radial.Nel());
        for(size_t iel=0;iel<radial.Nel();iel++) {
          // Radial functions in element
          size_t ifirst, ilast;
          radial.get_idx(iel,ifirst,ilast);
          // Density matrix
          helfem::Matrix Psub(Prad.block(ifirst,ifirst,ilast-ifirst+1,ilast-ifirst+1));
          helfem::Matrix bf(radial.get_bf(iel));
          helfem::Matrix df(radial.get_df(iel));

          d[iel]=2.0*(bf*Psub*df.transpose()).diagonal();
        }

        size_t Npts=d[0].size();
        helfem::Vector n = helfem::Vector::Zero(radial.Nel()*Npts+1);

        // TODO: implement gradient at the nucleus. This requires
        // third derivatives of the basis functions...

        // The other points
        for(size_t iel=0;iel<radial.Nel();iel++) {
          n.segment(1+iel*Npts,Npts)=d[iel];
        }

        return n;
      }

      helfem::Vector TwoDBasis::electron_density_laplacian(const helfem::Matrix & Prad) const {
        std::vector<helfem::Vector> l(radial.Nel());
        for(size_t iel=0;iel<radial.Nel();iel++) {
          // Radial functions in element
          size_t ifirst, ilast;
          radial.get_idx(iel,ifirst,ilast);
          // Density matrix
          helfem::Matrix Psub(Prad.block(ifirst,ifirst,ilast-ifirst+1,ilast-ifirst+1));
          helfem::Matrix bf(radial.get_bf(iel));
          helfem::Matrix df(radial.get_df(iel));
          helfem::Matrix lf(radial.get_lf(iel));
          helfem::Vector r(radial.get_r(iel));

          // Laplacian is df^2/dr^2 + 2/r df/dr
          helfem::Vector lap = 2.0*((df*Psub*df.transpose()).diagonal() + (bf*Psub*lf.transpose()).diagonal());
          lap.array() += 4.0*(bf*Psub*df.transpose()).diagonal().array()/r.array();
          l[iel]=lap;
        }

        size_t Npts=l[0].size();
        helfem::Vector n = helfem::Vector::Zero(radial.Nel()*Npts+1);

        // Skip the laplacian at the nucleus at least for now...
        for(size_t iel=0;iel<radial.Nel();iel++) {
          n.segment(1+iel*Npts,Npts)=l[iel];
        }

        return n;
      }

      helfem::Vector TwoDBasis::kinetic_energy_density(const helfem::Cube & Pl0) const {
        // Radial density matrices
        helfem::Matrix P = helfem::Matrix::Zero(Pl0[0].rows(), Pl0[0].cols());
        helfem::Matrix Pl = helfem::Matrix::Zero(Pl0[0].rows(), Pl0[0].cols());
        for(size_t l=0; l<Pl0.size(); l++) {
          P += Pl0[l];
          Pl += l*(l+1)*Pl0[l];
        }

        std::vector<helfem::Vector> t(radial.Nel());
        for(size_t iel=0;iel<radial.Nel();iel++) {
          // Radial functions in element
          size_t ifirst, ilast;
          radial.get_idx(iel,ifirst,ilast);

          // Radii
          helfem::Vector r(radial.get_r(iel));

          // Density matrix
          helfem::Matrix Psub(P.block(ifirst,ifirst,ilast-ifirst+1,ilast-ifirst+1));
          helfem::Matrix Psubl(Pl.block(ifirst,ifirst,ilast-ifirst+1,ilast-ifirst+1));

          // Basis function
          helfem::Matrix bf(radial.get_bf(iel));
          // Basis function derivative
          helfem::Matrix bf_rho(radial.get_df(iel));

          helfem::Vector term1((bf_rho * Psub * bf_rho.transpose()).diagonal());
          helfem::Vector term2((bf * Psubl * bf.transpose()).diagonal());
          term2.array() /= r.array().square();
          // The second term is tricky near the nucleus since only s
          // orbitals can contribute to the electron density but their
          // contribution is killed off by the l(l+1) factor
          term2 = term2.cwiseMax(0.0);
          t[iel] = 0.5*(term1+term2);
        }

        size_t Npts=t[0].size();
        helfem::Vector tn = helfem::Vector::Zero(radial.Nel()*Npts+1);

        // Skip the value at the nucleus at least for now...
        for(size_t iel=0;iel<radial.Nel();iel++) {
          tn.segment(1+iel*Npts,Npts)=t[iel];
        }

        return tn;
      }


      helfem::Vector TwoDBasis::xc_screening(const helfem::Matrix & Prad, int x_func, int c_func) const {
        helfem::Matrix v(xc_screening(Prad/2,Prad/2,x_func,c_func));
        return 0.5*(v.col(0)+v.col(1));
      }

      helfem::Matrix TwoDBasis::xc_screening(const helfem::Matrix & Parad, const helfem::Matrix & Pbrad, int x_func, int c_func) const {
        const double angfac=4.0*M_PI;
        // Get the electron density
        helfem::Vector rhoa(electron_density(Parad)/angfac);
        helfem::Vector rhob(electron_density(Pbrad)/angfac);
        // and the density gradient
        helfem::Vector grada(electron_density_gradient(Parad)/angfac);
        helfem::Vector gradb(electron_density_gradient(Pbrad)/angfac);
        // and the density Laplacian
        helfem::Vector lapla(electron_density_laplacian(Parad)/angfac);
        helfem::Vector laplb(electron_density_laplacian(Pbrad)/angfac);
        // Radial coordinates
        helfem::Vector r(radii());
        size_t Npoints(r.size());

        // and pack it for libxc. Store directly in transposed layout so
        // the column-major order is (na0, nb0, na1, nb1, ...).
        helfem::Matrix rho_libxc(2,Npoints);
        rho_libxc.row(0)=rhoa.transpose();
        rho_libxc.row(1)=rhob.transpose();

        // Reduced gradient, transposed layout as above.
        helfem::Matrix sigma_libxc(3,Npoints);
        sigma_libxc.row(0)=(grada.array()*grada.array()).matrix().transpose();
        sigma_libxc.row(1)=(grada.array()*gradb.array()).matrix().transpose();
        sigma_libxc.row(2)=(gradb.array()*gradb.array()).matrix().transpose();

        // Potential
        helfem::Matrix vxc = helfem::Matrix::Zero(2,Npoints);
        helfem::Matrix vsigma = helfem::Matrix::Zero(3,Npoints);

        // For GGA we also need the second derivative to calculate the
        // correction to the potential
        helfem::Matrix v2rhosigma = helfem::Matrix::Zero(6,Npoints);
        helfem::Matrix v2sigma2 = helfem::Matrix::Zero(6,Npoints);

        // Helper arrays
        helfem::Matrix vxc_wrk(2,Npoints);
        helfem::Matrix vsigma_wrk(3,Npoints);
        helfem::Matrix v2rho2_wrk(3,Npoints);
        helfem::Matrix v2rhosigma_wrk(6,Npoints);
        helfem::Matrix v2sigma2_wrk(6,Npoints);

        bool do_gga=false;

        if(x_func>0) {
          vxc_wrk.setZero();
          vsigma_wrk.setZero();
          v2rhosigma_wrk.setZero();
          v2sigma2_wrk.setZero();

          bool gga, mggat, mggal;
          ::is_gga_mgga(x_func, gga, mggat, mggal);
          if(mggat || mggal)
            throw std::logic_error("Mggas not supported!\n");

          xc_func_type func;
          if(xc_func_init(&func, x_func, XC_POLARIZED) != 0) {
            throw std::logic_error("Error initializing exchange functional!\n");
          }
          if(gga) {
            xc_gga_vxc(&func, rhoa.size(), rho_libxc.data(), sigma_libxc.data(), vxc_wrk.data(), vsigma_wrk.data());
            xc_gga_fxc(&func, rhoa.size(), rho_libxc.data(), sigma_libxc.data(), v2rho2_wrk.data(), v2rhosigma_wrk.data(), v2sigma2_wrk.data());
            do_gga=true;
            vsigma+=vsigma_wrk;
            v2rhosigma+=v2rhosigma_wrk;
            v2sigma2+=v2sigma2_wrk;
          } else {
            xc_lda_vxc(&func, rhoa.size(), rho_libxc.data(), vxc_wrk.data());
          }
          xc_func_end(&func);

          vxc+=vxc_wrk;
        }
        if(c_func>0) {
          vxc_wrk.setZero();
          vsigma_wrk.setZero();
          v2rhosigma_wrk.setZero();
          v2sigma2_wrk.setZero();

          bool gga, mggat, mggal;
          ::is_gga_mgga(c_func, gga, mggat, mggal);
          if(mggat || mggal)
            throw std::logic_error("Mggas not supported!\n");

          xc_func_type func;
          if(xc_func_init(&func, c_func, XC_POLARIZED) != 0) {
            throw std::logic_error("Error initializing correlation functional!\n");
          }
          if(gga) {
            xc_gga_vxc(&func, rhoa.size(), rho_libxc.data(), sigma_libxc.data(), vxc_wrk.data(), vsigma_wrk.data());
            xc_gga_fxc(&func, rhoa.size(), rho_libxc.data(), sigma_libxc.data(), v2rho2_wrk.data(), v2rhosigma_wrk.data(), v2sigma2_wrk.data());
            do_gga=true;
            vsigma+=vsigma_wrk;
            v2rhosigma+=v2rhosigma_wrk;
            v2sigma2+=v2sigma2_wrk;
          } else {
            xc_lda_vxc(&func, rhoa.size(), rho_libxc.data(), vxc_wrk.data());
          }

          vxc+=vxc_wrk;
        }

        // Add GGA correction to xc potential.
        if(do_gga) {
          helfem::Matrix corr = helfem::Matrix::Zero(2,vxc.cols());

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

        // Transpose: (2, Npoints) -> (Npoints, 2)
        helfem::Matrix vxcT = vxc.transpose();
        // Convert to radial potential (this is how it matches with GPAW)
        for(Eigen::Index ic=0;ic<vxcT.cols();ic++)
          vxcT.col(ic).array()*=r.array();

        return vxcT;
      }



      std::vector<std::pair<int, atomic::basis::NAORadialBasis>>
      extract_naos_per_l(const TwoDBasis & sad_basis,
                         const helfem::Cube & Ccube,
                         const std::vector<int> & keep_per_l) {
        const int lmax = static_cast<int>(Ccube.size()) - 1;
        if (static_cast<int>(keep_per_l.size()) != lmax + 1) {
          std::ostringstream oss;
          oss << "extract_naos_per_l: keep_per_l has size " << keep_per_l.size()
              << " but Ccube has " << Ccube.size()
              << " slices (expected " << lmax + 1 << ").\n";
          throw std::logic_error(oss.str());
        }
        std::vector<std::pair<int, atomic::basis::NAORadialBasis>> out;
        out.reserve(lmax + 1);
        for (int l = 0; l <= lmax; ++l) {
          const int keep = keep_per_l[l];
          if (keep == 0) continue;
          const helfem::Matrix & Cl = Ccube[l];
          if (keep > static_cast<int>(Cl.cols())) {
            std::ostringstream oss;
            oss << "extract_naos_per_l: requested " << keep
                << " NAOs for l=" << l << " but only "
                << Cl.cols() << " orbitals available.\n";
            throw std::logic_error(oss.str());
          }
          const helfem::Matrix Ckeep = (keep < 0) ? Cl
                                             : helfem::Matrix(Cl.leftCols(keep));
          out.emplace_back(l, atomic::basis::NAORadialBasis::from_owned_radial(
              sad_basis.get_radial(), Ckeep));
        }
        return out;
      }
    }
  }
}
