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
#include "TwoDBasis.h"
#include <ArmaEigen.h>
#include <CoulombExchangeFE.h>
#include "../general/radial_block_helper.h"
#include "../general/angular_index_helpers.h"
#include "basis.h"
#include "quadrature.h"
#include "chebyshev.h"
#include "../general/spherical_harmonics.h"
#include "../general/gaunt.h"
#include "utils.h"
#include "../general/scf_helpers.h"
#include <cassert>
#include <cfloat>
#include <helfem.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace helfem {
  namespace atomic {
    namespace basis {
      TwoDBasis::TwoDBasis() {
      }

      TwoDBasis::TwoDBasis(int Z_, modelpotential::nuclear_model_t model_,
                             double Rrms_,
                             const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly,
                             bool zeroder_, int n_quad,
                             const helfem::Vector & bval_e,
                             const Eigen::VectorXi & lval_e,
                             const Eigen::VectorXi & mval_e,
                             int Zl_, int Zr_, double Rhalf_) {
        // Nuclear charge
        Z=Z_;
        Zl=Zl_;
        Zr=Zr_;
        Rhalf=Rhalf_;
        model=model_;
        Rrms=Rrms_;

        // Bridge Eigen inputs to internal arma storage once.
        arma::vec bval(bval_e.size());
        std::memcpy(bval.memptr(), bval_e.data(), sizeof(double) * (size_t) bval_e.size());
        arma::ivec lval_arma(lval_e.size());
        for (Eigen::Index i = 0; i < lval_e.size(); ++i)
          lval_arma(i) = static_cast<arma::sword>(lval_e(i));
        arma::ivec mval_arma(mval_e.size());
        for (Eigen::Index i = 0; i < mval_e.size(); ++i)
          mval_arma(i) = static_cast<arma::sword>(mval_e(i));

        // Construct radial basis
        bool zero_func_left=true;
        bool zero_deriv_left=false;
        bool zero_func_right=true;
        zeroder=zeroder_;
        polynomial_basis::FiniteElementBasis fem(poly, helfem::to_eigen(bval), zero_func_left, zero_deriv_left, zero_func_right, zeroder);
        radial=FEMRadialBasis(fem, n_quad);

        // Construct angular basis
        lval=lval_arma;
        mval=mval_arma;
      }

      TwoDBasis::~TwoDBasis() {
      }

      int TwoDBasis::get_nuclear_model() const {
        return model;
      }

      double TwoDBasis::get_nuclear_size() const {
        return Rrms;
      }

      int TwoDBasis::get_Z() const {
        return Z;
      }

      int TwoDBasis::get_Zl() const {
        return Zl;
      }

      int TwoDBasis::get_Zr() const {
        return Zr;
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

      int TwoDBasis::get_poly_id() const {
        return radial.get_poly_id();
      }

      int TwoDBasis::get_poly_nnodes() const {
        return radial.get_poly_nnodes();
      }

      int TwoDBasis::get_zeroder() const {
        return zeroder;
      }

      size_t TwoDBasis::Ndummy() const {
        return lval.n_elem*radial.Nbf();
      }

      size_t TwoDBasis::Nbf() const {
        return Ndummy();
      }

      size_t TwoDBasis::Nrad() const {
        return radial.Nbf();
      }

      size_t TwoDBasis::Nang() const {
        return lval.n_elem;
      }

      arma::uvec TwoDBasis::pure_indices() const {
        return arma::linspace<arma::uvec>(0,Nbf()-1,Nbf());
      }

      arma::uvec TwoDBasis::m_indices(int m) const {
        return helfem::collect_shell_indices(mval.n_elem,
            [&](size_t)   { return radial.Nbf(); },
            [&](size_t i) { return mval(i) == m; });
      }

      arma::uvec TwoDBasis::lm_indices(int l, int m) const {
        return helfem::collect_shell_indices(mval.n_elem,
            [&](size_t)   { return radial.Nbf(); },
            [&](size_t i) { return mval(i) == m && lval(i) == l; });
      }

      std::vector<arma::uvec> TwoDBasis::get_sym_idx(int symm) const {
        std::vector<arma::uvec> idx;
        if(symm==0) {
          idx.resize(1);
          idx[0]=arma::linspace<arma::uvec>(0,Nbf()-1,Nbf());
        } else if(symm==1) {
          // Find unique m values
          arma::uvec muni(arma::find_unique(mval));
          arma::ivec mv(mval(muni));

          idx.resize(mv.n_elem);
          for(size_t i=0;i<mv.n_elem;i++)
            idx[i]=m_indices(mv(i));
        } else if(symm==2) {
          idx.resize(mval.n_elem);
          for(size_t i=0;i<mval.n_elem;i++) {
            idx[i]=lm_indices(lval(i),mval(i));
          }
        } else
          throw std::logic_error("Unknown symmetry\n");

        return idx;
      }

      helfem::Matrix TwoDBasis::Sinvh(bool chol, int sym) const {
        arma::mat S(helfem::to_arma(overlap()));

        // Half-inverse is
        if(sym==0) {
          return scf::form_Sinvh(helfem::to_eigen(S), chol);
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
            Sinvh(midx[i],cidx)=helfem::to_arma(scf::form_Sinvh(helfem::to_eigen(arma::mat(S(midx[i],midx[i]))),chol));
            // Increment offset
            ioff += midx[i].n_elem;
          }
          return helfem::to_eigen(Sinvh);
        }
      }

      void TwoDBasis::set_sub(arma::mat & M, size_t iang, size_t jang, const arma::mat & Mrad) const {
        M.submat(iang*radial.Nbf(),jang*radial.Nbf(),(iang+1)*radial.Nbf()-1,(jang+1)*radial.Nbf()-1)=Mrad;
      }

      void TwoDBasis::add_sub(arma::mat & M, size_t iang, size_t jang, const arma::mat & Mrad) const {
        M.submat(iang*radial.Nbf(),jang*radial.Nbf(),(iang+1)*radial.Nbf()-1,(jang+1)*radial.Nbf()-1)+=Mrad;
      }

      helfem::Matrix TwoDBasis::overlap() const {
        // Full overlap matrix built by scattering the radial R=0 integrals
        // along the angular diagonal.
        arma::mat Orad = helfem::to_arma(helfem::assemble_radial_diagonal(radial,
            [&](size_t iel) { return radial.radial_integral(0, iel); }));

        arma::mat O(Ndummy(),Ndummy());
        O.zeros();
        for(size_t iang=0;iang<lval.n_elem;iang++)
          set_sub(O,iang,iang,Orad);

        return helfem::to_eigen(remove_boundaries(O));
      }

      helfem::Matrix TwoDBasis::overlap(const TwoDBasis & rh) const {
        // Cross-basis overlap: scatter radial.overlap(rh.radial) along the
        // (l, m)-matched blocks of the two angular indexings.
        arma::mat S(Ndummy(), rh.Ndummy());
        S.zeros();
        arma::mat Srad(helfem::to_arma(radial.overlap(rh.radial)));

        for(size_t iang=0;iang<lval.n_elem;iang++)
          for(size_t jang=0;jang<rh.lval.n_elem;jang++)
            if(lval(iang) == rh.lval(jang) && mval(iang) == rh.mval(jang))
              S.submat(iang*radial.Nbf(), jang*rh.radial.Nbf(),
                       (iang+1)*radial.Nbf()-1, (jang+1)*rh.radial.Nbf()-1) = Srad;

        // Same boundary-condition handling as overlap(): the returned
        // matrix is indexed by pure_indices() on both sides so it can
        // be composed with Sinvh matrices without touching boundary rows.
        arma::uvec my_pure  = pure_indices();
        arma::uvec rh_pure  = rh.pure_indices();
        return helfem::to_eigen(arma::mat(S(my_pure, rh_pure)));
      }

      helfem::Matrix TwoDBasis::kinetic() const {
        // Build radial kinetic energy matrices
        arma::mat Trad = helfem::to_arma(helfem::assemble_radial_diagonal(radial,
            [&](size_t iel) { return radial.kinetic(iel); }));
        arma::mat Trad_l = helfem::to_arma(helfem::assemble_radial_diagonal(radial,
            [&](size_t iel) { return radial.kinetic_l(iel); }));

        // Full kinetic energy matrix
        arma::mat T(Ndummy(),Ndummy());
        T.zeros();
        // Fill elements
        for(size_t iang=0;iang<lval.n_elem;iang++) {
          set_sub(T,iang,iang,Trad);
          if(lval(iang)>0) {
            // We also get the l(l+1) term
            add_sub(T,iang,iang,lval(iang)*(lval(iang)+1)*Trad_l);
          }
        }

        return helfem::to_eigen(remove_boundaries(T));
      }

      helfem::Matrix TwoDBasis::nuclear() const {
        if(model != modelpotential::POINT_NUCLEUS) {
          modelpotential::ModelPotential *pot=modelpotential::get_nuclear_model(model,Z,Rrms);
          arma::mat Vnuc(model_potential(pot));
          delete pot;
          return helfem::to_eigen(Vnuc);
        } else {
          // Full nuclear attraction matrix
          arma::mat V(Ndummy(),Ndummy());
          V.zeros();

          if (Z != 0.0) {
            arma::mat Vrad = helfem::to_arma(helfem::assemble_radial_diagonal(radial,
                [&](size_t iel) { return radial.radial_integral(-1, iel); }));
            for (size_t iang = 0; iang < lval.n_elem; iang++)
              set_sub(V, iang, iang, -Z * Vrad);
          }

          if(Zl != 0.0 || Zr != 0.0) {
            // Auxiliary matrices
            size_t Nrad(radial.Nbf());
            int Lmax(2*arma::max(lval));
            std::vector<arma::mat> Vaux(Lmax+1);
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (int L = 0; L <= Lmax; L++) {
              Vaux[L] = helfem::to_arma(helfem::assemble_radial_diagonal(radial,
                  [&](size_t iel) { return radial.nuclear_offcenter(iel, Rhalf, L); }));
            }

            int gmax(std::max(arma::max(lval),arma::max(mval)));
            gaunt::Gaunt gaunt(gmax,2*gmax,gmax);

            /// Loop over basis set
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
            for(size_t iang=0;iang<lval.n_elem;iang++) {
              for(size_t jang=0;jang<lval.n_elem;jang++) {
                int li(lval(iang));
                int mi(mval(iang));

                int lj(lval(jang));
                int mj(mval(jang));

                // Zero contribution
                if(mi!=mj)
                  continue;

                // Loop over L
                for(int L=std::abs(li-lj);L<=li+lj;L++) {
                  double cpl(gaunt.coeff(li,mi,L,0,lj));
                  if(cpl==0.0)
                    continue;

                  const double signL = (L & 1) ? -1.0 : 1.0;
                  add_sub(V,iang,jang,cpl*(signL*Zl + Zr)*Vaux[L]);
                }
              }
            }
          }

          return helfem::to_eigen(remove_boundaries(V));
        }
      }

      arma::mat TwoDBasis::model_potential(const modelpotential::ModelPotential * pot) const {
        // Full nuclear attraction matrix
        arma::mat V(Ndummy(),Ndummy());
        V.zeros();

        arma::mat Vrad = helfem::to_arma(helfem::assemble_radial_diagonal(radial,
            [&](size_t iel) { return radial.model_potential(pot, iel); }));
	// Fill elements
	for(size_t iang=0;iang<lval.n_elem;iang++)
	  set_sub(V,iang,iang,Vrad);

        return remove_boundaries(V);
      }

      helfem::Matrix TwoDBasis::confinement(const int N, double r_0, const int iconf, const double V, const double shift_pot) const {
        // Full matrix
        arma::mat O(Ndummy(),Ndummy());
        O.zeros();

	if(iconf==0)
	  return helfem::to_eigen(remove_boundaries(O));

        arma::mat Orad = helfem::to_arma(helfem::assemble_radial_diagonal(radial,
            [&](size_t iel) {
              return radial.confinement_potential(iel, N, r_0, iconf, V, shift_pot);
            }));

        // Fill elements
        for(size_t iang=0;iang<lval.n_elem;iang++)
          set_sub(O,iang,iang,Orad);

        return helfem::to_eigen(remove_boundaries(O));
      }

      helfem::Matrix TwoDBasis::dipole_z() const {
        // Build radial elements
        arma::mat Orad = helfem::to_arma(helfem::assemble_radial_diagonal(radial,
            [&](size_t iel) { return radial.radial_integral(1, iel); }));

        // Full electric couplings
        arma::mat V(Ndummy(),Ndummy());
        V.zeros();

        int gmax(std::max(arma::max(lval),arma::max(mval)));
        gaunt::Gaunt gaunt(gmax,1,gmax);

        // Fill elements
        for(size_t iang=0;iang<lval.n_elem;iang++) {
          int li(lval(iang));
          int mi(mval(iang));

          for(size_t jang=0;jang<lval.n_elem;jang++) {
            int lj(lval(jang));
            int mj(mval(jang));

            // Calculate coupling
            double cpl(gaunt.cosine_coupling(lj,mj,li,mi));
            if(cpl!=0.0)
              set_sub(V,iang,jang,Orad*cpl);
          }
        }

        return helfem::to_eigen(remove_boundaries(V));
      }

      helfem::Matrix TwoDBasis::quadrupole_zz() const {
        // Build radial elements
        arma::mat Orad = helfem::to_arma(helfem::assemble_radial_diagonal(radial,
            [&](size_t iel) { return radial.radial_integral(2, iel); }));

        // Full electric couplings
        arma::mat V(Ndummy(),Ndummy());
        V.zeros();

        int gmax(std::max(arma::max(lval),arma::max(mval)));
        gaunt::Gaunt gaunt(gmax,2,gmax);

        // Fill elements
        for(size_t iang=0;iang<lval.n_elem;iang++) {
          int li(lval(iang));
          int mi(mval(iang));

          for(size_t jang=0;jang<lval.n_elem;jang++) {
            int lj(lval(jang));
            int mj(mval(jang));

            // Y_2^0 has m=0, so the m-sum forces mj == mi.
            if(mj != mi) continue;

            // Calculate coupling
            double cpl(gaunt.coeff(lj,mj,2,0,li));
            if(cpl!=0.0) {
              const double c0(2.0/5.0*sqrt(5.0*M_PI));
              cpl*=c0;
              set_sub(V,iang,jang,Orad*cpl);
            }
          }
        }

        return helfem::to_eigen(remove_boundaries(V));
      }

      helfem::Matrix TwoDBasis::Bz_field(double B) const {
        // Build radial elements
        arma::mat O0rad = helfem::to_arma(helfem::assemble_radial_diagonal(radial,
            [&](size_t iel) { return radial.radial_integral(0, iel); }));
        arma::mat O2rad = helfem::to_arma(helfem::assemble_radial_diagonal(radial,
            [&](size_t iel) { return radial.radial_integral(2, iel); }));

        // Full coupling
        arma::mat V(Ndummy(),Ndummy());
        V.zeros();

        int gmax(std::max(arma::max(lval),arma::max(mval)));
        gaunt::Gaunt gaunt(gmax,4,gmax);

        // Fill elements
        for(size_t iang=0;iang<lval.n_elem;iang++) {
          int li(lval(iang));
          int mi(mval(iang));

          for(size_t jang=0;jang<lval.n_elem;jang++) {
            int lj(lval(jang));
            int mj(mval(jang));

            // Calculate coupling
            double cpl(gaunt.sine2_coupling(lj,mj,li,mi));
            if(cpl!=0.0) {
              set_sub(V,iang,jang,B*B/8.0*O2rad*cpl);
            }

            if(li==lj && mi==mj) {
              add_sub(V,iang,jang,-0.5*B*mj*O0rad);
            }
          }
        }

        return helfem::to_eigen(remove_boundaries(V));
      }

      std::vector<std::vector<helfem::Matrix>>
      TwoDBasis::radial_df_factors(double tol) const {
        if (!prim_tei.size())
          throw std::logic_error(
              "Primitive teis have not been computed -- call "
              "compute_tei() before radial_df_factors().");

        const int N_L = 2 * arma::max(lval) + 1;
        const size_t Nrad = radial.Nbf();
        const size_t Nel  = radial.Nel();

        // Outer index = multipole L, inner index = Cholesky vector Q,
        // each an Nrad x Nrad helfem::Matrix (arma-free at the public
        // boundary; internal computation is arma-native for now).
        std::vector<std::vector<helfem::Matrix>> B(N_L);

        for (int L = 0; L < N_L; ++L) {
          // -- 1. Cached accessor lambdas for the per-L J helper.
          // Phase 2c: caches are now std::vector<helfem::Matrix>.
          auto rs = [&, L](size_t iel) -> const helfem::Matrix & {
            return disjoint_L[(size_t) L * Nel + iel];
          };
          auto rb = [&, L](size_t iel) -> const helfem::Matrix & {
            return disjoint_m1L[(size_t) L * Nel + iel];
          };
          auto tw = [&, L](size_t iel) -> const helfem::Matrix & {
            return prim_tei[Nel * Nel * (size_t) L + iel * Nel + iel];
          };

          // -- 2. Initial diagonal D(i, j) = R^L(i, j, i, j) over the
          //    redundant pair index. Compute via the J helper so that
          //    BOUNDARY-shared basis functions get all element
          //    contributions summed correctly. For an interior pair
          //    (i, j) both in element iel, J(P_ij)[i, j] is exactly the
          //    cached prim_tei diagonal; for a pair involving the
          //    shared boundary index, J(P_ij)[i, j] sums the within-
          //    elem_left, within-elem_right, and cross-element pieces
          //    that arise because the boundary function has support in
          //    two adjacent elements.
          //
          //    A naive prim_tei-only init misses the off-element
          //    pieces for boundary indices and corrupts the pivoted
          //    Cholesky -- D_pivot ends up too small, v_new is
          //    overscaled, and the reconstructed R^L diverges by ~1%
          //    on boundary-involved quadruples (verified bug, fixed
          //    here).
          //
          //    Iterate pairs per-element so cross-element (i, j) pairs
          //    (zero pair density -> D = 0) are not visited. Each
          //    pair is visited once if it lives in one element, twice
          //    (with identical J output) if both indices are the same
          //    boundary -- harmless small redundancy.
          arma::mat D(Nrad, Nrad, arma::fill::zeros);
          for (size_t iel = 0; iel < Nel; ++iel) {
            size_t ifirst, ilast;
            radial.get_idx(iel, ifirst, ilast);
            const size_t Ni = ilast - ifirst + 1;
            for (size_t j_loc = 0; j_loc < Ni; ++j_loc) {
              for (size_t i_loc = 0; i_loc <= j_loc; ++i_loc) {
                const size_t i = ifirst + i_loc;
                const size_t j = ifirst + j_loc;
                arma::mat P(Nrad, Nrad, arma::fill::zeros);
                if (i == j) {
                  P(i, i) = 1.0;
                } else {
                  P(i, j) = 0.5;
                  P(j, i) = 0.5;
                }
                arma::mat J = ::helfem::to_arma(
                    helfem::atomic::basis::
                        assemble_J_FE_one_multipole_cached(
                            radial, rs, rb, tw, ::helfem::to_eigen(P)));
                D(i, j) = J(i, j);
                D(j, i) = D(i, j);
              }
            }
          }

          // -- 3. Pivoted Cholesky on R^L. Each iteration picks the
          //    redundant-pair index (i*, j*) with maximum residual
          //    diagonal, computes the corresponding column of R^L via
          //    the J helper, subtracts projections onto previously found
          //    Cholesky vectors, normalises, appends, and updates the
          //    diagonal.
          std::vector<arma::mat> B_L_vecs;
          while (true) {
            const arma::uword pivot_flat = D.index_max();
            const double D_pivot = D(pivot_flat);
            if (D_pivot < tol) break;

            // Armadillo column-major flatten: i = flat % Nrad,
            // j = flat / Nrad.
            const arma::uword i_star = pivot_flat % Nrad;
            const arma::uword j_star = pivot_flat / Nrad;

            // Build the symmetric pair-density indicator P. For i == j:
            //   P = e_i e_i^T (single 1 on the diagonal).
            // For i != j:
            //   P = (e_i e_j^T + e_j e_i^T) / 2.
            // The J helper expects a symmetric P (it integrates against
            // the symmetric pair density), and J(P)[a, b] then equals
            // exactly R^L(a, b, i*, j*).
            arma::mat P(Nrad, Nrad, arma::fill::zeros);
            if (i_star == j_star) {
              P(i_star, i_star) = 1.0;
            } else {
              P(i_star, j_star) = 0.5;
              P(j_star, i_star) = 0.5;
            }
            arma::mat col = ::helfem::to_arma(
                helfem::atomic::basis::assemble_J_FE_one_multipole_cached(
                    radial, rs, rb, tw, ::helfem::to_eigen(P)));

            // Subtract projections onto already-found Cholesky vectors.
            for (const auto & V : B_L_vecs) {
              col -= V(i_star, j_star) * V;
            }

            // Normalise. By construction col(i*, j*) == D_pivot, so
            // v_new(i*, j*) == sqrt(D_pivot) -- pivot diagonal zeros
            // out exactly on the diagonal update below.
            arma::mat v_new = col / std::sqrt(D_pivot);

            // Update diagonal: D <- D - v_new % v_new (elementwise).
            D -= v_new % v_new;
            // Numerical: clamp at zero to avoid negative drift.
            D.transform([](double x) { return x < 0.0 ? 0.0 : x; });

            B_L_vecs.push_back(std::move(v_new));
          }

          // -- 4. Bridge to helfem::Matrix at the public boundary.
          B[L].reserve(B_L_vecs.size());
          for (auto & M : B_L_vecs)
            B[L].push_back(helfem::to_eigen(M));
        }

        return B;
      }

      void TwoDBasis::compute_tei(bool exchange) {
        // Delegate to the shared cache builders in CoulombExchangeFE.h.
        // The bare-Coulomb defaults (yukawa=false, lambda=0) populate
        // disjoint_L / disjoint_m1L from radial.radial_integral and
        // prim_tei (in-element only; cross-element is assembled on the
        // fly from the disjoint factors inside the cached J/K helpers).
        const int N_L = 2 * arma::max(lval) + 1;
        atomic::basis::compute_disjoint_radial_integrals(
            radial, N_L, disjoint_L, disjoint_m1L);
        atomic::basis::compute_in_element_tei(radial, N_L, prim_tei);
        // The exchange matrix is K(jk) = (ij|kl) P(il); we precompute the
        // (jk;il)-permuted form here so the cached K assembly path can
        // consume the cache without re-permuting.
        if (exchange) {
          atomic::basis::compute_in_element_ktei_from_tei(
              radial, N_L, prim_tei, prim_ktei);
        }
      }

      void TwoDBasis::compute_yukawa(double lambda_) {
        lambda = lambda_;
        yukawa = true;
        const int N_L = 2 * arma::max(lval) + 1;
        // Yukawa-mode disjoint factors (bessel i / bessel k) + in-element
        // Yukawa 2e + the exchange-permuted form needed by the cached K
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

      helfem::Matrix TwoDBasis::coulomb(const helfem::Matrix & P0_in) const {
        if(!prim_tei.size())
          throw std::logic_error("Primitive teis have not been computed!\n");

        // Phase 3: SCF surface takes Eigen, internals stay arma -- one
        // conversion at entry, one at exit.
        const arma::mat P0 = helfem::to_arma(P0_in);
        arma::mat P(expand_boundaries(P0));

        // Number of radial elements
        size_t Nel(radial.Nel());
        // Number of radial functions
        size_t Nrad(radial.Nbf());
        // Gaunt coefficient table
        int gmax(std::max(arma::max(lval),arma::max(mval)));
        gaunt::Gaunt gaunt(gmax,2*gmax,gmax);

        // maximal M value
        int Mmax=arma::max(mval)-arma::min(mval);

        // Radial helper matrices
        std::vector< std::vector<arma::mat> > Paux(2*arma::max(lval)+1);
        for(int L=0;L<(int) Paux.size();L++) {
          Paux[L].resize(2*Mmax+1);
          for(int M=-std::min(L,Mmax);M<=std::min(L,Mmax);M++) {
            Paux[L][M+Mmax].zeros(Nrad,Nrad);
          }
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
            int Lmin=std::max(std::abs(lk-ll),abs(M));
            int Lmax=lk+ll;
            for(int L=Lmin;L<=Lmax;L++) {
              // Calculate coupling coefficient
              double cpl(gaunt.coeff(lk,mk,L,M,ll));
              // Increment
              Paux[L][M+Mmax]+=cpl*P.submat(kang*Nrad,lang*Nrad,(kang+1)*Nrad-1,(lang+1)*Nrad-1);
            }
          }
        }

        // Helper matrices
        std::vector< std::vector<arma::mat> > Jaux(2*arma::max(lval)+1);
        for(int L=0;L<(int) Jaux.size();L++) {
          Jaux[L].resize(2*Mmax+1);
          for(int M=-std::min(L,Mmax);M<=std::min(L,Mmax);M++) {
            Jaux[L][M+Mmax].zeros(Nrad,Nrad);
          }
        }
        // Contract integrals. Per (L, M), delegate the FE assembly to
        // the shared helper with our SCF-cached per-(L, iel) integrals.
        for(int L=0;L<(int) Paux.size();L++) {
          const double Lfac=4.0*M_PI/(2*L+1);
          auto rs = [&,L](size_t iel) -> const helfem::Matrix & {
            return disjoint_L[L*Nel+iel];
          };
          auto rb = [&,L](size_t iel) -> const helfem::Matrix & {
            return disjoint_m1L[L*Nel+iel];
          };
          auto tw = [&,L](size_t iel) -> const helfem::Matrix & {
            return prim_tei[Nel*Nel*L + iel*Nel + iel];
          };
          for(int M=-std::min(L,Mmax);M<=std::min(L,Mmax);M++) {
            Jaux[L][M+Mmax] += Lfac * ::helfem::to_arma(
              helfem::atomic::basis::assemble_J_FE_one_multipole_cached(
                radial, rs, rb, tw, ::helfem::to_eigen(Paux[L][M+Mmax])));
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

            int Lmin=std::max(std::abs(lj-li),abs(M));
            int Lmax=lj+li;
            for(int L=Lmin;L<=Lmax;L++) {
              // Coupling
              double cpl(gaunt.coeff(lj,mj,L,M,li));
              if(cpl!=0.0) {
                J.submat(iang*Nrad,jang*Nrad,(iang+1)*Nrad-1,(jang+1)*Nrad-1)+=cpl*Jaux[L][M+Mmax];
              }
            }
          }
        }

        return helfem::to_eigen(remove_boundaries(J));
      }

      helfem::Matrix TwoDBasis::exchange(const helfem::Matrix & P0_in) const {
        if(!prim_ktei.size())
          throw std::logic_error("Primitive teis have not been computed!\n");

        const arma::mat P0 = helfem::to_arma(P0_in);
        arma::mat P(expand_boundaries(P0));

        // Gaunt coefficient table
        int gmax(std::max(arma::max(lval),arma::max(mval)));
        gaunt::Gaunt gaunt(gmax,2*gmax,gmax);

        // Number of radial elements
        size_t Nel(radial.Nel());
        // Number of radial basis functions
        size_t Nrad(radial.Nbf());

        // Full exchange matrix
        arma::mat K(Ndummy(),Ndummy());
        K.zeros();

        // Per-element K assembly is delegated to
        // assemble_K_FE_one_multipole_cached -- no per-thread scratch
        // needed here, the helper allocates its own.
#ifdef _OPENMP
#pragma omp parallel
#endif
        {
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
              size_t N_L(2*arma::max(lval)+1);
              std::vector<arma::mat> Rmat(N_L);
              for(size_t i=0;i<N_L;i++) {
                Rmat[i].zeros(Nrad,Nrad);
              }
              // Is there a coupling to the channel?
              std::vector<bool> couple(N_L,false);

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
                  int Lmin=std::max(std::max(std::abs(li-lj),std::abs(lk-ll)),abs(M));
                  int Lmax=std::min(li+lj,lk+ll);

                  for(int L=Lmin;L<=Lmax;L++) {
                    // Calculate total coupling coefficient
                    double cpl(gaunt.coeff(lj,mj,L,M,li)*gaunt.coeff(lk,mk,L,M,ll));
                    if(cpl==0.0)
                      continue;

                    // L factor
                    double Lfac=4.0*M_PI/(2*L+1);
                    Rmat[L]+=(Lfac*cpl)*P.submat(iang*Nrad,lang*Nrad,(iang+1)*Nrad-1,(lang+1)*Nrad-1);
                    couple[L]=true;
                  }
                }
              }

              // Per-L K assembly via the shared FE helper, accumulated
              // into the (jang, kang) angular block of K.
              arma::mat K_block(Nrad, Nrad, arma::fill::zeros);
              for(size_t L=0; L<N_L; ++L) {
                if(!couple[L]) continue;
                const size_t Lc = L;
                auto rs = [&,Lc](size_t iel) -> const helfem::Matrix & {
                  return disjoint_L[Lc*Nel+iel];
                };
                auto rb = [&,Lc](size_t iel) -> const helfem::Matrix & {
                  return disjoint_m1L[Lc*Nel+iel];
                };
                auto kt = [&,Lc](size_t iel) -> const helfem::Matrix & {
                  return prim_ktei[Nel*Nel*Lc + iel*Nel + iel];
                };
                K_block += ::helfem::to_arma(
                  helfem::atomic::basis::assemble_K_FE_one_multipole_cached(
                    radial, rs, rb, kt, ::helfem::to_eigen(Rmat[Lc])));
              }
              K.submat(jang*Nrad, kang*Nrad,
                       (jang+1)*Nrad-1, (kang+1)*Nrad-1) -= K_block;
            }
          }
        }

        return helfem::to_eigen(remove_boundaries(K));
      }

      helfem::Matrix TwoDBasis::rs_exchange(const helfem::Matrix & P0_in) const {
        if(!rs_ktei.size())
          throw std::logic_error("Primitive teis have not been computed!\n");

        const arma::mat P0 = helfem::to_arma(P0_in);
        arma::mat P(expand_boundaries(P0));

        // Gaunt coefficient table
        int gmax(std::max(arma::max(lval),arma::max(mval)));
        gaunt::Gaunt gaunt(gmax,2*gmax,gmax);

        // Number of radial elements
        size_t Nel(radial.Nel());
        // Number of radial basis functions
        size_t Nrad(radial.Nbf());

        // Full exchange matrix
        arma::mat K(Ndummy(),Ndummy());
        K.zeros();

        // Per-element K assembly is delegated to the cached helpers
        // in CoulombExchangeFE.h -- no per-thread scratch needed.
#ifdef _OPENMP
#pragma omp parallel
#endif
        {
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
              size_t N_L(2*arma::max(lval)+1);
              std::vector<arma::mat> Rmat(N_L);
              for(size_t i=0;i<N_L;i++) {
                Rmat[i].zeros(Nrad,Nrad);
              }
              // Is there a coupling to the channel?
              std::vector<bool> couple(N_L,false);

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
                  int Lmin=std::max(std::max(std::abs(li-lj),std::abs(lk-ll)),abs(M));
                  int Lmax=std::min(li+lj,lk+ll);

                  for(int L=Lmin;L<=Lmax;L++) {
                    // Calculate total coupling coefficient
                    double cpl(gaunt.coeff(lj,mj,L,M,li)*gaunt.coeff(lk,mk,L,M,ll));
                    if(cpl==0.0)
                      continue;

                    // L factor
                    double Lfac = yukawa ? 4.0*M_PI*lambda :  4.0*M_PI*lambda/(2*L+1);
                    Rmat[L]+=(Lfac*cpl)*P.submat(iang*Nrad,lang*Nrad,(iang+1)*Nrad-1,(lang+1)*Nrad-1);
                    couple[L]=true;
                  }
                }
              }

              if (yukawa) {
                // Same FE structure as bare exchange: per-L cached
                // helper, summed into the (jang, kang) angular block.
                arma::mat K_block(Nrad, Nrad, arma::fill::zeros);
                for(size_t L=0; L<N_L; ++L) {
                  if(!couple[L]) continue;
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
                  K_block += ::helfem::to_arma(
                    helfem::atomic::basis::assemble_K_FE_one_multipole_cached(
                      radial, rs, rb, kt, ::helfem::to_eigen(Rmat[Lc])));
                }
                K.submat(jang*Nrad, kang*Nrad,
                         (jang+1)*Nrad-1, (kang+1)*Nrad-1) -= K_block;
              } else {
                // Erfc: rs_ktei has cross-element entries for every
                // (iel, jel) pair -- delegate to the pairwise cached
                // helper, summed over L into the (jang, kang) block.
                arma::mat K_block(Nrad, Nrad, arma::fill::zeros);
                for(size_t L=0; L<N_L; ++L) {
                  if(!couple[L]) continue;
                  const size_t Lc = L;
                  auto kt = [&,Lc](size_t iel, size_t jel) -> const helfem::Matrix & {
                    return rs_ktei[Nel*Nel*Lc + iel*Nel + jel];
                  };
                  K_block += ::helfem::to_arma(
                    helfem::atomic::basis::assemble_K_FE_one_multipole_cached_pairwise(
                      radial, kt, ::helfem::to_eigen(Rmat[Lc])));
                }
                K.submat(jang*Nrad, kang*Nrad,
                         (jang+1)*Nrad-1, (kang+1)*Nrad-1) -= K_block;
              }
            }
          }
        }

        return helfem::to_eigen(remove_boundaries(K));
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

      std::vector<arma::mat> TwoDBasis::get_prim_tei() const {
        // Phase 2c: prim_tei is std::vector<helfem::Matrix> internally;
        // bridge to arma for the public accessor (low-frequency test/diag
        // path; consumers can be migrated later).
        std::vector<arma::mat> out;
        out.reserve(prim_tei.size());
        for (const auto & m : prim_tei) out.push_back(helfem::to_arma(m));
        return out;
      }

      arma::cx_mat TwoDBasis::eval_bf(size_t iel, double cth, double phi) const {
        // Evaluate spherical harmonics
        arma::cx_vec sph(lval.n_elem);
        for(size_t i=0;i<lval.n_elem;i++)
          sph(i)=::spherical_harmonics(lval(i),mval(i),cth,phi);

        // Evaluate radial functions
        arma::mat rad(helfem::to_arma(radial.get_bf(iel)));

        // Form supermatrix
        arma::cx_mat bf(rad.n_rows,lval.n_elem*rad.n_cols);
        for(size_t i=0;i<lval.n_elem;i++)
          bf.cols(i*rad.n_cols,(i+1)*rad.n_cols-1)=sph(i)*rad;

        return bf;
      }

      void TwoDBasis::eval_df(size_t iel, double cth, double phi, arma::cx_mat & dr, arma::cx_mat & dth, arma::cx_mat & dphi) const {
        // Evaluate spherical harmonics
        arma::cx_vec sph(lval.n_elem);
        for(size_t i=0;i<lval.n_elem;i++)
          sph(i)=::spherical_harmonics(lval(i),mval(i),cth,phi);

        // Evaluate radial functions
        arma::mat frad(helfem::to_arma(radial.get_bf(iel)));
        arma::mat drad(helfem::to_arma(radial.get_df(iel)));

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
        // sin^2(theta) = (1 - cth)(1 + cth) is algebraically identical to
        // 1 - cth*cth but avoids the catastrophic cancellation when cth is
        // close to +/- 1. The std::max(..., 0.0) is a paranoia floor for
        // the case where round-off pushes the product slightly negative.
        const double sinth = std::sqrt(std::max((1.0-cth)*(1.0+cth), 0.0));
        // cot(theta) is singular at the poles; for cth = +/- 1 the m*cot*Y
        // term is the m=0 case (contribution 0) or |m|>=1 with sph(i)=0 in
        // the limit, so a defensive 0 is the right value here.
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

      arma::cx_mat TwoDBasis::eval_lf(size_t iel, double cth, double phi) const {
        // Evaluate spherical harmonics
        arma::cx_vec sph(lval.n_elem);
        for(size_t i=0;i<lval.n_elem;i++)
          sph(i)=::spherical_harmonics(lval(i),mval(i),cth,phi);

        // Evaluate radial functions
        arma::vec r(helfem::to_arma(radial.get_r(iel)));
        arma::mat frad(helfem::to_arma(radial.get_bf(iel)));
        arma::mat drad(helfem::to_arma(radial.get_df(iel)));
        arma::mat lrad(helfem::to_arma(radial.get_lf(iel)));

        // Form supermatrix
        arma::cx_mat lf(frad.n_rows,lval.n_elem*frad.n_cols,arma::fill::zeros);
        // Loop over basis function indices
        for(size_t iang=0;iang<lval.n_elem;iang++)
          for(size_t irad=0;irad<frad.n_cols;irad++)
            // Loop over grid-point indices
            for(size_t igrid=0;igrid<frad.n_rows;igrid++)
              lf(igrid,iang*frad.n_cols+irad) = (lrad(igrid,irad) + 2*drad(igrid,irad)/r(igrid) - lval(iang)*(lval(iang)+1)*frad(igrid,irad)/(r(igrid)*r(igrid)))*sph(iang);

        //for(size_t i=0;i<lval.n_elem;i++)
        //lf.cols(i*frad.n_cols,(i+1)*frad.n_cols-1)=(arma::square(r)%lrad + 2*r%drad - lval(i)*(lval(i)+1)*frad) / arma::square(r) * sph(i);

        return lf;
      }

      arma::uvec TwoDBasis::bf_list(size_t iel) const {
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

      size_t TwoDBasis::get_rad_Nel() const {
        return radial.Nel();
      }

      std::pair<size_t, size_t>
      TwoDBasis::radial_element_range(size_t iel) const {
        size_t ifirst, ilast;
        radial.get_idx(iel, ifirst, ilast);
        return {ifirst, ilast};
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
