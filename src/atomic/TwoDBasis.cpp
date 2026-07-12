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
#include <algorithm>
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

        // Bridge the Eigen boundary input to internal arma storage once
        // (bval feeds the arma-based radial grid). lval/mval are stored
        // as Eigen::VectorXi directly.
        arma::vec bval(bval_e.size());
        std::memcpy(bval.memptr(), bval_e.data(), sizeof(double) * (size_t) bval_e.size());

        // Construct radial basis
        bool zero_func_left=true;
        bool zero_deriv_left=false;
        bool zero_func_right=true;
        zeroder=zeroder_;
        polynomial_basis::FiniteElementBasis fem(poly, helfem::to_eigen(bval), zero_func_left, zero_deriv_left, zero_func_right, zeroder);
        radial=FEMRadialBasis(fem, n_quad);

        // Construct angular basis
        lval=lval_e;
        mval=mval_e;
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
        // Bridge Eigen->arma at the public boundary; consumers
        // (checkpoint I/O, drivers) still take arma::ivec.
        arma::ivec out(lval.size());
        for (Eigen::Index i = 0; i < lval.size(); ++i)
          out(i) = static_cast<arma::sword>(lval(i));
        return out;
      }

      arma::ivec TwoDBasis::get_mval() const {
        arma::ivec out(mval.size());
        for (Eigen::Index i = 0; i < mval.size(); ++i)
          out(i) = static_cast<arma::sword>(mval(i));
        return out;
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
        return lval.size()*radial.Nbf();
      }

      size_t TwoDBasis::Nbf() const {
        return Ndummy();
      }

      size_t TwoDBasis::Nrad() const {
        return radial.Nbf();
      }

      size_t TwoDBasis::Nang() const {
        return lval.size();
      }

      arma::uvec TwoDBasis::m_indices(int m) const {
        return helfem::collect_shell_indices(mval.size(),
            [&](size_t)   { return radial.Nbf(); },
            [&](size_t i) { return mval(i) == m; });
      }

      arma::uvec TwoDBasis::lm_indices(int l, int m) const {
        return helfem::collect_shell_indices(mval.size(),
            [&](size_t)   { return radial.Nbf(); },
            [&](size_t i) { return mval(i) == m && lval(i) == l; });
      }

      std::vector<arma::uvec> TwoDBasis::get_sym_idx(int symm) const {
        std::vector<arma::uvec> idx;
        if(symm==0) {
          idx.resize(1);
          idx[0]=arma::linspace<arma::uvec>(0,Nbf()-1,Nbf());
        } else if(symm==1) {
          // Unique m values in ascending order (matches arma::find_unique).
          std::vector<int> mv;
          for (Eigen::Index i = 0; i < mval.size(); ++i)
            if (std::find(mv.begin(), mv.end(), mval(i)) == mv.end())
              mv.push_back(mval(i));
          std::sort(mv.begin(), mv.end());

          idx.resize(mv.size());
          for(size_t i=0;i<mv.size();i++)
            idx[i]=m_indices(mv[i]);
        } else if(symm==2) {
          idx.resize(mval.size());
          for(size_t i=0;i<mval.size();i++) {
            idx[i]=lm_indices(lval(i),mval(i));
          }
        } else
          throw std::logic_error("Unknown symmetry\n");

        return idx;
      }

      helfem::Matrix TwoDBasis::Sinvh(bool chol, int sym) const {
        const helfem::Matrix S = overlap();

        // Half-inverse is
        if(sym==0) {
          return scf::form_Sinvh(S, chol);
        } else {
          // Per-symmetry-block orthonormalization. get_sym_idx returns the
          // (scattered) AO index list for each block; the orthonormal
          // columns are packed contiguously, so Sinvh maps orthonormal ->
          // AO with block-diagonal structure (scattered rows, contiguous
          // columns per block).
          std::vector<arma::uvec> midx(get_sym_idx(sym));
          const Eigen::Index N = static_cast<Eigen::Index>(Nbf());
          helfem::Matrix Sinvh = helfem::Matrix::Zero(N, N);
          Eigen::Index ioff = 0;
          for(size_t i=0;i<midx.size();i++) {
            const Eigen::Index n = static_cast<Eigen::Index>(midx[i].n_elem);
            if(!n)
              continue;

            // AO indices of this block.
            std::vector<Eigen::Index> rows(n);
            for(Eigen::Index k=0;k<n;k++)
              rows[k] = static_cast<Eigen::Index>(midx[i](k));

            // Gather the block overlap, orthonormalize.
            helfem::Matrix Ssub(n, n);
            for(Eigen::Index a=0;a<n;a++)
              for(Eigen::Index b=0;b<n;b++)
                Ssub(a,b) = S(rows[a], rows[b]);
            const helfem::Matrix sub = scf::form_Sinvh(Ssub, chol);

            // Scatter: rows scattered by `rows`, columns contiguous
            // [ioff, ioff+n).
            for(Eigen::Index a=0;a<n;a++)
              for(Eigen::Index b=0;b<n;b++)
                Sinvh(rows[a], ioff+b) = sub(a,b);
            ioff += n;
          }
          return Sinvh;
        }
      }

      void TwoDBasis::set_sub(helfem::Matrix & M, size_t iang, size_t jang, const helfem::Matrix & Mrad) const {
        const Eigen::Index n = static_cast<Eigen::Index>(radial.Nbf());
        M.block(static_cast<Eigen::Index>(iang)*n, static_cast<Eigen::Index>(jang)*n, n, n) = Mrad;
      }

      void TwoDBasis::add_sub(helfem::Matrix & M, size_t iang, size_t jang, const helfem::Matrix & Mrad) const {
        const Eigen::Index n = static_cast<Eigen::Index>(radial.Nbf());
        M.block(static_cast<Eigen::Index>(iang)*n, static_cast<Eigen::Index>(jang)*n, n, n) += Mrad;
      }

      helfem::Matrix TwoDBasis::overlap() const {
        // Full overlap matrix built by scattering the radial R=0 integrals
        // along the angular diagonal. The atomic basis has no
        // boundary-condition reduction (pure_indices() is the identity),
        // so the assembled matrix is returned directly.
        const helfem::Matrix Orad = helfem::assemble_radial_diagonal(radial,
            [&](size_t iel) { return radial.radial_integral(0, iel); });

        helfem::Matrix O = helfem::Matrix::Zero(Ndummy(), Ndummy());
        for(size_t iang=0;iang<lval.size();iang++)
          set_sub(O,iang,iang,Orad);

        return O;
      }

      helfem::Matrix TwoDBasis::overlap(const TwoDBasis & rh) const {
        // Cross-basis overlap: scatter radial.overlap(rh.radial) along the
        // (l, m)-matched blocks of the two angular indexings. Each side's
        // pure_indices() is the identity, so no boundary slicing is needed.
        helfem::Matrix S = helfem::Matrix::Zero(Ndummy(), rh.Ndummy());
        const helfem::Matrix Srad = radial.overlap(rh.radial);
        const Eigen::Index n   = static_cast<Eigen::Index>(radial.Nbf());
        const Eigen::Index rhn = static_cast<Eigen::Index>(rh.radial.Nbf());

        for(size_t iang=0;iang<lval.size();iang++)
          for(size_t jang=0;jang<rh.lval.size();jang++)
            if(lval(iang) == rh.lval(jang) && mval(iang) == rh.mval(jang))
              S.block(static_cast<Eigen::Index>(iang)*n,
                      static_cast<Eigen::Index>(jang)*rhn, n, rhn) = Srad;

        return S;
      }

      helfem::Matrix TwoDBasis::kinetic() const {
        // Build radial kinetic energy matrices
        const helfem::Matrix Trad = helfem::assemble_radial_diagonal(radial,
            [&](size_t iel) { return radial.kinetic(iel); });
        const helfem::Matrix Trad_l = helfem::assemble_radial_diagonal(radial,
            [&](size_t iel) { return radial.kinetic_l(iel); });

        // Full kinetic energy matrix
        helfem::Matrix T = helfem::Matrix::Zero(Ndummy(), Ndummy());
        // Fill elements
        for(size_t iang=0;iang<lval.size();iang++) {
          set_sub(T,iang,iang,Trad);
          if(lval(iang)>0) {
            // We also get the l(l+1) term
            add_sub(T,iang,iang,static_cast<double>(lval(iang)*(lval(iang)+1))*Trad_l);
          }
        }

        return T;
      }

      helfem::Matrix TwoDBasis::nuclear() const {
        if(model != modelpotential::POINT_NUCLEUS) {
          modelpotential::ModelPotential *pot=modelpotential::get_nuclear_model(model,Z,Rrms);
          helfem::Matrix Vnuc = model_potential(pot);
          delete pot;
          return Vnuc;
        } else {
          // Full nuclear attraction matrix
          helfem::Matrix V = helfem::Matrix::Zero(Ndummy(), Ndummy());

          if (Z != 0.0) {
            const helfem::Matrix Vrad = helfem::assemble_radial_diagonal(radial,
                [&](size_t iel) { return radial.radial_integral(-1, iel); });
            for (size_t iang = 0; iang < lval.size(); iang++)
              set_sub(V, iang, iang, static_cast<double>(-Z) * Vrad);
          }

          if(Zl != 0.0 || Zr != 0.0) {
            // Auxiliary matrices
            int Lmax(2*lval.maxCoeff());
            std::vector<helfem::Matrix> Vaux(Lmax+1);
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (int L = 0; L <= Lmax; L++) {
              Vaux[L] = helfem::assemble_radial_diagonal(radial,
                  [&](size_t iel) { return radial.nuclear_offcenter(iel, Rhalf, L); });
            }

            int gmax(std::max(lval.maxCoeff(),mval.maxCoeff()));
            gaunt::Gaunt gaunt(gmax,2*gmax,gmax);

            /// Loop over basis set
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
            for(size_t iang=0;iang<lval.size();iang++) {
              for(size_t jang=0;jang<lval.size();jang++) {
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
                  add_sub(V,iang,jang,(cpl*(signL*Zl + Zr))*Vaux[L]);
                }
              }
            }
          }

          return V;
        }
      }

      helfem::Matrix TwoDBasis::model_potential(const modelpotential::ModelPotential * pot) const {
        // Full nuclear attraction matrix
        helfem::Matrix V = helfem::Matrix::Zero(Ndummy(), Ndummy());

        const helfem::Matrix Vrad = helfem::assemble_radial_diagonal(radial,
            [&](size_t iel) { return radial.model_potential(pot, iel); });
	// Fill elements
	for(size_t iang=0;iang<lval.size();iang++)
	  set_sub(V,iang,iang,Vrad);

        return V;
      }

      helfem::Matrix TwoDBasis::confinement(const int N, double r_0, const int iconf, const double V, const double shift_pot) const {
        // Full matrix
        helfem::Matrix O = helfem::Matrix::Zero(Ndummy(), Ndummy());

	if(iconf==0)
	  return O;

        const helfem::Matrix Orad = helfem::assemble_radial_diagonal(radial,
            [&](size_t iel) {
              return radial.confinement_potential(iel, N, r_0, iconf, V, shift_pot);
            });

        // Fill elements
        for(size_t iang=0;iang<lval.size();iang++)
          set_sub(O,iang,iang,Orad);

        return O;
      }

      helfem::Matrix TwoDBasis::dipole_z() const {
        // Build radial elements
        const helfem::Matrix Orad = helfem::assemble_radial_diagonal(radial,
            [&](size_t iel) { return radial.radial_integral(1, iel); });

        // Full electric couplings
        helfem::Matrix V = helfem::Matrix::Zero(Ndummy(), Ndummy());

        int gmax(std::max(lval.maxCoeff(),mval.maxCoeff()));
        gaunt::Gaunt gaunt(gmax,1,gmax);

        // Fill elements
        for(size_t iang=0;iang<lval.size();iang++) {
          int li(lval(iang));
          int mi(mval(iang));

          for(size_t jang=0;jang<lval.size();jang++) {
            int lj(lval(jang));
            int mj(mval(jang));

            // Calculate coupling
            double cpl(gaunt.cosine_coupling(lj,mj,li,mi));
            if(cpl!=0.0)
              set_sub(V,iang,jang,cpl*Orad);
          }
        }

        return V;
      }

      helfem::Matrix TwoDBasis::quadrupole_zz() const {
        // Build radial elements
        const helfem::Matrix Orad = helfem::assemble_radial_diagonal(radial,
            [&](size_t iel) { return radial.radial_integral(2, iel); });

        // Full electric couplings
        helfem::Matrix V = helfem::Matrix::Zero(Ndummy(), Ndummy());

        int gmax(std::max(lval.maxCoeff(),mval.maxCoeff()));
        gaunt::Gaunt gaunt(gmax,2,gmax);

        // Fill elements
        for(size_t iang=0;iang<lval.size();iang++) {
          int li(lval(iang));
          int mi(mval(iang));

          for(size_t jang=0;jang<lval.size();jang++) {
            int lj(lval(jang));
            int mj(mval(jang));

            // Y_2^0 has m=0, so the m-sum forces mj == mi.
            if(mj != mi) continue;

            // Calculate coupling
            double cpl(gaunt.coeff(lj,mj,2,0,li));
            if(cpl!=0.0) {
              const double c0(2.0/5.0*sqrt(5.0*M_PI));
              cpl*=c0;
              set_sub(V,iang,jang,cpl*Orad);
            }
          }
        }

        return V;
      }

      helfem::Matrix TwoDBasis::Bz_field(double B) const {
        // Build radial elements
        const helfem::Matrix O0rad = helfem::assemble_radial_diagonal(radial,
            [&](size_t iel) { return radial.radial_integral(0, iel); });
        const helfem::Matrix O2rad = helfem::assemble_radial_diagonal(radial,
            [&](size_t iel) { return radial.radial_integral(2, iel); });

        // Full coupling
        helfem::Matrix V = helfem::Matrix::Zero(Ndummy(), Ndummy());

        int gmax(std::max(lval.maxCoeff(),mval.maxCoeff()));
        gaunt::Gaunt gaunt(gmax,4,gmax);

        // Fill elements
        for(size_t iang=0;iang<lval.size();iang++) {
          int li(lval(iang));
          int mi(mval(iang));

          for(size_t jang=0;jang<lval.size();jang++) {
            int lj(lval(jang));
            int mj(mval(jang));

            // Calculate coupling
            double cpl(gaunt.sine2_coupling(lj,mj,li,mi));
            if(cpl!=0.0) {
              set_sub(V,iang,jang,(B*B/8.0*cpl)*O2rad);
            }

            if(li==lj && mi==mj) {
              add_sub(V,iang,jang,(-0.5*B*mj)*O0rad);
            }
          }
        }

        return V;
      }

      std::vector<std::vector<helfem::Matrix>>
      TwoDBasis::radial_df_factors(double tol) const {
        if (!prim_chol.size())
          throw std::logic_error(
              "Primitive teis have not been computed -- call "
              "compute_tei() before radial_df_factors().");

        const int N_L = 2 * lval.maxCoeff() + 1;
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
            return prim_chol[(size_t) L * Nel + iel];
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
          const Eigen::Index Nrad_e = static_cast<Eigen::Index>(Nrad);
          helfem::Matrix D = helfem::Matrix::Zero(Nrad_e, Nrad_e);
          for (size_t iel = 0; iel < Nel; ++iel) {
            size_t ifirst, ilast;
            radial.get_idx(iel, ifirst, ilast);
            const size_t Ni = ilast - ifirst + 1;
            for (size_t j_loc = 0; j_loc < Ni; ++j_loc) {
              for (size_t i_loc = 0; i_loc <= j_loc; ++i_loc) {
                const Eigen::Index i = static_cast<Eigen::Index>(ifirst + i_loc);
                const Eigen::Index j = static_cast<Eigen::Index>(ifirst + j_loc);
                helfem::Matrix P = helfem::Matrix::Zero(Nrad_e, Nrad_e);
                if (i == j) {
                  P(i, i) = 1.0;
                } else {
                  P(i, j) = 0.5;
                  P(j, i) = 0.5;
                }
                const helfem::Matrix J =
                    helfem::atomic::basis::assemble_J_FE_one_multipole_cached_chol(
                        radial, rs, rb, tw, P);
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
          std::vector<helfem::Matrix> B_L_vecs;
          while (true) {
            // Eigen is column-major, so maxCoeff scans in the same order
            // as the old arma index_max column-major flatten -- identical
            // pivot (i*, j*) selection and tie-breaking.
            Eigen::Index i_star = 0, j_star = 0;
            const double D_pivot = D.maxCoeff(&i_star, &j_star);
            if (D_pivot < tol) break;

            // Build the symmetric pair-density indicator P. For i == j:
            //   P = e_i e_i^T (single 1 on the diagonal).
            // For i != j:
            //   P = (e_i e_j^T + e_j e_i^T) / 2.
            // The J helper expects a symmetric P (it integrates against
            // the symmetric pair density), and J(P)[a, b] then equals
            // exactly R^L(a, b, i*, j*).
            helfem::Matrix P = helfem::Matrix::Zero(Nrad_e, Nrad_e);
            if (i_star == j_star) {
              P(i_star, i_star) = 1.0;
            } else {
              P(i_star, j_star) = 0.5;
              P(j_star, i_star) = 0.5;
            }
            helfem::Matrix col =
                helfem::atomic::basis::assemble_J_FE_one_multipole_cached_chol(
                    radial, rs, rb, tw, P);

            // Subtract projections onto already-found Cholesky vectors.
            for (const auto & V : B_L_vecs) {
              col -= V(i_star, j_star) * V;
            }

            // Normalise. By construction col(i*, j*) == D_pivot, so
            // v_new(i*, j*) == sqrt(D_pivot) -- pivot diagonal zeros
            // out exactly on the diagonal update below.
            helfem::Matrix v_new = col / std::sqrt(D_pivot);

            // Update diagonal: D <- D - v_new .* v_new (elementwise),
            // clamped at zero to avoid negative drift.
            D -= v_new.cwiseProduct(v_new);
            D = D.cwiseMax(0.0);

            B_L_vecs.push_back(std::move(v_new));
          }

          B[L] = std::move(B_L_vecs);
        }

        return B;
      }

      void TwoDBasis::compute_tei(bool exchange) {
        // Delegate to the shared cache builders in CoulombExchangeFE.h.
        // The bare-Coulomb defaults (yukawa=false, lambda=0) populate
        // disjoint_L / disjoint_m1L from radial.radial_integral and
        // prim_tei (in-element only; cross-element is assembled on the
        // fly from the disjoint factors inside the cached J/K helpers).
        const int N_L = 2 * lval.maxCoeff() + 1;
        atomic::basis::compute_disjoint_radial_integrals(
            radial, N_L, disjoint_L, disjoint_m1L);

        // In-element integrals, kept in factorized form: T = L L' with L of
        // shape (Ni^2 x r) and r ~ 30 rather than Ni^2 ~ 200. Both J and K are
        // assembled from this one factor -- K via the RI contraction -- so no
        // exchange-ordered tensor is built. (Its pairing is full rank, so
        // there would be nothing to gain by compressing it anyway.)
        const size_t Nel = radial.Nel();
        prim_chol.assign((size_t) N_L * Nel, helfem::Matrix());
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
        for (int L = 0; L < N_L; ++L)
          for (size_t iel = 0; iel < Nel; ++iel)
            prim_chol[(size_t) L * Nel + iel] =
                radial.twoe_integral_cholesky(L, iel, chol_tol);

        (void) exchange;
      }

      void TwoDBasis::compute_yukawa(double lambda_) {
        lambda = lambda_;
        yukawa = true;
        const int N_L = 2 * lval.maxCoeff() + 1;
        // Yukawa-mode disjoint factors (bessel i / bessel k) + in-element
        // Yukawa 2e + the exchange-permuted form needed by the cached K
        // assembly.
        atomic::basis::compute_disjoint_radial_integrals(
            radial, N_L, disjoint_iL, disjoint_kL, /*yukawa=*/true, lambda);

        // In-element Yukawa integrals, factorized exactly as the bare ones.
        // The rank bound is a property of the orbital product basis, not of the
        // kernel, so it carries over unchanged.
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

      helfem::Matrix TwoDBasis::coulomb(const helfem::Matrix & P0_in) const {
        if(!prim_chol.size())
          throw std::logic_error("Primitive teis have not been computed!\n");

        // The atomic basis has no boundary reduction (expand/remove_boundaries
        // are the identity), so the SCF-facing Eigen density is used directly.
        const helfem::Matrix & P = P0_in;

        // Number of radial elements
        size_t Nel(radial.Nel());
        // Number of radial functions
        const Eigen::Index Nrad = static_cast<Eigen::Index>(radial.Nbf());
        // Gaunt coefficient table
        int gmax(std::max(lval.maxCoeff(),mval.maxCoeff()));
        gaunt::Gaunt gaunt(gmax,2*gmax,gmax);

        // maximal M value
        int Mmax=mval.maxCoeff()-mval.minCoeff();

        // Radial helper matrices
        std::vector< std::vector<helfem::Matrix> > Paux(2*lval.maxCoeff()+1);
        for(int L=0;L<(int) Paux.size();L++) {
          Paux[L].resize(2*Mmax+1);
          for(int M=-std::min(L,Mmax);M<=std::min(L,Mmax);M++) {
            Paux[L][M+Mmax] = helfem::Matrix::Zero(Nrad,Nrad);
          }
        }

        // Form radial helpers: contract ket
        for(size_t kang=0;kang<lval.size();kang++) {
          for(size_t lang=0;lang<lval.size();lang++) {
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
              Paux[L][M+Mmax]+=cpl*P.block(static_cast<Eigen::Index>(kang)*Nrad,static_cast<Eigen::Index>(lang)*Nrad,Nrad,Nrad);
            }
          }
        }

        // Helper matrices
        std::vector< std::vector<helfem::Matrix> > Jaux(2*lval.maxCoeff()+1);
        for(int L=0;L<(int) Jaux.size();L++) {
          Jaux[L].resize(2*Mmax+1);
          for(int M=-std::min(L,Mmax);M<=std::min(L,Mmax);M++) {
            Jaux[L][M+Mmax] = helfem::Matrix::Zero(Nrad,Nrad);
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
            return prim_chol[(size_t) L * Nel + iel];
          };
          for(int M=-std::min(L,Mmax);M<=std::min(L,Mmax);M++) {
            Jaux[L][M+Mmax] += Lfac *
              helfem::atomic::basis::assemble_J_FE_one_multipole_cached_chol(
                radial, rs, rb, tw, Paux[L][M+Mmax]);
          }
        }

        // Full Coulomb matrix
        helfem::Matrix J = helfem::Matrix::Zero(Ndummy(),Ndummy());
        for(size_t iang=0;iang<lval.size();iang++) {
          for(size_t jang=0;jang<lval.size();jang++) {
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
                J.block(static_cast<Eigen::Index>(iang)*Nrad,static_cast<Eigen::Index>(jang)*Nrad,Nrad,Nrad)+=cpl*Jaux[L][M+Mmax];
              }
            }
          }
        }

        return J;
      }

      helfem::Matrix TwoDBasis::exchange(const helfem::Matrix & P0_in) const {
        if(!prim_chol.size())
          throw std::logic_error("Primitive teis have not been computed!\n");

        // No boundary reduction in the atomic basis (expand/remove are
        // the identity), so the Eigen density is used directly.
        const helfem::Matrix & P = P0_in;

        // Gaunt coefficient table
        int gmax(std::max(lval.maxCoeff(),mval.maxCoeff()));
        gaunt::Gaunt gaunt(gmax,2*gmax,gmax);

        // Number of radial elements
        size_t Nel(radial.Nel());
        // Number of radial basis functions
        const Eigen::Index Nrad = static_cast<Eigen::Index>(radial.Nbf());

        // Full exchange matrix
        helfem::Matrix K = helfem::Matrix::Zero(Ndummy(),Ndummy());

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
          for(size_t jang=0;jang<lval.size();jang++) {
            for(size_t kang=0;kang<lval.size();kang++) {
              int lj(lval(jang));
              int mj(mval(jang));

              int lk(lval(kang));
              int mk(mval(kang));

              // Form radial helpers
              size_t N_L(2*lval.maxCoeff()+1);
              std::vector<helfem::Matrix> Rmat(N_L);
              for(size_t i=0;i<N_L;i++) {
                Rmat[i] = helfem::Matrix::Zero(Nrad,Nrad);
              }
              // Is there a coupling to the channel?
              std::vector<bool> couple(N_L,false);

              // Perform angular sums
              for(size_t iang=0;iang<lval.size();iang++) {
                int li(lval(iang));
                int mi(mval(iang));

                for(size_t lang=0;lang<lval.size();lang++) {
                  int ll(lval(lang));
                  int ml(mval(lang));

                  // LH m value
                  int M(mj-mi);
                  // RH m value
                  int Mp(mk-ml);
                  if(M!=Mp)
                    continue;

                  // Do we have any density in this block?
                  double bdens(P.block(static_cast<Eigen::Index>(iang)*Nrad,static_cast<Eigen::Index>(lang)*Nrad,Nrad,Nrad).norm());
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
                    Rmat[L]+=(Lfac*cpl)*P.block(static_cast<Eigen::Index>(iang)*Nrad,static_cast<Eigen::Index>(lang)*Nrad,Nrad,Nrad);
                    couple[L]=true;
                  }
                }
              }

              // Per-L K assembly via the shared FE helper, accumulated
              // into the (jang, kang) angular block of K.
              helfem::Matrix K_block = helfem::Matrix::Zero(Nrad, Nrad);
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
                  return prim_chol[(size_t) Lc * Nel + iel];
                };
                // RI-K on the J-ordered Cholesky factor: K_ac = sum_p M_p P M_p'.
                // The exchange-ordered tensor is full rank, so this is the only
                // way to compress K -- and it means none needs to be stored.
                K_block += helfem::atomic::basis::assemble_K_FE_one_multipole_cached_chol(
                    radial, rs, rb, kt, Rmat[Lc]);
              }
              K.block(static_cast<Eigen::Index>(jang)*Nrad, static_cast<Eigen::Index>(kang)*Nrad, Nrad, Nrad) -= K_block;
            }
          }
        }

        return K;
      }

      helfem::Matrix TwoDBasis::rs_exchange(const helfem::Matrix & P0_in) const {
        if(!rs_ktei.size() && !rs_chol.size())
          throw std::logic_error("Primitive teis have not been computed!\n");

        // No boundary reduction in the atomic basis (expand/remove are
        // the identity), so the Eigen density is used directly.
        const helfem::Matrix & P = P0_in;

        // Gaunt coefficient table
        int gmax(std::max(lval.maxCoeff(),mval.maxCoeff()));
        gaunt::Gaunt gaunt(gmax,2*gmax,gmax);

        // Number of radial elements
        size_t Nel(radial.Nel());
        // Number of radial basis functions
        const Eigen::Index Nrad = static_cast<Eigen::Index>(radial.Nbf());

        // Full exchange matrix
        helfem::Matrix K = helfem::Matrix::Zero(Ndummy(),Ndummy());

        // Per-element K assembly is delegated to the cached helpers
        // in CoulombExchangeFE.h -- no per-thread scratch needed.
#ifdef _OPENMP
#pragma omp parallel
#endif
        {
#ifdef _OPENMP
#pragma omp for collapse(2)
#endif
          for(size_t jang=0;jang<lval.size();jang++) {
            for(size_t kang=0;kang<lval.size();kang++) {
              int lj(lval(jang));
              int mj(mval(jang));

              int lk(lval(kang));
              int mk(mval(kang));

              // Form radial helpers
              size_t N_L(2*lval.maxCoeff()+1);
              std::vector<helfem::Matrix> Rmat(N_L);
              for(size_t i=0;i<N_L;i++) {
                Rmat[i] = helfem::Matrix::Zero(Nrad,Nrad);
              }
              // Is there a coupling to the channel?
              std::vector<bool> couple(N_L,false);

              // Perform angular sums
              for(size_t iang=0;iang<lval.size();iang++) {
                int li(lval(iang));
                int mi(mval(iang));

                for(size_t lang=0;lang<lval.size();lang++) {
                  int ll(lval(lang));
                  int ml(mval(lang));

                  // LH m value
                  int M(mj-mi);
                  // RH m value
                  int Mp(mk-ml);
                  if(M!=Mp)
                    continue;

                  // Do we have any density in this block?
                  double bdens(P.block(static_cast<Eigen::Index>(iang)*Nrad,static_cast<Eigen::Index>(lang)*Nrad,Nrad,Nrad).norm());
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
                    Rmat[L]+=(Lfac*cpl)*P.block(static_cast<Eigen::Index>(iang)*Nrad,static_cast<Eigen::Index>(lang)*Nrad,Nrad,Nrad);
                    couple[L]=true;
                  }
                }
              }

              if (yukawa) {
                // Same FE structure as bare exchange: per-L cached
                // helper, summed into the (jang, kang) angular block.
                helfem::Matrix K_block = helfem::Matrix::Zero(Nrad, Nrad);
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
                    return rs_chol[(size_t) Lc * Nel + iel];
                  };
                  K_block += helfem::atomic::basis::assemble_K_FE_one_multipole_cached_chol(
                      radial, rs, rb, kt, Rmat[Lc]);
                }
                K.block(static_cast<Eigen::Index>(jang)*Nrad, static_cast<Eigen::Index>(kang)*Nrad, Nrad, Nrad) -= K_block;
              } else {
                // Erfc: rs_ktei has cross-element entries for every
                // (iel, jel) pair -- delegate to the pairwise cached
                // helper, summed over L into the (jang, kang) block.
                helfem::Matrix K_block = helfem::Matrix::Zero(Nrad, Nrad);
                for(size_t L=0; L<N_L; ++L) {
                  if(!couple[L]) continue;
                  const size_t Lc = L;
                  auto kt = [&,Lc](size_t iel, size_t jel) -> const helfem::Matrix & {
                    return rs_ktei[Nel*Nel*Lc + iel*Nel + jel];
                  };
                  K_block += helfem::atomic::basis::assemble_K_FE_one_multipole_cached_pairwise(
                      radial, kt, Rmat[Lc]);
                }
                K.block(static_cast<Eigen::Index>(jang)*Nrad, static_cast<Eigen::Index>(kang)*Nrad, Nrad, Nrad) -= K_block;
              }
            }
          }
        }

        return K;
      }

      arma::cx_mat TwoDBasis::eval_bf(size_t iel, double cth, double phi) const {
        // Evaluate spherical harmonics
        arma::cx_vec sph(lval.size());
        for(size_t i=0;i<lval.size();i++)
          sph(i)=::spherical_harmonics(lval(i),mval(i),cth,phi);

        // Evaluate radial functions
        arma::mat rad(helfem::to_arma(radial.get_bf(iel)));

        // Form supermatrix
        arma::cx_mat bf(rad.n_rows,lval.size()*rad.n_cols);
        for(size_t i=0;i<lval.size();i++)
          bf.cols(i*rad.n_cols,(i+1)*rad.n_cols-1)=sph(i)*rad;

        return bf;
      }

      void TwoDBasis::eval_df(size_t iel, double cth, double phi, arma::cx_mat & dr, arma::cx_mat & dth, arma::cx_mat & dphi) const {
        // Evaluate spherical harmonics
        arma::cx_vec sph(lval.size());
        for(size_t i=0;i<lval.size();i++)
          sph(i)=::spherical_harmonics(lval(i),mval(i),cth,phi);

        // Evaluate radial functions
        arma::mat frad(helfem::to_arma(radial.get_bf(iel)));
        arma::mat drad(helfem::to_arma(radial.get_df(iel)));

        // Form supermatrices
        dr.zeros(frad.n_rows,lval.size()*frad.n_cols);
        dth.zeros(frad.n_rows,lval.size()*frad.n_cols);
        dphi.zeros(frad.n_rows,lval.size()*frad.n_cols);

        // Radial one is easy
        for(size_t i=0;i<lval.size();i++)
          dr.cols(i*frad.n_cols,(i+1)*frad.n_cols-1)=sph(i)*drad;
        // and so is phi
        for(size_t i=0;i<lval.size();i++)
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
        for(size_t i=0;i<lval.size();i++) {
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
        arma::cx_vec sph(lval.size());
        for(size_t i=0;i<lval.size();i++)
          sph(i)=::spherical_harmonics(lval(i),mval(i),cth,phi);

        // Evaluate radial functions
        arma::vec r(helfem::to_arma(radial.get_r(iel)));
        arma::mat frad(helfem::to_arma(radial.get_bf(iel)));
        arma::mat drad(helfem::to_arma(radial.get_df(iel)));
        arma::mat lrad(helfem::to_arma(radial.get_lf(iel)));

        // Form supermatrix
        arma::cx_mat lf(frad.n_rows,lval.size()*frad.n_cols,arma::fill::zeros);
        // Loop over basis function indices
        for(size_t iang=0;iang<lval.size();iang++)
          for(size_t irad=0;irad<frad.n_cols;irad++)
            // Loop over grid-point indices
            for(size_t igrid=0;igrid<frad.n_rows;igrid++)
              lf(igrid,iang*frad.n_cols+irad) = (lrad(igrid,irad) + 2*drad(igrid,irad)/r(igrid) - lval(iang)*(lval(iang)+1)*frad(igrid,irad)/(r(igrid)*r(igrid)))*sph(iang);

        //for(size_t i=0;i<lval.size();i++)
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
        arma::uvec idx(Nr*lval.size());
        for(size_t iam=0;iam<lval.size();iam++)
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
