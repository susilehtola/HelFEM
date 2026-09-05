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
#include "../general/eigen_io.h"   // fmt_sci, for the residual report
#include <cassert>
#include <cfloat>
#include <algorithm>
#include <sstream>
#include <string>
#include <limits>
#include <type_traits>
#include <helfem.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace helfem {
  namespace atomic {
    namespace basis {

      // The routines that evaluate basis functions on a grid
      // (eval_bf / eval_df / eval_lf) return Eigen::MatrixXcd and go through the
      // double-only ::spherical_harmonics. They serve the DFT grid and the
      // analysis binaries, neither of which can run above double anyway (libxc
      // is a double-only C library), and none is on the Fock path. Rather than
      // split the class they are compiled for every T and guarded: at
      // T = double they are exactly the code they always were, and at any
      // other T they throw. (The quadrature-point accessors bval /
      // wrad / r are now precision-generic helfem::Vec<T> and need no
      // guard.)
      namespace {
        [[noreturn]] void double_only(const char *what) {
          throw std::logic_error(std::string(what) +
                                 " is only available at T = double "
                                 "(it uses the double-only spherical "
                                 "harmonics).\n");
        }
      }

      template <typename T>
      TwoDBasisT<T>::TwoDBasisT() {
      }

      template <typename T>
      TwoDBasisT<T>::TwoDBasisT(int Z, modelpotential::nuclear_model_t model_,
                                T Rrms_,
                                const std::shared_ptr<const helfem::polynomial_basis::PolynomialBasisT<T>> & poly,
                                bool zeroder, int n_quad,
                                const helfem::Vec<T> & bval,
                                const Eigen::VectorXi & lval_e,
                                const Eigen::VectorXi & mval_e,
                                int Zl, int Zr, T Rhalf) {
        // Nuclear charge
        Z_=Z;
        Zl_=Zl;
        Zr_=Zr;
        Rhalf_=Rhalf;
        model=model_;
        Rrms=Rrms_;

        // Construct radial basis
        bool zero_func_left=true;
        bool zero_deriv_left=false;
        bool zero_func_right=true;
        zeroder_=zeroder;
        polynomial_basis::FiniteElementBasisT<T> fem(poly, bval, zero_func_left, zero_deriv_left, zero_func_right, zeroder_);
        radial_=FEMRadialBasisT<T>(fem, n_quad);

        // Construct angular basis
        lval_=lval_e;
        mval_=mval_e;
      }

      template <typename T>
      TwoDBasisT<T>::~TwoDBasisT() {
      }

      template <typename T>
      int TwoDBasisT<T>::nuclear_model() const {
        return model;
      }

      template <typename T>
      T TwoDBasisT<T>::nuclear_size() const {
        return Rrms;
      }

      template <typename T>
      int TwoDBasisT<T>::Z() const {
        return Z_;
      }

      template <typename T>
      int TwoDBasisT<T>::Zl() const {
        return Zl_;
      }

      template <typename T>
      int TwoDBasisT<T>::Zr() const {
        return Zr_;
      }

      template <typename T>
      T TwoDBasisT<T>::Rhalf() const {
        return Rhalf_;
      }

      template <typename T>
      Eigen::VectorXi TwoDBasisT<T>::lval() const {
        return lval_;
      }

      template <typename T>
      Eigen::VectorXi TwoDBasisT<T>::mval() const {
        return mval_;
      }

      template <typename T>
      int TwoDBasisT<T>::nquad() const {
        return radial_.nquad();
      }

      template <typename T>
      helfem::Vec<T> TwoDBasisT<T>::bval() const {
        return radial_.bval();
      }

      template <typename T>
      int TwoDBasisT<T>::poly_id() const {
        return radial_.poly_id();
      }

      template <typename T>
      int TwoDBasisT<T>::poly_nnodes() const {
        return radial_.poly_nnodes();
      }

      template <typename T>
      int TwoDBasisT<T>::zeroder() const {
        return zeroder_;
      }

      template <typename T>
      size_t TwoDBasisT<T>::Ndummy() const {
        return lval_.size()*radial_.Nbf();
      }

      template <typename T>
      size_t TwoDBasisT<T>::Nbf() const {
        return Ndummy();
      }

      template <typename T>
      size_t TwoDBasisT<T>::Nrad() const {
        return radial_.Nbf();
      }

      template <typename T>
      size_t TwoDBasisT<T>::Nang() const {
        return lval_.size();
      }

      template <typename T>
      std::vector<Eigen::Index> TwoDBasisT<T>::m_indices(int m) const {
        return helfem::collect_shell_indices(mval_.size(),
            [&](size_t)   { return radial_.Nbf(); },
            [&](size_t i) { return mval_(i) == m; });
      }

      template <typename T>
      std::vector<Eigen::Index> TwoDBasisT<T>::lm_indices(int l, int m) const {
        return helfem::collect_shell_indices(mval_.size(),
            [&](size_t)   { return radial_.Nbf(); },
            [&](size_t i) { return mval_(i) == m && lval_(i) == l; });
      }

      template <typename T>
      std::vector<std::vector<Eigen::Index>> TwoDBasisT<T>::sym_idx(int symm) const {
        std::vector<std::vector<Eigen::Index>> idx;
        if(symm==0) {
          idx.resize(1);
          idx[0].resize(Nbf());
          for(Eigen::Index i=0;i<(Eigen::Index) Nbf();i++)
            idx[0][i]=i;
        } else if(symm==1) {
          // Unique m values in ascending order (matches arma::find_unique).
          std::vector<int> mv;
          for (Eigen::Index i = 0; i < mval_.size(); ++i)
            if (std::find(mv.begin(), mv.end(), mval_(i)) == mv.end())
              mv.push_back(mval_(i));
          std::sort(mv.begin(), mv.end());

          idx.resize(mv.size());
          for(size_t i=0;i<mv.size();i++)
            idx[i]=m_indices(mv[i]);
        } else if(symm==2) {
          idx.resize(mval_.size());
          for(size_t i=0;i<(size_t) mval_.size();i++) {
            idx[i]=lm_indices(lval_(i),mval_(i));
          }
        } else
          throw std::logic_error("Unknown symmetry\n");

        return idx;
      }

      template <typename T>
      std::vector<std::string> TwoDBasisT<T>::sym_labels(int symm) const {
        // Mirrors sym_idx block for block: same branches, same
        // ordering, same unique-m sort. Anything changed there has to be
        // changed here too.
        std::vector<std::string> labels;
        auto mlabel = [](int m) {
          std::ostringstream oss;
          // Explicit sign for nonzero m, but plain "m=0" rather than "m=+0".
          oss << "m=" << (m > 0 ? "+" : "") << m;
          return oss.str();
        };

        if(symm==0) {
          labels.push_back("all");
        } else if(symm==1) {
          std::vector<int> mv;
          for (Eigen::Index i = 0; i < mval_.size(); ++i)
            if (std::find(mv.begin(), mv.end(), mval_(i)) == mv.end())
              mv.push_back(mval_(i));
          std::sort(mv.begin(), mv.end());
          for(size_t i=0;i<mv.size();i++)
            labels.push_back(mlabel(mv[i]));
        } else if(symm==2) {
          // One block per (l,m) channel of the angular basis.
          for(size_t i=0;i<(size_t) mval_.size();i++) {
            std::ostringstream oss;
            oss << "l=" << lval_(i) << " " << mlabel(mval_(i));
            labels.push_back(oss.str());
          }
        } else
          throw std::logic_error("Unknown symmetry\n");

        return labels;
      }

      template <typename T>
      helfem::Mat<T> TwoDBasisT<T>::Sinvh(bool chol, int sym) const {
        const helfem::Mat<T> S = overlap();

        // Half-inverse is
        if(sym==0) {
          return scf::form_Sinvh<T>(S, chol);
        } else {
          // Per-symmetry-block orthonormalization. sym_idx returns the
          // (scattered) AO index list for each block; the orthonormal
          // columns are packed contiguously, so Sinvh maps orthonormal ->
          // AO with block-diagonal structure (scattered rows, contiguous
          // columns per block).
          std::vector<std::vector<Eigen::Index>> midx(sym_idx(sym));
          const Eigen::Index N = static_cast<Eigen::Index>(Nbf());
          helfem::Mat<T> Sinvh = helfem::Mat<T>::Zero(N, N);
          Eigen::Index ioff = 0;
          for(size_t i=0;i<midx.size();i++) {
            const Eigen::Index n = static_cast<Eigen::Index>(midx[i].size());
            if(!n)
              continue;

            // AO indices of this block.
            const std::vector<Eigen::Index> & rows = midx[i];

            // Gather the block overlap, orthonormalize.
            helfem::Mat<T> Ssub(n, n);
            for(Eigen::Index a=0;a<n;a++)
              for(Eigen::Index b=0;b<n;b++)
                Ssub(a,b) = S(rows[a], rows[b]);
            const helfem::Mat<T> sub = scf::form_Sinvh<T>(Ssub, chol);

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

      template <typename T>
      void TwoDBasisT<T>::set_sub(helfem::Mat<T> & M, size_t iang, size_t jang, const helfem::Mat<T> & Mrad) const {
        const Eigen::Index n = static_cast<Eigen::Index>(radial_.Nbf());
        M.block(static_cast<Eigen::Index>(iang)*n, static_cast<Eigen::Index>(jang)*n, n, n) = Mrad;
      }

      template <typename T>
      void TwoDBasisT<T>::add_sub(helfem::Mat<T> & M, size_t iang, size_t jang, const helfem::Mat<T> & Mrad) const {
        const Eigen::Index n = static_cast<Eigen::Index>(radial_.Nbf());
        M.block(static_cast<Eigen::Index>(iang)*n, static_cast<Eigen::Index>(jang)*n, n, n) += Mrad;
      }

      template <typename T>
      helfem::Mat<T> TwoDBasisT<T>::overlap() const {
        // Full overlap matrix built by scattering the radial R=0 integrals
        // along the angular diagonal. The atomic basis has no
        // boundary-condition reduction (pure_indices() is the identity),
        // so the assembled matrix is returned directly.
        const helfem::Mat<T> Orad = helfem::assemble_radial_diagonal(radial_,
            [&](size_t iel) { return radial_.radial_integral(0, iel); });

        helfem::Mat<T> O = helfem::Mat<T>::Zero(Ndummy(), Ndummy());
        for(size_t iang=0;iang<(size_t) lval_.size();iang++)
          set_sub(O,iang,iang,Orad);

        return O;
      }

      template <typename T>
      helfem::Mat<T> TwoDBasisT<T>::overlap(const TwoDBasisT<T> & rh) const {
        // Cross-basis overlap: scatter radial.overlap(rh.radial) along the
        // (l, m)-matched blocks of the two angular indexings. Each side's
        // pure_indices() is the identity, so no boundary slicing is needed.
        helfem::Mat<T> S = helfem::Mat<T>::Zero(Ndummy(), rh.Ndummy());
        const helfem::Mat<T> Srad = radial_.overlap(rh.radial_);
        const Eigen::Index n   = static_cast<Eigen::Index>(radial_.Nbf());
        const Eigen::Index rhn = static_cast<Eigen::Index>(rh.radial_.Nbf());

        for(size_t iang=0;iang<(size_t) lval_.size();iang++)
          for(size_t jang=0;jang<(size_t) rh.lval_.size();jang++)
            if(lval_(iang) == rh.lval_(jang) && mval_(iang) == rh.mval_(jang))
              S.block(static_cast<Eigen::Index>(iang)*n,
                      static_cast<Eigen::Index>(jang)*rhn, n, rhn) = Srad;

        return S;
      }

      template <typename T>
      helfem::Mat<T> TwoDBasisT<T>::kinetic() const {
        // Build radial kinetic energy matrices
        const helfem::Mat<T> Trad = helfem::assemble_radial_diagonal(radial_,
            [&](size_t iel) { return radial_.kinetic(iel); });
        const helfem::Mat<T> Trad_l = helfem::assemble_radial_diagonal(radial_,
            [&](size_t iel) { return radial_.kinetic_l(iel); });

        // Full kinetic energy matrix
        helfem::Mat<T> Tmat = helfem::Mat<T>::Zero(Ndummy(), Ndummy());
        // Fill elements
        for(size_t iang=0;iang<(size_t) lval_.size();iang++) {
          set_sub(Tmat,iang,iang,Trad);
          if(lval_(iang)>0) {
            // We also get the l(l+1) term
            add_sub(Tmat,iang,iang,static_cast<T>(lval_(iang)*(lval_(iang)+1))*Trad_l);
          }
        }

        return Tmat;
      }

      template <typename T>
      helfem::Mat<T> TwoDBasisT<T>::nuclear() const {
        if(model != modelpotential::POINT_NUCLEUS) {
          modelpotential::ModelPotentialT<T> *pot=modelpotential::nuclear_model<T>(model,Z_,Rrms);
          helfem::Mat<T> Vnuc = model_potential(pot);
          delete pot;
          return Vnuc;
        } else {
          // Full nuclear attraction matrix
          helfem::Mat<T> V = helfem::Mat<T>::Zero(Ndummy(), Ndummy());

          if (Z_ != 0) {
            const helfem::Mat<T> Vrad = helfem::assemble_radial_diagonal(radial_,
                [&](size_t iel) { return radial_.radial_integral(-1, iel); });
            for (size_t iang = 0; iang < (size_t) lval_.size(); iang++)
              set_sub(V, iang, iang, static_cast<T>(-Z_) * Vrad);
          }

          if(Zl_ != 0 || Zr_ != 0) {
            // Auxiliary matrices
            int Lmax(2*lval_.maxCoeff());
            std::vector<helfem::Mat<T>> Vaux(Lmax+1);
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (int L = 0; L <= Lmax; L++) {
              Vaux[L] = helfem::assemble_radial_diagonal(radial_,
                  [&](size_t iel) { return radial_.nuclear_offcenter(iel, Rhalf_, L); });
            }

            int gmax(std::max(lval_.maxCoeff(),mval_.maxCoeff()));
            gaunt::GauntT<T> gaunt(gmax,2*gmax,gmax);

            /// Loop over basis set
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
            for(size_t iang=0;iang<(size_t) lval_.size();iang++) {
              for(size_t jang=0;jang<(size_t) lval_.size();jang++) {
                int li(lval_(iang));
                int mi(mval_(iang));

                int lj(lval_(jang));
                int mj(mval_(jang));

                // Zero contribution
                if(mi!=mj)
                  continue;

                // Loop over L
                for(int L=std::abs(li-lj);L<=li+lj;L++) {
                  T cpl(gaunt.coeff(li,mi,L,0,lj));
                  if(cpl==T(0))
                    continue;

                  const T signL = (L & 1) ? T(-1) : T(1);
                  add_sub(V,iang,jang,(cpl*(signL*T(Zl_) + T(Zr_)))*Vaux[L]);
                }
              }
            }
          }

          return V;
        }
      }

      template <typename T>
      helfem::Mat<T> TwoDBasisT<T>::model_potential(const modelpotential::ModelPotentialT<T> * pot) const {
        // Full nuclear attraction matrix
        helfem::Mat<T> V = helfem::Mat<T>::Zero(Ndummy(), Ndummy());

        const helfem::Mat<T> Vrad = helfem::assemble_radial_diagonal(radial_,
            [&](size_t iel) { return radial_.model_potential(pot, iel); });
	// Fill elements
	for(size_t iang=0;iang<(size_t) lval_.size();iang++)
	  set_sub(V,iang,iang,Vrad);

        return V;
      }

      template <typename T>
      helfem::Mat<T> TwoDBasisT<T>::confinement(const int N, T r_0, const int iconf, const T V, const T shift_pot) const {
        // Full matrix
        helfem::Mat<T> O = helfem::Mat<T>::Zero(Ndummy(), Ndummy());

	if(iconf==0)
	  return O;

        const helfem::Mat<T> Orad = helfem::assemble_radial_diagonal(radial_,
            [&](size_t iel) {
              return radial_.confinement_potential(iel, N, r_0, iconf, V, shift_pot);
            });

        // Fill elements
        for(size_t iang=0;iang<(size_t) lval_.size();iang++)
          set_sub(O,iang,iang,Orad);

        return O;
      }

      template <typename T>
      helfem::Mat<T> TwoDBasisT<T>::dipole_z() const {
        // Build radial elements
        const helfem::Mat<T> Orad = helfem::assemble_radial_diagonal(radial_,
            [&](size_t iel) { return radial_.radial_integral(1, iel); });

        // Full electric couplings
        helfem::Mat<T> V = helfem::Mat<T>::Zero(Ndummy(), Ndummy());

        int gmax(std::max(lval_.maxCoeff(),mval_.maxCoeff()));
        gaunt::GauntT<T> gaunt(gmax,1,gmax);

        // Fill elements
        for(size_t iang=0;iang<(size_t) lval_.size();iang++) {
          int li(lval_(iang));
          int mi(mval_(iang));

          for(size_t jang=0;jang<(size_t) lval_.size();jang++) {
            int lj(lval_(jang));
            int mj(mval_(jang));

            // Calculate coupling
            T cpl(gaunt.cosine_coupling(lj,mj,li,mi));
            if(cpl!=T(0))
              set_sub(V,iang,jang,cpl*Orad);
          }
        }

        return V;
      }

      template <typename T>
      helfem::Mat<T> TwoDBasisT<T>::quadrupole_zz() const {
        // Build radial elements
        const helfem::Mat<T> Orad = helfem::assemble_radial_diagonal(radial_,
            [&](size_t iel) { return radial_.radial_integral(2, iel); });

        // Full electric couplings
        helfem::Mat<T> V = helfem::Mat<T>::Zero(Ndummy(), Ndummy());

        int gmax(std::max(lval_.maxCoeff(),mval_.maxCoeff()));
        gaunt::GauntT<T> gaunt(gmax,2,gmax);

        // Fill elements
        for(size_t iang=0;iang<(size_t) lval_.size();iang++) {
          int li(lval_(iang));
          int mi(mval_(iang));

          for(size_t jang=0;jang<(size_t) lval_.size();jang++) {
            int lj(lval_(jang));
            int mj(mval_(jang));

            // Y_2^0 has m=0, so the m-sum forces mj == mi.
            if(mj != mi) continue;

            // Calculate coupling
            T cpl(gaunt.coeff(lj,mj,2,0,li));
            if(cpl!=T(0)) {
              const T c0(T(2)/T(5)*std::sqrt(T(5)*utils::pi<T>()));
              cpl*=c0;
              set_sub(V,iang,jang,cpl*Orad);
            }
          }
        }

        return V;
      }

      template <typename T>
      helfem::Mat<T> TwoDBasisT<T>::Bz_field(T B) const {
        // Build radial elements
        const helfem::Mat<T> O0rad = helfem::assemble_radial_diagonal(radial_,
            [&](size_t iel) { return radial_.radial_integral(0, iel); });
        const helfem::Mat<T> O2rad = helfem::assemble_radial_diagonal(radial_,
            [&](size_t iel) { return radial_.radial_integral(2, iel); });

        // Full coupling
        helfem::Mat<T> V = helfem::Mat<T>::Zero(Ndummy(), Ndummy());

        int gmax(std::max(lval_.maxCoeff(),mval_.maxCoeff()));
        gaunt::GauntT<T> gaunt(gmax,4,gmax);

        // Fill elements
        for(size_t iang=0;iang<(size_t) lval_.size();iang++) {
          int li(lval_(iang));
          int mi(mval_(iang));

          for(size_t jang=0;jang<(size_t) lval_.size();jang++) {
            int lj(lval_(jang));
            int mj(mval_(jang));

            // Calculate coupling
            T cpl(gaunt.sine2_coupling(lj,mj,li,mi));
            if(cpl!=T(0)) {
              set_sub(V,iang,jang,(B*B/T(8)*cpl)*O2rad);
            }

            if(li==lj && mi==mj) {
              add_sub(V,iang,jang,(T(-0.5)*B*T(mj))*O0rad);
            }
          }
        }

        return V;
      }

      namespace {
        /// Report an incomplete density-fitting factorisation once per
        /// process. The caller gets the best factorisation achieved
        /// rather than a hang or a silent claim of success, and is told
        /// what residual it actually has: no caller can know in advance
        /// which tolerances a given basis can reach.
        template <typename T>
        void warn_df_incomplete(int L, size_t nvec, T achieved, T tol,
                                const char *why) {
          static bool warned = false;
          if (warned)
            return;
          warned = true;
          printf("Warning: TwoDBasis::radial_df_factors stopped at multipole "
                 "L=%d after %d vectors (%s) with residual %s, above the "
                 "requested %s; using the factorization obtained.\n",
                 L, (int) nvec, why,
                 helfem::io::fmt_sci(achieved).c_str(),
                 helfem::io::fmt_sci(tol).c_str());
          fflush(stdout);
        }
      }

      template <typename T>
      std::vector<std::vector<helfem::Mat<T>>>
      TwoDBasisT<T>::radial_df_factors(T tol) const {
        if (!prim_chol.size())
          throw std::logic_error(
              "Primitive teis have not been computed -- call "
              "compute_tei() before radial_df_factors().");

        const int N_L = 2 * lval_.maxCoeff() + 1;
        const size_t Nrad = radial_.Nbf();
        const size_t Nel  = radial_.Nel();

        // Outer index = multipole L, inner index = Cholesky vector Q,
        // each an Nrad x Nrad helfem::Mat<T>.
        std::vector<std::vector<helfem::Mat<T>>> B(N_L);

        for (int L = 0; L < N_L; ++L) {
          // -- 1. Cached accessor lambdas for the per-L J helper.
          auto rs = [&, L](size_t iel) -> const helfem::Mat<T> & {
            return disjoint_L[(size_t) L * Nel + iel];
          };
          auto rb = [&, L](size_t iel) -> const helfem::Mat<T> & {
            return disjoint_m1L[(size_t) L * Nel + iel];
          };
          auto tw = [&, L](size_t iel) -> const helfem::Mat<T> & {
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
          //    Cholesky.
          const Eigen::Index Nrad_e = static_cast<Eigen::Index>(Nrad);
          helfem::Mat<T> D = helfem::Mat<T>::Zero(Nrad_e, Nrad_e);
          for (size_t iel = 0; iel < Nel; ++iel) {
            size_t ifirst, ilast;
            radial_.idx(iel, ifirst, ilast);
            const size_t Ni = ilast - ifirst + 1;
            for (size_t j_loc = 0; j_loc < Ni; ++j_loc) {
              for (size_t i_loc = 0; i_loc <= j_loc; ++i_loc) {
                const Eigen::Index i = static_cast<Eigen::Index>(ifirst + i_loc);
                const Eigen::Index j = static_cast<Eigen::Index>(ifirst + j_loc);
                helfem::Mat<T> P = helfem::Mat<T>::Zero(Nrad_e, Nrad_e);
                if (i == j) {
                  P(i, i) = T(1);
                } else {
                  P(i, j) = T(0.5);
                  P(j, i) = T(0.5);
                }
                const helfem::Mat<T> J =
                    helfem::atomic::basis::assemble_J_FE_one_multipole_cached_chol(
                        radial_, rs, rb, tw, P);
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
          std::vector<helfem::Mat<T>> B_L_vecs;

          // Two bounds on the loop below. Without them the only exit was
          // D_pivot < tol, so a tol the metric cannot reach never
          // terminated -- no error, no output, just spinning. Reported
          // against a HIP3 basis, where 1000*eps was unattainable.
          //
          // The rank bound is the principled one: the residual lives on
          // the SYMMETRIC pair index, whose dimension is
          // Nrad(Nrad+1)/2, so the exact factorisation cannot need more
          // vectors than that and anything beyond it is round-off.
          //
          // The stall bound catches the same failure sooner. In exact
          // arithmetic the pivot diagonal is monotone non-increasing --
          // each step zeroes the element it consumes -- but in floating
          // point it settles onto the metric's own round-off floor,
          // which for an ill-conditioned basis sits well above a tight
          // tol. Ties are legitimate and systematic here (D is
          // symmetric, so every off-diagonal pair appears twice), which
          // is why the window is generous rather than tripping on the
          // first non-decrease.
          const size_t maxvec =
              (size_t) Nrad_e * (size_t) (Nrad_e + 1) / 2;
          const size_t stall_window = 50;
          T best_pivot = D.maxCoeff();
          size_t nstall = 0;

          while (true) {
            // Eigen is column-major, so maxCoeff scans in the same order
            // as the old arma index_max column-major flatten -- identical
            // pivot (i*, j*) selection and tie-breaking.
            Eigen::Index i_star = 0, j_star = 0;
            const T D_pivot = D.maxCoeff(&i_star, &j_star);
            if (D_pivot < tol) break;

            if (B_L_vecs.size() >= maxvec) {
              warn_df_incomplete(L, B_L_vecs.size(), D_pivot, tol, "rank");
              break;
            }
            if (D_pivot < best_pivot) {
              best_pivot = D_pivot;
              nstall = 0;
            } else if (++nstall >= stall_window) {
              warn_df_incomplete(L, B_L_vecs.size(), D_pivot, tol, "stalled");
              break;
            }

            // Build the symmetric pair-density indicator P. For i == j:
            //   P = e_i e_i^T (single 1 on the diagonal).
            // For i != j:
            //   P = (e_i e_j^T + e_j e_i^T) / 2.
            // The J helper expects a symmetric P (it integrates against
            // the symmetric pair density), and J(P)[a, b] then equals
            // exactly R^L(a, b, i*, j*).
            helfem::Mat<T> P = helfem::Mat<T>::Zero(Nrad_e, Nrad_e);
            if (i_star == j_star) {
              P(i_star, i_star) = T(1);
            } else {
              P(i_star, j_star) = T(0.5);
              P(j_star, i_star) = T(0.5);
            }
            helfem::Mat<T> col =
                helfem::atomic::basis::assemble_J_FE_one_multipole_cached_chol(
                    radial_, rs, rb, tw, P);

            // Subtract projections onto already-found Cholesky vectors.
            for (const auto & V : B_L_vecs) {
              col -= V(i_star, j_star) * V;
            }

            // Normalise. By construction col(i*, j*) == D_pivot, so
            // v_new(i*, j*) == sqrt(D_pivot) -- pivot diagonal zeros
            // out exactly on the diagonal update below.
            helfem::Mat<T> v_new = col / std::sqrt(D_pivot);

            // Update diagonal: D <- D - v_new .* v_new (elementwise),
            // clamped at zero to avoid negative drift.
            D -= v_new.cwiseProduct(v_new);
            D = D.cwiseMax(T(0));

            B_L_vecs.push_back(std::move(v_new));
          }

          B[L] = std::move(B_L_vecs);
        }

        return B;
      }

      template <typename T>
      void TwoDBasisT<T>::compute_tei(bool exchange) {
        // Delegate to the shared cache builders in CoulombExchangeFE.h.
        // The bare-Coulomb defaults (yukawa=false, lambda=0) populate
        // disjoint_L / disjoint_m1L from radial.radial_integral and
        // prim_chol (in-element only; cross-element is assembled on the
        // fly from the disjoint factors inside the cached J/K helpers).
        const int N_L = 2 * lval_.maxCoeff() + 1;
        atomic::basis::compute_disjoint_radial_integrals(
            radial_, N_L, disjoint_L, disjoint_m1L);

        // In-element integrals, kept in factorized form: T = L L' with L of
        // shape (Ni^2 x r) and r ~ 30 rather than Ni^2 ~ 200. Both J and K are
        // assembled from this one factor -- K via the RI contraction -- so no
        // exchange-ordered tensor is built. (Its pairing is full rank, so
        // there would be nothing to gain by compressing it anyway.)
        const size_t Nel = radial_.Nel();
        prim_chol.assign((size_t) N_L * Nel, helfem::Mat<T>());
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
        for (int L = 0; L < N_L; ++L)
          for (size_t iel = 0; iel < Nel; ++iel)
            prim_chol[(size_t) L * Nel + iel] =
                radial_.twoe_integral_cholesky(L, iel, chol_tol);

        (void) exchange;
      }

      template <typename T>
      void TwoDBasisT<T>::compute_yukawa(T lambda_) {
        lambda = lambda_;
        yukawa = true;
        const int N_L = 2 * lval_.maxCoeff() + 1;
        // Yukawa-mode disjoint factors (bessel i / bessel k) + in-element
        // Yukawa 2e.
        atomic::basis::compute_disjoint_radial_integrals(
            radial_, N_L, disjoint_iL, disjoint_kL, /*yukawa=*/true, lambda);

        // In-element Yukawa integrals, factorized exactly as the bare ones.
        // The rank bound is a property of the orbital product basis, not of the
        // kernel, so it carries over unchanged.
        const size_t Nel = radial_.Nel();
        rs_chol.assign((size_t) N_L * Nel, helfem::Mat<T>());
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
        for (int L = 0; L < N_L; ++L)
          for (size_t iel = 0; iel < Nel; ++iel)
            rs_chol[(size_t) L * Nel + iel] =
                radial_.yukawa_integral_cholesky(L, lambda, iel, chol_tol);
      }

      template <typename T>
      void TwoDBasisT<T>::compute_erfc(T mu) {
        lambda = mu;
        yukawa = false;
        // Erfc kernel doesn't factorise -- no disjoint integrals; all
        // (iel, jel) pairs are stored explicitly in rs_ktei.
        disjoint_iL.clear();
        disjoint_kL.clear();
        const int N_L = 2 * lval_.maxCoeff() + 1;
        atomic::basis::compute_erfc_ktei(radial_, N_L, lambda, rs_ktei);
      }

      template <typename T>
      helfem::Mat<T> TwoDBasisT<T>::coulomb(const helfem::Mat<T> & P0_in) const {
        if(!prim_chol.size())
          throw std::logic_error("Primitive teis have not been computed!\n");

        // The atomic basis has no boundary reduction (expand/remove_boundaries
        // are the identity), so the SCF-facing Eigen density is used directly.
        const helfem::Mat<T> & P = P0_in;

        // Number of radial elements
        size_t Nel(radial_.Nel());
        // Number of radial functions
        const Eigen::Index Nrad = static_cast<Eigen::Index>(radial_.Nbf());
        // Gaunt coefficient table
        int gmax(std::max(lval_.maxCoeff(),mval_.maxCoeff()));
        gaunt::GauntT<T> gaunt(gmax,2*gmax,gmax);

        // maximal M value
        int Mmax=mval_.maxCoeff()-mval_.minCoeff();

        // Radial helper matrices
        std::vector< std::vector<helfem::Mat<T>> > Paux(2*lval_.maxCoeff()+1);
        for(int L=0;L<(int) Paux.size();L++) {
          Paux[L].resize(2*Mmax+1);
          for(int M=-std::min(L,Mmax);M<=std::min(L,Mmax);M++) {
            Paux[L][M+Mmax] = helfem::Mat<T>::Zero(Nrad,Nrad);
          }
        }

        // Form radial helpers: contract ket
        for(size_t kang=0;kang<(size_t) lval_.size();kang++) {
          for(size_t lang=0;lang<(size_t) lval_.size();lang++) {
            // l and m values
            int lk(lval_(kang));
            int mk(mval_(kang));
            int ll(lval_(lang));
            int ml(mval_(lang));
            // RH m value
            int M(mk-ml);
            // M values match. Loop over possible couplings
            int Lmin=std::max(std::abs(lk-ll),abs(M));
            int Lmax=lk+ll;
            for(int L=Lmin;L<=Lmax;L++) {
              // Calculate coupling coefficient
              T cpl(gaunt.coeff(lk,mk,L,M,ll));
              // Increment
              Paux[L][M+Mmax]+=cpl*P.block(static_cast<Eigen::Index>(kang)*Nrad,static_cast<Eigen::Index>(lang)*Nrad,Nrad,Nrad);
            }
          }
        }

        // Helper matrices
        std::vector< std::vector<helfem::Mat<T>> > Jaux(2*lval_.maxCoeff()+1);
        for(int L=0;L<(int) Jaux.size();L++) {
          Jaux[L].resize(2*Mmax+1);
          for(int M=-std::min(L,Mmax);M<=std::min(L,Mmax);M++) {
            Jaux[L][M+Mmax] = helfem::Mat<T>::Zero(Nrad,Nrad);
          }
        }
        // Contract integrals. Per (L, M), delegate the FE assembly to
        // the shared helper with our SCF-cached per-(L, iel) integrals.
        for(int L=0;L<(int) Paux.size();L++) {
          const T Lfac=T(4)*utils::pi<T>()/T(2*L+1);
          auto rs = [&,L](size_t iel) -> const helfem::Mat<T> & {
            return disjoint_L[L*Nel+iel];
          };
          auto rb = [&,L](size_t iel) -> const helfem::Mat<T> & {
            return disjoint_m1L[L*Nel+iel];
          };
          auto tw = [&,L](size_t iel) -> const helfem::Mat<T> & {
            return prim_chol[(size_t) L * Nel + iel];
          };
          for(int M=-std::min(L,Mmax);M<=std::min(L,Mmax);M++) {
            Jaux[L][M+Mmax] += Lfac *
              helfem::atomic::basis::assemble_J_FE_one_multipole_cached_chol(
                radial_, rs, rb, tw, Paux[L][M+Mmax]);
          }
        }

        // Full Coulomb matrix
        helfem::Mat<T> J = helfem::Mat<T>::Zero(Ndummy(),Ndummy());
        for(size_t iang=0;iang<(size_t) lval_.size();iang++) {
          for(size_t jang=0;jang<(size_t) lval_.size();jang++) {
            // l and m values
            int li(lval_(iang));
            int mi(mval_(iang));
            int lj(lval_(jang));
            int mj(mval_(jang));
            // LH m value
            int M(mj-mi);

            int Lmin=std::max(std::abs(lj-li),abs(M));
            int Lmax=lj+li;
            for(int L=Lmin;L<=Lmax;L++) {
              // Coupling
              T cpl(gaunt.coeff(lj,mj,L,M,li));
              if(cpl!=T(0)) {
                J.block(static_cast<Eigen::Index>(iang)*Nrad,static_cast<Eigen::Index>(jang)*Nrad,Nrad,Nrad)+=cpl*Jaux[L][M+Mmax];
              }
            }
          }
        }

        return J;
      }

      template <typename T>
      helfem::Mat<T> TwoDBasisT<T>::exchange(const helfem::Mat<T> & P0_in) const {
        if(!prim_chol.size())
          throw std::logic_error("Primitive teis have not been computed!\n");

        // No boundary reduction in the atomic basis (expand/remove are
        // the identity), so the Eigen density is used directly.
        const helfem::Mat<T> & P = P0_in;

        // Gaunt coefficient table
        int gmax(std::max(lval_.maxCoeff(),mval_.maxCoeff()));
        gaunt::GauntT<T> gaunt(gmax,2*gmax,gmax);

        // Number of radial elements
        size_t Nel(radial_.Nel());
        // Number of radial basis functions
        const Eigen::Index Nrad = static_cast<Eigen::Index>(radial_.Nbf());

        // Density-block screening threshold. eps(T) rather than DBL_EPSILON:
        // at T = double this is the same 10*DBL_EPSILON it always was, and at
        // higher T it tightens with the arithmetic instead of throwing away
        // blocks that are now perfectly resolvable.
        const T bdens_thr = T(10)*std::numeric_limits<T>::epsilon();

        // Full exchange matrix
        helfem::Mat<T> K = helfem::Mat<T>::Zero(Ndummy(),Ndummy());

        // Per-element K assembly is delegated to
        // assemble_K_FE_one_multipole_cached_chol -- no per-thread scratch
        // needed here, the helper allocates its own.
#ifdef _OPENMP
#pragma omp parallel
#endif
        {
#ifdef _OPENMP
#pragma omp for collapse(2)
#endif
          for(size_t jang=0;jang<(size_t) lval_.size();jang++) {
            for(size_t kang=0;kang<(size_t) lval_.size();kang++) {
              int lj(lval_(jang));
              int mj(mval_(jang));

              int lk(lval_(kang));
              int mk(mval_(kang));

              // Form radial helpers
              size_t N_L(2*lval_.maxCoeff()+1);
              std::vector<helfem::Mat<T>> Rmat(N_L);
              for(size_t i=0;i<N_L;i++) {
                Rmat[i] = helfem::Mat<T>::Zero(Nrad,Nrad);
              }
              // Is there a coupling to the channel?
              std::vector<bool> couple(N_L,false);

              // Perform angular sums
              for(size_t iang=0;iang<(size_t) lval_.size();iang++) {
                int li(lval_(iang));
                int mi(mval_(iang));

                for(size_t lang=0;lang<(size_t) lval_.size();lang++) {
                  int ll(lval_(lang));
                  int ml(mval_(lang));

                  // LH m value
                  int M(mj-mi);
                  // RH m value
                  int Mp(mk-ml);
                  if(M!=Mp)
                    continue;

                  // Do we have any density in this block?
                  T bdens(P.block(static_cast<Eigen::Index>(iang)*Nrad,static_cast<Eigen::Index>(lang)*Nrad,Nrad,Nrad).norm());
                  if(bdens<bdens_thr)
                    continue;

                  // M values match. Loop over possible couplings
                  int Lmin=std::max(std::max(std::abs(li-lj),std::abs(lk-ll)),abs(M));
                  int Lmax=std::min(li+lj,lk+ll);

                  for(int L=Lmin;L<=Lmax;L++) {
                    // Calculate total coupling coefficient
                    T cpl(gaunt.coeff(lj,mj,L,M,li)*gaunt.coeff(lk,mk,L,M,ll));
                    if(cpl==T(0))
                      continue;

                    // L factor
                    T Lfac=T(4)*utils::pi<T>()/T(2*L+1);
                    Rmat[L]+=(Lfac*cpl)*P.block(static_cast<Eigen::Index>(iang)*Nrad,static_cast<Eigen::Index>(lang)*Nrad,Nrad,Nrad);
                    couple[L]=true;
                  }
                }
              }

              // Per-L K assembly via the shared FE helper, accumulated
              // into the (jang, kang) angular block of K.
              helfem::Mat<T> K_block = helfem::Mat<T>::Zero(Nrad, Nrad);
              for(size_t L=0; L<N_L; ++L) {
                if(!couple[L]) continue;
                const size_t Lc = L;
                auto rs = [&,Lc](size_t iel) -> const helfem::Mat<T> & {
                  return disjoint_L[Lc*Nel+iel];
                };
                auto rb = [&,Lc](size_t iel) -> const helfem::Mat<T> & {
                  return disjoint_m1L[Lc*Nel+iel];
                };
                auto kt = [&,Lc](size_t iel) -> const helfem::Mat<T> & {
                  return prim_chol[(size_t) Lc * Nel + iel];
                };
                // RI-K on the J-ordered Cholesky factor: K_ac = sum_p M_p P M_p'.
                // The exchange-ordered tensor is full rank, so this is the only
                // way to compress K -- and it means none needs to be stored.
                K_block += helfem::atomic::basis::assemble_K_FE_one_multipole_cached_chol(
                    radial_, rs, rb, kt, Rmat[Lc]);
              }
              K.block(static_cast<Eigen::Index>(jang)*Nrad, static_cast<Eigen::Index>(kang)*Nrad, Nrad, Nrad) -= K_block;
            }
          }
        }

        return K;
      }

      template <typename T>
      helfem::Mat<T> TwoDBasisT<T>::rs_exchange(const helfem::Mat<T> & P0_in) const {
        if(!rs_ktei.size() && !rs_chol.size())
          throw std::logic_error("Primitive teis have not been computed!\n");

        // No boundary reduction in the atomic basis (expand/remove are
        // the identity), so the Eigen density is used directly.
        const helfem::Mat<T> & P = P0_in;

        // Gaunt coefficient table
        int gmax(std::max(lval_.maxCoeff(),mval_.maxCoeff()));
        gaunt::GauntT<T> gaunt(gmax,2*gmax,gmax);

        // Number of radial elements
        size_t Nel(radial_.Nel());
        // Number of radial basis functions
        const Eigen::Index Nrad = static_cast<Eigen::Index>(radial_.Nbf());

        const T bdens_thr = T(10)*std::numeric_limits<T>::epsilon();

        // Full exchange matrix
        helfem::Mat<T> K = helfem::Mat<T>::Zero(Ndummy(),Ndummy());

        // Per-element K assembly is delegated to the cached helpers
        // in CoulombExchangeFE.h -- no per-thread scratch needed.
#ifdef _OPENMP
#pragma omp parallel
#endif
        {
#ifdef _OPENMP
#pragma omp for collapse(2)
#endif
          for(size_t jang=0;jang<(size_t) lval_.size();jang++) {
            for(size_t kang=0;kang<(size_t) lval_.size();kang++) {
              int lj(lval_(jang));
              int mj(mval_(jang));

              int lk(lval_(kang));
              int mk(mval_(kang));

              // Form radial helpers
              size_t N_L(2*lval_.maxCoeff()+1);
              std::vector<helfem::Mat<T>> Rmat(N_L);
              for(size_t i=0;i<N_L;i++) {
                Rmat[i] = helfem::Mat<T>::Zero(Nrad,Nrad);
              }
              // Is there a coupling to the channel?
              std::vector<bool> couple(N_L,false);

              // Perform angular sums
              for(size_t iang=0;iang<(size_t) lval_.size();iang++) {
                int li(lval_(iang));
                int mi(mval_(iang));

                for(size_t lang=0;lang<(size_t) lval_.size();lang++) {
                  int ll(lval_(lang));
                  int ml(mval_(lang));

                  // LH m value
                  int M(mj-mi);
                  // RH m value
                  int Mp(mk-ml);
                  if(M!=Mp)
                    continue;

                  // Do we have any density in this block?
                  T bdens(P.block(static_cast<Eigen::Index>(iang)*Nrad,static_cast<Eigen::Index>(lang)*Nrad,Nrad,Nrad).norm());
                  if(bdens<bdens_thr)
                    continue;

                  // M values match. Loop over possible couplings
                  int Lmin=std::max(std::max(std::abs(li-lj),std::abs(lk-ll)),abs(M));
                  int Lmax=std::min(li+lj,lk+ll);

                  for(int L=Lmin;L<=Lmax;L++) {
                    // Calculate total coupling coefficient
                    T cpl(gaunt.coeff(lj,mj,L,M,li)*gaunt.coeff(lk,mk,L,M,ll));
                    if(cpl==T(0))
                      continue;

                    // L factor
                    T Lfac = yukawa ? T(4)*utils::pi<T>()*lambda
                                    : T(4)*utils::pi<T>()*lambda/T(2*L+1);
                    Rmat[L]+=(Lfac*cpl)*P.block(static_cast<Eigen::Index>(iang)*Nrad,static_cast<Eigen::Index>(lang)*Nrad,Nrad,Nrad);
                    couple[L]=true;
                  }
                }
              }

              if (yukawa) {
                // Same FE structure as bare exchange: per-L cached
                // helper, summed into the (jang, kang) angular block.
                helfem::Mat<T> K_block = helfem::Mat<T>::Zero(Nrad, Nrad);
                for(size_t L=0; L<N_L; ++L) {
                  if(!couple[L]) continue;
                  const size_t Lc = L;
                  auto rs = [&,Lc](size_t iel) -> const helfem::Mat<T> & {
                    return disjoint_iL[Lc*Nel+iel];
                  };
                  auto rb = [&,Lc](size_t iel) -> const helfem::Mat<T> & {
                    return disjoint_kL[Lc*Nel+iel];
                  };
                  auto kt = [&,Lc](size_t iel) -> const helfem::Mat<T> & {
                    return rs_chol[(size_t) Lc * Nel + iel];
                  };
                  K_block += helfem::atomic::basis::assemble_K_FE_one_multipole_cached_chol(
                      radial_, rs, rb, kt, Rmat[Lc]);
                }
                K.block(static_cast<Eigen::Index>(jang)*Nrad, static_cast<Eigen::Index>(kang)*Nrad, Nrad, Nrad) -= K_block;
              } else {
                // Erfc: rs_ktei has cross-element entries for every
                // (iel, jel) pair -- delegate to the pairwise cached
                // helper, summed over L into the (jang, kang) block.
                helfem::Mat<T> K_block = helfem::Mat<T>::Zero(Nrad, Nrad);
                for(size_t L=0; L<N_L; ++L) {
                  if(!couple[L]) continue;
                  const size_t Lc = L;
                  auto kt = [&,Lc](size_t iel, size_t jel) -> const helfem::Mat<T> & {
                    return rs_ktei[Nel*Nel*Lc + iel*Nel + jel];
                  };
                  K_block += helfem::atomic::basis::assemble_K_FE_one_multipole_cached_pairwise(
                      radial_, kt, Rmat[Lc]);
                }
                K.block(static_cast<Eigen::Index>(jang)*Nrad, static_cast<Eigen::Index>(kang)*Nrad, Nrad, Nrad) -= K_block;
              }
            }
          }
        }

        return K;
      }

      template <typename T>
      Eigen::MatrixXcd TwoDBasisT<T>::eval_bf(size_t iel, double cth, double phi) const {
        if constexpr (std::is_same_v<T, double>) {
          // Evaluate spherical harmonics
          Eigen::VectorXcd sph(lval_.size());
          for(size_t i=0;i<(size_t) lval_.size();i++)
            sph(i)=::spherical_harmonics(lval_(i),mval_(i),cth,phi);

          // Evaluate radial functions
          helfem::Matrix rad(radial_.bf(iel));

          // Form supermatrix
          Eigen::MatrixXcd bf(rad.rows(),lval_.size()*rad.cols());
          for(size_t i=0;i<(size_t) lval_.size();i++)
            bf.middleCols(i*rad.cols(),rad.cols())=sph(i)*rad.cast<std::complex<double>>();

          return bf;
        } else {
          (void) iel; (void) cth; (void) phi;
          double_only("eval_bf");
        }
      }

      template <typename T>
      void TwoDBasisT<T>::eval_df(size_t iel, double cth, double phi, Eigen::MatrixXcd & dr, Eigen::MatrixXcd & dth, Eigen::MatrixXcd & dphi) const {
        if constexpr (std::is_same_v<T, double>) {
          // Evaluate spherical harmonics
          Eigen::VectorXcd sph(lval_.size());
          for(size_t i=0;i<(size_t) lval_.size();i++)
            sph(i)=::spherical_harmonics(lval_(i),mval_(i),cth,phi);

          // Evaluate radial functions
          helfem::Matrix frad(radial_.bf(iel));
          helfem::Matrix drad(radial_.df(iel));

          // Form supermatrices
          dr=Eigen::MatrixXcd::Zero(frad.rows(),lval_.size()*frad.cols());
          dth=Eigen::MatrixXcd::Zero(frad.rows(),lval_.size()*frad.cols());
          dphi=Eigen::MatrixXcd::Zero(frad.rows(),lval_.size()*frad.cols());

          // Radial one is easy
          for(size_t i=0;i<(size_t) lval_.size();i++)
            dr.middleCols(i*frad.cols(),frad.cols())=sph(i)*drad.cast<std::complex<double>>();
          // and so is phi
          for(size_t i=0;i<(size_t) lval_.size();i++)
            dphi.middleCols(i*frad.cols(),frad.cols())=(std::complex<double>(0.0,mval_(i))*sph(i))*frad.cast<std::complex<double>>();
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
          for(size_t i=0;i<(size_t) lval_.size();i++) {
            int l(lval_(i));
            int m(mval_(i));

            // Angular factor
            std::complex<double> angfac(m*cotth*sph(i));
            if(mval_(i)<lval_(i))
              angfac+=sqrt((l-m)*(l+m+1))*std::exp(std::complex<double>(0,-phi))*::spherical_harmonics(lval_(i),mval_(i)+1,cth,phi);

            dth.middleCols(i*frad.cols(),frad.cols())=angfac*frad.cast<std::complex<double>>();
          }
        } else {
          (void) iel; (void) cth; (void) phi; (void) dr; (void) dth; (void) dphi;
          double_only("eval_df");
        }
      }

      template <typename T>
      Eigen::MatrixXcd TwoDBasisT<T>::eval_lf(size_t iel, double cth, double phi) const {
        if constexpr (std::is_same_v<T, double>) {
          // Evaluate spherical harmonics
          Eigen::VectorXcd sph(lval_.size());
          for(size_t i=0;i<(size_t) lval_.size();i++)
            sph(i)=::spherical_harmonics(lval_(i),mval_(i),cth,phi);

          // Evaluate radial functions
          helfem::Vector r(radial_.r(iel));
          helfem::Matrix frad(radial_.bf(iel));
          helfem::Matrix drad(radial_.df(iel));
          helfem::Matrix lrad(radial_.lf(iel));

          // Form supermatrix
          Eigen::MatrixXcd lf(Eigen::MatrixXcd::Zero(frad.rows(),lval_.size()*frad.cols()));
          // Loop over basis function indices
          for(size_t iang=0;iang<(size_t) lval_.size();iang++)
            for(Eigen::Index irad=0;irad<frad.cols();irad++)
              // Loop over grid-point indices
              for(Eigen::Index igrid=0;igrid<frad.rows();igrid++)
                lf(igrid,iang*frad.cols()+irad) = (lrad(igrid,irad) + 2*drad(igrid,irad)/r(igrid) - lval_(iang)*(lval_(iang)+1)*frad(igrid,irad)/(r(igrid)*r(igrid)))*sph(iang);

          return lf;
        } else {
          (void) iel; (void) cth; (void) phi;
          double_only("eval_lf");
        }
      }

      template <typename T>
      std::vector<Eigen::Index> TwoDBasisT<T>::bf_list(size_t iel) const {
        // Radial functions in element
        size_t ifirst, ilast;
        radial_.idx(iel,ifirst,ilast);
        // Number of radial functions in element
        size_t Nr(ilast-ifirst+1);

        // Total number of radial functions
        size_t Nradf(radial_.Nbf());

        // List of functions in the element
        std::vector<Eigen::Index> idx(Nr*lval_.size());
        for(size_t iam=0;iam<(size_t) lval_.size();iam++)
          for(size_t j=0;j<Nr;j++)
            idx[iam*Nr+j]=(Eigen::Index) (Nradf*iam+ifirst+j);

        return idx;
      }

      template <typename T>
      size_t TwoDBasisT<T>::rad_Nel() const {
        return radial_.Nel();
      }

      template <typename T>
      std::pair<size_t, size_t>
      TwoDBasisT<T>::radial_element_range(size_t iel) const {
        size_t ifirst, ilast;
        radial_.idx(iel, ifirst, ilast);
        return {ifirst, ilast};
      }

      template <typename T>
      helfem::Vec<T> TwoDBasisT<T>::wrad(size_t iel) const {
        return radial_.wrad(iel);
      }

      template <typename T>
      helfem::Vec<T> TwoDBasisT<T>::r(size_t iel) const {
        return radial_.r(iel);
      }

      template class TwoDBasisT<double>;
      template class TwoDBasisT<long double>;
#ifdef HELFEM_HAVE_FLOAT128
      template class TwoDBasisT<_Float128>;
#endif

    }
  }
}
