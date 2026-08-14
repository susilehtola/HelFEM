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

#include "twodquadrature.h"
#include "../general/atomdb.h"
#include "chebyshev.h"
#include "../general/lcao.h"
#include "../general/spherical_harmonics.h"
#include "../general/model_potential.h"
#include "../sadatom/scf.h"
#include "utils.h"
#include <algorithm>
#include <map>
#include <limits>
#include <cmath>

// PBE ground states determined with 10 radial elements
int pbe_ground_states[118][4] = {
  { 1, 0, 0, 0},     //   1  H
  { 2, 0, 0, 0},     //   2  He
  { 3, 0, 0, 0},     //   3  Li
  { 4, 0, 0, 0},     //   4  Be
  { 4, 1, 0, 0},     //   5  B
  { 4, 2, 0, 0},     //   6  C
  { 4, 3, 0, 0},     //   7  N
  { 4, 4, 0, 0},     //   8  O
  { 4, 5, 0, 0},     //   9  F
  { 4, 6, 0, 0},     //  10  Ne
  { 5, 6, 0, 0},     //  11  Na
  { 6, 6, 0, 0},     //  12  Mg
  { 6, 7, 0, 0},     //  13  Al
  { 6, 8, 0, 0},     //  14  Si
  { 6, 9, 0, 0},     //  15  P
  { 6,10, 0, 0},     //  16  S
  { 6,11, 0, 0},     //  17  Cl
  { 6,12, 0, 0},     //  18  Ar
  { 7,12, 0, 0},     //  19  K
  { 8,12, 0, 0},     //  20  Ca
  { 8,13, 0, 0},     //  21  Sc
  { 8,12, 2, 0},     //  22  Ti
  { 8,12, 3, 0},     //  23  V
  { 8,12, 4, 0},     //  24  Cr
  { 6,12, 7, 0},     //  25  Mn
  { 6,12, 8, 0},     //  26  Fe
  { 6,12, 9, 0},     //  27  Co
  { 6,12,10, 0},     //  28  Ni
  { 7,12,10, 0},     //  29  Cu
  { 8,12,10, 0},     //  30  Zn
  { 8,13,10, 0},     //  31  Ga
  { 8,14,10, 0},     //  32  Ge
  { 8,15,10, 0},     //  33  As
  { 8,16,10, 0},     //  34  Se
  { 8,17,10, 0},     //  35  Br
  { 8,18,10, 0},     //  36  Kr
  { 9,18,10, 0},     //  37  Rb
  {10,18,10, 0},     //  38  Sr
  {10,19,10, 0},     //  39  Y
  {10,18,12, 0},     //  40  Zr
  {10,18,13, 0},     //  41  Nb
  { 8,18,16, 0},     //  42  Mo
  { 8,18,17, 0},     //  43  Tc
  { 8,18,18, 0},     //  44  Ru
  { 8,18,19, 0},     //  45  Rh
  { 8,18,20, 0},     //  46  Pd
  { 9,18,20, 0},     //  47  Ag
  {10,18,20, 0},     //  48  Cd
  {10,19,20, 0},     //  49  In
  {10,20,20, 0},     //  50  Sn
  {10,21,20, 0},     //  51  Sb
  {10,22,20, 0},     //  52  Te
  {10,23,20, 0},     //  53  I
  {10,24,20, 0},     //  54  Xe
  {11,24,20, 0},     //  55  Cs
  {12,24,20, 0},     //  56  Ba
  {12,24,21, 0},     //  57  La
  {12,24,22, 0},     //  58  Ce
  {12,24,21, 2},     //  59  Pr
  {12,24,20, 4},     //  60  Nd
  {12,24,20, 5},     //  61  Pm
  {12,24,20, 6},     //  62  Sm
  {12,24,20, 7},     //  63  Eu
  {11,24,20, 9},     //  64  Gd
  {10,24,20,11},     //  65  Tb
  {10,24,20,12},     //  66  Dy
  {10,24,20,13},     //  67  Ho
  {10,24,20,14},     //  68  Er
  {11,24,20,14},     //  69  Tm
  {12,24,20,14},     //  70  Yb
  {12,25,20,14},     //  71  Lu
  {12,24,22,14},     //  72  Hf
  {12,24,23,14},     //  73  Ta
  {10,24,26,14},     //  74  W
  {10,24,27,14},     //  75  Re
  {10,24,28,14},     //  76  Os
  {10,24,29,14},     //  77  Ir
  {10,24,30,14},     //  78  Pt
  {11,24,30,14},     //  79  Au
  {12,24,30,14},     //  80  Hg
  {12,25,30,14},     //  81  Tl
  {12,26,30,14},     //  82  Pb
  {12,27,30,14},     //  83  Bi
  {12,28,30,14},     //  84  Po
  {12,29,30,14},     //  85  At
  {12,30,30,14},     //  86  Rn
  {13,30,30,14},     //  87  Fr
  {14,30,30,14},     //  88  Ra
  {14,30,31,14},     //  89  Ac
  {14,30,32,14},     //  90  Th
  {14,30,30,17},     //  91  Pa
  {14,30,30,18},     //  92  U
  {14,30,30,19},     //  93  Np
  {13,30,30,21},     //  94  Pu
  {12,30,30,23},     //  95  Am
  {12,30,30,24},     //  96  Cm
  {12,30,30,25},     //  97  Bk
  {12,30,30,26},     //  98  Cf
  {12,30,30,27},     //  99  Es
  {12,30,30,28},     // 100  Fm
  {13,30,30,28},     // 101  Md
  {14,30,30,28},     // 102  No
  {14,30,31,28},     // 103  Lr
  {14,30,32,28},     // 104  Rf
  {14,30,33,28},     // 105  Db
  {12,30,36,28},     // 106  Sg
  {12,30,37,28},     // 107  Bh
  {12,30,38,28},     // 108  Hs
  {12,30,39,28},     // 109  Mt
  {12,30,40,28},     // 110  Ds
  {13,30,40,28},     // 111  Rg
  {14,30,40,28},     // 112  Cn
  {14,31,40,28},     // 113  Nh
  {14,32,40,28},     // 114  Fl
  {14,33,40,28},     // 115  Mc
  {14,34,40,28},     // 116  Lv
  {14,35,40,28},     // 117  Ts
  {14,36,40,28}     // 118  Og
};

namespace helfem {
  namespace diatomic {
    namespace twodquad {
      TwoDGridWorker::TwoDGridWorker() {
      }

      TwoDGridWorker::TwoDGridWorker(const helfem::diatomic::basis::TwoDBasis * basp_, int lang) : basp(basp_) {
        // Get angular grid (chebyshev shim is Eigen-typed)
        chebyshev::chebyshev(lang,cth,wang);
      }

      TwoDGridWorker::~TwoDGridWorker() {
      }

      void TwoDGridWorker::compute_bf(size_t iel, size_t irad, int m_) {
        // Store m
        m=m_;
        // Update function list
        bf_ind=basp->bf_list_dummy(iel,m);

        // Get radial weights. Only do one radial quadrature point at a
        // time, since this is an easy way to save a lot of memory.
        r=helfem::Vector::Zero(1);
        r(0)=basp->r(iel)(irad);
        wrad=helfem::Vector::Zero(1);
        wrad(0)=basp->wrad(iel)(irad);

        double Rhalf(basp->Rhalf());

        // Calculate helpers
        helfem::Vector shmu(r.array().sinh());

        helfem::Vector sth(cth.size());
        for(size_t ia=0;ia<(size_t) cth.size();ia++)
          sth(ia)=sqrt(1.0 - cth(ia)*cth(ia));

        // Update total weights
        wtot=helfem::Vector::Zero(wrad.size()*wang.size());
        for(size_t ia=0;ia<(size_t) wang.size();ia++)
          for(size_t ir=0;ir<(size_t) wrad.size();ir++) {
            size_t idx=ia*wrad.size()+ir;
            // sin(th) is already contained within wang, but we don't want to divide by it since it may be zero. Phi integrals yield 2 pi
            wtot(idx)=2.0*M_PI*wang(ia)*wrad(ir)*std::pow(Rhalf,3)*shmu(ir)*(std::pow(shmu(ir),2)+std::pow(sth(ia),2));
          }

        // Compute basis function values
        bf=helfem::Matrix::Zero(bf_ind.size(),wtot.size());

        // The element's FEM polynomials depend only on the element, so
        // evaluate them ONCE here rather than once per angular point:
        // eval_bf(iel,irad,...) evaluates the whole element at every
        // quadrature point and then keeps a single row, so calling it inside
        // the angular loop redid that work cth.size() times over. Hoisted
        // above the parallel region, which also keeps it read-only shared.
        const helfem::Matrix rad_all(basp->rad_bf(iel));

        // Loop over angular grid
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(size_t ia=0;ia<(size_t) cth.size();ia++) {
          // Evaluate basis functions at angular point
          helfem::Matrix abf(basp->eval_bf(iel, irad, cth(ia), m, rad_all));
          if((size_t) abf.cols() != bf_ind.size()) {
            std::ostringstream oss;
            oss << "Mismatch! Have " << bf_ind.size() << " basis function indices but " << abf.cols() << " basis functions!\n";
            throw std::logic_error(oss.str());
          }
          // Store functions
          bf.middleCols(ia*wrad.size(),wrad.size())=abf.transpose();
        }
      }

      void TwoDGridWorker::model_potential(const modelpotential::ModelPotential * p1, const modelpotential::ModelPotential * p2) {
        double Rhalf(basp->Rhalf());
        helfem::Vector chmu(r.array().cosh());

        itg=helfem::Matrix::Zero(1,wtot.size());
        for(size_t ia=0;ia<(size_t) wang.size();ia++)
          for(size_t ir=0;ir<(size_t) wrad.size();ir++) {
            size_t idx=ia*wrad.size()+ir;

            double r1=Rhalf*(chmu(ir) + cth(ia));
            double r2=Rhalf*(chmu(ir) - cth(ia));

	    double V1(p1->V(r1));
	    double V2(p2->V(r2));
	    if(std::isnormal(V1))
	      itg(0,idx)+=V1;
	    if(std::isnormal(V2))
	      itg(0,idx)+=V2;
          }
      }

      void TwoDGridWorker::multiply_Plm(int l, int m, probe_t p) {
        helfem::Vector chmu(r.array().cosh());
        helfem::Vector shmu(r.array().sinh());

        // The cthval rationals can drift slightly outside [-1, 1] under
        // round-off near the cusps; clamp before std::acos so we never
        // feed it a NaN.
        auto eval_Plm = [l, m](double cthval) {
          return std::sph_legendre(static_cast<unsigned>(l),
                                   static_cast<unsigned>(std::abs(m)),
                                   std::acos(std::clamp(cthval, -1.0, 1.0)));
        };

        if(p==PROBE_LEFT) {
          for(size_t ia=0;ia<wang.size();ia++)
            for(size_t ir=0;ir<wrad.size();ir++) {
              size_t idx=ia*wrad.size()+ir;
              double cthval = (1.0 + chmu(ir)*cth(ia))/(chmu(ir) + cth(ia));
              double Plm = eval_Plm(cthval);
              for(size_t ix=0;ix<itg.rows();ix++)
                itg(ix,idx)*=Plm;
            }

        } else if(p==PROBE_RIGHT) {
          for(size_t ia=0;ia<wang.size();ia++)
            for(size_t ir=0;ir<wrad.size();ir++) {
              size_t idx=ia*wrad.size()+ir;
              double cthval = (1.0 - chmu(ir)*cth(ia))/(chmu(ir) - cth(ia));
              double Plm = eval_Plm(cthval);
              for(size_t ix=0;ix<itg.rows();ix++)
                itg(ix,idx)*=Plm;
            }

        } else if(p==PROBE_MIDDLE) {
          for(size_t ia=0;ia<wang.size();ia++)
            for(size_t ir=0;ir<wrad.size();ir++) {
              size_t idx=ia*wrad.size()+ir;
              double cthval = cth(ia);
              double Plm = eval_Plm(cthval);
              for(size_t ix=0;ix<itg.rows();ix++)
                itg(ix,idx)*=Plm;
            }
        }
      }

      void TwoDGridWorker::ao_projection(const std::function<helfem::Vector(double r)> & compute_ao, probe_t p) {
        double Rhalf(basp->Rhalf());
        helfem::Vector chmu(r.array().cosh());

        itg=helfem::Matrix::Zero(compute_ao(0.0).size(),wtot.size());
        if(p==PROBE_LEFT) {
          for(size_t ia=0;ia<wang.size();ia++)
            for(size_t ir=0;ir<wrad.size();ir++) {
              size_t idx=ia*wrad.size()+ir;

              double ra(Rhalf*(chmu(ir) + cth(ia)));
              itg.col(idx) = compute_ao(ra);
            }

        } else if(p==PROBE_RIGHT) {
          for(size_t ia=0;ia<wang.size();ia++)
            for(size_t ir=0;ir<wrad.size();ir++) {
              size_t idx=ia*wrad.size()+ir;

              double rb(Rhalf*(chmu(ir) - cth(ia)));
              itg.col(idx) = compute_ao(rb);
            }

        } else if(p==PROBE_MIDDLE) {
          for(size_t ia=0;ia<wang.size();ia++)
            for(size_t ir=0;ir<wrad.size();ir++) {
              size_t idx=ia*wrad.size()+ir;

              // chmu^2 + cth^2 - 1 = sinh^2(mu) + cth^2 is mathematically
              // non-negative but can underflow to a tiny negative under
              // round-off (small mu, small cth); clamp before sqrt.
              double rc(Rhalf*std::sqrt(std::max(chmu(ir)*chmu(ir) + cth(ia)*cth(ia) - 1.0, 0.0)));
              itg.col(idx) = compute_ao(rc);
            }
        }
      }

      void TwoDGridWorker::gto(int l, const helfem::Vector & expn, probe_t p) {
        std::function<helfem::Vector(double r)> compute_gto = [expn, l](double r) {
          helfem::Vector f(expn.size());
          for(size_t ix=0;ix<(size_t) expn.size();ix++)
            f(ix)=lcao::radial_GTO(r,l,expn(ix));
          return f;
        };
        ao_projection(compute_gto, p);
      }

      void TwoDGridWorker::sto(int l, const helfem::Vector & expn, probe_t p) {
        std::function<helfem::Vector(double r)> compute_sto = [expn, l](double r) {
          helfem::Vector f(expn.size());
          for(size_t ix=0;ix<(size_t) expn.size();ix++)
            f(ix)=lcao::radial_STO(r,l,expn(ix));
          return f;
        };
        ao_projection(compute_sto, p);
      }

      void TwoDGridWorker::eval_pot(helfem::Matrix & Vo) const {
        if(itg.rows() != 1)
          throw std::logic_error("Should only have one column in integrand!\n");
        // Elementwise product of the (single-row) integrand with the total
        // quadrature weights, used as the diagonal of the weighting matrix.
        helfem::Vector w = itg.row(0).transpose().array() * wtot.array();
        Vo(bf_ind,bf_ind)+=bf*w.asDiagonal()*bf.transpose();
      }

      void TwoDGridWorker::eval_proj(helfem::Matrix & Vo) const {
        Vo(Eigen::all,bf_ind)+=itg*wtot.asDiagonal()*bf.transpose();
      }

      void TwoDGridWorker::eval_proj_overlap(helfem::Matrix & Vo) const {
        Vo+=itg*wtot.asDiagonal()*itg.transpose();
      }

      TwoDGrid::TwoDGrid() {
      }

      TwoDGrid::TwoDGrid(const helfem::diatomic::basis::TwoDBasis * basp_, int lang_) : basp(basp_), lang(lang_) {
      }

      TwoDGrid::~TwoDGrid() {
      }

      helfem::Matrix TwoDGrid::model_potential(const modelpotential::ModelPotential * p1, const modelpotential::ModelPotential * p2) {
        helfem::Matrix H = helfem::Matrix::Zero(basp->Ndummy(),basp->Ndummy());

        // Get unique m values in basis set. eval_pot accumulates into H
        // additively per m, so the iteration order does not affect the result.
        const Eigen::VectorXi mvals(basp->mval());
        std::vector<int> muni;
        for(Eigen::Index i=0;i<mvals.size();i++)
          if(std::find(muni.begin(),muni.end(),mvals(i))==muni.end())
            muni.push_back(mvals(i));
        std::sort(muni.begin(),muni.end());
        {
          TwoDGridWorker grid(basp,lang);

          for(size_t im=0;im<muni.size();im++) {
            for(size_t iel=0;iel<basp->rad_Nel();iel++) {
              for(size_t irad=0;irad<(size_t) basp->r(iel).size();irad++) {
                grid.compute_bf(iel,irad,muni[im]);
                grid.model_potential(p1, p2);
                grid.eval_pot(H);
              }
            }
          }
        }

        // Use the Eigen-native boundary removal (same cached pure index list
        // as the Fock path); no arma round trip.
        return basp->remove_boundaries(H);
      }

      helfem::Matrix TwoDGrid::atomdb_projection(int Z, int l, int m, probe_t p) {
        const helfem::atomdb::Atom at(Z);
        // The radial functions the database stores; ao_projection takes
        // any callable of r, so nothing else has to change.
        auto compute_ao = [&at, l](double r) { return at.orbitals(l, r); };

        const int norb = helfem::atomdb::norb(Z, l);
        helfem::Matrix S = helfem::Matrix::Zero(std::max(norb, 0), basp->Ndummy());
        if (norb <= 0)
          return S(Eigen::all, basp->pure_indices());

        TwoDGridWorker grid(basp, lang);
        for(size_t iel=0;iel<basp->rad_Nel();iel++) {
          for(size_t irad=0;irad<(size_t) basp->r(iel).size();irad++) {
            grid.compute_bf(iel,irad,m);
            grid.ao_projection(compute_ao, p);
            grid.multiply_Plm(l, m, p);
            grid.eval_proj(S);
          }
        }
        return S(Eigen::all,basp->pure_indices());
      }

      helfem::Matrix TwoDGrid::atomdb_overlap(int Z, int l, int m, probe_t p) {
        const helfem::atomdb::Atom at(Z);
        auto compute_ao = [&at, l](double r) { return at.orbitals(l, r); };

        const int norb = helfem::atomdb::norb(Z, l);
        helfem::Matrix S = helfem::Matrix::Zero(std::max(norb, 0), std::max(norb, 0));
        if (norb <= 0)
          return S;

        TwoDGridWorker grid(basp, lang);
        for(size_t iel=0;iel<basp->rad_Nel();iel++) {
          for(size_t irad=0;irad<(size_t) basp->r(iel).size();irad++) {
            grid.compute_bf(iel,irad,m);
            grid.ao_projection(compute_ao, p);
            grid.multiply_Plm(l, m, p);
            grid.eval_proj_overlap(S);
          }
        }
        return S;
      }

      // Panel-graded AO projection: one quadrature whose panels are
      // smooth for BOTH factors of the integrand at every exponent.
      //
      // The basis side is covered exactly by construction: panels never
      // cross a mu-element boundary, so the basis functions restricted
      // to any panel are polynomials. The AO side is covered by grading:
      // within each element, panels are bisected until the AO's radial
      // argument r_p spans no more than a couple of its decay lengths
      // and its angular argument cos(theta_p) is narrow, so the AO is a
      // low-degree function on every panel it can reach. Panels the AO
      // cannot reach are dropped, and the recursion around the probe
      // point itself stops once a panel's whole extent is deep inside
      // the AO peak (its contribution is O((r_hi/scale)^(l+3)) of the
      // total). Tensor Gauss on the accepted panels is then spectrally
      // accurate for every exponent -- tight, moderate or diffuse --
      // with no rule switch anywhere.
      //
      // The panel bounds are exact, not sampled: r_p and cos(theta_p)
      // are monotone in mu and in eta for all three probes (e.g. left
      // probe: dr_p/deta = a > 0, dcth_p/deta = sinh^2(mu)/(cosh(mu) +
      // eta)^2 >= 0), so corner evaluations bound them. The one special
      // case is a corner that touches the probe itself (r_p = 0), where
      // the angle is undefined and is treated as spanning [-1, 1].
      //
      // Conventions match the (mu, nu) grid path: the same
      // spherical_harmonics-at-phi=0 basis values, the same
      // sph_legendre AO angular factor, the same analytic 2 pi from the
      // phi integral. One deliberate difference: the midbond probe's
      // angular argument is the true polar angle about the midpoint,
      // cos(theta_c) = z / r_c, where the old grid path used the
      // prolate eta. The two agree for l = 0 (both constant); for l > 0
      // the old choice made the "midbond AO" a hybrid object rather
      // than a spherical harmonic about the midpoint.
      std::vector<helfem::Matrix> TwoDGrid::graded_projections(int lmin, int lmax_ao, int m,
                                                                const helfem::Vector & expn,
                                                                probe_t p, bool sto_probe) const {
        const int nl = lmax_ao - lmin + 1;
        const double a = basp->Rhalf();
        const size_t Nrad = basp->Nrad();
        const std::vector<Eigen::Index> pure = basp->pure_indices();
        const Eigen::VectorXi shell_m = basp->mval();

        // The phi integral is analytic: it kills every m' != m block
        // and yields the same 2 pi the grid path folds into its
        // weights. Collect the surviving functions.
        std::vector<Eigen::Index> mfun;
        for(size_t i = 0; i < pure.size(); i++)
          if(shell_m(pure[i] / Nrad) == m)
            mfun.push_back((Eigen::Index) i);

        std::vector<helfem::Matrix> SL(nl, helfem::Matrix::Zero(expn.size(), pure.size()));
        if(mfun.empty())
          return SL;

        // Lean per-element evaluation tables: each surviving function is
        // (angular shell) x (global radial function); within an element
        // only the radial functions [ifirst, ilast] are non-zero, so per
        // element we keep (output column, shell, local radial index).
        const Eigen::VectorXi shell_l = basp->lval();
        const helfem::diatomic::basis::RadialBasis & rad = basp->radial();
        std::vector<int> live_shells;
        for(Eigen::Index i = 0; i < shell_m.size(); i++)
          if(shell_m(i) == m)
            live_shells.push_back((int) i);
        struct Entry { Eigen::Index col; int shell; Eigen::Index jloc; };
        std::vector<std::vector<Entry>> eltab(rad.Nel());
        for(size_t iel = 0; iel < rad.Nel(); iel++) {
          size_t ifirst, ilast;
          rad.idx(iel, ifirst, ilast);
          for(Eigen::Index k = 0; k < (Eigen::Index) mfun.size(); k++) {
            const Eigen::Index dummy = pure[mfun[k]];
            const size_t g = dummy % Nrad;
            if(g >= ifirst && g <= ilast)
              eltab[iel].push_back({mfun[k], (int) (dummy / Nrad), (Eigen::Index) (g - ifirst)});
          }
        }

        // AO geometry at a point, in cancellation-free form: near an
        // on-focus probe both cosh(mu) - 1 and 1 +- eta vanish, so the
        // textbook expressions lose all their digits exactly where the
        // grading operates.
        const auto geom = [a, p](double mu, double eta, double & rp, double & cthp) {
          const double sh2 = std::sinh(0.5 * mu);
          const double chm1 = 2.0 * sh2 * sh2;   // cosh(mu) - 1
          const double ch = 1.0 + chm1;
          if(p == PROBE_LEFT) {
            const double q = 1.0 + eta;
            rp = a * (chm1 + q);
            cthp = (rp > 0.0) ? (ch * q - chm1) / (chm1 + q) : 1.0;
          } else if(p == PROBE_RIGHT) {
            const double q = 1.0 - eta;
            rp = a * (chm1 + q);
            cthp = (rp > 0.0) ? (ch * q - chm1) / (chm1 + q) : 1.0;
          } else {
            const double sh = std::sinh(mu);
            rp = a * std::hypot(sh, eta);
            cthp = (rp > 0.0) ? a * ch * eta / rp : 1.0;
          }
          cthp = std::clamp(cthp, -1.0, 1.0);
        };

        const auto eval_Plm = [m](int L, double u) {
          return std::sph_legendre(static_cast<unsigned>(L),
                                   static_cast<unsigned>(std::abs(m)),
                                   std::acos(std::clamp(u, -1.0, 1.0)));
        };

        // Panel rules. Full orders cover the element-wide basis
        // polynomials (degree ~ nnodes for the radial factor, ~ Lmax
        // for the angular one); deeply graded panels see only a tiny
        // slice of those polynomials, which is effectively low degree,
        // so they get lean orders.
        const int Lmax = basp->lval().maxCoeff();
        helfem::Vector xf_mu, wf_mu, xf_eta, wf_eta, xl_mu, wl_mu, xl_eta, wl_eta;
        lobatto::lobatto_compute(16, xf_mu, wf_mu);
        lobatto::lobatto_compute((Lmax + lmax_ao) / 2 + 12, xf_eta, wf_eta);
        lobatto::lobatto_compute(10, xl_mu, wl_mu);
        lobatto::lobatto_compute(12, xl_eta, wl_eta);

        const helfem::Vector bval(basp->bval());

        // Octave batching: exponents within a factor two share one
        // panelisation, graded for the tightest member and reaching as
        // far as the most diffuse one. The basis evaluation -- the
        // dominant cost -- is then paid once per point for the whole
        // bucket instead of once per exponent.
        std::map<int, std::vector<Eigen::Index>> buckets;
        for(Eigen::Index ix = 0; ix < expn.size(); ix++)
          buckets[(int) std::floor(std::log2(expn(ix)))].push_back(ix);
        std::vector<std::vector<Eigen::Index>> bucket_list;
        for(auto & b : buckets)
          bucket_list.push_back(b.second);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for(size_t ib = 0; ib < bucket_list.size(); ib++) {
          const std::vector<Eigen::Index> & bucket = bucket_list[ib];
          double scale = std::numeric_limits<double>::infinity(), rmax_ao = 0.0;
          for(Eigen::Index ix : bucket) {
            const double s = sto_probe ? 1.0 / expn(ix) : 1.0 / std::sqrt(expn(ix));
            scale = std::min(scale, s);
            rmax_ao = std::max(rmax_ao, (sto_probe ? 32.0 : std::sqrt(32.0)) * s);
          }

          std::vector<double> wao(nl * bucket.size());
          double m1el = 0.0, m2el = 0.0;

          const std::function<void(size_t,double,double,double,double,double,int)> panel =
              [&](size_t iel, double m1, double m2, double e1, double e2, double elwidth, int depth) {
            // Exact panel bounds from the corners (monotonicity; see
            // above). A probe-touching corner leaves the angle free.
            double r11, r12, r21, r22, c11, c12, c21, c22;
            geom(m1, e1, r11, c11); geom(m1, e2, r12, c12);
            geom(m2, e1, r21, c21); geom(m2, e2, r22, c22);
            double rlo = std::min(std::min(r11, r12), std::min(r21, r22));
            double rhi = std::max(std::max(r11, r12), std::max(r21, r22));
            if(p == PROBE_MIDDLE && e1 < 0.0 && e2 > 0.0) {
              double rm, cm;
              geom(m1, 0.0, rm, cm);
              rlo = std::min(rlo, rm);
            }
            double clo = std::min(std::min(c11, c12), std::min(c21, c22));
            double chi = std::max(std::max(c11, c12), std::max(c21, c22));
            if(rlo <= 0.0) { clo = -1.0; chi = 1.0; }

            if(rlo > rmax_ao)
              return;                        // the AO cannot reach this panel

            const bool tiny = rhi < 1e-3 * scale;   // deep inside the AO peak
            const bool rok = (rhi - rlo) <= 2.0 * scale;
            // Far from the probe (proper annulus panels), cos(theta_p)
            // is a tame analytic function of eta no matter how wide its
            // range, and the full-order eta rule integrates it; only
            // near the probe does the mapping distort violently.
            const bool cok = (chi - clo) <= 0.6 || rlo >= 2.0 * (rhi - rlo);
            if(!tiny && !(rok && cok) && depth < 48) {
              // Bisect the direction that contributes more of the
              // offending spread. r_p is monotone in mu, so corner
              // differences measure the mu direction exactly; along eta
              // the midpoint probe has an interior minimum at eta = 0
              // (all four corners of a symmetric panel see the SAME
              // r_p), so the eta span must include it or the chooser
              // goes blind and bisects mu forever.
              const double dr_mu  = std::max(std::abs(r21 - r11), std::abs(r22 - r12));
              double dr_eta = std::max(std::abs(r12 - r11), std::abs(r22 - r21));
              if(p == PROBE_MIDDLE && e1 < 0.0 && e2 > 0.0) {
                double rmid1, rmid2, cdum;
                geom(m1, 0.0, rmid1, cdum);
                geom(m2, 0.0, rmid2, cdum);
                dr_eta = std::max({dr_eta, r11 - rmid1, r12 - rmid1, r21 - rmid2, r22 - rmid2});
              }
              const double dc_mu  = std::max(std::abs(c21 - c11), std::abs(c22 - c12));
              const double dc_eta = std::max(std::abs(c12 - c11), std::abs(c22 - c21));
              const bool r_worse = ((rhi - rlo) / (2.0 * scale)) >= ((chi - clo) / 0.6);
              bool split_mu = r_worse ? (dr_mu > dr_eta) : (dc_mu > dc_eta);
              // degenerate metrics (e.g. a probe-corner panel): fall
              // back to bisecting the longer side in relative units
              if((r_worse && dr_mu <= 0.0 && dr_eta <= 0.0) ||
                 (!r_worse && dc_mu <= 0.0 && dc_eta <= 0.0))
                split_mu = ((m2 - m1) / elwidth) >= (0.5 * (e2 - e1));
              if(split_mu) {
                const double mm = 0.5 * (m1 + m2);
                panel(iel, m1, mm, e1, e2, elwidth, depth + 1);
                panel(iel, mm, m2, e1, e2, elwidth, depth + 1);
              } else {
                const double em = 0.5 * (e1 + e2);
                panel(iel, m1, m2, e1, em, elwidth, depth + 1);
                panel(iel, m1, m2, em, e2, elwidth, depth + 1);
              }
              return;
            }

            // Integrate. Lean rule once the panel is a small slice of
            // the element in both directions.
            const bool lean = (m2 - m1) < 0.0625 * elwidth && (e2 - e1) < 0.125;
            const helfem::Vector & xmu = lean ? xl_mu : xf_mu, & wmu = lean ? wl_mu : wf_mu;
            const helfem::Vector & xet = lean ? xl_eta : xf_eta, & wet = lean ? wl_eta : wf_eta;
            const double mmid = 0.5 * (m2 + m1), mhw = 0.5 * (m2 - m1);
            const double emid = 0.5 * (e2 + e1), ehw = 0.5 * (e2 - e1);
            // Only the bucket members whose AO reaches this panel do
            // any work here; the tight members die on the outer annuli.
            std::vector<size_t> livek;
            for(size_t k = 0; k < bucket.size(); k++) {
              const double ex = expn(bucket[k]);
              const bool reaches = sto_probe ? (ex * rlo <= 32.0)
                                             : (ex * rlo * rlo <= 32.0);
              if(reaches)
                livek.push_back(k);
            }
            if(livek.empty())
              return;

            // Per-panel factor tables: the radial basis values depend
            // only on mu and the shell harmonics only on eta, so they
            // are evaluated once per node line, not once per point.
            // (The Lobatto endpoints can land an ulp outside the
            // element under round-off; clamp so the lookups succeed.)
            helfem::Vector xloc(xmu.size());
            for(Eigen::Index iq = 0; iq < xmu.size(); iq++) {
              const double mu = std::clamp(mmid + mhw * xmu(iq), m1, m2);
              xloc(iq) = std::clamp(2.0 * (mu - m1el) / (m2el - m1el) - 1.0, -1.0, 1.0);
            }
            const helfem::Matrix rmat = rad.bf(iel, xloc);   // (nmu x nlocal)
            helfem::Matrix sph_tab(xet.size(), shell_l.size());
            for(Eigen::Index jq = 0; jq < xet.size(); jq++) {
              const double eta = std::clamp(emid + ehw * xet(jq), -1.0, 1.0);
              for(int s : live_shells)
                sph_tab(jq, s) = std::real(::spherical_harmonics(shell_l(s), m, eta, 0.0));
            }

            for(Eigen::Index iq = 0; iq < xmu.size(); iq++) {
              const double mu = std::clamp(mmid + mhw * xmu(iq), m1, m2);
              const double shmu = std::sinh(mu);
              for(Eigen::Index jq = 0; jq < xet.size(); jq++) {
                const double eta = std::clamp(emid + ehw * xet(jq), -1.0, 1.0);
                double rp, cthp;
                geom(mu, eta, rp, cthp);
                if(rp <= 0.0)
                  continue;
                // dV = 2 pi a^3 sinh(mu) (sinh^2(mu) + 1 - eta^2) dmu deta
                const double jac = a * a * a * shmu * (shmu * shmu + 1.0 - eta * eta);
                const double wgeo = 2.0 * M_PI * mhw * wmu(iq) * ehw * wet(jq) * jac;
                bool live = false;
                for(int il = 0; il < nl; il++) {
                  const double wPl = wgeo * eval_Plm(lmin + il, cthp);
                  for(size_t kk = 0; kk < livek.size(); kk++) {
                    const double ex = expn(bucket[livek[kk]]);
                    const double Rao = sto_probe ? lcao::radial_STO(rp, lmin + il, ex)
                                                 : lcao::radial_GTO(rp, lmin + il, ex);
                    wao[il * livek.size() + kk] = wPl * Rao;
                    live = live || (Rao != 0.0);
                  }
                }
                if(!live)
                  continue;
                for(const Entry & en : eltab[iel]) {
                  const double b = sph_tab(jq, en.shell) * rmat(iq, en.jloc);
                  for(int il = 0; il < nl; il++) {
                    helfem::Matrix & Sl = SL[il];
                    const double * w = &wao[il * livek.size()];
                    for(size_t kk = 0; kk < livek.size(); kk++)
                      Sl(bucket[livek[kk]], en.col) += w[kk] * b;
                  }
                }
              }
            }
          };

          for(size_t iel = 0; iel + 1 < (size_t) bval.size(); iel++) {
            m1el = bval(iel);
            m2el = bval(iel + 1);
            panel(iel, m1el, m2el, -1.0, 1.0, m2el - m1el, 0);
          }
        }

        return SL;
      }

      helfem::Matrix TwoDGrid::gto_projection(int l, int m, const helfem::Vector & expn, probe_t p) {
        return graded_projections(l, l, m, expn, p, /*sto_probe=*/false)[0];
      }

      helfem::Matrix TwoDGrid::gto_overlap(int l, int m, const helfem::Vector & expn, probe_t p) {
        helfem::Matrix S = helfem::Matrix::Zero(expn.size(),expn.size());
        TwoDGridWorker grid(basp,lang);

        for(size_t iel=0;iel<basp->rad_Nel();iel++) {
          for(size_t irad=0;irad<(size_t) basp->r(iel).size();irad++) {
            grid.compute_bf(iel,irad,m);
            grid.gto(l, expn, p);
            grid.multiply_Plm(l, m, p);
            grid.eval_proj_overlap(S);
          }
        }

        return S;
      }

      helfem::Matrix TwoDGrid::sto_projection(int l, int m, const helfem::Vector & expn, probe_t p) {
        return graded_projections(l, l, m, expn, p, /*sto_probe=*/true)[0];
      }

      helfem::Matrix TwoDGrid::sto_overlap(int l, int m, const helfem::Vector & expn, probe_t p) {
        helfem::Matrix S = helfem::Matrix::Zero(expn.size(),expn.size());
        TwoDGridWorker grid(basp,lang);

        for(size_t iel=0;iel<basp->rad_Nel();iel++) {
          for(size_t irad=0;irad<(size_t) basp->r(iel).size();irad++) {
            grid.compute_bf(iel,irad,m);
            grid.sto(l, expn, p);
            grid.multiply_Plm(l, m, p);
            grid.eval_proj_overlap(S);
          }
        }

        return S;
      }

      helfem::Matrix TwoDGrid::atomic_projection(int l, int m, probe_t p) {
        helfem::Matrix C;
        sadatom::basis::TwoDBasis basis;
        if(p == PROBE_LEFT) {
          if((size_t) l>=(size_t) lh_occs.size())
            return helfem::Matrix();
          int nocc = std::ceil(lh_occs(l)/(2.0*(2.0*l+1.0)));
          if(nocc==0)
            // empty matrix
            return helfem::Matrix();
          C = lh_orbs[l].leftCols(nocc);
          basis = lh_basis;
        } else if(p == PROBE_RIGHT) {
          if((size_t) l>=(size_t) rh_occs.size())
            return helfem::Matrix();
          int nocc = std::ceil(rh_occs(l)/(2.0*(2.0*l+1.0)));
          if(nocc==0)
            // empty matrix
            return helfem::Matrix();
          C = rh_orbs[l].leftCols(nocc);
          basis = rh_basis;
        } else
          throw std::logic_error("No AOs on bond center!\n");

        // eval_orbs is Eigen-typed (returns helfem::Vector).
        std::function<helfem::Vector(double r)> eval_ao = [basis, C](double r) {
          return basis.eval_orbs(C, r);
        };

        helfem::Matrix S = helfem::Matrix::Zero(C.cols(),basp->Ndummy());
        TwoDGridWorker grid(basp,lang);

        for(size_t iel=0;iel<basp->rad_Nel();iel++) {
          for(size_t irad=0;irad<(size_t) basp->r(iel).size();irad++) {
            grid.compute_bf(iel,irad,m);
            grid.ao_projection(eval_ao, p);
            grid.multiply_Plm(l, m, p);
            grid.eval_proj(S);
          }
        }

        return S(Eigen::all,basp->pure_indices());
      }

      void TwoDGrid::compute_atoms(int Zl, int Zr) {
        // Atomic-orbital guess for the two nuclei via the shared
        // sadatom OOO SCF helper (helfem::sadatom::scf::run_atomic_scf).
        // PBE (x_func = 101, c_func = 130), restricted-only, with the
        // per-l occupation frozen to the tabulated PBE ground-state
        // configuration in `pbe_ground_states`. This drops the
        // dependency on the bespoke SCFSolver / DIIS / L-BFGS
        // machinery that used to live in src/sadatom/solver.cpp.
        constexpr int primbas = 4;
        constexpr int Nnodes  = 15;
        constexpr int x_func  = 101; // libxc: gga_x_pbe
        constexpr int c_func  = 130; // libxc: gga_c_pbe
        constexpr int Nelem   = 5;
        constexpr double Rmax = 40.0;
        constexpr int igrid   = 4;
        constexpr double zexp = 2.0;
        constexpr double dftthr = 1e-12;

        auto poly = std::shared_ptr<const polynomial_basis::PolynomialBasis>(
            polynomial_basis::make_basis(primbas, Nnodes));
        const int Nquad = 5 * poly->nbf();

        auto lmax_for_Z = [](int Z){
          for (int l = 3; l >= 0; --l)
            if (pbe_ground_states[Z-1][l] > 0) return l;
          return -1;
        };
        auto per_l_occ = [&lmax_for_Z](int Z) {
          Eigen::VectorXi o(lmax_for_Z(Z) + 1);
          for (Eigen::Index l = 0; l < o.size(); ++l)
            o(l) = pbe_ground_states[Z-1][l];
          return o;
        };

        auto run_side = [&](int Z, sadatom::basis::TwoDBasis & basis_out,
                             helfem::Cube & orbs_out, Eigen::VectorXi & occs_out) {
          const int lmax = lmax_for_Z(Z);
          const Eigen::VectorXi occ = per_l_occ(Z);
          const int Ntot = static_cast<int>(occ.sum());

          sadatom::scf::AtomicSCFOptions opts;
          opts.Z              = Z;
          opts.lmax           = lmax;
          opts.poly           = poly;
          opts.Nquad          = Nquad;
          opts.bval           = atomic::basis::form_grid(
              modelpotential::POINT_NUCLEUS, 0.0, Nelem, Rmax, igrid, zexp,
              0, 0, 0.0, Z, 0, 0, 0.0);
          opts.nela           = Ntot / 2;  // restricted closed-shell (PBE ground occs
          opts.nelb           = Ntot / 2;  // are always even totals per l).
          opts.restricted     = true;
          opts.x_func         = x_func;
          opts.c_func         = c_func;
          opts.dftthr         = dftthr;
          // fixed_per_l_* are real-valued (fractional occupations are
          // allowed); the tabulated PBE ground-state config is integral.
          opts.fixed_per_l_a  = occ.cast<double>();
          opts.verbosity      = 0;
          auto res = sadatom::scf::run_atomic_scf(opts);
          basis_out = res.basis;
          orbs_out  = res.orbs_a;
          occs_out  = res.occs_a;
        };

        if (Zl > 0) run_side(Zl, lh_basis, lh_orbs, lh_occs);
        if (Zr > 0) run_side(Zr, rh_basis, rh_orbs, rh_occs);
      }
    }
  }
}
