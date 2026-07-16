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
#include "chebyshev.h"
#include "../general/lcao.h"
#include "../general/model_potential.h"
#include "../sadatom/scf.h"
#include "utils.h"
#include <ArmaEigen.h>
#include <algorithm>
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
        r(0)=basp->get_r(iel)(irad);
        wrad=helfem::Vector::Zero(1);
        wrad(0)=basp->get_wrad(iel)(irad);

        double Rhalf(basp->get_Rhalf());

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
        const helfem::Matrix rad_all(basp->get_rad_bf(iel));

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
        double Rhalf(basp->get_Rhalf());
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
        double Rhalf(basp->get_Rhalf());
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
        const Eigen::VectorXi mvals(basp->get_mval());
        std::vector<int> muni;
        for(Eigen::Index i=0;i<mvals.size();i++)
          if(std::find(muni.begin(),muni.end(),mvals(i))==muni.end())
            muni.push_back(mvals(i));
        std::sort(muni.begin(),muni.end());
        {
          TwoDGridWorker grid(basp,lang);

          for(size_t im=0;im<muni.size();im++) {
            for(size_t iel=0;iel<basp->get_rad_Nel();iel++) {
              for(size_t irad=0;irad<(size_t) basp->get_r(iel).size();irad++) {
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

      helfem::Matrix TwoDGrid::gto_projection(int l, int m, const helfem::Vector & expn, probe_t p) {
        helfem::Matrix S = helfem::Matrix::Zero(expn.size(),basp->Ndummy());
        TwoDGridWorker grid(basp,lang);

        for(size_t iel=0;iel<basp->get_rad_Nel();iel++) {
          for(size_t irad=0;irad<(size_t) basp->get_r(iel).size();irad++) {
            grid.compute_bf(iel,irad,m);
            grid.gto(l, expn, p);
            grid.multiply_Plm(l, m, p);
            grid.eval_proj(S);
          }
        }

        return S(Eigen::all,basp->pure_indices());
      }

      helfem::Matrix TwoDGrid::gto_overlap(int l, int m, const helfem::Vector & expn, probe_t p) {
        helfem::Matrix S = helfem::Matrix::Zero(expn.size(),expn.size());
        TwoDGridWorker grid(basp,lang);

        for(size_t iel=0;iel<basp->get_rad_Nel();iel++) {
          for(size_t irad=0;irad<(size_t) basp->get_r(iel).size();irad++) {
            grid.compute_bf(iel,irad,m);
            grid.gto(l, expn, p);
            grid.multiply_Plm(l, m, p);
            grid.eval_proj_overlap(S);
          }
        }

        return S;
      }

      helfem::Matrix TwoDGrid::sto_projection(int l, int m, const helfem::Vector & expn, probe_t p) {
        helfem::Matrix S = helfem::Matrix::Zero(expn.size(),basp->Ndummy());
        TwoDGridWorker grid(basp,lang);

        for(size_t iel=0;iel<basp->get_rad_Nel();iel++) {
          for(size_t irad=0;irad<(size_t) basp->get_r(iel).size();irad++) {
            grid.compute_bf(iel,irad,m);
            grid.sto(l, expn, p);
            grid.multiply_Plm(l, m, p);
            grid.eval_proj(S);
          }
        }

        return S(Eigen::all,basp->pure_indices());
      }

      helfem::Matrix TwoDGrid::sto_overlap(int l, int m, const helfem::Vector & expn, probe_t p) {
        helfem::Matrix S = helfem::Matrix::Zero(expn.size(),expn.size());
        TwoDGridWorker grid(basp,lang);

        for(size_t iel=0;iel<basp->get_rad_Nel();iel++) {
          for(size_t irad=0;irad<(size_t) basp->get_r(iel).size();irad++) {
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

        for(size_t iel=0;iel<basp->get_rad_Nel();iel++) {
          for(size_t irad=0;irad<(size_t) basp->get_r(iel).size();irad++) {
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
            polynomial_basis::get_basis(primbas, Nnodes));
        const int Nquad = 5 * poly->get_nbf();

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
          // atomic::basis::form_grid still returns arma::vec; bridge once.
          opts.bval           = helfem::to_eigen(atomic::basis::form_grid(
              modelpotential::POINT_NUCLEUS, 0.0, Nelem, Rmax, igrid, zexp,
              0, 0, 0.0, Z, 0, 0, 0.0));
          opts.nela           = Ntot / 2;  // restricted closed-shell (PBE ground occs
          opts.nelb           = Ntot / 2;  // are always even totals per l).
          opts.restricted     = true;
          opts.x_func         = x_func;
          opts.c_func         = c_func;
          opts.dftthr         = dftthr;
          opts.fixed_per_l_a  = occ;
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
