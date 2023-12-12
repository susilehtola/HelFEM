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

#include "twodquadrature.h"
#include "chebyshev.h"
#include "../general/lcao.h"
#include "../general/model_potential.h"
#include "utils.h"
extern "C" {
// Legendre polynomials
#include <gsl/gsl_sf_legendre.h>
}

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
        // Get angular grid
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
        r.zeros(1);
        r(0)=basp->get_r(iel)(irad);
        wrad.zeros(1);
        wrad(0)=basp->get_wrad(iel)(irad);

        double Rhalf(basp->get_Rhalf());

        // Calculate helpers
        arma::vec shmu(arma::sinh(r));

        arma::vec sth(cth.n_elem);
        for(size_t ia=0;ia<cth.n_elem;ia++)
          sth(ia)=sqrt(1.0 - cth(ia)*cth(ia));

        // Update total weights
        wtot.zeros(wrad.n_elem*wang.n_elem);
        for(size_t ia=0;ia<wang.n_elem;ia++)
          for(size_t ir=0;ir<wrad.n_elem;ir++) {
            size_t idx=ia*wrad.n_elem+ir;
            // sin(th) is already contained within wang, but we don't want to divide by it since it may be zero. Phi integrals yield 2 pi
            wtot(idx)=2.0*M_PI*wang(ia)*wrad(ir)*std::pow(Rhalf,3)*shmu(ir)*(std::pow(shmu(ir),2)+std::pow(sth(ia),2));
          }

        // Compute basis function values
        bf.zeros(bf_ind.n_elem,wtot.n_elem);
        // Loop over angular grid
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(size_t ia=0;ia<cth.n_elem;ia++) {
          // Evaluate basis functions at angular point
          arma::mat abf(basp->eval_bf(iel, irad, cth(ia), m));
          if(abf.n_cols != bf_ind.n_elem) {
            std::ostringstream oss;
            oss << "Mismatch! Have " << bf_ind.n_elem << " basis function indices but " << abf.n_cols << " basis functions!\n";
            throw std::logic_error(oss.str());
          }
          // Store functions
          bf.cols(ia*wrad.n_elem,(ia+1)*wrad.n_elem-1)=arma::trans(abf);
        }
      }

      void TwoDGridWorker::model_potential(const modelpotential::ModelPotential * p1, const modelpotential::ModelPotential * p2) {
        double Rhalf(basp->get_Rhalf());
        arma::vec chmu(arma::cosh(r));

        itg.zeros(1,wtot.n_elem);
        for(size_t ia=0;ia<wang.n_elem;ia++)
          for(size_t ir=0;ir<wrad.n_elem;ir++) {
            size_t idx=ia*wrad.n_elem+ir;

            double r1=Rhalf*(chmu(ir) + cth(ia));
            double r2=Rhalf*(chmu(ir) - cth(ia));

	    double V1(p1->V(r1));
	    double V2(p2->V(r2));
	    if(std::isnormal(V1))
	      itg(idx)+=V1;
	    if(std::isnormal(V2))
	      itg(idx)+=V2;
          }
      }

      void TwoDGridWorker::unit_pot() {
        itg.ones(1,wtot.n_elem);
      }

      void TwoDGridWorker::multiply_Plm(int l, int m, probe_t p) {
        arma::vec chmu(arma::cosh(r));
        arma::vec shmu(arma::sinh(r));

        if(p==PROBE_LEFT) {
          for(size_t ia=0;ia<wang.n_elem;ia++)
            for(size_t ir=0;ir<wrad.n_elem;ir++) {
              size_t idx=ia*wrad.n_elem+ir;
              double cthval = (1.0 + chmu(ir)*cth(ia))/(chmu(ir) + cth(ia));
              double Plm = gsl_sf_legendre_sphPlm(l,std::abs(m),cthval);
              for(size_t ix=0;ix<itg.n_rows;ix++)
                itg(ix,idx)*=Plm;
            }

        } else if(p==PROBE_RIGHT) {
          for(size_t ia=0;ia<wang.n_elem;ia++)
            for(size_t ir=0;ir<wrad.n_elem;ir++) {
              size_t idx=ia*wrad.n_elem+ir;
              double cthval = (1.0 - chmu(ir)*cth(ia))/(chmu(ir) - cth(ia));
              double Plm = gsl_sf_legendre_sphPlm(l,std::abs(m),cthval);
              for(size_t ix=0;ix<itg.n_rows;ix++)
                itg(ix,idx)*=Plm;
            }

        } else if(p==PROBE_MIDDLE) {
          for(size_t ia=0;ia<wang.n_elem;ia++)
            for(size_t ir=0;ir<wrad.n_elem;ir++) {
              size_t idx=ia*wrad.n_elem+ir;
              double cthval = cth(ia);
              double Plm = gsl_sf_legendre_sphPlm(l,std::abs(m),cthval);
              for(size_t ix=0;ix<itg.n_rows;ix++)
                itg(ix,idx)*=Plm;
            }
        }
      }

      void TwoDGridWorker::ao_projection(const std::function<arma::vec(double r)> & compute_ao, probe_t p) {
        double Rhalf(basp->get_Rhalf());
        arma::vec chmu(arma::cosh(r));

        itg.zeros(compute_ao(0.0).n_elem,wtot.n_elem);
        if(p==PROBE_LEFT) {
          for(size_t ia=0;ia<wang.n_elem;ia++)
            for(size_t ir=0;ir<wrad.n_elem;ir++) {
              size_t idx=ia*wrad.n_elem+ir;

              double ra(Rhalf*(chmu(ir) + cth(ia)));
              itg.col(idx) = compute_ao(ra);
            }

        } else if(p==PROBE_RIGHT) {
          for(size_t ia=0;ia<wang.n_elem;ia++)
            for(size_t ir=0;ir<wrad.n_elem;ir++) {
              size_t idx=ia*wrad.n_elem+ir;

              double rb(Rhalf*(chmu(ir) - cth(ia)));
              itg.col(idx) = compute_ao(rb);
            }

        } else if(p==PROBE_MIDDLE) {
          for(size_t ia=0;ia<wang.n_elem;ia++)
            for(size_t ir=0;ir<wrad.n_elem;ir++) {
              size_t idx=ia*wrad.n_elem+ir;

              double rc(Rhalf*sqrt(std::pow(chmu(ir),2) + std::pow(cth(ia),2) -1.0));
              itg.col(idx) = compute_ao(rc);
            }
        }
      }

      void TwoDGridWorker::gto(int l, const arma::vec & expn, probe_t p) {
        std::function<arma::vec(double r)> compute_gto = [expn, l](double r) {
          arma::vec f(expn.n_elem);
          for(size_t ix=0;ix<expn.n_elem;ix++)
            f(ix)=lcao::radial_GTO(r,l,expn(ix));
          return f;
        };
        ao_projection(compute_gto, p);
      }

      void TwoDGridWorker::sto(int l, const arma::vec & expn, probe_t p) {
        std::function<arma::vec(double r)> compute_sto = [expn, l](double r) {
          arma::vec f(expn.n_elem);
          for(size_t ix=0;ix<expn.n_elem;ix++)
            f(ix)=lcao::radial_STO(r,l,expn(ix));
          return f;
        };
        ao_projection(compute_sto, p);
      }

      void TwoDGridWorker::eval_pot(arma::mat & Vo) const {
        if(itg.n_rows != 1)
          throw std::logic_error("Should only have one column in integrand!\n");
        Vo.submat(bf_ind,bf_ind)+=bf*arma::diagmat(itg%wtot)*arma::trans(bf);
      }

      void TwoDGridWorker::eval_proj(arma::mat & Vo) const {
        Vo.cols(bf_ind)+=itg*arma::diagmat(wtot)*arma::trans(bf);
      }

      void TwoDGridWorker::eval_proj_overlap(arma::mat & Vo) const {
        Vo+=itg*arma::diagmat(wtot)*arma::trans(itg);
      }

      TwoDGrid::TwoDGrid() {
      }

      TwoDGrid::TwoDGrid(const helfem::diatomic::basis::TwoDBasis * basp_, int lang_) : basp(basp_), lang(lang_) {
      }

      TwoDGrid::~TwoDGrid() {
      }

      arma::mat TwoDGrid::model_potential(const modelpotential::ModelPotential * p1, const modelpotential::ModelPotential * p2) {
        arma::mat H;
        H.zeros(basp->Ndummy(),basp->Ndummy());

        // Get unique m values in basis set
        arma::ivec muni(basp->get_mval());
        muni=muni(arma::find_unique(muni));
        {
          TwoDGridWorker grid(basp,lang);

          for(size_t im=0;im<muni.n_elem;im++) {
            for(size_t iel=0;iel<basp->get_rad_Nel();iel++) {
              for(size_t irad=0;irad<basp->get_r(iel).n_elem;irad++) {
                grid.compute_bf(iel,irad,muni(im));
                grid.model_potential(p1, p2);
                grid.eval_pot(H);
              }
            }
          }
        }

        H=basp->remove_boundaries(H);

        return H;
      }

      arma::mat TwoDGrid::overlap() {
        arma::mat S;
        S.zeros(basp->Ndummy(),basp->Ndummy());

        // Get unique m values in basis set
        arma::ivec muni(basp->get_mval());
        muni=muni(arma::find_unique(muni));
        {
          TwoDGridWorker grid(basp,lang);

          for(size_t im=0;im<muni.n_elem;im++) {
            for(size_t iel=0;iel<basp->get_rad_Nel();iel++) {
              for(size_t irad=0;irad<basp->get_r(iel).n_elem;irad++) {
                grid.compute_bf(iel,irad,muni(im));
                grid.unit_pot();
                grid.eval_pot(S);
              }
            }
          }
        }

        S=basp->remove_boundaries(S);

        return S;
      }

      arma::mat TwoDGrid::gto_projection(int l, int m, const arma::vec & expn, probe_t p) {
        arma::mat S;
        S.zeros(expn.n_elem,basp->Ndummy());
        TwoDGridWorker grid(basp,lang);

        for(size_t iel=0;iel<basp->get_rad_Nel();iel++) {
          for(size_t irad=0;irad<basp->get_r(iel).n_elem;irad++) {
            grid.compute_bf(iel,irad,m);
            grid.gto(l, expn, p);
            grid.multiply_Plm(l, m, p);
            grid.eval_proj(S);
          }
        }

        S=S.cols(basp->pure_indices());

        return S;
      }

      arma::mat TwoDGrid::gto_overlap(int l, int m, const arma::vec & expn, probe_t p) {
        arma::mat S;
        S.zeros(expn.n_elem,expn.n_elem);
        TwoDGridWorker grid(basp,lang);

        for(size_t iel=0;iel<basp->get_rad_Nel();iel++) {
          for(size_t irad=0;irad<basp->get_r(iel).n_elem;irad++) {
            grid.compute_bf(iel,irad,m);
            grid.gto(l, expn, p);
            grid.multiply_Plm(l, m, p);
            grid.eval_proj_overlap(S);
          }
        }

        return S;
      }

      arma::mat TwoDGrid::sto_projection(int l, int m, const arma::vec & expn, probe_t p) {
        arma::mat S;
        S.zeros(expn.n_elem,basp->Ndummy());
        TwoDGridWorker grid(basp,lang);

        for(size_t iel=0;iel<basp->get_rad_Nel();iel++) {
          for(size_t irad=0;irad<basp->get_r(iel).n_elem;irad++) {
            grid.compute_bf(iel,irad,m);
            grid.sto(l, expn, p);
            grid.multiply_Plm(l, m, p);
            grid.eval_proj(S);
          }
        }

        S=S.cols(basp->pure_indices());

        return S;
      }

      arma::mat TwoDGrid::sto_overlap(int l, int m, const arma::vec & expn, probe_t p) {
        arma::mat S;
        S.zeros(expn.n_elem,expn.n_elem);
        TwoDGridWorker grid(basp,lang);

        for(size_t iel=0;iel<basp->get_rad_Nel();iel++) {
          for(size_t irad=0;irad<basp->get_r(iel).n_elem;irad++) {
            grid.compute_bf(iel,irad,m);
            grid.sto(l, expn, p);
            grid.multiply_Plm(l, m, p);
            grid.eval_proj_overlap(S);
          }
        }

        return S;
      }

      arma::mat TwoDGrid::atomic_projection(int l, int m, probe_t p) {
        arma::mat C;
        sadatom::basis::TwoDBasis basis;
        if(p == PROBE_LEFT) {
          if((size_t) l>=lh_occs.n_elem)
            return C;
          int nocc = std::ceil(lh_occs(l)/(2.0*(2.0*l+1.0)));
          if(nocc==0)
            // empty matrix
            return C;
          C = lh_orbs.slice(l).cols(0,nocc-1);
          basis = lh_basis;
        } else if(p == PROBE_RIGHT) {
          if((size_t) l>=rh_occs.n_elem)
            return C;
          int nocc = std::ceil(rh_occs(l)/(2.0*(2.0*l+1.0)));
          if(nocc==0)
            // empty matrix
            return C;
          C = rh_orbs.slice(l).cols(0, nocc-1);
          basis = rh_basis;
        } else
          throw std::logic_error("No AOs on bond center!\n");

        std::function<arma::vec(double r)> eval_ao = [basis, C](double r) {
          return basis.eval_orbs(C, r);
        };

        arma::mat S;
        S.zeros(C.n_cols,basp->Ndummy());
        TwoDGridWorker grid(basp,lang);

        for(size_t iel=0;iel<basp->get_rad_Nel();iel++) {
          for(size_t irad=0;irad<basp->get_r(iel).n_elem;irad++) {
            grid.compute_bf(iel,irad,m);
            grid.ao_projection(eval_ao, p);
            grid.multiply_Plm(l, m, p);
            grid.eval_proj(S);
          }
        }

        S=S.cols(basp->pure_indices());

        return S;
      }

      void TwoDGrid::compute_atoms(int Zl, int Zr) {
        int primbas=4;
        int Nnodes=15;
        auto poly(std::shared_ptr<const polynomial_basis::PolynomialBasis>(polynomial_basis::get_basis(primbas,Nnodes)));
        int Nquad=5*poly->get_nbf();
        int taylor_order = poly->get_nprim()-1;

        // PBE xc
        int x_func = 101;
        int c_func = 130;

        // Radial basis
        modelpotential::nuclear_model_t finitenuc = modelpotential::POINT_NUCLEUS;
        double Rrms = 0.0;
        int Nelem = 5;
        double Rmax = 40.0;
        int igrid = 4;
        double zexp = 2.0;
        int Nelem0 = 0;
        int igrid0 = 4;
        double zexp0 = 2.0;
        bool zeroder = false;

        arma::vec blval=atomic::basis::form_grid((modelpotential::nuclear_model_t) finitenuc, Rrms, Nelem, Rmax, igrid, zexp, Nelem0, igrid0, zexp0, Zl, 0, 0, 0.0);
        arma::vec brval=atomic::basis::form_grid((modelpotential::nuclear_model_t) finitenuc, Rrms, Nelem, Rmax, igrid, zexp, Nelem0, igrid0, zexp0, Zr, 0, 0, 0.0);

        double shift = 1.0;
        double convthr = 1e-7;
        double dftthr = 1e-12;
        double diiseps = 1e-2;
        double diisthr = 1e-3;
        int diisorder = 10;
        int maxit = 128;

        std::function<int(int)> lmax = [](int Z){
          for(int l=3;l>=0;l--)
            if(pbe_ground_states[Z-1][l]>0)
              return l;
          return -1;
        };

        std::function<arma::ivec(int)> get_occs = [lmax](int Z){
          arma::ivec occs(lmax(Z)+1);
          for(size_t l=0;l<occs.n_elem;l++)
            occs(l) = pbe_ground_states[Z-1][l];
          return occs;
        };

        int iguess=2;

        if(Zl>0) {
          sadatom::solver::SCFSolver lsolver(Zl, finitenuc, Rrms, lmax(Zl), poly, zeroder, Nquad, blval, taylor_order, x_func, c_func, maxit, shift, convthr, dftthr, diiseps, diisthr, diisorder);
          helfem::sadatom::solver::rconf_t lconf;
          lconf.orbs=sadatom::solver::OrbitalChannel(true);
          lsolver.Initialize(lconf.orbs,iguess);
          lconf.orbs.SetOccs(get_occs(Zl));
          lsolver.Solve(lconf);
          lh_basis = lsolver.Basis();
          lh_orbs = lconf.orbs.Coeffs();
          lh_occs = lconf.orbs.Occs();
        }
        if(Zr>0) {
          sadatom::solver::SCFSolver rsolver(Zr, finitenuc, Rrms, lmax(Zr), poly, zeroder, Nquad, brval, taylor_order, x_func, c_func, maxit, shift, convthr, dftthr, diiseps, diisthr, diisorder);
          helfem::sadatom::solver::rconf_t rconf;
          rconf.orbs=sadatom::solver::OrbitalChannel(true);
          rsolver.Initialize(rconf.orbs,iguess);
          rconf.orbs.SetOccs(get_occs(Zr));
          rsolver.Solve(rconf);
          rh_basis = rsolver.Basis();
          rh_orbs = rconf.orbs.Coeffs();
          rh_occs = rconf.orbs.Occs();
        }
      }
    }
  }
}
