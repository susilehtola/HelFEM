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
#include "../general/utils.h"

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
        bf_ind=basp->bf_list(iel,m);

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

      void TwoDGridWorker::gto(int l, const arma::vec & expn, probe_t p) {
        double Rhalf(basp->get_Rhalf());
        arma::vec chmu(arma::cosh(r));

        itg.zeros(expn.n_elem,wtot.n_elem);
        if(p==PROBE_LEFT) {
          for(size_t ia=0;ia<wang.n_elem;ia++)
            for(size_t ir=0;ir<wrad.n_elem;ir++) {
              size_t idx=ia*wrad.n_elem+ir;

              double ra(Rhalf*(chmu(ir) + cth(ia)));
              for(size_t ix=0;ix<expn.n_elem;ix++)
                itg(ix,idx)=lcao::radial_GTO(ra,l,expn(ix));
            }

        } else if(p==PROBE_RIGHT) {
          for(size_t ia=0;ia<wang.n_elem;ia++)
            for(size_t ir=0;ir<wrad.n_elem;ir++) {
              size_t idx=ia*wrad.n_elem+ir;

              double rb(Rhalf*(chmu(ir) - cth(ia)));
              for(size_t ix=0;ix<expn.n_elem;ix++)
                itg(ix,idx)=lcao::radial_GTO(rb,l,expn(ix));
            }

        } else if(p==PROBE_MIDDLE) {
          for(size_t ia=0;ia<wang.n_elem;ia++)
            for(size_t ir=0;ir<wrad.n_elem;ir++) {
              size_t idx=ia*wrad.n_elem+ir;

              double rc(Rhalf*sqrt(std::pow(chmu(ir),2) + std::pow(cth(ia),2) -1.0));
              for(size_t ix=0;ix<expn.n_elem;ix++)
                itg(ix,idx)=lcao::radial_GTO(rc,l,expn(ix));
            }
        }

        // Assure normalization
        itg/=sqrt(4.0*M_PI);
      }

      void TwoDGridWorker::sto(int l, const arma::vec & expn, probe_t p) {
        double Rhalf(basp->get_Rhalf());
        arma::vec chmu(arma::cosh(r));

        itg.zeros(expn.n_elem,wtot.n_elem);
        if(p==PROBE_LEFT) {
          for(size_t ia=0;ia<wang.n_elem;ia++)
            for(size_t ir=0;ir<wrad.n_elem;ir++) {
              size_t idx=ia*wrad.n_elem+ir;
              double ra(Rhalf*(chmu(ir) + cth(ia)));
              for(size_t ix=0;ix<expn.n_elem;ix++)
                itg(ix,idx)=lcao::radial_STO(ra,l,expn(ix));
            }

        } else if(p==PROBE_RIGHT) {
          for(size_t ia=0;ia<wang.n_elem;ia++)
            for(size_t ir=0;ir<wrad.n_elem;ir++) {
              size_t idx=ia*wrad.n_elem+ir;
              double rb(Rhalf*(chmu(ir) - cth(ia)));
              for(size_t ix=0;ix<expn.n_elem;ix++)
                itg(ix,idx)=lcao::radial_STO(rb,l,expn(ix));
            }

        } else if(p==PROBE_MIDDLE) {
          for(size_t ia=0;ia<wang.n_elem;ia++)
            for(size_t ir=0;ir<wrad.n_elem;ir++) {
              size_t idx=ia*wrad.n_elem+ir;
              double rc(Rhalf*sqrt(std::pow(chmu(ir),2) + std::pow(cth(ia),2) -1.0));
              for(size_t ix=0;ix<expn.n_elem;ix++)
                itg(ix,idx)=lcao::radial_STO(rc,l,expn(ix));
            }
        }

        // Assure normalization
        itg/=sqrt(4.0*M_PI);
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
            grid.eval_proj_overlap(S);
          }
        }

        return S;
      }
    }
  }
}
