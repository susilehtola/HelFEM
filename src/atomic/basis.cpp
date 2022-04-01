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
      static arma::vec concatenate_grid(const arma::vec & left, const arma::vec & right) {
        if(!left.n_elem)
          return right;
        if(!right.n_elem)
          return left;

        if(left(0) != 0.0)
          throw std::logic_error("left vector doesn't start from zero");
        if(right(0) != 0.0)
          throw std::logic_error("right vector doesn't start from zero");

        // Concatenated vector
        arma::vec ret(left.n_elem + right.n_elem - 1);
        ret.subvec(0,left.n_elem-1)=left;
        ret.subvec(left.n_elem,ret.n_elem-1)=right.subvec(1,right.n_elem-1) + left(left.n_elem-1)*arma::ones<arma::vec>(right.n_elem-1);
        return ret;
      }

      arma::vec normal_grid(int num_el, double rmax, int igrid, double zexp) {
        return utils::get_grid(rmax,num_el,igrid,zexp);
      }

      arma::vec finite_nuclear_grid(int num_el, double rmax, int igrid, double zexp, int num_el_nuc, double rnuc, int igrid_nuc, double zexp_nuc) {
        if(num_el_nuc) {
          // Grid for the finite nucleus
          arma::vec bnuc(utils::get_grid(rnuc,num_el_nuc,igrid_nuc,zexp_nuc));
          // and the one for the electrons
          arma::vec belec(utils::get_grid(rmax-rnuc,num_el,igrid,zexp));

          arma::vec bnucel(concatenate_grid(bnuc,bnuc));
          return concatenate_grid(bnucel,belec);
        } else {
          return utils::get_grid(rmax,num_el,igrid,zexp);
        }
      }

      arma::vec offcenter_nuclear_grid(int num_el0, int Zm, int Zlr, double Rhalf, int num_el, double rmax, int igrid, double zexp) {
        // First boundary at
        int b0used = (Zm != 0);
        double b0=Zm*Rhalf/(Zm+Zlr);
        // Second boundary at
        int b1used = (Zlr != 0);
        double b1=Rhalf;
        // Last boundary at
        double b2=rmax;

        printf("b0 = %e, b0used = %i\n",b0,b0used);
        printf("b1 = %e, b1used = %i\n",b1,b1used);
        printf("b2 = %e\n",b2);

        // Get grids
        arma::vec bval0, bval1;
        if(b0used) {
          // 0 to b0
          bval0=utils::get_grid(b0,num_el0,igrid,zexp);
        }
        if(b1used) {
          // b0 to b1

          // Reverse grid to get tighter spacing around nucleus
          bval1=-arma::reverse(utils::get_grid(b1-b0,num_el0,igrid,zexp));
          bval1+=arma::ones<arma::vec>(bval1.n_elem)*(b1-b0);
          // Assert numerical exactness
          bval1(0)=0.0;
          bval1(bval1.n_elem-1)=b1-b0;
        }
        arma::vec bval2=utils::get_grid(b2-b1,num_el,igrid,zexp);

        arma::vec bval;
        if(b0used && b1used) {
          bval=concatenate_grid(bval0,bval1);
        } else if(b0used) {
          bval=bval0;
        } else if(b1used) {
          bval=bval1;
        }
        if(b0used || b1used) {
          bval=concatenate_grid(bval,bval2);
        } else {
          bval=bval2;
        }

        return bval;
      }

      arma::vec form_grid(modelpotential::nuclear_model_t model, double Rrms, int Nelem, double Rmax, int igrid, double zexp, int Nelem0, int igrid0, double zexp0, int Z, int Zl, int Zr, double Rhalf) {
        // Construct the radial basis
        arma::vec bval;
        if(model != modelpotential::POINT_NUCLEUS) {
          printf("Finite-nucleus grid\n");

          if(Zl != 0 || Zr != 0)
            throw std::logic_error("Off-center nuclei not supported in finite nucleus mode!\n");

          double rnuc;
          if(model == modelpotential::HOLLOW_NUCLEUS)
            rnuc = Rrms;
          else if(model == modelpotential::SPHERICAL_NUCLEUS)
            rnuc = sqrt(5.0/3.0)*Rrms;
          else if(model == modelpotential::GAUSSIAN_NUCLEUS)
            rnuc = 3*Rrms;
          else
            throw std::logic_error("Nuclear grid not handled!\n");

          bval=atomic::basis::finite_nuclear_grid(Nelem,Rmax,igrid,zexp,Nelem0,rnuc,igrid0,zexp0);

        } else if(Zl != 0 || Zr != 0) {
          printf("Off-center grid\n");
          bval=atomic::basis::offcenter_nuclear_grid(Nelem0,Z,std::max(Zl,Zr),Rhalf,Nelem,Rmax,igrid,zexp);
        } else {
          printf("Normal grid\n");
          bval=atomic::basis::normal_grid(Nelem,Rmax,igrid,zexp);
        }

        bval.print("Grid");

        return bval;
      }

      void angular_basis(int lmax, int mmax, arma::ivec & lval, arma::ivec & mval) {
        size_t nang=0;
        for(int l=0;l<=lmax;l++)
          nang+=2*std::min(mmax,l)+1;

        // Allocate memory
        lval.zeros(nang);
        mval.zeros(nang);

        // Store values
        size_t iang=0;
        for(int mabs=0;mabs<=mmax;mabs++)
          for(int l=mabs;l<=lmax;l++) {
            lval(iang)=l;
            mval(iang)=mabs;
            iang++;
            if(mabs>0) {
              lval(iang)=l;
              mval(iang)=-mabs;
              iang++;
            }
          }
        if(iang!=lval.n_elem)
          throw std::logic_error("Error.\n");
      }
    }
  }
}
