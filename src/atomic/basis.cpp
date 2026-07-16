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
#include "quadrature.h"
#include "chebyshev.h"
#include "../general/spherical_harmonics.h"
#include "../general/gaunt.h"
#include "utils.h"
#include "../general/scf_helpers.h"
#include "../general/eigen_io.h"
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <helfem.h>
#include <ArmaEigen.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace helfem {
  namespace atomic {
    namespace basis {
      static helfem::Vector concatenate_grid(const helfem::Vector & left, const helfem::Vector & right) {
        if(!left.size())
          return right;
        if(!right.size())
          return left;

        if(left(0) != 0.0)
          throw std::logic_error("left vector doesn't start from zero");
        if(right(0) != 0.0)
          throw std::logic_error("right vector doesn't start from zero");

        // Concatenated vector
        helfem::Vector ret(left.size() + right.size() - 1);
        ret.segment(0,left.size())=left;
        ret.segment(left.size(),right.size()-1)=(right.segment(1,right.size()-1).array() + left(left.size()-1)).matrix();
        return ret;
      }

      helfem::Vector normal_grid(int num_el, double rmax, int igrid, double zexp) {
        return utils::get_grid(rmax,num_el,igrid,zexp);
      }

      helfem::Vector finite_nuclear_grid(int num_el, double rmax, int igrid, double zexp, int num_el_nuc, double rnuc, int igrid_nuc, double zexp_nuc) {
        if(num_el_nuc) {
          // Grid for the finite nucleus
          helfem::Vector bnuc(utils::get_grid(rnuc,num_el_nuc,igrid_nuc,zexp_nuc));
          // and the one for the electrons
          helfem::Vector belec(utils::get_grid(rmax-rnuc,num_el,igrid,zexp));

          helfem::Vector bnucel(concatenate_grid(bnuc,bnuc));
          return concatenate_grid(bnucel,belec);
        } else {
          return utils::get_grid(rmax,num_el,igrid,zexp);
        }
      }

      helfem::Vector offcenter_nuclear_grid(int num_el0, int Zm, int Zlr, double Rhalf, int num_el, double rmax, int igrid, double zexp) {
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
        helfem::Vector bval0, bval1;
        if(b0used) {
          // 0 to b0
          bval0=utils::get_grid(b0,num_el0,igrid,zexp);
        }
        if(b1used) {
          // b0 to b1

          // Reverse grid to get tighter spacing around nucleus
          helfem::Vector b1grid(utils::get_grid(b1-b0,num_el0,igrid,zexp));
          bval1=-b1grid.reverse();
          bval1.array()+=(b1-b0);
          // Assert numerical exactness
          bval1(0)=0.0;
          bval1(bval1.size()-1)=b1-b0;
        }
        helfem::Vector bval2=utils::get_grid(b2-b1,num_el,igrid,zexp);

        helfem::Vector bval;
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

      helfem::Vector form_grid(modelpotential::nuclear_model_t model, double Rrms, int Nelem, double Rmax, int igrid, double zexp, int Nelem0, int igrid0, double zexp0, int Z, int Zl, int Zr, double Rhalf) {
	return form_grid(model, Rrms, Nelem, Rmax, igrid, zexp, Nelem0, igrid0, zexp0, Z, Zl, Zr, Rhalf, false, 0.0);
      }

      helfem::Vector form_grid(modelpotential::nuclear_model_t model, double Rrms, int Nelem, double Rmax, int igrid, double zexp, int Nelem0, int igrid0, double zexp0, int Z, int Zl, int Zr, double Rhalf, bool add_el, double shift_conf) {
        // Construct the radial basis
        helfem::Vector bval;
        if(model != modelpotential::POINT_NUCLEUS && model != modelpotential::REGULARIZED_NUCLEUS) {
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

	if(add_el) {
	  // Check that r is not in bval
	  bool in_bval = false;
	  for (Eigen::Index i = 0; i < bval.size(); i++)
	    if (bval(i) == shift_conf)
	      in_bval = true;

	  // Add
	  if (!in_bval) {
	    helfem::Vector newbval(bval.size() + 1);
	    newbval.segment(0, bval.size()) = bval;
	    newbval(bval.size()) = shift_conf;
	    std::sort(newbval.data(), newbval.data() + newbval.size());
	    bval = newbval;
	  }

	}

        helfem::io::print_matrix("Grid", helfem::Matrix(bval));

        return bval;
      }

      void angular_basis(int lmax, int mmax, Eigen::VectorXi & lval, Eigen::VectorXi & mval) {
        size_t nang=0;
        for(int l=0;l<=lmax;l++)
          nang+=2*std::min(mmax,l)+1;

        // Allocate memory
        lval=Eigen::VectorXi::Zero(nang);
        mval=Eigen::VectorXi::Zero(nang);

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
        if(iang!=(size_t) lval.size())
          throw std::logic_error("Error.\n");
      }
    }
  }
}
