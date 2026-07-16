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
#include "angular.h"
#include "spherical_harmonics.h"
#include <ArmaEigen.h>

int main(void) {
  int lsph=5;
  for(int lang=2;lang<=3*lsph;lang++) {

    // Get quadrature rule (angular_* is Eigen-typed; this analysis test
    // still runs its overlap check with arma, so bridge once here).
    helfem::Vector cth_e, phi_e, w_e;
    //helfem::angular::angular_chebyshev(lang,cth_e,phi_e,w_e);
    helfem::angular::angular_lobatto(lang,cth_e,phi_e,w_e);
    arma::vec cth(helfem::to_arma(cth_e)), phi(helfem::to_arma(phi_e)), w(helfem::to_arma(w_e));
    
    // Evaluate spherical harmonics
    int nsph=(lsph+1)*(lsph+1);
    arma::cx_mat sph(nsph,cth.n_elem);
    for(size_t iang=0;iang<cth.n_elem;iang++) {
      int ioff=0;
      for(int l=0;l<=lsph;l++)
        for(int m=-l;m<=l;m++)
          sph(ioff++,iang)=spherical_harmonics(l,m,cth(iang),phi(iang));
      if(ioff!=nsph)
        throw std::logic_error("Count wrong!\n");
    }

    // Calculate quadrature
    arma::mat ovl(arma::abs(sph*arma::diagmat(w)*arma::trans(sph)));
    //ovl.print("Overlap");
    ovl-=arma::eye<arma::mat>(ovl.n_rows,ovl.n_cols);
    printf("Difference from orthonormality with lang=%i is %e max %e\n",lang,arma::norm(ovl,"fro"),arma::max(arma::max(ovl)));
  }
  
  return 0;
}

  
