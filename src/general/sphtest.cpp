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
#include "angular.h"
#include "spherical_harmonics.h"

int main(void) {
  int lsph=5;
  for(int lang=2;lang<=3*lsph;lang++) {

    // Get quadrature rule
    arma::vec cth, phi, w;
    //helfem::angular::angular_chebyshev(lang,cth,phi,w);
    helfem::angular::angular_lobatto(lang,cth,phi,w);
    
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

  
