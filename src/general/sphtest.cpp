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

    // Get quadrature rule (angular_* is Eigen-typed).
    helfem::Vector cth, phi, w;
    //helfem::angular::angular_chebyshev(lang,cth,phi,w);
    helfem::angular::angular_lobatto(lang,cth,phi,w);

    // Evaluate spherical harmonics
    int nsph=(lsph+1)*(lsph+1);
    Eigen::MatrixXcd sph(nsph,cth.size());
    for(Eigen::Index iang=0;iang<cth.size();iang++) {
      int ioff=0;
      for(int l=0;l<=lsph;l++)
        for(int m=-l;m<=l;m++)
          sph(ioff++,iang)=spherical_harmonics(l,m,cth(iang),phi(iang));
      if(ioff!=nsph)
        throw std::logic_error("Count wrong!\n");
    }

    // Calculate quadrature
    const Eigen::VectorXcd wc = w.cast<std::complex<double>>();
    Eigen::MatrixXd ovl((sph*wc.asDiagonal()*sph.adjoint()).cwiseAbs());
    //helfem::io::print_matrix("Overlap", ovl);
    ovl-=Eigen::MatrixXd::Identity(ovl.rows(),ovl.cols());
    printf("Difference from orthonormality with lang=%i is %e max %e\n",lang,ovl.norm(),ovl.maxCoeff());
  }
  
  return 0;
}

  
