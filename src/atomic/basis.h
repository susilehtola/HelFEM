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
#ifndef ATOMIC_BASIS_H
#define ATOMIC_BASIS_H

#include <armadillo>
#include "../general/model_potential.h"
#include "../general/sap.h"
#include <RadialBasis.h>
#include "TwoDBasis.h"

namespace helfem {
  namespace atomic {
    namespace basis {
      /// Get the element grid for a normal calculation
      arma::vec normal_grid(int num_el, double rmax, int igrid, double zexp);
      /// Get the element grid for a finite nucleus
      arma::vec finite_nuclear_grid(int num_el, double rmax, int igrid, double zexp, int num_el_nuc, double rnuc, int igrid_nuc, double zexp_nuc);
      /// Get the element grid in the case of off-center nuclei
      arma::vec offcenter_nuclear_grid(int num_el0, int Zm, int Zlr, double Rhalf, int num_el, double rmax, int igrid, double zexp);
      /// Form the grid in the general case, using the above routines
      arma::vec form_grid(modelpotential::nuclear_model_t model, double Rrms, int Nelem, double Rmax, int igrid, double zexp, int Nelem0, int igrid0, double zexp0, int Z, int Zl, int Zr, double Rhalf);

      /// Constructs an angular basis
      void angular_basis(int lmax, int mmax, arma::ivec & lval, arma::ivec & mval);
    }
  }
}


#endif
