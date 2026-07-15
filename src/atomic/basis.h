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
      helfem::Vector normal_grid(int num_el, double rmax, int igrid, double zexp);
      /// Get the element grid for a finite nucleus
      arma::vec finite_nuclear_grid(int num_el, double rmax, int igrid, double zexp, int num_el_nuc, double rnuc, int igrid_nuc, double zexp_nuc);
      /// Get the element grid in the case of off-center nuclei
      arma::vec offcenter_nuclear_grid(int num_el0, int Zm, int Zlr, double Rhalf, int num_el, double rmax, int igrid, double zexp);
      /// Form the grid in the general case, using the above routines
      arma::vec form_grid(modelpotential::nuclear_model_t model, double Rrms, int Nelem, double Rmax, int igrid, double zexp, int Nelem0, int igrid0, double zexp0, int Z, int Zl, int Zr, double Rhalf);

      /// Form the grid in case of added boundary due to confinement
      arma::vec form_grid(modelpotential::nuclear_model_t model, double Rrms, int Nelem, double Rmax, int igrid, double zexp, int Nelem0, int igrid0, double zexp0, int Z, int Zl, int Zr, double Rhalf, bool add_el, double r);

      /// Constructs an angular basis
      void angular_basis(int lmax, int mmax, Eigen::VectorXi & lval, Eigen::VectorXi & mval);
    }
  }
}


#endif
