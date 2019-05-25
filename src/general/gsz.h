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
#ifndef GSZ_H
#define GSZ_H

#include <armadillo>

namespace helfem {
  namespace GSZ {
    /// Get tabulated parameter values from Green-Sellin-Zachor 1969 paper
    void GSZ_parameters(int Z, double & d_Z, double & H_Z);

    /// Calculate GSZ charge at r for nucleus with charge Z and parameters d_Z and H_Z
    arma::vec Z_GSZ(const arma::vec & r, double Z, double d_Z, double H_Z);
    /// Calculate GSZ charge at r for nucleus with charge Z, using default parameter values
    arma::vec Z_GSZ(const arma::vec & r, int Z);

    /// Calculate Thomas-Fermi potential (arXiv:physics/0511017)
    arma::vec Z_thomasfermi(const arma::vec & r, int Z);
  }
}

#endif
