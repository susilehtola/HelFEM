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
#ifndef UTILS_H
#define UTILS_H

#include <armadillo>

namespace helfem {
  namespace utils {
    /// inverse cosh
    double arcosh(double x);
    /// inverse cosh
    arma::vec arcosh(const arma::vec & x);
    /// inverse sinh
    double arsinh(double x);
    /// inverse sinh
    arma::vec arsinh(const arma::vec & x);

    /// Form two-electron integrals from product of large-r and small-r radial moment matrices
    arma::mat product_tei(const arma::mat & big, const arma::mat & small);

    /// Check that the two-electron integral has proper symmetry i<->j and k<->l
    void check_tei_symmetry(const arma::mat & tei, size_t Ni, size_t Nj, size_t Nk, size_t Nl);

    /// Permute indices (ij|kl) -> (jk|il)
    arma::mat exchange_tei(const arma::mat & tei, size_t Ni, size_t Nj, size_t Nk, size_t Nl);

    /**
     * Form radial grid for a calculation, ranging from r=0 to r=rmax.
     *
     * igrid: 0 for linear grid
     *        1 for quadratic grid
     *        2 for generalized polynomial grid with exponent zexp
     *        3 for generalized exponential grid with parameter zexp
     */
    arma::vec get_grid(double rmax, int num_el, int igrid, double zexp);

    /**
     * Calculate SAP potential
     */
    arma::vec sap_potential(int Z, const arma::vec & r);

    /// Case independent string comparison
    int stricmp(const std::string & str1, const std::string & str2);
  }
}

#endif
