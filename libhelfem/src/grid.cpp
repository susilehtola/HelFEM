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
#include <helfem.h>

arma::vec helfem::utils::get_grid(double rmax, int num_el, int igrid, double zexp) {
  // Boundary values
  arma::vec bval;

  // Get boundary values
  switch (igrid) {
  // linear grid
  case (1):
    if (helfem::verbose)
      printf("Using linear grid\n");
    bval = arma::linspace<arma::vec>(0, rmax, num_el + 1);
    break;

  // quadratic grid (Schweizer et al 1999)
  case (2):
    if (helfem::verbose)
      printf("Using quadratic grid\n");
    bval.zeros(num_el + 1);
    for (int i = 0; i <= num_el; i++)
      bval(i) = i * i * rmax / (num_el * num_el);
    break;

  case (3):
    // generalized polynomial grid, monotonic decrease till zexp~3, after that fails to work
    if (helfem::verbose)
      printf("Using generalized polynomial grid, zexp = %e\n", zexp);
    bval.zeros(num_el + 1);
    for (int i = 0; i <= num_el; i++)
      bval(i) = rmax * std::pow(i * 1.0 / num_el, zexp);
    break;

  // generalized exponential grid, monotonic decrease till zexp~2, after that fails to work
  case (4):
    if (helfem::verbose)
      printf("Using generalized exponential grid, zexp = %e\n", zexp);
    bval = arma::exp(arma::pow(arma::linspace<arma::vec>(
                                   0, std::pow(log(rmax + 1), 1.0 / zexp), num_el + 1),
                               zexp)) -
           arma::ones<arma::vec>(num_el + 1);
    break;

  // geometric grid, Cancès and Mourad 2018
  case (5):
    if (helfem::verbose)
      printf("Using geometric grid of doi:10.2140/camcos.2018.13.139, s = %e\n", zexp);
    if(zexp<=0.0 or zexp>=1.0)
      throw std::logic_error("Invalid value for s parameter!\n");
    // We compute the h_k, see p. 158
    {
      arma::vec hk(num_el);
      hk(num_el-1) = (1.0-zexp)/(1.0-std::pow(zexp,num_el))*rmax;
      for(int iel=num_el-2;iel>=0;iel--)
        hk(iel)=zexp*hk(iel+1);
      // and then set the nodes
      bval.zeros(num_el+1);
      for(int iel=0;iel<num_el;iel++)
        bval(iel+1) = bval(iel) + hk(iel);
    }
    break;

  default:
    throw std::logic_error("Invalid choice for grid\n");
  }

  // Make sure start and end points are numerically exact
  bval(0) = 0.0;
  bval(bval.n_elem - 1) = rmax;

  return bval;
}
