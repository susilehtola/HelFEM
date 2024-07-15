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

#include "gsz.h"

// Values for d_Z from Green 1969 paper
static const double d_Z_GSZ[]={1.00, 0.215, 0.563, 0.858, 0.979, 0.880, 0.776, 0.708, 0.575, 0.500, 0.561, 0.621, 0.729, 0.817, 0.868, 0.885, 0.881, 0.862, 1.006, 1.154, 1.116, 1.060, 0.996, 0.837, 0.866, 0.807, 0.751, 0.700, 0.606, 0.612, 0.631, 0.649, 0.663, 0.675, 0.684, 0.689, 0.744, 0.798, 0.855, 0.866, 0.831, 0.825, 0.855, 0.803, 0.788, 0.737, 0.754, 0.775, 0.810, 0.841, 0.870, 0.896, 0.919, 0.940, 1.022, 1.108, 1.150, 1.081, 0.970, 0.938, 0.905, 0.873, 0.842, 0.862, 0.830, 0.754, 0.728, 0.702, 0.677, 0.654, 0.665, 0.672, 0.676, 0.679, 0.680, 0.680, 0.679, 0.661, 0.657, 0.671, 0.690, 0.708, 0.726, 0.744, 0.761, 0.777, 0.818, 0.859, 0.899, 0.927, 0.887, 0.880, 0.872, 0.832, 0.822, 0.842, 0.830, 0.790, 0.778, 0.766, 0.754, 0.742, 0.755};

namespace helfem {
  namespace GSZ {
    void GSZ_parameters(int Z, double & d, double & H) {
      if((size_t) Z>sizeof(d_Z_GSZ)/sizeof(d_Z_GSZ[0])) {
	std::ostringstream oss;
	oss << "No GSZ parameters for Z = " << Z << "!\n";
	throw std::logic_error(oss.str());
      }

      d=d_Z_GSZ[Z];
      H=d*std::pow(Z-1,0.4);
    }

    double Z_GSZ(double r, double Z, double d_Z, double H_Z) {
      return 1.0 + (Z-1)*1.0/(1.0 + (exp(r/d_Z) - 1.0)*H_Z);
    }

    arma::vec Z_GSZ(const arma::vec & r, double Z, double d_Z, double H_Z) {
      arma::vec Zv(r.n_elem);
      for(size_t i=0;i<r.n_elem;i++)
        Zv[i]=Z_GSZ(r(i),Z,d_Z,H_Z);
      return Zv;
    }

    arma::vec Z_GSZ(const arma::vec & r, int Z) {
      double d, H;
      GSZ_parameters(Z,d,H);
      return Z_GSZ(r,Z,d,H);
    }

    double Z_thomasfermi(double r, int Z) {
      // arXiv physics/0511017
      const double alpha = 0.7280642371;
      const double beta = -0.5430794693;
      const double gamma = 0.3612163121;
      double x = r*cbrt(128*Z/(9.0*M_PI*M_PI));
      return Z*std::pow(1 + alpha*sqrt(x) + beta*x*exp(-gamma*sqrt(x)),2)*exp(-2*alpha*sqrt(x));
    }

    arma::vec Z_thomasfermi(const arma::vec & r, int Z) {
      arma::vec Zeff(r.n_elem);
      for(size_t i=0;i<r.n_elem;i++) {
        Zeff(i)=Z_thomasfermi(r(i),Z);
      }
      return Zeff;
    }

  }
}
