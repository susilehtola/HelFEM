/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "spherical_harmonics.h"
#include <cmath>
#include <cfloat>

std::complex<double> spherical_harmonics(int l, int m, double cth, double phi) {
  /* Calculate value of spherical harmonic Y_{lm} = N_{lm} P_{lm} (\cos \theta) e^{i m \phi} */
  if(m<0) { // std::sph_legendre requires m>=0; use Y_l^{-m} = (-1)^m conj(Y_l^m)
    return conj(pow(-1.0,m)*spherical_harmonics(l,-m,cth,phi));
  }

  std::complex<double> ylm;

  // First, calculate the exponential
  ylm=pow(M_E,std::complex<double>(0.0,m*phi)); // e^(im phi)
  // std::sph_legendre returns the normalised associated Legendre polynomial
  // including the Condon-Shortley phase. It takes theta, not cos(theta).
  ylm*=std::sph_legendre(static_cast<unsigned>(l), static_cast<unsigned>(m), std::acos(cth));

  return ylm;
}
