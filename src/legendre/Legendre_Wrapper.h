/*
Copyright (c) 2018 Susi Lehtola
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef LEGENDRE_WRAPPER_H
#define LEGENDRE_WRAPPER_H

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * Calculates a value of the regular Legendre function.
 */
double calc_Plm_val(int l, int m, double xi);
/**
 * Calculates a value of the normalized regular Legendre function.
 */
double calc_norm_Plm_val(int l, int m, double xi);

/**
 * Calculates the regular Legendre functions. Plm should be an array
 * of size (lmax+1)*(mmax+1). The values are stored in Fortran order
 * i.e. column-major.
 */
void calc_Plm_arr(double *Plm, int lmax, int mmax, double xi);
/**
 * Calculates the normalized regular Legendre functions. Plm should be
 * an array of size (lmax+1)*(mmax+1). The values are stored in
 * Fortran order i.e. column-major.
 */
void calc_norm_Plm_arr(double *Plm, int lmax, int mmax, double xi);

/**
 * Calculates a value of the irregular Legendre function.
 */
double calc_Qlm_val(int l, int m, double xi);
/**
 * Calculates the irregular Legendre functions. Qlm should be an array
 * of size (lmax+1)*(mmax+1). The values are stored in Fortran order
 * i.e. column-major.
 */
void calc_Qlm_arr(double *Qlm, int lmax, int mmax, double xi);

#ifdef __cplusplus
}
#endif

#endif
