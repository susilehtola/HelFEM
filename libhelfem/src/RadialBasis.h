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
#ifndef ATOMIC_BASIS_RADIALBASIS_H
#define ATOMIC_BASIS_RADIALBASIS_H

#include <helfem/ModelPotential.h>
#include "polynomial_basis.h"
#include <armadillo>

namespace helfem {
namespace atomic {
namespace basis {
/// Radial basis set
class RadialBasis {
  /// Quadrature points
  arma::vec xq;
  /// Quadrature weights
  arma::vec wq;

  /// Polynomial basis
  const polynomial_basis::PolynomialBasis *poly;
  /// Primitive polynomial basis functions evaluated on the quadrature grid
  arma::mat bf;
  /// Primitive polynomial basis function derivatives evaluated on the quadrature grid
  arma::mat df;

  /// Element boundary values
  arma::vec bval;

  /// Used basis function indices in element
  arma::uvec basis_indices(size_t iel) const;
  /// Get basis functions in element
  polynomial_basis::PolynomialBasis *
  get_basis(const polynomial_basis::PolynomialBasis *poly, size_t iel) const;

public:
  /// Dummy constructor
  RadialBasis();
  /// Construct radial basis
  RadialBasis(const polynomial_basis::PolynomialBasis *poly, int n_quad,
              const arma::vec &bval);

  /// Explicit copy constructor because of shared pointer
  RadialBasis(const RadialBasis &rh);
  /// Explicit assignment operator because of shared pointer
  RadialBasis &operator=(const RadialBasis &rh);
  /// Explicit destructor because of shared pointer
  ~RadialBasis();

  /// Add an element boundary
  void add_boundary(double r);

  /// Get polynomial basis
  polynomial_basis::PolynomialBasis *get_poly() const;
  /// Get basis functions in element
  arma::mat get_basis(const arma::mat &b, size_t iel) const;

  /// Get number of quadrature points
  int get_nquad() const;
  /// Get boundary values
  arma::vec get_bval() const;
  /// Get polynomial basis identifier
  int get_poly_id() const;
  /// Get polynomial basis order
  int get_poly_order() const;

  /// Get number of overlapping functions
  size_t get_noverlap() const;
  /// Number of basis functions
  size_t Nbf() const;
  /// Number of primitive functions in element
  size_t Nprim(size_t iel) const;
  /// Number of primitive functions in element
  size_t max_Nprim() const;

  /// Number of elements
  size_t Nel() const;
  /// Get function indices
  void get_idx(size_t iel, size_t &ifirst, size_t &ilast) const;

  /// Form density matrix
  arma::mat form_density(const arma::mat &Cl, const arma::mat &Cr, size_t nocc) const;

  /// Compute radial matrix elements <r^n> in element (overlap is n=0, nuclear is n=-1)
  arma::mat radial_integral(const arma::mat &bf_c, int n, size_t iel) const;
  /// Compute radial matrix elements <r^n> in element (overlap is n=0, nuclear is n=-1)
  arma::mat radial_integral(int n, size_t iel) const;

  /// Compute Bessel i_L integral
  arma::mat bessel_il_integral(int L, double lambda, size_t iel) const;
  /// Compute Bessel k_L integral
  arma::mat bessel_kl_integral(int L, double lambda, size_t iel) const;

  /// Compute overlap matrix in element
  arma::mat overlap(size_t iel) const;
  /// Compute primitive kinetic energy matrix in element (excluding l part)
  arma::mat kinetic(size_t iel) const;
  /// Compute l part of kinetic energy matrix
  arma::mat kinetic_l(size_t iel) const;
  /// Compute nuclear attraction matrix in element
  arma::mat nuclear(size_t iel) const;
  /// Compute model potential matrix in element
  arma::mat model_potential(const modelpotential::ModelPotential *nuc, size_t iel) const;
  /// Compute off-center nuclear attraction matrix in element
  arma::mat nuclear_offcenter(size_t iel, double Rhalf, int L) const;

  /// Compute primitive two-electron integral
  arma::mat twoe_integral(int L, size_t iel) const;
  /// Compute primitive Yukawa-screened two-electron integral
  arma::mat yukawa_integral(int L, double lambda, size_t iel) const;
  /// Compute primitive complementary error function two-electron integral
  arma::mat erfc_integral(int L, double lambda, size_t iel, size_t jel) const;
  /// Compute a spherically symmetric potential
  arma::mat spherical_potential(size_t iel) const;

  /// Compute cross-basis integral
  arma::mat radial_integral(const RadialBasis &rh, int n, bool lhder = false,
                            bool rhder = false) const;
  /// Compute cross-basis model potential integral
  arma::mat model_potential(const RadialBasis &rh,
                            const modelpotential::ModelPotential *model, bool lhder = false,
                            bool rhder = false) const;
  /// Compute projection
  arma::mat overlap(const RadialBasis &rh) const;

  /// Evaluate basis functions at quadrature points
  arma::mat get_bf(size_t iel) const;
  /// Evaluate derivatives of basis functions at quadrature points
  arma::mat get_df(size_t iel) const;
  /// Evaluate second derivatives of basis functions at quadrature points
  arma::mat get_lf(size_t iel) const;
  /// Get quadrature weights
  arma::vec get_wrad(size_t iel) const;
  /// Get r values
  arma::vec get_r(size_t iel) const;

  /// Evaluate nuclear density
  double nuclear_density(const arma::mat &P) const;
  /// Evaluate nuclear density gradient
  double nuclear_density_gradient(const arma::mat &P) const;
  /// Evaluate orbitals at nucleus
  arma::rowvec nuclear_orbital(const arma::mat &C) const;
};
} // namespace basis
} // namespace atomic
} // namespace helfem

#endif
