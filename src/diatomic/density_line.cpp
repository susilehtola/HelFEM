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
#include "../general/cmdline.h"
#include "../general/checkpoint.h"
#include "../general/constants.h"
#include "../general/timer.h"
#include "utils.h"
#include "basis.h"
#include "Matrix.h"
#include "ArmaEigen.h"
#include <cfloat>
#include <climits>
#include <complex>
#include <fstream>

using namespace helfem;

int main(int argc, char **argv) {
  cmdline::parser parser;

  parser.add<std::string>("load", 0, "load guess from checkpoint", false, "");
  parser.add<double>("x", 0, "value of x", false, 0.0);
  parser.add<double>("y", 0, "value of y", false, 0.0);
  parser.add<double>("zmin", 0, "z min", false, -5.0);
  parser.add<double>("zmax", 0, "z max", false, 5.0);
  parser.add<int>("Nz", 0, "number of points in z", false, 101);
  parser.add<std::string>("savedens", 0, "save density to file", false, "density.dat");
  parser.parse_check(argc, argv);

  const std::string load = parser.get<std::string>("load");
  const double x         = parser.get<double>("x");
  const double y         = parser.get<double>("y");
  const double zmin      = parser.get<double>("zmin");
  const double zmax      = parser.get<double>("zmax");
  const std::string savedens = parser.get<std::string>("savedens");
  const int Nz           = parser.get<int>("Nz");

  // Load checkpoint. Basis + density matrices are still arma-typed on
  // the diatomic side; bridge to Eigen at the read boundary.
  Checkpoint loadchk(load, false);
  diatomic::basis::TwoDBasis basis;
  loadchk.read(basis);
  arma::mat Pa_a, Pb_a;
  loadchk.read("Pa", Pa_a);
  loadchk.read("Pb", Pb_a);
  const helfem::Matrix Pa = helfem::to_eigen(Pa_a);
  const helfem::Matrix Pb = helfem::to_eigen(Pb_a);
  double Rhalf;
  loadchk.read("Rhalf", Rhalf);

  const helfem::Vector z = helfem::Vector::LinSpaced(Nz, zmin, zmax);

  const double phi  = std::atan2(y, x);
  const double xysq = x * x + y * y;

  // Density on the line (z col + alpha col + beta col + total col).
  helfem::Matrix den = helfem::Matrix::Zero(Nz, 4);
  for (Eigen::Index iz = 0; iz < z.size(); ++iz) {
    const double zi = z(iz);
    const double ra = std::sqrt(std::pow(zi + Rhalf, 2) + xysq);
    const double rb = std::sqrt(std::pow(zi - Rhalf, 2) + xysq);
    const double xi = (ra + rb) / (2.0 * Rhalf);
    double eta      = (ra - rb) / (2.0 * Rhalf);
    if (eta < -1.0) eta = -1.0;
    if (eta >  1.0) eta =  1.0;
    const double mu = utils::arcosh(xi);
    if (mu > basis.get_mumax()) {
      den(iz, 0) = zi;
      continue;
    }
    // basis.eval_bf returns arma::cx_vec; bridge to Eigen VectorXcd.
    const arma::cx_vec bf_a = basis.eval_bf(mu, eta, phi);
    Eigen::VectorXcd bf(bf_a.n_elem);
    for (arma::uword i = 0; i < bf_a.n_elem; ++i) bf(i) = bf_a(i);
    // rho_spin = Re(bf^* . P . bf) with P symmetric real.
    const double rhoa = (bf.adjoint() * Pa * bf).real().value();
    const double rhob = (bf.adjoint() * Pb * bf).real().value();
    den(iz, 0) = zi;
    den(iz, 1) = rhoa;
    den(iz, 2) = rhob;
    den(iz, 3) = rhoa + rhob;
  }

  printf("Saving density to file %s\n", savedens.c_str());
  std::ofstream out(savedens);
  for (Eigen::Index i = 0; i < den.rows(); ++i) {
    for (Eigen::Index j = 0; j < den.cols(); ++j) {
      if (j) out << " ";
      out << den(i, j);
    }
    out << "\n";
  }

  return 0;
}
