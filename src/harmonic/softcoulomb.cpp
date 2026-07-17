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
#include "PolynomialBasis.h"
#include "FiniteElementBasis.h"
#include "Matrix.h"
#include <chebyshev.h>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>

using namespace helfem;

namespace {
  helfem::Matrix overlap(const polynomial_basis::FiniteElementBasis & fem,
                         const helfem::Vector & x, const helfem::Vector & wx) {
    return fem.matrix_element(false, false, x, wx, nullptr);
  }

  helfem::Matrix potential(const polynomial_basis::FiniteElementBasis & fem,
                           const helfem::Vector & x, const helfem::Vector & wx,
                           double z, double x0, double alpha, bool abs) {
    std::function<double(double)> soft_coulomb;
    if (abs)
      soft_coulomb = [z, x0, alpha](double x) { return -z / (std::abs(x - x0) + alpha); };
    else
      soft_coulomb = [z, x0, alpha](double x) {
        return -z / std::sqrt((x - x0) * (x - x0) + alpha * alpha);
      };
    return fem.matrix_element(false, false, x, wx, soft_coulomb);
  }

  helfem::Matrix kinetic(const polynomial_basis::FiniteElementBasis & fem,
                         const helfem::Vector & x, const helfem::Vector & wx) {
    return 0.5 * fem.matrix_element(true, true, x, wx, nullptr);
  }
}

int main(int argc, char **argv) {
  cmdline::parser parser;

  parser.add<double>("xmax",   0, "practical infinity in au",   false, 40.0);
  parser.add<int>("nelem",     0, "number of elements",         false, 5);
  parser.add<int>("nnodes",    0, "number of elements",         false, 15);
  parser.add<int>("primbas",   0, "primitive basis",            false, 4);
  parser.add<int>("nquad",     0, "primitive basis",            false, -1);
  parser.add<int>("Z1",        0, "primitive basis",            true);
  parser.add<int>("Z2",        0, "primitive basis",            true);
  parser.add<double>("R",      0, "Bond length",                true);
  parser.add<double>("alpha",  0, "Coulomb regularization parameter", true);
  parser.add<bool>("abs",      0, "Use 1/(|x-x0|+alpha) instead of 1/sqrt( (x-x0)^2 + alpha^2 ) as potential", false, 0);
  parser.add<std::string>("save", 0, "Checkpoint to save results to", false, "softcoulomb.chk");

  parser.parse_check(argc, argv);
  const double xmax   = parser.get<double>("xmax");
  const int    Nelem  = parser.get<int>("nelem");
  const int    Nnodes = parser.get<int>("nnodes");
  const int    primbas = parser.get<int>("primbas");
        int    Nquad  = parser.get<int>("nquad");
  const int    Z1     = parser.get<int>("Z1");
  const int    Z2     = parser.get<int>("Z2");
  const double R      = parser.get<double>("R");
  const double alpha  = parser.get<double>("alpha");
  const bool   abs    = parser.get<bool>("abs");
  const std::string save = parser.get<std::string>("save");

  if (abs)
    printf("Using potential V(x) = -Z / ( |x-x0| + alpha )\n");
  else
    printf("Using potential V(x) = -Z / sqrt( (x-x0)^2 + alpha^2 )\n");

  auto poly = std::shared_ptr<const polynomial_basis::PolynomialBasis>(
      polynomial_basis::get_basis(primbas, Nnodes));
  if (Nquad < 0) Nquad = 5 * poly->get_nbf();

  const helfem::Vector x = helfem::Vector::LinSpaced(Nelem + 1, -xmax, xmax);

  polynomial_basis::FiniteElementBasis fem(poly, x,
      /*zero_func_left=*/true,  /*zero_deriv_left=*/true,
      /*zero_func_right=*/true, /*zero_deriv_right=*/true);

  helfem::Vector xq, wq;
  helfem::chebyshev::chebyshev<double>(Nquad, xq, wq);

  const size_t Nbf = fem.get_nbf();
  printf("Basis set contains %i functions\n", (int) Nbf);

  const helfem::Matrix S = overlap(fem, xq, wq);
  const helfem::Matrix V = potential(fem, xq, wq, Z1, -0.5 * R, alpha, abs)
                         + potential(fem, xq, wq, Z2,  0.5 * R, alpha, abs);
  const helfem::Matrix T = kinetic(fem, xq, wq);
  const helfem::Matrix H = T + V;

  Eigen::SelfAdjointEigenSolver<helfem::Matrix> Ses(S);
  const helfem::Vector Sval = Ses.eigenvalues();
  const helfem::Matrix Svec = Ses.eigenvectors();
  printf("Smallest value of overlap matrix is % e, condition number is %e\n",
         Sval(0), Sval(Sval.size() - 1) / Sval(0));
  const helfem::Vector Sdiag = S.diagonal().cwiseAbs();
  printf("Smallest and largest bf norms are %e and %e\n",
         Sdiag.minCoeff(), Sdiag.maxCoeff());

  const helfem::Matrix Sinvh =
      Svec * Sval.array().pow(-0.5).matrix().asDiagonal() * Svec.transpose();

  const helfem::Matrix Horth = Sinvh.transpose() * H * Sinvh;

  Eigen::SelfAdjointEigenSolver<helfem::Matrix> Hes(Horth);
  const helfem::Vector E = Hes.eigenvalues();
  helfem::Matrix C = Sinvh * Hes.eigenvectors();

  for (int i = 0; i < 10 && i < E.size(); ++i)
    printf("E[%i] = % .15e\n", i, E(i));

  helfem::Matrix Smo = C.transpose() * S * C;
  Smo -= helfem::Matrix::Identity(Smo.rows(), Smo.cols());
  printf("Orbital orthonormality devation is %e\n", Smo.norm());

  // Evaluate the basis set: 0th derivative -- native Eigen through the
  // FE interface.
  const helfem::Matrix bfval = fem.eval_dnf(xq, 0);
  const helfem::Matrix phival = bfval * C;
  const helfem::Vector coords = fem.eval_coord(xq);
  const helfem::Vector weights = fem.eval_weights(wq);

  helfem::Matrix Sgrid = phival.transpose() * weights.asDiagonal() * phival;
  Sgrid -= helfem::Matrix::Identity(Sgrid.rows(), Sgrid.cols());
  printf("Orbital orthonormality devation on grid is %e\n", Sgrid.norm());

  Checkpoint chkpt(save, true);
  chkpt.write("bf",      bfval);
  chkpt.write("C",       C);
  chkpt.write("phi",     phival);
  chkpt.write("coords",  coords);
  chkpt.write("weights", weights);

  return 0;
}
