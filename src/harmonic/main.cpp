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

#include "PolynomialBasis.h"
#include "FiniteElementBasis.h"
#include "Matrix.h"
#include <lib1dfem/chebyshev.h>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>

using namespace helfem;

namespace {
  helfem::Matrix overlap(const polynomial_basis::FiniteElementBasis & fem,
                         const helfem::Vector & x, const helfem::Vector & wx) {
    return fem.matrix_element(false, false, x, wx, nullptr);
  }

  double square_potential(double r) { return r * r; }

  helfem::Matrix potential(const polynomial_basis::FiniteElementBasis & fem,
                           const helfem::Vector & x, const helfem::Vector & wx) {
    return fem.matrix_element(false, false, x, wx, square_potential);
  }

  helfem::Matrix kinetic(const polynomial_basis::FiniteElementBasis & fem,
                         const helfem::Vector & x, const helfem::Vector & wx) {
    return fem.matrix_element(true, true, x, wx, nullptr);
  }

  void write_raw_ascii(const std::string & path, const helfem::Vector & v) {
    std::ofstream out(path);
    for (Eigen::Index i = 0; i < v.size(); ++i)
      out << v(i) << "\n";
  }
  void write_raw_ascii(const std::string & path, const helfem::Matrix & m) {
    std::ofstream out(path);
    for (Eigen::Index i = 0; i < m.rows(); ++i) {
      for (Eigen::Index j = 0; j < m.cols(); ++j) {
        if (j) out << " ";
        out << m(i, j);
      }
      out << "\n";
    }
  }
}

int main(int argc, char **argv) {
  if (argc != 6) {
    printf("Usage: %s xmax Nel Nnode primbas Nquad\n", argv[0]);
    return 1;
  }

  const double xmax   = std::atof(argv[1]);
  const int    Nelem  = std::atoi(argv[2]);
  const int    Nnodes = std::atoi(argv[3]);
  const int    primbas = std::atoi(argv[4]);
  const int    Nquad  = std::atoi(argv[5]);

  printf("Running calculation with xmax=%e and %i elements.\n", xmax, Nelem);
  printf("Using %i point quadrature rule.\n", Nquad);

  auto poly = std::shared_ptr<const polynomial_basis::PolynomialBasis>(
      polynomial_basis::get_basis(primbas, Nnodes));

  // Radial grid: linspace(-xmax, xmax, Nelem+1) on helfem::Vector.
  const helfem::Vector r = helfem::Vector::LinSpaced(Nelem + 1, -xmax, xmax);

  polynomial_basis::FiniteElementBasis fem(poly, r,
      /*zero_func_left=*/true,  /*zero_deriv_left=*/true,
      /*zero_func_right=*/true, /*zero_deriv_right=*/true);

  helfem::Vector xq, wq;
  helfem::lib1dfem::chebyshev::chebyshev<double>(Nquad, xq, wq);

  helfem::Matrix bf, dbf;
  poly->eval_dnf(xq, bf,  0, 1.0);
  poly->eval_dnf(xq, dbf, 1, 1.0);

  write_raw_ascii("x.dat",  xq);
  write_raw_ascii("bf.dat", bf);
  write_raw_ascii("dbf.dat", dbf);

  const size_t Nbf = fem.get_nbf();
  printf("Basis set contains %i functions\n", (int) Nbf);

  const helfem::Matrix S = overlap  (fem, xq, wq);
  const helfem::Matrix V = potential(fem, xq, wq);
  const helfem::Matrix T = kinetic  (fem, xq, wq);
  const helfem::Matrix H = T + V;

  // Symmetric orthonormalisation: Sinvh = Svec * diag(Sval^{-1/2}) * Svec^T.
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

  printf("Eigenvalues\n");
  const Eigen::Index neig = std::min<Eigen::Index>(E.size(), 8);
  for (Eigen::Index i = 0; i < neig; ++i)
    printf("%i % 10.6f % 10.6f\n", (int) i, E(i), E(i) - (2 * i + 1));

  helfem::Matrix Smo = C.transpose() * S * C;
  Smo -= helfem::Matrix::Identity(Smo.rows(), Smo.cols());
  printf("Orbital orthonormality devation is %e\n", Smo.norm());

  return 0;
}
