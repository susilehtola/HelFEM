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
#include "quadrature.h"
#include "PolynomialBasis.h"
#include "LIPBasis.h"
#include "Matrix.h"
#include "../general/eigen_io.h"
#include <lib1dfem/chebyshev.h>
#include <lib1dfem/lobatto.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>

using namespace helfem;

void run(double R, int n_quad) {
  // Basis functions on [0, R]: x/R, (R-x)/R.

  // Get primitive polynomial representation for LIP
  helfem::Vector x, w;
  helfem::lib1dfem::lobatto::lobatto_compute<double>(2, x, w);
  auto pbas = std::shared_ptr<const polynomial_basis::PolynomialBasis>(
      new polynomial_basis::LIPBasis(x, 0));

  // Gauss-Chebyshev nodes for the outer TEI quadrature.
  helfem::Vector xq, wq;
  helfem::lib1dfem::chebyshev::chebyshev<double>(n_quad, xq, wq);

  // Inner integral by quadrature.
  const helfem::Matrix teiinner =
      quadrature::twoe_inner_integral(0, R, xq, wq, pbas, 0);

  // Radial points at which the inner integral is sampled.
  helfem::Vector r = 0.5 * R * (helfem::Vector::Ones(xq.size()) + xq);

  // Analytical inner integrals for L=0 on this element.
  helfem::Matrix teiishould(r.size(), 4);
  for (Eigen::Index i = 0; i < r.size(); ++i) {
    const double ri  = r(i);
    const double R2  = R * R;
    teiishould(i, 0) = 1.0 - ri / R + (ri * ri) / (3.0 * R2);
    teiishould(i, 1) = ri * (-2.0 * ri + 3.0 * R) / (6.0 * R2);
    teiishould(i, 2) = teiishould(i, 1);
    teiishould(i, 3) = (ri * ri) / (3.0 * R2);
  }

  io::write_raw_ascii("r.dat",     r);
  io::write_raw_ascii("teii_q.dat", teiinner);
  io::write_raw_ascii("teii.dat",   teiishould);

  const helfem::Matrix teiidiff = teiishould - teiinner;
  printf("Error in inner integral is %e\n", teiidiff.norm());

  // Full inner+outer integral.
  helfem::Matrix teiq = quadrature::twoe_integral(0, R, xq, wq, pbas, 0);

  // Maple analytical values for the eight independent (ij|kl) entries.
  helfem::Matrix tei = helfem::Matrix::Zero(4, 4);
  // 1111
  tei(0, 0) = 47.0 / 180.0;
  // 111{2} = 1121
  tei(0, 1) = 11.0 / 360.0;
  tei(0, 2) = tei(0, 1);
  // 1122
  tei(0, 3) = 1.0 / 90.0;
  // 1211
  tei(1, 0) = 1.0 / 10.0;
  // 1212 = 1221
  tei(1, 1) = 1.0 / 40.0;
  tei(1, 2) = tei(1, 1);
  // 1222
  tei(1, 3) = 1.0 / 60.0;
  // 2111 = 1211, 2112 = 1212, 2121 = 1221, 2122 = 1222 by symmetry.
  tei(2, 0) = tei(1, 0);
  tei(2, 1) = tei(1, 1);
  tei(2, 2) = tei(1, 2);
  tei(2, 3) = tei(1, 3);
  // 2211
  tei(3, 0) = 3.0 / 20.0;
  // 2212 = 2221
  tei(3, 1) = 7.0 / 120.0;
  tei(3, 2) = tei(3, 1);
  // 2222
  tei(3, 3) = 1.0 / 15.0;
  tei = 4.0 * M_PI * (tei + tei.transpose().eval()) * R;

  io::print_matrix("Analytical", tei);
  io::print_matrix("Quadrature", teiq);
  io::print_matrix("Difference", helfem::Matrix(teiq - tei));
}

int main(int argc, char **argv) {
  if (argc != 3) {
    printf("Usage: %s nquad R\n", argv[0]);
    return 1;
  }
  const int    nquad = std::atoi(argv[1]);
  const double R     = std::atof(argv[2]);
  run(R, nquad);
  return 0;
}
