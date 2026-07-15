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

// How compressible are the ATOMIC in-element two-electron integrals?
//
// Same question as src/diatomic/teirank.cpp, and the same situation: the
// cross-element contributions to J and K are already contracted in factorized
// (disjoint r^L / r^-L-1) form, so the in-element block is the only 4-index
// object left. It is stored as an (Ni^2 x Ni^2) matrix per (element, L), and
// again as an exchange-ordered copy (prim_ktei), so storage grows as nnodes^4.
//
// Here the kernel is the single-channel r_<^L / r_>^(L+1), which is a positive
// kernel -- so unlike the diatomic 2-channel case (indefinite for odd |M|) a
// genuine pivoted Cholesky should apply, with no sign bookkeeping.
//
// libhelfem already provides FEMRadialBasis::twoe_integral_cholesky(); this
// probe measures the rank it finds, checks that L L' reproduces the exact
// tensor, and confirms that the exchange PAIRING is full rank -- i.e. that K
// must go through RI rather than compressing the exchange-ordered tensor.

#include "../general/cmdline.h"
#include "basis.h"
#include "../general/tei_utils.h"
#include "FiniteElementBasis.h"
#include "RadialBasis.h"
#include "PolynomialBasis.h"
#include <ArmaEigen.h>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <cstdio>

using namespace helfem;

int main(int argc, char **argv) {
  cmdline::parser parser;
  parser.add<double>("Rmax", 0, "practical infinity", false, 40.0);
  parser.add<int>("nelem", 0, "number of elements", false, 5);
  parser.add<int>("nnodes", 0, "nodes per element", false, 15);
  parser.add<int>("primbas", 0, "primitive radial basis", false, 4);
  parser.add<int>("Lmax", 0, "largest multipole to probe", false, 6);
  parser.add<double>("tol", 0, "Cholesky tolerance", false, 1e-12);
  parser.parse_check(argc, argv);

  const double Rmax = parser.get<double>("Rmax");
  const int Nelem = parser.get<int>("nelem");
  const int Nnodes = parser.get<int>("nnodes");
  const int primbas = parser.get<int>("primbas");
  const int Lmax = parser.get<int>("Lmax");
  const double tol = parser.get<double>("tol");

  auto poly = std::shared_ptr<const polynomial_basis::PolynomialBasis>(
      polynomial_basis::get_basis(primbas, Nnodes));
  const int Nquad = 5 * poly->get_nbf();

  const helfem::Vector bval = atomic::basis::normal_grid(Nelem, Rmax, 4, 1.0);
  polynomial_basis::FiniteElementBasis fem(poly, bval,
                                            true, false, true, false);
  atomic::basis::FEMRadialBasis radial(fem, Nquad);

  printf("nnodes = %i, nquad = %i, tol = %.0e\n\n", Nnodes, Nquad, tol);
  printf("%3s %3s | %6s | %6s | %11s | %11s | %11s\n",
         "iel", "L", "Ni^2", "rank", "|T-LL'|/|T|", "min eig", "K-pairing rank");
  printf("---------------------------------------------------------------------------------\n");

  for (size_t iel = 0; iel < (size_t) std::min(Nelem, 3); iel++) {
    for (int L = 0; L <= Lmax; L++) {
      const helfem::Matrix T = radial.twoe_integral(L, iel);
      const Eigen::Index n = T.rows();
      const size_t Ni = (size_t) std::lround(std::sqrt((double) n));

      // Existing libhelfem pivoted Cholesky
      const helfem::Matrix Lf = radial.twoe_integral_cholesky(L, iel, tol);
      const double err = (T - Lf * Lf.transpose()).norm() / T.norm();

      // Is it really PSD?
      Eigen::SelfAdjointEigenSolver<helfem::Matrix> es(
          helfem::Matrix(0.5 * (T + T.transpose())), Eigen::EigenvaluesOnly);
      const double emin = es.eigenvalues().minCoeff();
      const double emax = es.eigenvalues().cwiseAbs().maxCoeff();

      // Exchange pairing: is it compressible at all?
      const helfem::Matrix Kt(helfem::to_eigen(
          utils::exchange_tei(helfem::to_arma(T), Ni, Ni, Ni, Ni)));
      Eigen::JacobiSVD<helfem::Matrix> svd(Kt);
      const helfem::Vector sv = svd.singularValues();
      int krank = 0;
      for (Eigen::Index i = 0; i < sv.size(); i++)
        if (sv(i) / sv(0) > 1e-12) krank++;

      printf("%3i %3i | %6i | %6i | %11.2e | %11.2e | %6i / %i\n",
             (int) iel, L, (int) n, (int) Lf.cols(), err, emin / emax,
             krank, (int) n);
    }
    printf("\n");
  }
  return 0;
}
