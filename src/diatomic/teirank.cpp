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

// How compressible are the in-element two-electron integrals?
//
// The in-element block is the only 4-index object left in the diatomic code:
// the cross-element contributions to J and K are already contracted in the
// factorized (disjoint P/Q) form. It is stored as four (Nprim^2 x Nprim^2)
// matrices per (element, L, |M|) -- T00, T02, T20, T22 -- and enters both J and
// K through the symmetric 2-channel kernel
//
//   W = [  T00  -T02 ]
//       [ -T02'  T22 ]
//
// This probe reports its eigenvalue spectrum and the rank surviving a relative
// threshold -- i.e. exactly what a thresholded Cholesky (or, if the kernel
// turns out indefinite, an eigen-based low-rank) factorization would keep.
// A rank r replaces the O(Nprim^4) storage and contraction by O(Nprim^2 * r).

#include "../general/cmdline.h"
#include "basis.h"
#include "utils.h"
#include "../atomic/basis.h"
#include <ArmaEigen.h>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <cstdio>

using namespace helfem;

int main(int argc, char **argv) {
  cmdline::parser parser;
  parser.add<double>("Rbond", 0, "internuclear distance", false, 2.07);
  parser.add<double>("Rmax", 0, "practical infinity", false, 15.0);
  parser.add<int>("nelem", 0, "number of elements", false, 4);
  parser.add<int>("nnodes", 0, "nodes per element", false, 15);
  parser.add<int>("primbas", 0, "primitive radial basis", false, 4);
  parser.add<int>("lmax", 0, "maximum l", false, 3);
  parser.add<int>("mmax", 0, "maximum m", false, 2);
  parser.parse_check(argc, argv);

  const double Rbond = parser.get<double>("Rbond");
  const double Rmax = parser.get<double>("Rmax");
  const int Nelem = parser.get<int>("nelem");
  const int Nnodes = parser.get<int>("nnodes");
  const int primbas = parser.get<int>("primbas");
  const int lmax = parser.get<int>("lmax");
  const int mmax = parser.get<int>("mmax");

  auto poly = std::shared_ptr<const polynomial_basis::PolynomialBasis>(
      polynomial_basis::get_basis(primbas, Nnodes));
  const int Nquad = 5 * poly->get_nbf();

  arma::ivec lmmax(mmax + 1);
  lmmax.ones();
  lmmax *= lmax;
  arma::ivec lval, mval;
  diatomic::basis::lm_to_l_m(lmmax, lval, mval);

  const double Rhalf = 0.5 * Rbond;
  const double mumax = utils::arcosh(Rmax / Rhalf);
  arma::vec bval(atomic::basis::normal_grid(Nelem, mumax, 4, 1.0));

  diatomic::basis::TwoDBasis basis(7, 7, Rhalf, poly, Nquad, bval, lval, mval);

  const int nprim = (int) poly->get_nbf();
  const int n = nprim * nprim;
  printf("nnodes = %i -> Nprim = %i, Nprim^2 = %i, nquad = %i\n",
         Nnodes, nprim, n, Nquad);
  printf("In-element kernel W is %i x %i.\n", 2 * n, 2 * n);
  printf("Stored today: 4 x (%i x %i) doubles = %.2f MB per (element, L, |M|)\n\n",
         n, n, 4.0 * n * n * 8.0 / 1024 / 1024);

  printf("                    COULOMB pairing (%i x %i)                | EXCHANGE pairing (%i x %i)\n",
         2*n, 2*n, n, 4*n);
  printf("%3s %3s %4s | %11s | %-26s | %-10s | %-26s | %8s %8s\n", "iel", "L", "|M|", "max|eig|",
         "rank at 1e-6/-8/-10/-12", "definite?", "rank at 1e-6/-8/-10/-12", "|W-BSB'|", "RI-K err");
  printf("--------------------------------------------------------------------------------------------------------\n");

  // Build the factorization once
  basis.compute_tei(true);

  const int Lmax = 2 * lmax + 2;
  for (size_t iel = 0; iel < (size_t) std::min(Nelem, 2); iel++) {
    for (int L = 0; L <= Lmax; L++) {
      for (int M = 0; M <= std::min(L, 2 * mmax); M++) {
        helfem::Matrix W;
        try {
          W = basis.in_element_kernel(iel, L, M);
        } catch (...) {
          continue;   // (L,|M|) not in the table
        }

        Eigen::SelfAdjointEigenSolver<helfem::Matrix> es(W, Eigen::EigenvaluesOnly);
        const helfem::Vector ev = es.eigenvalues();
        const double amax = ev.cwiseAbs().maxCoeff();
        if (amax == 0.0) continue;
        const double emin = ev.minCoeff();

        int r[4] = {0, 0, 0, 0};
        const double thr[4] = {1e-6, 1e-8, 1e-10, 1e-12};
        for (Eigen::Index i = 0; i < ev.size(); i++) {
          const double a = std::abs(ev(i)) / amax;
          for (int t = 0; t < 4; t++)
            if (a > thr[t]) r[t]++;
        }

        char buf[64];
        snprintf(buf, sizeof(buf), "%4i %4i %4i %4i", r[0], r[1], r[2], r[3]);

        // Same integrals, exchange pairing. Not symmetric in general, so use
        // singular values.
        const helfem::Matrix Kcat = basis.in_element_kernel_exchange(iel, L, M);
        Eigen::JacobiSVD<helfem::Matrix> svd(Kcat);
        const helfem::Vector sv = svd.singularValues();
        const double smax = sv(0);
        int k[4] = {0,0,0,0};
        for (Eigen::Index i = 0; i < sv.size(); i++)
          for (int t = 0; t < 4; t++)
            if (sv(i)/smax > thr[t]) k[t]++;
        char kbuf[64];
        snprintf(kbuf, sizeof(kbuf), "%4i %4i %4i %4i", k[0], k[1], k[2], k[3]);

        // Self-check: does the stored factorization reproduce W, and does the
        // RI-K contraction reproduce the exact exchange-ordered contraction?
        const std::pair<double,double> chk = basis.check_cd(iel, L, M);

        printf("%3i %3i %4i | %11.3e | %-26s | %-10s | %-26s | %8.1e %8.1e\n",
               (int) iel, L, M, amax, buf,
               (emin < -1e-10 * amax) ? "INDEFINITE" : "PSD", kbuf,
               chk.first, chk.second);
      }
    }
    printf("\n");
  }
  printf("Full rank would be %i. A rank-r factorization costs O(Nprim^2 * r)\n"
         "instead of O(Nprim^4) = %i.\n", 2 * n, n * n);
  return 0;
}
