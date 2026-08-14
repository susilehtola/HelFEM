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

// Checks TwoDGrid::atomdb_projection, the projection of the tabulated
// atomic orbitals onto the diatomic basis that a projected initial guess
// is built from.
//
// The projection reuses the quadrature the completeness and importance
// profiles run on, so a structural error is unlikely; what is easy to get
// wrong is a NORMALIZATION or an ANGULAR FACTOR, and neither would make
// the guess fail visibly -- it would simply converge more slowly, which
// is invisible. Hence two checks with independent failure modes.
//
// 1. atomdb_overlap must come out as the identity. The stored orbitals
//    are orthonormal, so evaluating them on the diatomic grid and
//    integrating tests the radial evaluation, the r(mu,nu) geometry and
//    the quadrature weight, WITHOUT involving the basis at all. A radial
//    function off by a factor of r fails here.
//
// 2. The projection must recover the orbital's norm. Expanding the
//    projected orbital in the orthonormal basis and taking its length
//    gives the fraction of the orbital the diatomic basis captures; for
//    an atom the basis is built for that must be ~1. Less than 1 means an
//    incomplete basis, MORE than 1 is impossible and means the projection
//    or the angular factor is wrong.

#include "../general/cmdline.h"
#include "../general/constants.h"
#include "../general/elements.h"
#include "../general/atomdb.h"
#include "basis.h"
#include "twodquadrature.h"
#include "../atomic/basis.h"
#include "utils.h"
#include <cstdio>
#include <cmath>
#include <helfem.h>

using namespace helfem;

int main(int argc, char **argv) {
  helfem::set_verbosity(false);

  cmdline::parser parser;
  parser.add<std::string>("Z1", 0, "nuclear charge of the atom", false, "Be");
  parser.add<double>("Rbond", 0, "distance to the dummy centre", false, 1.0);
  parser.add<int>("lmax", 0, "maximum l", false, 8);
  parser.add<int>("mmax", 0, "maximum m", false, 2);
  parser.add<int>("nelem", 0, "number of elements", false, 5);
  parser.add<int>("nnodes", 0, "nodes per element", false, 15);
  parser.add<double>("Rmax", 0, "practical infinity", false, 40.0);
  parser.add<int>("primbas", 0, "primitive basis", false, 4);
  parser.add<double>("thresh", 0, "how much of the norm may be missing", false, 1e-3);
  parser.parse_check(argc, argv);

  const int Z1 = element_Z(parser.get<std::string>("Z1"));
  const double Rbond = parser.get<double>("Rbond");
  const int lmax = parser.get<int>("lmax");
  const int mmax = parser.get<int>("mmax");
  const int Nelem = parser.get<int>("nelem");
  const int Nnodes = parser.get<int>("nnodes");
  const double Rmax = parser.get<double>("Rmax");
  const int primbas = parser.get<int>("primbas");
  const double thresh = parser.get<double>("thresh");

  auto poly = std::shared_ptr<const polynomial_basis::PolynomialBasis>(
      polynomial_basis::make_basis(primbas, Nnodes));
  const int Nquad = 5 * poly->nbf();

  const Eigen::VectorXi lmmax = Eigen::VectorXi::Constant(mmax + 1, lmax);
  Eigen::VectorXi lval, mval;
  diatomic::basis::lm_to_l_m(lmmax, lval, mval);

  const double Rhalf = 0.5 * Rbond;
  const double mumax = utils::arcosh(Rmax / Rhalf);
  const helfem::Vector bval = atomic::basis::normal_grid(Nelem, mumax, 4, 1.0);

  // A single atom with a zero-charge partner: the basis then has to
  // represent an atomic solution, so the projection ought to be complete.
  diatomic::basis::TwoDBasis basis(Z1, 0, Rhalf, poly, Nquad, bval, lval, mval);
  const int lang = 4 * lval.maxCoeff() + 12;
  diatomic::twodquad::TwoDGrid grid(&basis, lang);

  const helfem::Matrix Sinvh = basis.Sinvh(false, 0);

  printf("Z = %i, %i angular x %i radial = %i functions, lang = %i\n",
         Z1, (int) basis.Nang(), (int) basis.Nrad(), (int) basis.Nbf(), lang);

  int nfail = 0;
  double worst_ortho = 0.0, worst_missing = 0.0, worst_excess = 0.0;

  for (int l = 0; l <= std::min(lmax, helfem::atomdb::lmax()); l++) {
    const int norb = helfem::atomdb::norb(Z1, l);
    if (norb <= 0)
      continue;
    for (int m = -std::min(l, mmax); m <= std::min(l, mmax); m++) {
      const helfem::Matrix ovl =
          grid.atomdb_overlap(Z1, l, m, diatomic::twodquad::PROBE_LEFT);
      const helfem::Matrix proj =
          grid.atomdb_projection(Z1, l, m, diatomic::twodquad::PROBE_LEFT);

      // (1) the stored orbitals are orthonormal on this grid
      for (int a = 0; a < norb; a++)
        for (int b = 0; b < norb; b++) {
          const double ref = (a == b) ? 1.0 : 0.0;
          worst_ortho = std::max(worst_ortho, std::abs(ovl(a, b) - ref));
        }

      // (2) the projection recovers the norm
      const helfem::Matrix C = proj * Sinvh;
      for (int a = 0; a < norb; a++) {
        const double recovered = C.row(a).squaredNorm();
        const double missing = 1.0 - recovered;
        if (missing > 0.0) worst_missing = std::max(worst_missing, missing);
        else               worst_excess  = std::max(worst_excess, -missing);
        printf("  l=%i m=%+d orbital %i: <psi|psi>=%.10f  recovered %.10f\n",
               l, m, a, ovl(a, a), recovered);
      }
    }
  }

  printf("\nWorst deviation of the tabulated orbitals from orthonormality: %.3e\n",
         worst_ortho);
  printf("Worst norm MISSING from the projection (basis incompleteness): %.3e\n",
         worst_missing);
  printf("Worst norm IN EXCESS of unity (impossible; a bug): %.3e\n", worst_excess);

  if (worst_ortho > 1e-6) {
    printf("\n** The tabulated orbitals are not orthonormal on the diatomic grid.\n"
           "   That is the radial evaluation or the quadrature, not the basis.\n");
    nfail++;
  }
  if (worst_excess > 1e-8) {
    printf("\n** A projection recovered MORE than the whole orbital, which no\n"
           "   finite basis can do. Normalization or angular factor is wrong.\n");
    nfail++;
  }
  if (worst_missing > thresh) {
    printf("\n** The projection loses more of the orbital than expected.\n");
    nfail++;
  }

  printf("\n%s\n", nfail ? "FAILED" : "All checks passed.");
  return nfail ? 1 : 0;
}
