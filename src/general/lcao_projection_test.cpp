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

// Unit tests for the AO -> FEM projection primitive.
//
// The sweep (ao_profiles) is covered end to end by gensap --completeness.
// What is NOT covered there is the way an optimizer drives this: build an
// AO basis and its cross-overlap once, then project repeatedly. These
// tests pin that path, since it is the one libatomscf consumes.

#include "lcao_projection.h"
#include <Eigen/Eigenvalues>
#include <cmath>
#include <cstdio>
#include <stdexcept>

using namespace helfem;

static int failures = 0;

static void check(bool ok, const char *what) {
  printf("  %-58s %s\n", what, ok ? "ok" : "FAILED");
  if (!ok)
    failures++;
}

int main() {
  printf("AO -> FEM projection tests\n");

  // Any FEM radial basis serves as the "solution" basis here: the
  // projection machinery only ever sees its grid.
  const lcao::AOBasis sol = lcao::make_ao_basis(40.0, 4, 15, 12);
  const Matrix Ssol = sol.rad.overlap();
  Eigen::SelfAdjointEigenSolver<Matrix> es(Ssol);
  const Vector sval = es.eigenvalues();
  const Matrix svec = es.eigenvectors();
  Matrix Sinvh(svec.rows(), svec.cols());
  for (Eigen::Index i = 0; i < sval.size(); i++)
    Sinvh.col(i) = svec.col(i) / std::sqrt(sval(i));

  auto eval_gto = [](double r, int l, double ex) {
    return lcao::radial_GTO(r, l, ex);
  };

  const double alpha = 1.3;
  const int l = 1;

  // Build once, project many times: the pattern an exponent optimizer uses.
  const lcao::AOBasis ao = lcao::make_ao_basis(lcao::gto_rmax(alpha));
  const Matrix Sx = sol.rad.overlap(ao.rad);
  const Vector b = lcao::project_ao(ao, Sx, l, alpha, eval_gto);

  // Rebuilding the AO basis and the cross overlap must change nothing:
  // that equivalence is what makes reuse safe.
  const lcao::AOBasis ao2 = lcao::make_ao_basis(lcao::gto_rmax(alpha));
  const Matrix Sx2 = sol.rad.overlap(ao2.rad);
  const Vector b2 = lcao::project_ao(ao2, Sx2, l, alpha, eval_gto);
  check((b - b2).cwiseAbs().maxCoeff() == 0.0,
        "reusing an AO basis reproduces a rebuilt one exactly");

  // The AO is normalized on its own basis by construction.
  const Vector c = lcao::ao_coefficients(ao, l, alpha, eval_gto);
  check(std::abs(c.dot(ao.Sao * c) - 1.0) < 1e-12,
        "AO expansion is normalized on its own basis");

  // A projection onto an orthonormal space cannot exceed the AO's norm.
  const double Y = lcao::completeness(Sinvh, b);
  check(Y > 0.0 && Y <= 1.0 + 1e-10, "completeness lies in (0, 1]");

  // This AO is well inside the solution basis's range, so it should be
  // nearly fully represented.
  check(Y > 0.99, "a well-resolved GTO is almost completely represented");

  // Importance against a single occupied column is a plain projection,
  // and cannot exceed the completeness measured over the whole space.
  const Matrix Cocc = Sinvh.leftCols(3);
  check(lcao::importance(Cocc, b) <= Y + 1e-10,
        "importance never exceeds completeness");

  // The box heuristics are the documented ones.
  check(std::abs(lcao::gto_rmax(50.0) - 1.0) < 1e-15, "gto_rmax(50) == 1");
  check(std::abs(lcao::sto_rmax(60.0) - 1.0) < 1e-15, "sto_rmax(60) == 1");

  // The LIP restriction is enforced, not assumed: the nodal-coefficient
  // construction is invalid for a Hermite or Legendre basis.
  bool threw = false;
  try {
    lcao::make_ao_basis(1.0, /*primbas=*/5);
  } catch (const std::logic_error &) {
    threw = true;
  }
  check(threw, "a non-LIP primitive basis is refused");

  printf("%s\n", failures ? "FAILURES" : "All tests passed.");
  return failures ? 1 : 0;
}
