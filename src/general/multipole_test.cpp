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

// Tests for FEMRadialBasis::multipole_potential.
//
// The monopole potential of a normalized 1s Slater density is analytic,
//     V(r) = [1 - e^(-2 z r) (1 + z r)] / r,     V(0) = z,
// so the potential can be checked absolutely rather than against another
// numerical route. The second test then uses it the way libatomscf will:
// as the weight of a matrix element evaluated on a DIFFERENT basis.

#include "lcao_projection.h"
#include <Eigen/Eigenvalues>
#include <cmath>
#include <cstdio>

using namespace helfem;

static int failures = 0;
static void check(bool ok, const char *what, double val) {
  printf("  %-54s %-12.3e %s\n", what, val, ok ? "ok" : "FAILED");
  if (!ok)
    failures++;
}

/// Exact monopole potential of a normalized 1s Slater density.
static double slater_potential(double r, double zeta) {
  if (r <= 0.0)
    return zeta;
  return (1.0 - std::exp(-2.0 * zeta * r) * (1.0 + zeta * r)) / r;
}

int main() {
  printf("Multipole potential tests\n");
  const double zeta = 1.7;

  // A FEM basis fine enough that the interpolation error is well below the
  // accuracy we are testing the potential to.
  const lcao::AOBasis fem = lcao::make_ao_basis(lcao::sto_rmax(zeta), 4, 15, 12);
  auto eval_sto = [](double r, int l, double z) {
    return lcao::radial_STO(r, l, z);
  };
  // Normalized density of a single 1s orbital: rho = c c^T with c'Sc = 1.
  const Vector c = lcao::ao_coefficients(fem, 0, zeta, eval_sto);
  const Matrix D = c * c.transpose();

  const auto V = fem.rad.multipole_potential(D, 0);

  // Absolute check against the closed form, across four decades in r.
  double worst = 0.0;
  for (double r : {1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0}) {
    const double got = V(r), want = slater_potential(r, zeta);
    worst = std::max(worst, std::abs(got - want) / std::abs(want));
  }
  check(worst < 1e-9, "V(r) matches the analytic 1s Slater potential", worst);

  // The total charge is one electron, so far outside the density the
  // potential must be exactly 1/r.
  const double far = 60.0;
  check(std::abs(V(far) * far - 1.0) < 1e-9, "V(r) -> 1/r outside the density",
        std::abs(V(far) * far - 1.0));

  // At the origin the potential is the nuclear-cusp value zeta.
  check(std::abs(V(0.0) - zeta) < 1e-12, "V(0) equals zeta",
        std::abs(V(0.0) - zeta));

  // Now the intended use: the Coulomb matrix of this frozen density in a
  // DIFFERENT basis, with no projection of one basis onto the other. A
  // tight Gaussian is the case that motivates it -- projecting it onto the
  // reference grid would destroy its norm, while the integral against the
  // potential is perfectly well defined.
  auto eval_gto = [](double r, int l, double a) {
    return lcao::radial_GTO(r, l, a);
  };
  printf("  Coulomb element <g|V|g> of a normalized 1s GTO:\n");
  printf("    %-14s %-20s %-20s %s\n", "alpha", "mixed-basis", "V at its centroid",
         "ratio");
  for (double alpha : {1.0, 1e2, 1e4, 1e6, 1e8}) {
    const lcao::AOBasis ao = lcao::make_ao_basis(lcao::gto_rmax(alpha));
    const Vector ca = lcao::ao_coefficients(ao, 0, alpha, eval_gto);
    // <g|V|g> on the AO's own grid, V evaluated wherever it is asked.
    const Matrix J = ao.rad.matrix_element(
        atomic::basis::FEMRadialBasis::BasisKind::B0,
        atomic::basis::FEMRadialBasis::BasisKind::B0, V);
    const double elem = ca.dot(J * ca);
    // A normalized 1s GTO has <r> = 2/sqrt(pi*alpha); as alpha grows the
    // element must approach V there, and ultimately V(0) = zeta.
    const double rbar = 2.0 / std::sqrt(M_PI * alpha);
    printf("    %-14.3e %-20.12f %-20.12f %.6f\n", alpha, elem,
           slater_potential(rbar, zeta), elem / slater_potential(rbar, zeta));
  }
  // The alpha -> infinity limit samples the potential at the nucleus.
  {
    const double alpha = 1e10;
    const lcao::AOBasis ao = lcao::make_ao_basis(lcao::gto_rmax(alpha));
    const Vector ca = lcao::ao_coefficients(ao, 0, alpha, eval_gto);
    const Matrix J = ao.rad.matrix_element(
        atomic::basis::FEMRadialBasis::BasisKind::B0,
        atomic::basis::FEMRadialBasis::BasisKind::B0, V);
    const double elem = ca.dot(J * ca);
    check(std::abs(elem - zeta) / zeta < 1e-3,
          "a very tight GTO samples V at the nucleus", std::abs(elem - zeta) / zeta);
  }

  printf("%s\n", failures ? "FAILURES" : "All tests passed.");
  return failures ? 1 : 0;
}
