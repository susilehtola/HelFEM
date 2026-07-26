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

/* Dumps the SAP effective charge of every tabulated atom on a radial
   grid, as "Z r Zeff" rows. tools/gen_sap_table.py turns the output into
   the table in sap.cpp.

   Regenerating sap.cpp through this path rather than from gensap's
   result_<El>.dat files makes the two representations of the SAP
   potential the same object: the interpolated table becomes a tabulation
   of exactly what SAPFEAtom evaluates, so the only thing separating them
   is the interpolation. The .dat files cannot do that -- they come out of
   a fresh SCF seeded from the checkpoint, whose density is close to, but
   not identical with, the orbitals the database ships.

   The grid is the radial quadrature grid of the wave functions' own
   basis, plus the origin. It is dense where the density varies fastest,
   which is what a table wants too.
*/

#include "atomdb.h"
#include <xc_funcs.h>
#include <cstdio>
#include <vector>

using namespace helfem;

int main(void) {
  /* The radial grid, taken from the shared basis so that it needs no
     parameters of its own. */
  const atomdb::Atom probe(1);
  const atomic::basis::FEMRadialBasis & rb = probe.basis();
  std::vector<double> r;
  r.push_back(0.0);
  for (size_t iel = 0; iel < rb.Nel(); iel++) {
    const helfem::Vector ri = rb.get_r(iel);
    for (Eigen::Index ip = 0; ip < ri.size(); ip++)
      r.push_back(ri(ip));
  }
  fprintf(stderr, "%zu radial points out to r = %.6f\n", r.size(), r.back());

  /* One evaluation costs a partial-element quadrature, so the whole
     table is a few CPU-minutes; the atoms are independent, so spread
     them and print afterwards in order. */
  const int maxZ = atomdb::max_Z();
  std::vector<std::vector<double>> zeff(maxZ);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int Z = 1; Z <= maxZ; Z++) {
    const atomdb::Atom atom(Z);
    std::vector<double> z(r.size());
    for (size_t ip = 0; ip < r.size(); ip++)
      z[ip] = atom.effective_charge(r[ip], XC_LDA_X, 0);
    zeff[Z - 1] = std::move(z);
  }

  for (int Z = 1; Z <= maxZ; Z++)
    for (size_t ip = 0; ip < r.size(); ip++)
      printf("%3i %.17e %.17e\n", Z, r[ip], zeff[Z - 1][ip]);
  return 0;
}
