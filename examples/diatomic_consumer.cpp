/*
 * HelFEM diatomic basis -- example for external consumers.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * What this example does: builds a diatomic prolate-spheroidal basis and
 * calls the Coulomb and exchange builders, using nothing but an INSTALLED
 * HelFEM. No HelFEM source directory is on the include path.
 *
 * It exists because that use case was, until the library split, impossible
 * to satisfy: the diatomic machinery lived in helfem-common, which only
 * existed under HELFEM_BINARIES=ON and carried HDF5 + libxc as PUBLIC
 * dependencies -- neither of which the diatomic basis uses -- and no
 * diatomic headers were installed at all.
 *
 * Build it against an install tree configured with HELFEM_BINARIES=OFF:
 *
 *     cmake -B build -DCMAKE_PREFIX_PATH=<helfem-install-prefix>
 *     cmake --build build
 *     ./build/diatomic_consumer
 *
 * with a CMakeLists.txt of:
 *
 *     cmake_minimum_required(VERSION 3.20)
 *     project(diatomic_consumer CXX)
 *     find_package(helfem REQUIRED)
 *     add_executable(diatomic_consumer diatomic_consumer.cpp)
 *     target_link_libraries(diatomic_consumer PRIVATE helfem::fem)
 *
 * helfem::fem is the dependency-light half: FE geometry machinery, no HDF5,
 * no libxc, no executables. helfem::helfem-common remains the full surface
 * (DFT grids, checkpointing, SCF drivers) and still needs both.
 *
 * Keeping this compiling is what stops the packaging from regressing. Three
 * separate faults -- an unconditional find_dependency(OpenOrbitalOptimizer),
 * an unconditionally exported helfem_otr leaving a dangling
 * OpenTrustRegion::opentrustregion, and an export set publishing
 * helfem::helfem-fem rather than helfem::fem -- were invisible to the in-tree
 * build and surfaced only here.
 */

#include <helfem/diatomic/basis.h>
#include <helfem/PolynomialBasis.h>
#include <helfem/Matrix.h>

#include <cmath>
#include <cstdio>
#include <memory>

int main() {
  using namespace helfem;

  // H2 at the equilibrium distance, on a small grid.
  const int Z1 = 1, Z2 = 1, lmax = 2, mmax = 1, nnodes = 6, nelem = 3;
  const double Rhalf = 0.7, Rmax = 20.0;

  std::shared_ptr<const polynomial_basis::PolynomialBasis> poly(
      polynomial_basis::make_basis(4, nnodes));

  // lm_to_l_m takes the per-|m| lmax vector: entry k is the lmax for |m| = k.
  Eigen::VectorXi lmmax(mmax + 1);
  for (int k = 0; k <= mmax; k++) lmmax(k) = lmax;
  Eigen::VectorXi lval, mval;
  diatomic::basis::lm_to_l_m(lmmax, lval, mval);

  // The radial variable is mu, with r = Rhalf cosh(mu); the practical
  // infinity Rmax therefore sits at acosh(Rmax / Rhalf).
  const helfem::Vector bval =
      helfem::Vector::LinSpaced(nelem + 1, 0.0, std::acosh(Rmax / Rhalf));

  diatomic::basis::TwoDBasis basis(Z1, Z2, Rhalf, poly, 5 * poly->nbf(),
                                   bval, lval, mval);
  basis.compute_tei(true);

  const Eigen::Index nbf = (Eigen::Index) basis.Nbf();
  const helfem::Matrix S = basis.overlap();
  const helfem::Matrix H = basis.kinetic() + basis.nuclear();

  helfem::Matrix P = helfem::Matrix::Identity(nbf, nbf) * 0.01;
  const helfem::Matrix J = basis.coulomb(P);
  // NOTE: exchange() returns the SIGNED contribution added to a spin
  // channel's Fock matrix, not a positive K. For a closed shell the spin
  // density is P/2, so the effective potential is coulomb(P) + exchange(P/2).
  const helfem::Matrix K = basis.exchange(P);

  printf("Nbf=%d  |S|=%.6e  |H|=%.6e  |J|=%.6e  |K|=%.6e\n",
         (int) nbf, S.norm(), H.norm(), J.norm(), K.norm());

  const bool ok = nbf > 0 && S.norm() > 0.0 && J.norm() > 0.0 && K.norm() > 0.0;
  printf("%s\n", ok ? "OK" : "FAILED");
  return ok ? 0 : 1;
}
