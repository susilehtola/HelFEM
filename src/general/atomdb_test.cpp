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

/* Checks the tabulated atomic wave functions and the quantities derived
   from them at arbitrary r.

   The interesting check is the third one below. atomdb evaluates the
   enclosed charge and the Hartree screening from partial-element
   integrals of the density matrix; the test recomputes both by
   quadrature of the pointwise density, which shares no code with that
   path beyond the basis evaluation itself. The two agreeing to roundoff
   is what says the partial-element integration is right.
*/

#include "atomdb.h"
#include "elements.h"
#include "sap.h"
#include "../sadatom/basis.h"
#include <lobatto.h>
#include <xc_funcs.h>
#include <cstdio>
#include <cmath>

using namespace helfem;

/* Integrate f over [a, b] with an n-point Gauss-Lobatto rule on nsub
   subintervals.

   The caller must never let [a, b] straddle an element boundary. Inside
   one element the radial density is a polynomial of degree 2*(nnodes-1)
   = 38, which a 40-point Lobatto rule integrates exactly; across a
   boundary it is a different polynomial on each side, and the rule falls
   back to algebraic convergence. */
template <typename F>
static double quad(const F &f, double a, double b, int nsub = 1, int n = 40) {
  helfem::Vector x, w;
  helfem::lobatto::lobatto_compute(n, x, w);
  double sum = 0.0;
  for (int is = 0; is < nsub; is++) {
    const double lo = a + (b - a) * is / nsub;
    const double hi = a + (b - a) * (is + 1) / nsub;
    const double mid = 0.5 * (lo + hi), half = 0.5 * (hi - lo);
    for (Eigen::Index ip = 0; ip < x.size(); ip++)
      sum += half * w(ip) * f(mid + half * x(ip));
  }
  return sum;
}

int main(void) {
  const int maxZ = atomdb::max_Z();
  const helfem::Vector bval(atomdb::element_boundaries());
  const double Rmax = bval(bval.size() - 1);

  printf("Database: Z = 1..%i, lmax = %i, %i radial functions on %i elements "
         "out to r = %.3f\n\n",
         maxZ, atomdb::lmax(), atomdb::Nbf(), (int)bval.size() - 1, Rmax);
  printf("%3s %-3s %10s %12s %12s %12s %12s\n", "Z", "El", "N(stored)", "err Q(r)",
         "err r*V_H", "Zeff(0)-Z", "d(SAP tab)");

  double worst_q = 0.0, worst_v = 0.0, worst_lim = 0.0;
  int nfail = 0;

  for (int Z = 1; Z <= maxZ; Z++) {
    const atomdb::Atom atom(Z);
    const double N = atom.nelectrons();

    /* Sample points, deliberately including radii inside every element
       as well as the element boundaries themselves. */
    double eq = 0.0, ev = 0.0, dsap = 0.0;
    for (Eigen::Index iel = 0; iel + 1 < bval.size(); iel++) {
      for (int is = 1; is <= 2; is++) {
        const double r = bval(iel) + (bval(iel + 1) - bval(iel)) * is / 3.0;

        /* Enclosed charge by quadrature of the pointwise density, one
           element at a time so every panel stays inside one polynomial. */
        const auto rho4pir2 = [&atom](double x) { return atom.radial_density(x); };
        double Qref = quad(rho4pir2, bval(iel), r);
        for (Eigen::Index jel = 0; jel < iel; jel++)
          Qref += quad(rho4pir2, bval(jel), bval(jel + 1));
        eq = std::max(eq, std::abs(Qref - atom.enclosed_charge(r)));

        /* Hartree screening: r*V_H = Q(<r) + r * integral_{r'>r} rho dV / r'.
           The 1/r' makes this integrand non-polynomial, so give it a few
           panels per element. */
        const auto rho4pir = [&atom](double x) {
          return (x > 0.0) ? atom.radial_density(x) / x : 0.0;
        };
        double Mout = quad(rho4pir, r, bval(iel + 1), 4);
        for (Eigen::Index jel = iel + 1; jel + 1 < bval.size(); jel++)
          Mout += quad(rho4pir, bval(jel), bval(jel + 1), 4);
        ev = std::max(ev, std::abs(Qref + r * Mout - atom.hartree_screening(r)));

        /* The tabulated SAP charge is the same object, interpolated. */
        dsap = std::max(dsap,
                        std::abs(atom.effective_charge(r, XC_LDA_X, 0) -
                                 ::sap_effective_charge(Z, r)));
      }
    }

    /* Limits: all of the nuclear charge is felt at the origin, and only
       the net charge outside the practical infinity. */
    const double z0 = atom.effective_charge(1e-12, XC_LDA_X, 0);
    const double zinf = atom.effective_charge(2.0 * Rmax, XC_LDA_X, 0);
    const double elim = std::max(std::abs(z0 - Z), std::abs(zinf - (Z - N)));

    printf("%3i %-3s %10.6f %12.3e %12.3e %12.3e %12.3e\n", Z,
           element_symbols[Z].c_str(), N, eq, ev, z0 - Z, dsap);

    worst_q = std::max(worst_q, eq);
    worst_v = std::max(worst_v, ev);
    worst_lim = std::max(worst_lim, elim);
    /* The stored orbitals must carry essentially all the electrons: what
       is missing is the occupation of the orbitals dropped below the
       storage threshold, a few times 1e-6 at worst. */
    if (std::abs(N - Z) > 1e-4) {
      printf("  ** Z=%i: stored orbitals carry %.6f electrons, expected %i\n", Z, N, Z);
      nfail++;
    }
  }

  /* The exchange-correlation screening against the grid implementation
     it mirrors, sadatom::basis::TwoDBasis::xc_screening, for the same
     density matrix. LDA exercises only the potential itself; a GGA also
     exercises the divergence correction, which is where the density
     gradient, the Laplacian and the functional's second derivatives all
     have to line up. */
  {
    printf("\nExchange-correlation screening vs. the grid implementation:\n");
    std::shared_ptr<const polynomial_basis::PolynomialBasis> poly(
        polynomial_basis::get_basis(atomdb::data::primbas, atomdb::data::nnodes));
    const helfem::Vector bv(atomdb::element_boundaries());
    sadatom::basis::TwoDBasis tdb(1, modelpotential::POINT_NUCLEUS, 0.0, poly, false,
                                  atomdb::data::nquad, bv, atomdb::lmax());
    const helfem::Vector rq(tdb.radii());

    struct { const char *name; int x, c; } cases[] = {
        {"LDA exchange       ", XC_LDA_X, 0},
        {"LDA exchange+VWN   ", XC_LDA_X, XC_LDA_C_VWN},
        {"PBE exchange       ", XC_GGA_X_PBE, 0},
        {"PBE exchange+corr. ", XC_GGA_X_PBE, XC_GGA_C_PBE},
    };
    for (const auto &cs : cases) {
      double worst = 0.0;
      int worstZ = 0;
      for (int Z : {2, 8, 26, 54, 79}) {
        const atomdb::Atom atom(Z);
        helfem::Matrix P = helfem::Matrix::Zero(atomdb::Nbf(), atomdb::Nbf());
        for (int l = 0; l <= atomdb::lmax(); l++) {
          if (!atomdb::norb(Z, l))
            continue;
          const helfem::Matrix C = atomdb::coefficients(Z, l);
          P += C * atomdb::occupations(Z, l).asDiagonal() * C.transpose();
        }
        const helfem::Vector ref(tdb.xc_screening(P, cs.x, cs.c));
        /* Skip the nucleus, where the grid version has no Laplacian. */
        for (Eigen::Index ip = 1; ip < rq.size(); ip++) {
          const double d = std::abs(atom.xc_screening(rq(ip), cs.x, cs.c) - ref(ip));
          if (d > worst) { worst = d; worstZ = Z; }
        }
      }
      printf("  %s max deviation %.3e (Z = %i)\n", cs.name, worst, worstZ);
      if (worst > 1e-10) {
        printf("  ** pointwise and grid screening disagree\n");
        nfail++;
      }
    }
  }

  // The stored orbitals must reproduce the stored density:
  //   rho(r) = sum_l sum_n occ_nl R_nl(r)^2 / (4 pi)
  // This pins the normalization convention of Atom::orbitals -- get_bf
  // returns B(r)/r, so the contraction is R(r) and not r*R(r) -- so that
  // a projected guess does not have to rediscover it, and gets it wrong
  // by a factor of r if it does.
  {
    double worst_orb = 0.0;
    int worst_orb_Z = 0;
    for (int Z = 1; Z <= helfem::atomdb::max_Z(); Z++) {
      const helfem::atomdb::Atom at(Z);
      for (double r : {0.05, 0.2, 0.7, 1.5, 3.0, 8.0, 20.0}) {
        double rho = 0.0;
        for (int l = 0; l <= helfem::atomdb::lmax(); l++) {
          const int n = helfem::atomdb::norb(Z, l);
          if (n <= 0)
            continue;
          const helfem::Vector occ = helfem::atomdb::occupations(Z, l);
          const helfem::Vector R = at.orbitals(l, r);
          for (int i = 0; i < n; i++)
            rho += occ(i) * R(i) * R(i);
        }
        rho /= 4.0 * M_PI;
        const double ref = at.density(r);
        const double dev = std::abs(rho - ref) / std::max(std::abs(ref), 1e-12);
        if (dev > worst_orb) { worst_orb = dev; worst_orb_Z = Z; }
      }
    }
    printf("\nWorst orbital-vs-density reconstruction (relative): %.3e (Z = %i)\n",
           worst_orb, worst_orb_Z);
    if (worst_orb > 1e-12) {
      printf("** Atom::orbitals does not reproduce Atom::density.\n");
      nfail++;
    }
  }

  printf("\nWorst enclosed-charge deviation from quadrature: %.3e\n", worst_q);
  printf("Worst Hartree-screening deviation from quadrature: %.3e\n", worst_v);
  printf("Worst deviation from the exact r -> 0 / r -> inf limits: %.3e\n", worst_lim);

  if (worst_q > 1e-9 || worst_v > 1e-9) {
    printf("\n** The partial-element integration disagrees with quadrature of the "
           "pointwise density.\n");
    nfail++;
  }
  if (worst_lim > 1e-8) {
    printf("\n** The effective charge does not reach its exact limits.\n");
    nfail++;
  }

  printf("\n%s\n", nfail ? "FAILED" : "All checks passed.");
  return nfail ? 1 : 0;
}
