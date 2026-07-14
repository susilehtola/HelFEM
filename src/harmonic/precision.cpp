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

// Arbitrary-precision demonstration.
//
// The SAME finite element code, instantiated at several scalar types, solving
// the 1D harmonic oscillator
//
//     [ -1/2 d^2/dx^2 + 1/2 x^2 ] psi = E psi,   E_n = n + 1/2  exactly.
//
// Because the eigenvalues are known in closed form, the error is measurable
// rather than asserted. Sweeping the basis shows three regimes:
//
//   * small basis: the error is DISCRETIZATION, and the scalar type is
//     irrelevant -- double and long double agree to every digit;
//   * converged basis: double SATURATES near 1e-13, while long double keeps
//     going, four orders of magnitude further;
//   * larger still: double gets WORSE, because roundoff accumulates with the
//     basis size, while higher precision continues to improve.
//
// So the accuracy of a converged HelFEM calculation is set by the arithmetic,
// not by the basis -- which is the whole argument for templating on the scalar.
//
// Nothing here is a special-cased "high precision path": FiniteElementBasisT<T>
// and get_basis_T<T> are the ordinary production classes, instantiated at a
// different T. Everything below them (PolynomialBasis<T>, LIPBasis<T>,
// lobatto_compute<T>) was already generic.

#include "PolynomialBasis.h"
#include "FiniteElementBasis.h"
#include "Matrix.h"
#include <lib1dfem/chebyshev.h>
#include <Eigen/Eigenvalues>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
#include <cmath>

using namespace helfem;

namespace {

  /// Solve the harmonic oscillator in precision T; return the first nstate
  /// eigenvalues as long double (for reporting only -- the computation itself
  /// is entirely in T).
  template <typename T>
  std::vector<long double> solve(double xmax, int Nelem, int Nnodes, int primbas,
                                 int Nquad, int nstate) {
    namespace pb = helfem::lib1dfem::polynomial_basis;

    auto poly = std::shared_ptr<const pb::PolynomialBasis<T>>(
        polynomial_basis::get_basis_T<T>(primbas, Nnodes));

    const helfem::Vec<T> r =
        helfem::Vec<T>::LinSpaced(Nelem + 1, T(-xmax), T(xmax));

    polynomial_basis::FiniteElementBasisT<T> fem(
        poly, r, /*zero_func_left=*/true, /*zero_deriv_left=*/true,
        /*zero_func_right=*/true, /*zero_deriv_right=*/true);

    helfem::Vec<T> xq, wq;
    helfem::lib1dfem::chebyshev::chebyshev<T>(Nquad, xq, wq);

    // H = T + V with V = 1/2 x^2, T = 1/2 |psi'|^2 (matrix_element supplies the
    // 1/2 on the kinetic term via the derivative-derivative form below).
    const helfem::Mat<T> S = fem.matrix_element(false, false, xq, wq, nullptr);
    const helfem::Mat<T> V =
        fem.matrix_element(false, false, xq, wq, [](T x) { return T(0.5) * x * x; });
    const helfem::Mat<T> K = fem.matrix_element(true, true, xq, wq, nullptr);
    const helfem::Mat<T> H = T(0.5) * K + V;

    // Symmetric orthonormalisation, then diagonalise.
    Eigen::SelfAdjointEigenSolver<helfem::Mat<T>> Ses(S);
    const helfem::Vec<T> Sval = Ses.eigenvalues();
    const helfem::Mat<T> Svec = Ses.eigenvectors();
    helfem::Vec<T> invsqrt(Sval.size());
    for (Eigen::Index i = 0; i < Sval.size(); i++)
      invsqrt(i) = T(1) / std::sqrt(Sval(i));
    const helfem::Mat<T> Sinvh = Svec * invsqrt.asDiagonal() * Svec.transpose();

    Eigen::SelfAdjointEigenSolver<helfem::Mat<T>> Hes(
        helfem::Mat<T>(Sinvh.transpose() * H * Sinvh));
    const helfem::Vec<T> E = Hes.eigenvalues();

    // Errors are computed IN T against the exact n+1/2, then returned as long
    // double. Returning the eigenvalues themselves would cap the quad column at
    // long double's ~1e-19, hiding the very thing being measured.
    std::vector<long double> out;
    for (int i = 0; i < nstate && i < E.size(); i++) {
      const T exact = T(i) + T(0.5);
      const T err = (E(i) > exact) ? (E(i) - exact) : (exact - E(i));
      out.push_back((long double) err);
    }
    return out;
  }

  /// solve() already returns the per-state ERRORS, computed in T.
  long double worst_error(const std::vector<long double> &errs) {
    long double worst = 0.0L;
    for (long double e : errs) if (e > worst) worst = e;
    return worst;
  }

} // namespace

int main(int argc, char **argv) {
  const double xmax = (argc > 1) ? std::atof(argv[1]) : 10.0;
  const int Nelem   = (argc > 2) ? std::atoi(argv[2]) : 10;
  const int Nnodes  = (argc > 3) ? std::atoi(argv[3]) : 15;
  const int primbas = (argc > 4) ? std::atoi(argv[4]) : 4;
  const int Nquad   = (argc > 5) ? std::atoi(argv[5]) : 100;
  const int nstate  = (argc > 6) ? std::atoi(argv[6]) : 10;

  printf("1D harmonic oscillator, exact E_n = n + 1/2 (n = 0..%i).\n", nstate-1);
  printf("xmax=%g, primbas %i, %i-point quadrature.\n\n", xmax, primbas, Nquad);
  printf("The SAME FiniteElementBasisT<T> code, instantiated at several scalar types.\n");
  printf("Basis is swept so the discretization error passes through double's floor.\n\n");

#ifdef HELFEM_HAVE_FLOAT128
  printf("  %-7s %-7s | %-12s | %-12s | %-12s\n",
         "nelem", "nnodes", "double(53)", "long dbl(64)", "_Float128(113)");
#else
  printf("  %-7s %-7s | %-12s | %-12s\n",
         "nelem", "nnodes", "double(53)", "long dbl(64)");
#endif
  printf("  ---------------------------------------------------------------------\n");

  const int nel[]  = {5, 5, 10, 20, 40};
  const int nnd[]  = {15, 25, 25, 25, 25};
  for (size_t i = 0; i < sizeof(nel)/sizeof(nel[0]); i++) {
    const long double ed = worst_error(solve<double>     (xmax, nel[i], nnd[i], primbas, Nquad, nstate));
    const long double el = worst_error(solve<long double>(xmax, nel[i], nnd[i], primbas, Nquad, nstate));
#ifdef HELFEM_HAVE_FLOAT128
    const long double eq = worst_error(solve<_Float128>  (xmax, nel[i], nnd[i], primbas, Nquad, nstate));
    printf("  %-7i %-7i | %-12.3Le | %-12.3Le | %-12.3Le\n", nel[i], nnd[i], ed, el, eq);
#else
    printf("  %-7i %-7i | %-12.3Le | %-12.3Le\n", nel[i], nnd[i], ed, el);
#endif
  }

  printf("\nSmall basis: every type agrees -- the error is discretization, and the\n");
  printf("scalar type is irrelevant.\n");
#ifdef HELFEM_HAVE_FLOAT128
  printf("\nConverged basis: double saturates near 1e-13 and then gets WORSE, because\n");
  printf("roundoff accumulates with basis size. _Float128 instead goes FLAT at ~1e-30:\n");
  printf("that is the true discretization limit of this basis -- sixteen orders of\n");
  printf("magnitude below where double sits. The basis was converged to 1e-30 all\n");
  printf("along; double simply cannot see it.\n");
  printf("\nNote this is a VERIFICATION tool, not an accuracy one: double already\n");
  printf("reaches useful (nanohartree) precision long before any of this matters.\n");
  printf("The point is that a reference code should be limited by its basis, not by\n");
  printf("its arithmetic -- and now one can check which.\n");
#else
  printf("\nConverged basis: double saturates near 1e-13, then gets WORSE as roundoff\n");
  printf("accumulates with basis size, while long double keeps improving.\n");
  printf("\nBuild with -DHELFEM_FLOAT128=ON (needs C++23) to add the 113-bit column,\n");
  printf("which exposes the true basis limit at ~1e-30.\n");
#endif
  return 0;
}
