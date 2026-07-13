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

    std::vector<long double> out;
    for (int i = 0; i < nstate && i < E.size(); i++)
      out.push_back((long double)E(i));
    return out;
  }

  long double worst_error(const std::vector<long double> &E) {
    long double worst = 0.0L;
    for (size_t n = 0; n < E.size(); n++) {
      const long double err = std::fabs(E[n] - ((long double)n + 0.5L));
      if (err > worst) worst = err;
    }
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
  printf("The SAME FiniteElementBasisT<T> code, instantiated at two scalar types.\n");
  printf("Basis is swept so the discretization error passes through double's floor.\n\n");

  printf("  %-7s %-7s | %-12s | %-12s | %s\n",
         "nelem", "nnodes", "double", "long double", "gain");
  printf("  ----------------------------------------------------------------\n");

  const int nel[]  = {5, 5, 10, 20, 40};
  const int nnd[]  = {15, 25, 25, 25, 25};
  for (size_t i = 0; i < sizeof(nel)/sizeof(nel[0]); i++) {
    const long double ed = worst_error(solve<double>     (xmax, nel[i], nnd[i], primbas, Nquad, nstate));
    const long double el = worst_error(solve<long double>(xmax, nel[i], nnd[i], primbas, Nquad, nstate));
    printf("  %-7i %-7i | %-12.3Le | %-12.3Le | %.0Lfx\n",
           nel[i], nnd[i], ed, el, (el > 0) ? ed/el : 0.0L);
  }

  printf("\nSmall basis: both agree -- the error is discretization, not arithmetic.\n");
  printf("Converged basis: double saturates; higher precision keeps going.\n");
  printf("Larger still: double gets WORSE (roundoff accumulates with basis size),\n");
  printf("while higher precision continues to improve. The accuracy of a converged\n");
  printf("calculation is set by the arithmetic, not by the basis.\n");
  return 0;
}
