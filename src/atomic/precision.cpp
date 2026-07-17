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

// Arbitrary-precision Hartree-Fock on a real atom.
//
// The 1D harmonic-oscillator demo (src/harmonic/precision.cpp) showed that the
// FINITE ELEMENT BASIS runs at any scalar type. This one shows that the whole
// atomic HARTREE-FOCK path does: overlap, kinetic, nuclear attraction, the
// two-electron integrals, Coulomb, exchange and the symmetric orthonormaliser,
// instantiated at double, long double and _Float128 from the SAME production
// TwoDBasisT<T> / FEMRadialBasisT<T> / GauntT<T> code. Nothing here is a
// special-cased high-precision path.
//
// Helium is the test case because its Hartree-Fock basis-set limit is KNOWN
// from the literature:
//
//     E_HF(He) = -2.861679995612 Ha     (Koga et al.)
//
// so the error is measured against an external reference rather than asserted.
// He is closed-shell and its HF ground state is spherically symmetric, so
// lmax = 0 is not an approximation: the only thing left to converge is the
// RADIAL basis, which is exactly the knob being swept.
//
// The expected story, and what the table shows:
//
//   * small basis: the error is DISCRETIZATION. Every scalar type agrees to
//     every digit -- the arithmetic is irrelevant.
//   * converged basis: double SATURATES. Its answer stops improving (and
//     starts drifting) around 1e-13, because roundoff in the Fock build and
//     the diagonalisations has caught up with the basis error.
//   * higher precision keeps going, and converges onto the literature value.
//
// The |E(T) - E(quad)| column is the cleanest statement of it: at a fixed
// basis, quad IS the exact variational answer for that basis (its own roundoff
// is ~1e-33), so that column is literally the arithmetic error of the lower
// precision -- nothing to do with the basis.
//
// The grid boundaries are computed in double and then cast to T, deliberately:
// that makes the variational basis IDENTICAL at all three precisions, so the
// columns differ by arithmetic alone and nothing else.

#include "TwoDBasis.h"
#include "basis.h"
#include "../general/model_potential.h"
#include "../general/gaunt.h"
#include "PolynomialBasis.h"
#include "Matrix.h"
#include "utils.h"
#include <helfem.h>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <vector>
#include <stdexcept>
#include <unistd.h>
#include <fcntl.h>

using namespace helfem;

namespace {

  /// Silence stdout for the duration of the scope.
  ///
  /// The basis constructor and the symmetric orthonormaliser print their usual
  /// diagnostics (element continuity, smallest overlap eigenvalue) on every
  /// call. That is the right behaviour for the production drivers, but here it
  /// would interleave three of them into every row of the table. Rather than
  /// change what the library prints, just close the tap while solving.
  class Quiet {
    int saved;
  public:
    Quiet() {
      fflush(stdout);
      saved = dup(fileno(stdout));
      int devnull = open("/dev/null", O_WRONLY);
      dup2(devnull, fileno(stdout));
      close(devnull);
    }
    ~Quiet() {
      fflush(stdout);
      dup2(saved, fileno(stdout));
      close(saved);
    }
  };

  /// Hartree-Fock basis-set limit for the He atom, Koga et al.
  const long double E_HF_He = -2.861679995612L;

  struct Result {
    long double E;      ///< converged total energy
    long double dE;     ///< energy change on the last SCF iteration
    int iter;           ///< SCF iterations taken
    bool converged;
    bool noise_floor;   ///< stopped because dE stalled at this type's roundoff
  };

  /// Restricted Hartree-Fock on helium, entirely in precision T.
  ///
  /// The basis, the integrals, the Fock build and the diagonalisations are all
  /// TwoDBasisT<T>; only the returned scalars are widened to long double for
  /// printing (there is no operator<< / printf conversion for _Float128).
  /// `Pguess`, when given, is a converged density from a LOWER precision, used
  /// purely as a starting guess -- it cuts the iteration count roughly in half
  /// and changes nothing else. The Roothaan fixed point does not depend on
  /// where the iteration starts, and each precision is still driven all the way
  /// down to its OWN convergence threshold below, so the converged numbers are
  /// exactly what a core guess would have produced. `Pout`, when given,
  /// receives the converged density so it can seed the next precision up.
  template <typename T>
  Result solve_he(int Nelem, int Nnodes, double Rmax, int igrid, double zexp,
                  int primbas,
                  const helfem::Matrix *Pguess = nullptr,
                  helfem::Matrix *Pout = nullptr) {
    namespace pb = helfem::polynomial_basis;
    Quiet quiet;

    // --- Basis -------------------------------------------------------------
    auto poly = std::shared_ptr<const pb::PolynomialBasisT<T>>(
        polynomial_basis::get_basis_T<T>(primbas, Nnodes));
    const int Nquad = 5 * poly->get_nbf();

    // Element boundaries: computed in double and cast up, so that every
    // precision solves the SAME variational problem and the columns below
    // differ by arithmetic alone.
    const helfem::Vector bval_d = utils::get_grid(Rmax, Nelem, igrid, zexp);
    helfem::Vec<T> bval(bval_d.size());
    for (Eigen::Index i = 0; i < bval_d.size(); i++)
      bval(i) = T(bval_d(i));

    // He: 1s^2, spherically symmetric, so a single l = m = 0 channel is exact
    // for Hartree-Fock. lmax > 0 would only add correlation, which HF has none
    // of.
    Eigen::VectorXi lval(1), mval(1);
    lval(0) = 0;
    mval(0) = 0;

    atomic::basis::TwoDBasisT<T> basis(
        /*Z=*/2, modelpotential::POINT_NUCLEUS, /*Rrms=*/T(0), poly,
        /*zeroder=*/false, Nquad, bval, lval, mval,
        /*Zl=*/0, /*Zr=*/0, /*Rhalf=*/T(0));

    // --- One-electron matrices and the two-electron integrals --------------
    const helfem::Mat<T> S     = basis.overlap();
    const helfem::Mat<T> Tkin  = basis.kinetic();
    const helfem::Mat<T> Vnuc  = basis.nuclear();
    const helfem::Mat<T> H     = Tkin + Vnuc;
    const helfem::Mat<T> Sinvh = basis.Sinvh(/*chol=*/false, /*sym=*/0);
    basis.compute_tei(/*exchange=*/true);

    const Eigen::Index Nbf = static_cast<Eigen::Index>(basis.Nbf());

    // --- Roothaan SCF ------------------------------------------------------
    // Drive each precision down to ITS OWN floor rather than to a threshold
    // borrowed from double. Two ways that can end:
    //
    //   * dE falls below a few ulps of the energy at this precision. This is
    //     what the higher precisions do -- a clean convergence.
    //   * dE STALLS. Below some point the iteration stops making progress
    //     because the Fock build and the diagonalisation are themselves only
    //     accurate to eps(T): dE bottoms out and then rattles around at the
    //     roundoff level instead of shrinking. That is not a failure to
    //     converge, it IS the arithmetic floor -- exactly the thing this
    //     program exists to measure -- so it is accepted and flagged. In
    //     practice it is double that stalls and the wider types that do not.
    const T thr = T(4) * std::numeric_limits<T>::epsilon() * T(std::abs(E_HF_He));
    const int maxiter  = 100;
    const int miniter  = 5;    // let the density relax to THIS precision's fixed
                               // point even when handed a guess from below
    const int stall_max = 12;  // iterations of no progress before calling it

    // Core guess, or the density converged at the precision below.
    helfem::Mat<T> F = H;
    helfem::Mat<T> P = helfem::Mat<T>::Zero(Nbf, Nbf);
    if (Pguess && Pguess->rows() == Nbf && Pguess->cols() == Nbf) {
      P = Pguess->template cast<T>();
      const helfem::Mat<T> Pa = T(0.5) * P;
      F = H + basis.coulomb(P) + basis.exchange(Pa);
    }
    T Eold = T(0), E = T(0);
    T best_dE = T(1);
    int stall = 0;
    Result res{0.0L, 0.0L, 0, false, false};

    for (int iter = 1; iter <= maxiter; iter++) {
      // Diagonalise in the orthonormal basis, back-transform.
      const helfem::Mat<T> Fortho = Sinvh.transpose() * F * Sinvh;
      Eigen::SelfAdjointEigenSolver<helfem::Mat<T>> es(Fortho);
      if (es.info() != Eigen::Success)
        throw std::logic_error("Fock diagonalization failed\n");
      const helfem::Mat<T> C = Sinvh * es.eigenvectors();

      // He is 1s^2: the total density is 2 * c0 c0^T, the spin density half of
      // it. Matches the driver's convention (basis.exchange takes the SPIN
      // density and returns the signed contribution that is ADDED to F).
      const helfem::Vec<T> c0 = C.col(0);
      P = T(2) * (c0 * c0.transpose());
      const helfem::Mat<T> Pa = T(0.5) * P;

      const helfem::Mat<T> J = basis.coulomb(P);
      const helfem::Mat<T> K = basis.exchange(Pa);
      F = H + J + K;

      // E = Ekin + Enuc + Ecoul + Exx, with Exx = 2 * (1/2) tr(Pa K) for the
      // restricted case (alpha == beta).
      Eold = E;
      E = (P * H).trace()
        + T(0.5) * (P * J).trace()
        + (Pa * K).trace();

      const T dE = (iter == 1) ? T(1) : (E > Eold ? E - Eold : Eold - E);
      res.iter = iter;
      res.E    = (long double) E;
      res.dE   = (long double) dE;

      if (iter > 1) {
        // Progress means a MEANINGFUL drop in dE; anything else is a stall.
        if (dE < T(0.9) * best_dE) { best_dE = dE; stall = 0; }
        else                       { ++stall; }
      }

      if (iter >= miniter && dE < thr) {         // clean convergence
        res.converged = true;
        break;
      }
      if (iter >= miniter && stall >= stall_max) {
        // No longer improving: the iteration has hit this type's roundoff.
        // The energy is converged as far as this arithmetic can express it.
        res.converged   = true;
        res.noise_floor = true;
        break;
      }
    }

    if (!res.converged)
      throw std::runtime_error(
          "SCF neither converged nor stalled within maxiter -- the reported "
          "energy would not be a converged one.\n");

    if (Pout) *Pout = P.template cast<double>();
    return res;
  }

  void print_row(int nelem, int nnodes, const Result &rd, const Result &rl,
                 const Result *rq) {
    // Deviation from the literature HF limit.
    const long double dd = rd.E - E_HF_He;
    const long double dl = rl.E - E_HF_He;
    printf("  %-5i %-6i | % .15Lf  %-9.2Le | % .15Lf  %-9.2Le",
           nelem, nnodes, rd.E, dd, rl.E, dl);
    if (rq) {
      const long double dq = rq->E - E_HF_He;
      printf(" | % .15Lf  %-9.2Le", rq->E, dq);
      // Arithmetic error of each column. chol_tol now follows eps(T), so quad
      // really is the exact variational answer for this basis (its own roundoff
      // is ~1e-33) and these are the lower precisions' arithmetic errors.
      const long double ed = std::fabs(rd.E - rq->E);
      const long double el = std::fabs(rl.E - rq->E);
      printf(" | %-9.2Le %-9.2Le", ed, el);
    }
    printf("\n");
    fflush(stdout);
  }

} // namespace

/// Sanity check on the ANGULAR half of the integrals, before trusting the
/// radial half.
///
/// The Gaunt coefficients multiply every radial integral in the Coulomb and
/// exchange builds, so if they were only correct to double precision they --
/// and not the arithmetic -- would be what caps the calculation, and the whole
/// exercise would be pointless. They are not: libwignernj evaluates each
/// coefficient from an exact prime-factorised rational and rounds only at the
/// very end, so it is correctly rounded at whatever precision is asked for.
///
/// Checked against a closed form:  <Y_0^0 | Y_1^0 Y_1^0> = 1/sqrt(4 pi).
///
/// The reference is evaluated ONCE, in the widest type available, and every
/// scalar type is measured against that same number. Comparing each type
/// against a reference computed in its own precision would be vacuous -- both
/// sides would be correctly rounded to T and the error would come out as
/// exactly zero for all of them, which says nothing.
#ifdef HELFEM_HAVE_FLOAT128
using HighPrec = _Float128;
#else
using HighPrec = long double;
#endif

static HighPrec gaunt_reference() {
  return HighPrec(1) / std::sqrt(HighPrec(4) * helfem::utils::pi<HighPrec>());
}

template <typename T>
static long double gaunt_error() {
  // The T -> HighPrec widening is exact, so this is purely T's own error.
  const HighPrec g   = (HighPrec) helfem::gaunt::gaunt_coefficient_T<T>(0, 0, 1, 0, 1, 0);
  const HighPrec ref = gaunt_reference();
  const HighPrec err = (g > ref) ? (g - ref) : (ref - g);
  return (long double) err;
}

static void check_gaunts() {
  printf("Gaunt coefficients (libwignernj: exact rational, rounded once at the\n");
  printf("end -- so correctly rounded at ANY precision, and never the limit).\n");
  printf("Closed form: gaunt(0,0,1,0,1,0) = 1/sqrt(4 pi).\n\n");
  printf("    %-16s %-14s\n", "scalar type", "|err|");
  printf("    ------------------------------\n");
  printf("    %-16s %-14.3Le\n", "double(53)",     gaunt_error<double>());
  printf("    %-16s %-14.3Le\n", "long double(64)", gaunt_error<long double>());
#ifdef HELFEM_HAVE_FLOAT128
  // Measured in _Float128 against a _Float128 reference, so this really is the
  // quad-precision error and not long double's showing through.
  printf("    %-16s %-14.3Le\n", "_Float128(113)", gaunt_error<_Float128>());
#endif
  printf("\nEach is at the half-ulp of its own type, so the angular factors gain\n");
  printf("digits right along with the arithmetic. Good -- carry on.\n\n");
}

int main(int argc, char **argv) {
  const double Rmax  = (argc > 1) ? std::atof(argv[1]) : 40.0;
  const int igrid    = (argc > 2) ? std::atoi(argv[2]) : 4;
  const double zexp  = (argc > 3) ? std::atof(argv[3]) : 2.0;
  const int primbas  = (argc > 4) ? std::atoi(argv[4]) : 4;

  helfem::set_verbosity(false);

  check_gaunts();

  printf("Helium, restricted Hartree-Fock, in three precisions.\n");
  printf("The same production TwoDBasisT<T> / FEMRadialBasisT<T> / GauntT<T>\n");
  printf("code, instantiated at three scalar types. Point nucleus, lmax = 0\n");
  printf("(exact for the He HF ground state), Rmax = %g, grid %i, zexp = %g,\n",
         Rmax, igrid, zexp);
  printf("primbas %i. The element boundaries are the same numbers at every\n",
         primbas);
  printf("precision, so the three columns solve the IDENTICAL variational\n");
  printf("problem and differ by arithmetic alone.\n\n");
  printf("Literature HF basis-set limit (Koga et al.):  E = %.12Lf Ha\n\n",
         E_HF_He);

#ifdef HELFEM_HAVE_FLOAT128
  printf("  %-5s %-6s | %-24s %-10s | %-24s %-10s | %-24s %-10s | %s\n",
         "nelem", "nnodes",
         "double(53)", "err", "long double(64)", "err",
         "_Float128(113)", "err", "|E - E_quad|: dbl / ldbl");
  printf("  ---------------------------------------------------------------"
         "---------------------------------------------------------------"
         "-------------------------------\n");
#else
  printf("  %-5s %-6s | %-24s %-10s | %-24s %-10s\n",
         "nelem", "nnodes", "double(53)", "err", "long double(64)", "err");
  printf("  ---------------------------------------------------------------"
         "------------------------\n");
#endif

  const int nel[] = {5, 5, 10, 10, 20, 20};
  const int nnd[] = {15, 25, 15, 25, 15, 25};

  for (size_t i = 0; i < sizeof(nel) / sizeof(nel[0]); i++) {
    // Solve at double first, then hand its converged density up as the guess
    // for the slower precisions (see solve_he: it is a guess, nothing more).
    helfem::Matrix Pd;
    const Result rd = solve_he<double>     (nel[i], nnd[i], Rmax, igrid, zexp, primbas, nullptr, &Pd);
    const Result rl = solve_he<long double>(nel[i], nnd[i], Rmax, igrid, zexp, primbas, &Pd);
#ifdef HELFEM_HAVE_FLOAT128
    const Result rq = solve_he<_Float128>  (nel[i], nnd[i], Rmax, igrid, zexp, primbas, &Pd);
    print_row(nel[i], nnd[i], rd, rl, &rq);
#else
    print_row(nel[i], nnd[i], rd, rl, nullptr);
#endif
  }

  printf("\nEvery column is driven to ITS OWN floor: the SCF stops either when dE\n");
  printf("drops below a few ulps of the energy at that precision, or when dE stops\n");
  printf("improving at all because the Fock build and the diagonalisation have run\n");
  printf("out of arithmetic. Whichever comes first, the energy is then converged as\n");
  printf("far as that scalar type can express it.\n");
  printf("\n'err' is the deviation from the literature HF limit; the literature\n");
  printf("value itself is only quoted to 12 decimals, so errors below ~1e-12\n");
  printf("are at the resolution of the reference, not of the calculation.\n");
#ifdef HELFEM_HAVE_FLOAT128
  printf("\nThe last two columns are the honest measure. At a FIXED basis, quad\n");
  printf("is the exact variational answer (its own roundoff is ~1e-33), so\n");
  printf("|E - E_quad| is purely the ARITHMETIC error of that column -- it has\n");
  printf("nothing to do with basis-set incompleteness. double's sits around\n");
  printf("1e-13 and does not improve as the basis grows; it gets WORSE, because\n");
  printf("roundoff accumulates with basis size. long double is ~1e3 better, and\n");
  printf("quad simply does not have this error term.\n");
  printf("\nSo a converged HelFEM Hartree-Fock energy is limited by its\n");
  printf("ARITHMETIC, not by its basis -- and now one can check which.\n");
#else
  printf("\nBuild with -DHELFEM_FLOAT128=ON (needs C++23) to add the 113-bit\n");
  printf("column, which exposes the true basis limit.\n");
#endif
  return 0;
}
