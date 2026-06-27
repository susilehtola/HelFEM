/*
 * HelFEM atomic radial FEM export — example for external consumers
 * (e.g. a BSD-licensed atomic-SCF library wanting to use HelFEM as a
 * "numerical atomic orbital" backend).
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * What this example does:
 *
 *   1. Constructs an atomic radial FEM basis with the SAME defaults the
 *      `atomic` executable uses (so the consumer gets numerically-exact
 *      atomic HF out of the box).
 *
 *   2. Computes the one-electron matrices S, T, T_l, V exposed by libhelfem
 *      and verifies them by solving hydrogenic eigenproblems:
 *          - H Z=1, l=0:   E_1s = -0.5 Eh           (one-electron)
 *          - H Z=2, l=1:   E_2p = -Z^2/2/(l+1)^2     (centrifugal correct)
 *
 *   3. Runs a closed-shell He HF SCF (Z=2, single l=0 shell) and
 *      verifies E = -2.86167999561 Eh. This uses TwoDBasis::coulomb/
 *      exchange because, for a single-l shell, those reduce to the
 *      monopole (k=0) radial J/K.
 *
 *   4. Documents the integration conventions (see comments at the bottom).
 *
 * Build (against an installed HelFEM):
 *   g++ -std=c++17 -O2 -DARMA_64BIT_WORD -DARMA_BLAS_LONG \
 *       -I $PREFIX/include -I $HELFEM_SRC/src/atomic \
 *       -I $HELFEM_SRC/src/general -I $HELFEM_SRC/libhelfem/include \
 *       atomic_radial_export.cpp \
 *       -L $PREFIX/lib -lhelfem-common -lhelfem -llegendre \
 *       -larmadillo -lflexiblas64 -lhdf5 -lhdf5_cpp
 *
 * The ARMA_64BIT_WORD / ARMA_BLAS_LONG / flexiblas64 trio MUST match how
 * libhelfem was built. Mismatch silently corrupts the heap inside LAPACK
 * (the consumer's "free(): invalid next size" report had this cause).
 */

#include <TwoDBasis.h>
#include <RadialBasis.h>
#include <PolynomialBasis.h>
#include <FiniteElementBasis.h>
#include <armadillo>
#include <cstdio>

namespace helfem { namespace utils {
  arma::vec get_grid(double Rmax, int num_el, int igrid, double zexp);
}}

using namespace helfem;

namespace {

// Build the atomic radial basis with documented defaults.
//
// Documented defaults (extracted from src/atomic/main.cpp):
//   primbas       = 4         (LIPs with Gauss-Lobatto nodes)
//   Nnodes        = 15        (15-node LIP per element)
//   Rmax          = 40 au
//   Nelem         = 20
//   igrid         = 4         (exponential)
//   zexp          = 2.0
//   n_quad        = 5 * poly->get_nbf()
//   taylor_order  = poly->get_nprim() - 1
//
// FEM boundary flags match TwoDBasis exactly:
//   zero_func_left=true (Dirichlet at r=0)
//   zero_deriv_left=false
//   zero_func_right=true (Dirichlet at r=Rmax)
//   zero_deriv_right=false (=zeroder, false for ordinary atoms)
struct Defaults { int primbas, Nnodes, Nelem, igrid; double Rmax, zexp; };
constexpr Defaults DEF{4, 15, 20, 4, 40.0, 2.0};

atomic::basis::FEMRadialBasis make_radial(const Defaults & d = DEF) {
  auto poly = std::shared_ptr<const polynomial_basis::PolynomialBasis>(
      polynomial_basis::get_basis(d.primbas, d.Nnodes));
  arma::vec bval = utils::get_grid(d.Rmax, d.Nelem, d.igrid, d.zexp);
  polynomial_basis::FiniteElementBasis fem(poly, bval,
      /*zero_func_left*/true,  /*zero_deriv_left*/false,
      /*zero_func_right*/true, /*zero_deriv_right*/false);
  int n_quad       = 5 * poly->get_nbf();
  int taylor_order = poly->get_nprim() - 1;
  return atomic::basis::FEMRadialBasis(fem, n_quad, taylor_order);
}

double diag_lowest(const arma::mat & H, const arma::mat & S) {
  arma::vec sval; arma::mat svec;
  arma::eig_sym(sval, svec, S);
  arma::uvec keep = arma::find(sval > 1e-10);
  arma::mat X = svec.cols(keep) * arma::diagmat(1.0/arma::sqrt(sval.elem(keep)));
  arma::vec ev; arma::eig_sym(ev, X.t() * H * X);
  return ev(0);
}

// Returns false on failure (any test off by more than `tol`).
bool one_electron_tests(const atomic::basis::FEMRadialBasis & radial) {
  arma::mat S = radial.overlap();
  arma::mat T = radial.kinetic();
  arma::mat Tl = radial.kinetic_l();
  arma::mat V = radial.nuclear();
  std::printf("Nbf = %llu\n", (unsigned long long)S.n_rows);

  struct Case { int l; double Z; const char * tag; };
  Case cases[] = {{0, 1.0, "H  1s"}, {0, 2.0, "He+ 1s"},
                  {1, 1.0, "H  2p"}, {1, 2.0, "He+ 2p"},
                  {2, 1.0, "H  3d"}};
  bool ok = true;
  for(auto c : cases) {
    arma::mat H = T + double(c.l*(c.l+1))*Tl + c.Z*V;
    double e = diag_lowest(H, S);
    double ref = -c.Z*c.Z/(2.0*(c.l+1)*(c.l+1));
    double err = std::abs(e - ref);
    bool pass = err < 1e-6;
    std::printf("  %-7s   E = %+.10f   ref %+.10f   err %.2e %s\n",
                c.tag, e, ref, err, pass?"":"  *** FAIL ***");
    if(!pass) ok = false;
  }
  return ok;
}

bool he_hf_test() {
  std::printf("\nHe HF (Z=2, single l=0 shell, closed-shell 1s^2):\n");
  int Z = 2;
  auto poly = std::shared_ptr<const polynomial_basis::PolynomialBasis>(
      polynomial_basis::get_basis(DEF.primbas, DEF.Nnodes));
  arma::vec bval = utils::get_grid(DEF.Rmax, DEF.Nelem, DEF.igrid, DEF.zexp);
  arma::ivec lval = {0}, mval = {0};
  atomic::basis::TwoDBasis basis(Z, modelpotential::POINT_NUCLEUS, /*Rrms*/0.0,
      poly, /*zeroder*/false,
      5*poly->get_nbf(), bval, poly->get_nprim()-1,
      lval, mval, /*Zl*/0, /*Zr*/0, /*Rhalf*/0.0);
  basis.compute_tei(/*exchange*/true);

  arma::mat S = basis.overlap();
  arma::mat T = basis.kinetic();
  arma::mat V = basis.nuclear();
  arma::mat Hc = T + V;
  arma::mat Sinvh = basis.Sinvh(/*chol*/false, /*sym*/0);

  arma::vec ev; arma::mat C;
  arma::eig_sym(ev, C, Sinvh.t()*Hc*Sinvh);
  C = Sinvh * C;

  double Eprev = 0.0;
  for(int iter = 0; iter < 60; iter++) {
    arma::vec c = C.col(0);
    arma::mat P = 2.0 * c * c.t();
    arma::mat Pa = 0.5*P;
    arma::mat J = basis.coulomb(P);
    arma::mat K = basis.exchange(Pa);              // HelFEM's K is sign-included
    double Ekin = arma::trace(P*T);
    double Epot = arma::trace(P*V);
    double Ecoul = 0.5*arma::trace(P*J);
    double Exx  = arma::trace(Pa*K);               // Pa=Pb, so 2 * 0.5 * Tr(Pa K)
    double E = Ekin + Epot + Ecoul + Exx;
    if(iter > 0 && std::abs(E-Eprev) < 1e-12) {
      const double ref = -2.86167999561;
      double err = std::abs(E - ref);
      std::printf("  Converged at iter %2d   E = %.11f   ref %.11f   err %.2e %s\n",
                  iter, E, ref, err, err<1e-6 ? "" : "  *** FAIL ***");
      return err < 1e-6;
    }
    Eprev = E;
    arma::eig_sym(ev, C, Sinvh.t() * (Hc+J+K) * Sinvh);
    C = Sinvh * C;
  }
  std::printf("  SCF did not converge\n");
  return false;
}

} // namespace

int main() {
  std::printf("=== one-electron / hydrogenic tests ===\n");
  auto radial = make_radial();
  bool ok1 = one_electron_tests(radial);
  bool ok2 = he_hf_test();
  std::printf("\nOverall: %s\n", (ok1 && ok2) ? "PASS" : "FAIL");
  return (ok1 && ok2) ? 0 : 1;
}

/* ============================================================================
 * INTEGRATION CONVENTIONS (verified empirically by the tests above)
 * ============================================================================
 *
 * Basis representation
 * --------------------
 *   The radial basis represents  u(r) = r * R(r)  (the radial wave function
 *   times r), where R(r) is what appears in psi = R(r) Y_lm. The FEM
 *   primitive polynomials B(r) are u(r) at quadrature points; near the
 *   origin a Taylor expansion of B(r)/r is used for numerical safety.
 *
 *   Integration measure is `dr`. There is NO implicit r^2 Jacobian.
 *
 * One-electron matrices returned by RadialBasis
 * ---------------------------------------------
 *   S  = overlap()      = integral u_i(r) u_j(r) dr
 *                       = quantum-mechanical radial overlap (r^2 absorbed in u^2)
 *   T  = kinetic()      = (1/2) integral u'_i(r) u'_j(r) dr
 *                       = radial kinetic operator, EXCLUDING centrifugal
 *   Tl = kinetic_l()    = (1/2) integral u_i(r) u_j(r) / r^2 dr
 *                       = half of <u | 1/r^2 | u>
 *   V  = nuclear()      = - integral u_i(r) u_j(r) / r dr
 *                       = matrix element of -1/r in u-basis
 *                       (Z = 1 implicit; sign already negative)
 *
 *   The l-dependent one-electron Hamiltonian is:
 *
 *      H(l) = T  +  l*(l+1) * Tl  +  Z * V
 *
 *   Note the factor is l*(l+1), NOT l*(l+1)/2; the 1/2 is already in Tl.
 *
 * Nuclear charge convention
 * -------------------------
 *   `radial.nuclear()` returns the matrix elements of -1/r with sign
 *   included. To assemble  -Z/r  for arbitrary Z, the caller multiplies
 *   by +Z (NOT -Z).
 *
 *   For finite-nucleus models, use the TwoDBasis ctor with the appropriate
 *   `nuclear_model_t` and `Rrms`; the model_potential machinery in
 *   libhelfem (ModelPotential / nucleus headers) handles the smearing.
 *
 * Two-electron (Coulomb / exchange) convention
 * --------------------------------------------
 *   For a SINGLE-l shell (lval = {l}), TwoDBasis::coulomb(P) and
 *   exchange(P) reduce to the monopole (k=0) radial J and K. P is the
 *   TOTAL density matrix (Pa+Pb); coulomb(P) and exchange(Pa per spin)
 *   are sign-included so Fock = H_core + J + K assembles correctly.
 *
 *   For an external consumer that needs per-multipole-k radial J/K
 *   (Slater R^k integrals contracted with one density block (la, lb)),
 *   the corresponding contraction logic currently lives INSIDE
 *   TwoDBasis::coulomb/exchange in src/atomic/TwoDBasis.cpp (between the
 *   per-L radial assembly and the Gaunt-coefficient angular sum). See
 *   examples/DESIGN_NOTE.md for the recommended public accessor.
 *
 * ILP64 / LP64 BLAS interface
 * ---------------------------
 *   libhelfem is built against an ILP64 BLAS (libflexiblas64, 8-byte
 *   integer arguments) and Armadillo macros ARMA_64BIT_WORD /
 *   ARMA_BLAS_LONG. The consumer MUST compile with the same flags and
 *   link the same BLAS. A LP64 (4-byte) BLAS linked into the same
 *   process corrupts the heap inside LAPACK routines because integer
 *   sizes mismatch silently.
 *
 *   Symptom: `free(): invalid next size` or `malloc(): invalid next
 *   size` during the first LAPACK call after building S/T/V.
 * ============================================================================
 */
