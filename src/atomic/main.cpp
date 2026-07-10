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

// Atomic (2D angular+radial) driver using OpenOrbitalOptimizer.
// Supports:
//   - Restricted (closed-shell) DFT: single particle type, per-symmetry blocks,
//     max_occupation = 2 per orbital, total P used for J + XC.
//   - Unrestricted DFT: two particle types (alpha, beta) each split into the
//     same symmetry blocks, max_occupation = 1 per orbital, Pa+Pb used for J,
//     (Pa, Pb) used for the spin-polarized XC path.
//
// The per-symmetry decomposition (--symmetry=0/1/2) is a first-class use of
// OOO's per-block Aufbau + occupation logic, not an afterthought. Each block
// has its own Sinvh_k = symmetric orthonormalization of S restricted to that
// block's basis functions; the Fock builder assembles the AO Fock matrix
// once, then extracts and orthonormalizes per block.
//
// HF (x_func == 0 && c_func == 0) is still deferred (needs basis.exchange()
// bridge -- same status as sadatom_ooo / diatomic_ooo).

#include "../general/cmdline.h"
#include "../general/constants.h"
#include "../general/dftfuncs.h"
#include "../general/elements.h"
#include "../general/scf_helpers.h"
#include "../general/checkpoint.h"

#include "openorbitaloptimizer/scfsolver.hpp"

#include "basis.h"
#include "dftgrid.h"
#include <ArmaEigen.h>
#include <cfloat>

using namespace helfem;

int main(int argc, char **argv) {
  cmdline::parser parser;

  parser.add<std::string>("Z", 0, "nuclear charge", true);
  parser.add<int>("Q", 0, "charge state", false, 0);
  parser.add<int>("M", 0, "spin multiplicity (2S+1); mutually exclusive with nela/nelb", false, 0);
  parser.add<int>("nela", 0, "number of alpha electrons (leave 0 to derive from Q/M)", false, 0);
  parser.add<int>("nelb", 0, "number of beta electrons (leave 0 to derive from Q/M)", false, 0);
  parser.add<int>("lmax", 0, "maximum l quantum number", true);
  parser.add<int>("mmax", 0, "maximum m quantum number", true);
  parser.add<double>("Rmax", 0, "practical infinity in au", false, 40.0);
  parser.add<int>("grid", 0, "type of grid: 1 for linear, 2 for quadratic, 3 for polynomial, 4 for exponential", false, 4);
  parser.add<int>("grid0", 0, "type of grid: 1 for linear, 2 for quadratic, 3 for polynomial, 4 for exponential", false, 4);
  parser.add<double>("zexp", 0, "parameter in radial grid", false, 2.0);
  parser.add<double>("zexp0", 0, "parameter in radial grid", false, 2.0);
  parser.add<int>("nelem", 0, "number of elements", true);
  parser.add<int>("nelem0", 0, "number of elements between center and off-center nuclei", false, 0);
  parser.add<int>("nnodes", 0, "number of nodes per element", false, 15);
  parser.add<int>("nquad", 0, "number of quadrature points", false, 0);
  parser.add<std::string>("method", 0, "DFT method to use", false, "lda_x");
  parser.add<int>("ldft", 0, "theta rule for dft quadrature (0 for auto)", false, 0);
  parser.add<int>("mdft", 0, "phi rule for dft quadrature (0 for auto)", false, 0);
  parser.add<double>("dftthr", 0, "density threshold for dft", false, 1e-12);
  parser.add<int>("primbas", 0, "primitive radial basis", false, 4);
  parser.add<int>("finitenuc", 0, "finite nuclear model", false, 0);
  parser.add<double>("Rrms", 0, "finite nuclear rms radius", false, 0.0);
  parser.add<int>("restricted", 0, "spin-restricted: 1 restricted, 0 unrestricted, -1 auto from nela/nelb", false, -1);
  parser.add<int>("symmetry", 0, "orbital symmetry: 0 none, 1 per-m, 2 per-(l,m)", false, 1);
  parser.add<std::string>("x_pars", 0, "file for parameters for exchange functional", false, "");
  parser.add<std::string>("c_pars", 0, "file for parameters for correlation functional", false, "");
  parser.add<bool>("zeroder", 0, "zero derivative at Rmax?", false, false);

  // External static fields (parity with bespoke atomic driver).
  parser.add<double>("Ez",  0, "electric dipole field",     false, 0.0);
  parser.add<double>("Qzz", 0, "electric quadrupole field", false, 0.0);
  parser.add<double>("Bz",  0, "magnetic dipole field",     false, 0.0);

  // Confinement potential (parity with bespoke atomic driver).
  parser.add<int>   ("iconf",        0, "Confinement potential: 1 polynomial, 2 exponential, 3 barrier, 4 Junquera et al.", false, 0);
  parser.add<int>   ("conf_N",       0, "Exponent in confinement potential",   false, 0);
  parser.add<double>("conf_R",       0, "Confinement radius",                  false, 0.0);
  parser.add<double>("conf_barrier", 0, "Confinement barrier height",          false, 0.0);
  parser.add<double>("shift_conf",   0, "Where does confinement start?",       false, 0.0);
  parser.add<bool>  ("add_conf",     0, "Add element boundary at shifted confinement radius?", false, true);

  // Fock symmetry averaging over m values (parity with bespoke atomic).
  parser.add<bool>("maverage", 0, "average Fock matrix over m values", false, false);

  // Frozen per-block occupations read from occs.dat (parity with bespoke
  // atomic's --readocc, but as a bool since OOO's fixed-per-block
  // occupation API applies for the whole SCF, not for the first N
  // Fock builds only).
  parser.add<bool>("readocc", 0, "read frozen per-block occupations from occs.dat", false, false);

  // Initial guess model potential. Bespoke atomic defaulted to iguess=2
  // (SAP); we keep that here since SAP typically converges materially
  // faster than the bare core-Hamiltonian guess. Ignored when --load
  // supplies orbitals from a checkpoint.
  parser.add<int>("iguess", 0, "initial guess: 0 core Hamiltonian, 1 GSZ, 2 SAP, 3 Thomas-Fermi", false, 2);

  // Load orbital guess from a checkpoint written by an earlier run.
  // The old basis + AO densities Pa, Pb are read; densities are
  // projected into the current basis via P_new = P_proj * P_old *
  // P_proj^T with P_proj = S_new^-1 * S12, and used to seed OOO's
  // initial orbitals + occupations.
  parser.add<std::string>("load", 0, "load orbital guess from checkpoint file", false, "");
  // Save basis + final AO densities + electron counts to a checkpoint
  // for downstream reuse (--load, density_line, etc.).
  parser.add<std::string>("save", 0, "save results to checkpoint file", false, "");

  parser.parse_check(argc, argv);

  const int    Z          = get_Z(parser.get<std::string>("Z"));
        int    Q          = parser.get<int>("Q");
        int    M          = parser.get<int>("M");
        int    nela       = parser.get<int>("nela");
        int    nelb       = parser.get<int>("nelb");
  const int    lmax       = parser.get<int>("lmax");
  const int    mmax       = parser.get<int>("mmax");
  const double Rmax       = parser.get<double>("Rmax");
  const int    igrid      = parser.get<int>("grid");
  const int    igrid0     = parser.get<int>("grid0");
  const double zexp       = parser.get<double>("zexp");
  const double zexp0      = parser.get<double>("zexp0");
  const int    Nelem      = parser.get<int>("nelem");
  const int    Nelem0     = parser.get<int>("nelem0");
  const int    Nnodes     = parser.get<int>("nnodes");
        int    Nquad      = parser.get<int>("nquad");
  const std::string method = parser.get<std::string>("method");
  const int    ldft_arg   = parser.get<int>("ldft");
  const int    mdft_arg   = parser.get<int>("mdft");
  const double dftthr     = parser.get<double>("dftthr");
  const int    primbas    = parser.get<int>("primbas");
  const int    finitenuc  = parser.get<int>("finitenuc");
  const double Rrms       = parser.get<double>("Rrms");
        int    restr      = parser.get<int>("restricted");
  const int    symm       = parser.get<int>("symmetry");
  const std::string xparf = parser.get<std::string>("x_pars");
  const std::string cparf = parser.get<std::string>("c_pars");
  const bool   zeroder    = parser.get<bool>("zeroder");

  const double Ez           = parser.get<double>("Ez");
  const double Qzz          = parser.get<double>("Qzz");
  const double Bz           = parser.get<double>("Bz");
  const int    iconf        = parser.get<int>("iconf");
  const int    conf_N       = parser.get<int>("conf_N");
  const double conf_R       = parser.get<double>("conf_R");
  const double conf_barrier = parser.get<double>("conf_barrier");
  const double shift_conf   = parser.get<double>("shift_conf");
  const bool   add_conf     = parser.get<bool>("add_conf");
  const bool   maverage     = parser.get<bool>("maverage");
  const bool   readocc      = parser.get<bool>("readocc");
  const int    iguess       = parser.get<int>("iguess");
  const std::string loadfile = parser.get<std::string>("load");
  const std::string savefile = parser.get<std::string>("save");

  // Derive nela/nelb from Q, M -- same convention as the bespoke atomic
  // driver, so --M is the natural way to specify open-shell states.
  // parse_nela_nelb(nela, nelb, Q, M, Ztot) fills in nela/nelb when
  // both are zero on entry (using Q and M); if nela/nelb are given,
  // it recomputes Q from them.
  scf::parse_nela_nelb(nela, nelb, Q, M, Z);
  if (restr == -1) {
    // Auto: closed shell -> restricted, else unrestricted.
    restr = (nela == nelb) ? 1 : 0;
  }
  const bool restricted = (restr != 0);
  const int Ntot = nela + nelb;
  if (restricted && nela != nelb)
    throw std::logic_error("Restricted mode requires nela == nelb (closed shell). "
                            "Use --restricted=0 (or leave -1 for auto) for open-shell.");

  // symm=2 (per-(l,m)) breaks in the presence of an external electric or
  // magnetic field -- the field mixes l or m sectors respectively, so the
  // per-(l,m) block-diagonal Fock ansatz is no longer valid. Fall back to
  // symm=1 (per-m only). Matches src/atomic/main.cpp lines 280..288.
  int symm_eff = symm;
  if (symm_eff == 2 && (Ez != 0.0 || Qzz != 0.0)) {
    printf("Warning - asked for full orbital symmetry in presence of electric field. Relaxing restriction.\n");
    symm_eff = 1;
  }
  if (symm_eff == 2 && Bz != 0.0) {
    printf("Warning - asked for full orbital symmetry in presence of magnetic field. Relaxing restriction.\n");
    symm_eff = 1;
  }

  arma::vec x_pars, c_pars;
  if (xparf.size()) x_pars = helfem::to_arma(scf::parse_xc_params(xparf));
  if (cparf.size()) c_pars = helfem::to_arma(scf::parse_xc_params(cparf));

  int x_func, c_func;
  ::parse_xc_func(x_func, c_func, method);
  ::print_info(x_func, c_func);
  if (!is_supported(x_func) || !is_supported(c_func))
    throw std::logic_error("The specified functional is not supported in HelFEM.\n");

  // Range-separation parameters. Following libxc/HelFEM naming:
  //   kfrac  = full-range HF exchange fraction   (alpha in CAM)
  //   kshort = short-range HF exchange fraction  (beta in CAM)
  //   omega  = range-separation parameter        (mu in erfc, lambda in Yukawa)
  // For pure DFT: kfrac = kshort = 0. For pure HF: kfrac = 1, kshort = 0.
  // For hybrids (B3LYP, PBE0): kfrac in (0,1), kshort = 0.
  // For range-separated hybrids (CAM-B3LYP, LC-BLYP, wB97X): kfrac + kshort
  // nonzero and omega > 0, with basis.rs_exchange(P) giving the SR piece
  // using either erfc or Yukawa kernel (functional-dependent, discovered
  // via is_range_separated).
  double kfrac, kshort, omega;
  range_separation(x_func, omega, kfrac, kshort);
  const bool have_exx = (kfrac != 0.0 || kshort != 0.0);
  const bool have_xc  = (x_func != 0 || c_func != 0);
  bool rs_erfc = false, rs_yukawa = false;
  if (kshort != 0.0)
    is_range_separated(x_func, rs_erfc, rs_yukawa);

  auto poly = std::shared_ptr<const polynomial_basis::PolynomialBasis>(
      polynomial_basis::get_basis(primbas, Nnodes));
  if (Nquad == 0) Nquad = 5 * poly->get_nbf();
  else if (Nquad < 2 * poly->get_nbf())
    throw std::logic_error("Insufficient radial quadrature.\n");

  const int Zl = 0, Zr = 0;
  const double Rhalf = 0.0;

  arma::ivec lval, mval;
  atomic::basis::angular_basis(lmax, mmax, lval, mval);
  arma::vec bval = atomic::basis::form_grid(
      (modelpotential::nuclear_model_t) finitenuc, Rrms, Nelem, Rmax,
      igrid, zexp, Nelem0, igrid0, zexp0, Z, Zl, Zr, Rhalf,
      iconf ? add_conf : false, shift_conf);

  atomic::basis::TwoDBasis basis(Z, (modelpotential::nuclear_model_t) finitenuc,
                                  Rrms, poly, zeroder, Nquad,
                                  helfem::to_eigen(bval),
                                  helfem::to_eigen(lval),
                                  helfem::to_eigen(mval),
                                  Zl, Zr, Rhalf);
  const size_t Nbf = basis.Nbf();
  printf("Basis set: %i angular shells x %i radial = %i basis functions\n",
          (int) basis.Nang(), (int) basis.Nrad(), (int) Nbf);
  printf("Mode: %s, symmetry=%d, nela=%d nelb=%d\n",
          restricted ? "restricted" : "unrestricted", symm, nela, nelb);

  // --- One-electron + overlap in arma (chemistry-layer buffers stay arma).
  const arma::mat S    = helfem::to_arma(basis.overlap());
  const arma::mat T    = helfem::to_arma(basis.kinetic());
  const arma::mat Vnuc = helfem::to_arma(basis.nuclear());

  // --- External static-field one-electron matrices. All four are added
  //     into H0 (and hence into every Fock build), and their trace
  //     contribution is broken out in the energy print for parity with
  //     the bespoke atomic driver. Left as zeros when the corresponding
  //     coupling is off, so the additions cost nothing but a trace pass.
  arma::mat Vconf(Nbf, Nbf, arma::fill::zeros);
  if (iconf) {
    printf("Computing confinement potential\n");
    Vconf = helfem::to_arma(basis.confinement(conf_N, conf_R, iconf, conf_barrier, shift_conf));
  }
  const arma::mat dip  = helfem::to_arma(basis.dipole_z());
  const arma::mat quad = helfem::to_arma(basis.quadrupole_zz());
  const arma::mat Vel  = Ez * dip + Qzz * quad / 3.0;
  const arma::mat Vmag = helfem::to_arma(basis.Bz_field(Bz));
  const bool have_efield = (Ez != 0.0 || Qzz != 0.0);
  const bool have_bfield = (Bz != 0.0);
  const bool have_conf   = (iconf != 0);

  // l_idx groups AO basis-function index sets by (l, m): the l-th outer
  // entry contains the BF-index arrays for the m values that are
  // actually present in the basis (|m| <= min(l, mmax)). The Fock
  // symmetry-average step below averages each l-block's Fock over that
  // m subset, enforcing degenerate orbital energies for orbitals of the
  // same l but different m (a physical symmetry of the atomic
  // Hamiltonian that finite-precision SCF can drift out of).
  //
  // Note: the bespoke atomic driver builds an m = -l..+l range
  // unconditionally, which crashes fock_symmetry_average when the basis
  // has mmax < lmax (empty index arrays hit a 0x0 = NxN assignment).
  // Filtering by lm_indices(l, m).n_elem > 0 keeps parity when
  // mmax == lmax and degrades gracefully when mmax < lmax (partial
  // m-average over the represented m subset -- a no-op when only m=0
  // is present, which is what you want).
  std::vector<std::vector<arma::uvec>> l_idx;
  if (maverage) {
    const arma::ivec l_all = basis.get_lval();
    const int lmax_bf = arma::max(l_all);
    l_idx.assign(lmax_bf + 1, {});
    for (int l = 0; l <= lmax_bf; ++l)
      for (int m = -l; m <= l; ++m) {
        arma::uvec idx = basis.lm_indices(l, m);
        if (idx.n_elem) l_idx[l].push_back(idx);
      }
  }

  // --- Symmetry decomposition. symm==0 collapses to one block containing
  //     all basis functions.
  std::vector<arma::uvec> dsym;
  if (symm_eff == 0) {
    arma::uvec all(Nbf);
    for (size_t i = 0; i < Nbf; ++i) all(i) = i;
    dsym.push_back(all);
  } else {
    dsym = basis.get_sym_idx(symm_eff);
  }
  const size_t nsym = dsym.size();

  // --- Per-block Sinvh_k = symmetric orthonormalization of S(dsym[k], dsym[k]).
  std::vector<arma::mat> Sinvh_arma(nsym);
  for (size_t k = 0; k < nsym; ++k) {
    if (!dsym[k].n_elem) continue;
    const arma::mat Sk = S(dsym[k], dsym[k]);
    Sinvh_arma[k] = helfem::to_arma(scf::form_Sinvh(helfem::to_eigen(Sk), /*chol*/false));
  }

  const int ldft = ldft_arg > 0 ? ldft_arg : 4 * lmax + 12;
  const int mdft = mdft_arg > 0 ? mdft_arg : 4 * mmax + 12;
  auto grid = helfem::atomic::dftgrid::DFTGrid(&basis, ldft, mdft);
  basis.compute_tei(have_exx);
  // Precompute the range-separated exchange kernel if the functional uses
  // one. Yukawa (screened Coulomb) and erfc (complementary error-function)
  // are the two kernels libxc's CAM functionals ask for.
  if (rs_yukawa) basis.compute_yukawa(omega);
  else if (rs_erfc) basis.compute_erfc(omega);

  // --- OOO block layout.
  using OOO_Real = double;
  const size_t nparttype = restricted ? 1 : 2;
  OpenOrbitalOptimizer::IndexVector number_of_blocks_per_particle_type(nparttype);
  Eigen::Matrix<OOO_Real, Eigen::Dynamic, 1> maximum_occupation(nsym * nparttype);
  Eigen::Matrix<OOO_Real, Eigen::Dynamic, 1> number_of_particles(nparttype);
  std::vector<std::string> block_descriptions;
  block_descriptions.reserve(nsym * nparttype);

  for (size_t t = 0; t < nparttype; ++t) {
    number_of_blocks_per_particle_type(t) = static_cast<int>(nsym);
    number_of_particles(t) = static_cast<OOO_Real>(restricted ? Ntot : (t == 0 ? nela : nelb));
    for (size_t k = 0; k < nsym; ++k) {
      maximum_occupation(t * nsym + k) = restricted ? 2.0 : 1.0;
      block_descriptions.push_back(
          (nparttype == 1 ? "" : (t == 0 ? "a:" : "b:")) + std::string("sym") + std::to_string(k));
    }
  }

  // Accumulate a per-block density (C_k * diag(occ_k) * C_k^T) into the
  // full-Nbf density matrix P_full through the block's basis-function
  // scatter index dsym[k].
  auto accumulate_density = [&](arma::mat & P_full, size_t k,
                                 const helfem::Matrix & orb_e,
                                 const helfem::Vector & occ_e) {
    if (!dsym[k].n_elem) return;
    const arma::mat orb  = helfem::to_arma(orb_e);
    const arma::vec occ  = helfem::to_arma(occ_e);
    const arma::mat C_k  = Sinvh_arma[k] * orb;
    const arma::mat P_k  = C_k * arma::diagmat(occ) * C_k.t();
    P_full(dsym[k], dsym[k]) += P_k;
  };

  // Extract F_full(dsym[k], dsym[k]), transform to the block's
  // orthonormal basis via Sinvh_k^T . F_k . Sinvh_k, and stash into
  // fock[b] as helfem::Matrix.
  auto orthonormalize_block =
      [&](OpenOrbitalOptimizer::FockMatrix<OOO_Real> & fock, size_t b,
          const arma::mat & F_full, size_t k) {
    if (!dsym[k].n_elem) {
      fock[b] = helfem::Matrix::Zero(0, 0);
      return;
    }
    const arma::mat Fk_sub = F_full(dsym[k], dsym[k]);
    const arma::mat F_orth = Sinvh_arma[k].t() * Fk_sub * Sinvh_arma[k];
    fock[b] = helfem::to_eigen(F_orth);
  };

  OpenOrbitalOptimizer::FockBuilder<OOO_Real, OOO_Real> fock_builder =
      [&](const OpenOrbitalOptimizer::DensityMatrix<OOO_Real, OOO_Real> & dm) {
    const auto & orbitals    = dm.first;
    const auto & occupations = dm.second;

    // --- Density assembly. Restricted mode has a single closed-shell
    //     channel with max_occ = 2, so P comes straight from the alpha
    //     channel; no Pa/Pb split, no *0.5 double-scatter, no unused Pb.
    OpenOrbitalOptimizer::FockMatrix<OOO_Real> fock(nsym * nparttype);
    arma::mat P(Nbf, Nbf, arma::fill::zeros);
    // Spin-density buffers used for both the XC branch and the HF
    // exchange branch; for restricted the alpha density is 0.5 * P.
    arma::mat Pa, Pb;
    double Exc = 0.0;
    double nelnum = 0.0, ekin_grid = 0.0;
    arma::mat XCa, XCb;

    if (restricted) {
      for (size_t k = 0; k < nsym; ++k)
        accumulate_density(P, k, orbitals[k], occupations[k]);
      if (have_xc)
        grid.eval_Fxc(x_func, x_pars, c_func, c_pars, P, XCa, Exc, nelnum, ekin_grid, dftthr);
      if (have_exx) Pa = 0.5 * P;
    } else {
      Pa.zeros(Nbf, Nbf);
      Pb.zeros(Nbf, Nbf);
      for (size_t k = 0; k < nsym; ++k) {
        accumulate_density(Pa, k, orbitals[k],        occupations[k]);
        accumulate_density(Pb, k, orbitals[nsym + k], occupations[nsym + k]);
      }
      P = Pa + Pb;
      if (have_xc)
        grid.eval_Fxc(x_func, x_pars, c_func, c_pars, Pa, Pb, XCa, XCb,
                       Exc, nelnum, ekin_grid, nelb > 0, dftthr);
    }

    const double Ekin = arma::trace(P * T);
    const double Enuc = arma::trace(P * Vnuc);
    // External-field contributions. Vel/Vmag/Vconf are zero matrices when
    // the corresponding coupling is disabled, so the traces cost O(Nbf).
    // The -Bz/2 * (nela-nelb) piece is the Ms.Bz spin-Zeeman term:
    // restricted mode has nela == nelb so it vanishes there.
    const double Eefield = have_efield ? arma::trace(P * Vel)  : 0.0;
    const double Emfield = have_bfield
        ? arma::trace(P * Vmag) - 0.5 * Bz * (nela - nelb) : 0.0;
    const double Econf   = have_conf   ? arma::trace(P * Vconf) : 0.0;

    const arma::mat J = helfem::to_arma(basis.coulomb(helfem::to_eigen(P)));
    const double Ecoul = 0.5 * arma::trace(P * J);

    // --- HF exchange: K = kfrac * basis.exchange(P) + kshort * basis.rs_exchange(P).
    //     Sign convention is baked into basis.exchange (it returns the
    //     signed contribution that gets ADDED to the Fock matrix), so
    //     Exx = 0.5 * trace(P_spin * K_spin) per channel with a +
    //     sign matches the bespoke atomic driver's line 754.
    //     For RS hybrids, rs_exchange uses the erfc or Yukawa kernel
    //     precomputed above.
    arma::mat Ka, Kb;
    double Exx = 0.0;
    if (have_exx) {
      Ka.zeros(Nbf, Nbf);
      if (kfrac  != 0.0) Ka += kfrac  * helfem::to_arma(basis.exchange(helfem::to_eigen(Pa)));
      if (kshort != 0.0) Ka += kshort * helfem::to_arma(basis.rs_exchange(helfem::to_eigen(Pa)));
      Exx = 0.5 * arma::trace(Pa * Ka);
      if (!restricted) {
        Kb.zeros(Nbf, Nbf);
        if (kfrac  != 0.0) Kb += kfrac  * helfem::to_arma(basis.exchange(helfem::to_eigen(Pb)));
        if (kshort != 0.0) Kb += kshort * helfem::to_arma(basis.rs_exchange(helfem::to_eigen(Pb)));
        Exx += 0.5 * arma::trace(Pb * Kb);
      } else {
        // In restricted mode alpha == beta density, and we only assemble
        // one Fock. Skipping the K(Pb) call saves one exchange build;
        // the beta contribution equals the alpha one so double it.
        Exx *= 2.0;
      }
    }

    const double Etot = Ekin + Enuc + Eefield + Emfield + Econf
                       + Ecoul + Exc + Exx;
    printf("kinetic %.10f nuclear %.10f Coulomb %.10f XC %.10f Exx %.10f",
            Ekin, Enuc, Ecoul, Exc, Exx);
    if (have_efield) printf(" Eefield %.10f", Eefield);
    if (have_bfield) printf(" Emfield %.10f", Emfield);
    if (have_conf)   printf(" Econf %.10f",   Econf);
    printf("  total %.10f  (nel err %.3e)\n",
            Etot, nelnum - static_cast<double>(Ntot));
    fflush(stdout);

    // --- Fock assembly. Restricted: one AO Fock, orthonormalize per block.
    //     Unrestricted: separate Fa/Fb AO Fock matrices for the two channels.
    // Pre-assembled 1-electron core matrix. Vel/Vmag/Vconf are zero when
    // their coupling is off so this is unchanged from H0 = T + Vnuc in
    // the no-field case.
    const arma::mat H1 = T + Vnuc + Vel + Vmag + Vconf;
    // Apply the maverage post-processor once per channel; noop otherwise.
    auto apply_mavg = [&](arma::mat & F) {
      if (maverage)
        F = helfem::to_arma(scf::fock_symmetry_average(helfem::to_eigen(F), l_idx));
    };
    if (restricted) {
      arma::mat F_ao = H1 + J;
      if (have_xc)  F_ao += XCa;
      if (have_exx) F_ao += Ka;
      apply_mavg(F_ao);
      for (size_t k = 0; k < nsym; ++k)
        orthonormalize_block(fock, k, F_ao, k);
    } else {
      arma::mat Fa_ao = H1 + J;
      arma::mat Fb_ao = H1 + J;
      if (have_xc)  { Fa_ao += XCa; Fb_ao += XCb; }
      if (have_exx) { Fa_ao += Ka;  Fb_ao += Kb;  }
      // Spin-Zeeman splitting: alpha <- -Bz/2 * S, beta <- +Bz/2 * S.
      // Only matters in unrestricted mode -- in restricted the two
      // channels are equal by construction.
      if (have_bfield) {
        Fa_ao -= 0.5 * Bz * S;
        Fb_ao += 0.5 * Bz * S;
      }
      apply_mavg(Fa_ao);
      apply_mavg(Fb_ao);
      for (size_t k = 0; k < nsym; ++k) {
        orthonormalize_block(fock, k,        Fa_ao, k);
        orthonormalize_block(fock, nsym + k, Fb_ao, k);
      }
    }
    return std::make_pair(Etot, fock);
  };

  // Initial-guess Hamiltonian per block per particle type. Include the
  // external-field 1e matrices so the guess feels the perturbing
  // potential from the start. iguess selects the electron-nuclear
  // channel:
  //   0 core Hamiltonian: bare Vnuc
  //   1..3 model potential (GSZ / SAP / Thomas-Fermi) instead of Vnuc.
  //     Restores the bespoke-atomic guess menu; SAP is the default
  //     because it typically converges materially faster than core-H.
  arma::mat Vguess;
  if (iguess == 0) {
    printf("Guess orbitals from core Hamiltonian\n");
    Vguess = Vnuc;
  } else {
    modelpotential::ModelPotential * model = nullptr;
    switch (iguess) {
    case 1:
      printf("Guess orbitals from GSZ screened nucleus\n");
      model = new modelpotential::GSZAtom(Z);
      break;
    case 2:
      printf("Guess orbitals from SAP screened nucleus\n");
      model = new modelpotential::SAPAtom(Z);
      break;
    case 3:
      printf("Guess orbitals from Thomas-Fermi screened nucleus\n");
      model = new modelpotential::TFAtom(Z);
      break;
    default:
      throw std::logic_error("Unsupported iguess value (expected 0..3).\n");
    }
    Vguess = helfem::to_arma(basis.model_potential(model));
    delete model;
  }
  const arma::mat H0 = T + Vguess + Vel + Vmag + Vconf;
  OpenOrbitalOptimizer::FockMatrix<OOO_Real> CoreH(nsym * nparttype);
  for (size_t t = 0; t < nparttype; ++t) {
    for (size_t k = 0; k < nsym; ++k) {
      if (!dsym[k].n_elem) {
        CoreH[t * nsym + k] = helfem::Matrix::Zero(0, 0);
        continue;
      }
      arma::mat H_sub = H0(dsym[k], dsym[k]);
      // Same spin-Zeeman split as in the Fock builder (only meaningful
      // in unrestricted mode; harmless in restricted since we hit both
      // channels equally).
      if (have_bfield && nparttype == 2)
        H_sub += (t == 0 ? -0.5 : 0.5) * Bz * arma::mat(S(dsym[k], dsym[k]));
      const arma::mat H_orth = Sinvh_arma[k].t() * H_sub * Sinvh_arma[k];
      CoreH[t * nsym + k] = helfem::to_eigen(H_orth);
    }
  }

  OpenOrbitalOptimizer::SCFSolver<OOO_Real, OOO_Real> scfsolver(
      number_of_blocks_per_particle_type, maximum_occupation,
      number_of_particles, fock_builder, block_descriptions);

  // --readocc: parse occs.dat and hand OOO a fixed per-block particle
  // count, bypassing Aufbau for the whole SCF. Bespoke atomic reads
  // three columns (nocca, noccb, m) under symm=1 and four columns
  // (nocca, noccb, l, m) under symm=2 (symm=0 is not supported since
  // there is no per-block index to key on).
  //
  // Semantic difference from bespoke: this is a bool -- OOO's
  // fixed_number_of_particles_per_block API cannot be released
  // mid-SCF, so we freeze for the entire run instead of the first N
  // Fock builds only. In practice readocc is nearly always used to
  // pin a specific configuration for the whole SCF so this matches
  // the common case exactly.
  if (readocc) {
    if (symm_eff == 0)
      throw std::logic_error("--readocc requires --symmetry=1 or 2 (need per-block index).");
    arma::imat occs;
    occs.load("occs.dat", arma::raw_ascii);
    if (symm_eff == 1 && occs.n_cols != 3)
      throw std::logic_error("occs.dat: expected 3 columns (nocca, noccb, m) for symmetry=1.");
    if (symm_eff == 2 && occs.n_cols != 4)
      throw std::logic_error("occs.dat: expected 4 columns (nocca, noccb, l, m) for symmetry=2.");

    // Match each occs row to a dsym block by comparing BF-index sets.
    // The dsym[k] arrays come from basis.get_sym_idx(symm), so an m-key
    // row hits basis.m_indices(m) which is exactly one of the dsym entries.
    Eigen::Matrix<OOO_Real, Eigen::Dynamic, 1> fixed_particles =
        Eigen::Matrix<OOO_Real, Eigen::Dynamic, 1>::Zero(nsym * nparttype);
    int total_a = 0, total_b = 0;
    for (arma::uword i = 0; i < occs.n_rows; ++i) {
      const int nocca_i = static_cast<int>(occs(i, 0));
      const int noccb_i = static_cast<int>(occs(i, 1));
      arma::uvec row_idx;
      if (symm_eff == 1)
        row_idx = basis.m_indices(occs(i, 2));
      else
        row_idx = basis.lm_indices(occs(i, 2), occs(i, 3));
      if (!row_idx.n_elem)
        throw std::logic_error("occs.dat: row references a symmetry block with no basis functions.");
      int matched_block = -1;
      for (size_t k = 0; k < nsym; ++k) {
        if (dsym[k].n_elem == row_idx.n_elem &&
            arma::all(dsym[k] == row_idx)) { matched_block = static_cast<int>(k); break; }
      }
      if (matched_block < 0)
        throw std::logic_error("occs.dat: row does not match any current symmetry block.");
      total_a += nocca_i;
      total_b += noccb_i;
      if (restricted) {
        if (nocca_i != noccb_i)
          throw std::logic_error("occs.dat: nocca != noccb on a row is incompatible with restricted mode.");
        fixed_particles(matched_block) = static_cast<OOO_Real>(nocca_i + noccb_i);
      } else {
        fixed_particles(matched_block)         = static_cast<OOO_Real>(nocca_i);
        fixed_particles(nsym + matched_block)  = static_cast<OOO_Real>(noccb_i);
      }
    }
    if (total_a != nela)
      throw std::logic_error("occs.dat: sum of nocca does not match --nela.");
    if (total_b != nelb)
      throw std::logic_error("occs.dat: sum of noccb does not match --nelb.");
    scfsolver.fixed_number_of_particles_per_block(fixed_particles);
  }

  // --load: project a saved AO density from an old basis into the
  // current basis and initialise OOO with per-block orbitals derived
  // from the projected density's eigenvectors. If no file was given
  // OOO is seeded with the core Hamiltonian as before.
  if (loadfile.size()) {
    Checkpoint loadchk(loadfile, /*writemode=*/false);
    atomic::basis::TwoDBasis oldbasis;
    loadchk.read(oldbasis);
    arma::mat Pa_old, Pb_old;
    loadchk.read("Pa", Pa_old);
    loadchk.read("Pb", Pb_old);
    int nela_old = 0, nelb_old = 0;
    loadchk.read("nela", nela_old);
    loadchk.read("nelb", nelb_old);
    if (nela_old != nela || nelb_old != nelb)
      throw std::logic_error("--load: checkpoint nela/nelb do not match current run.");

    // Cross-basis overlap S12 = <this | oldbasis>, shape (Nbf, Nbf_old).
    const arma::mat S12         = helfem::to_arma(basis.overlap(oldbasis));
    // Full-basis Sinvh (symmetric orthonormalisation, no per-symmetry
    // block). S^-1 = Sinvh * Sinvh^T since Sinvh = S^{-1/2}.
    const arma::mat Sinvh_full  = helfem::to_arma(basis.Sinvh(/*chol*/false, /*sym*/0));
    const arma::mat Pproj       = Sinvh_full * arma::trans(Sinvh_full) * S12;

    // Project the old AO densities to the current basis, then rescale
    // so the total trace P*S recovers the exact electron count (the
    // projection itself is not particle-conserving when the two bases
    // do not span the same subspace).
    arma::mat Pa_new = Pproj * Pa_old * arma::trans(Pproj);
    arma::mat Pb_new = Pproj * Pb_old * arma::trans(Pproj);
    const double na = arma::trace(Pa_new * S);
    const double nb = arma::trace(Pb_new * S);
    if (na > 0 && nela > 0) Pa_new *= static_cast<double>(nela) / na;
    if (nb > 0 && nelb > 0) Pb_new *= static_cast<double>(nelb) / nb;

    // Build per-block (orbitals, occupations) for OOO from the
    // projected density. For a block k with basis-function index set
    // dsym[k] and orthonormaliser Sinvh_k, we diagonalise
    //   P_orth = Sinvh_k^T . P_spin(dsym[k], dsym[k]) . Sinvh_k
    // and hand OOO the eigenvectors (orthonormal orbitals in the
    // block basis) and eigenvalues (per-orbital occupations, sorted
    // largest first for Aufbau semantics).
    OpenOrbitalOptimizer::Orbitals<OOO_Real>            loaded_orbs(nsym * nparttype);
    OpenOrbitalOptimizer::OrbitalOccupations<OOO_Real>  loaded_occs(nsym * nparttype);
    const double max_occ_restr = 2.0;
    const double max_occ_open  = 1.0;
    auto fill_block = [&](size_t base, size_t k, const arma::mat & Pspin,
                           double max_occ) {
      if (!dsym[k].n_elem) {
        loaded_orbs[base + k] = helfem::Matrix::Zero(0, 0);
        loaded_occs[base + k] = helfem::Vector::Zero(0);
        return;
      }
      const arma::mat Pblk  = Pspin(dsym[k], dsym[k]);
      const arma::mat Porth = arma::trans(Sinvh_arma[k]) * Pblk * Sinvh_arma[k];
      arma::vec occ_eigs;
      arma::mat vec_eigs;
      if (!arma::eig_sym(occ_eigs, vec_eigs, Porth))
        throw std::logic_error("--load: eigendecomposition of projected block density failed");
      // Ascending -> descending so highest-occupation orbitals come first.
      const arma::uword n = vec_eigs.n_cols;
      arma::mat V(vec_eigs.n_rows, n);
      arma::vec w(n);
      for (arma::uword i = 0; i < n; ++i) {
        V.col(i) = vec_eigs.col(n - 1 - i);
        w(i)     = std::min(std::max(occ_eigs(n - 1 - i), 0.0), max_occ);
      }
      loaded_orbs[base + k] = helfem::to_eigen(V);
      loaded_occs[base + k] = helfem::to_eigen(w);
    };
    if (restricted) {
      // Single channel: eigenvalues of total P should be in [0, 2].
      const arma::mat P_total = Pa_new + Pb_new;
      for (size_t k = 0; k < nsym; ++k) fill_block(0, k, P_total, max_occ_restr);
    } else {
      for (size_t k = 0; k < nsym; ++k) {
        fill_block(0,    k, Pa_new, max_occ_open);
        fill_block(nsym, k, Pb_new, max_occ_open);
      }
    }
    scfsolver.initialize_with_orbitals(loaded_orbs, loaded_occs);
  } else {
    scfsolver.initialize_with_fock(CoreH);
  }

  scfsolver.run();

  // --save: reconstruct the AO densities from the converged per-block
  // orbitals + occupations and write them plus the basis to a
  // checkpoint. Layout matches --load's expectation and is what
  // density_line / density_grid read.
  if (savefile.size()) {
    Checkpoint savechk(savefile, /*writemode=*/true);
    savechk.write(basis);
    const auto final_orbs = scfsolver.get_orbitals();
    const auto final_occs = scfsolver.get_orbital_occupations();

    arma::mat Pa_final(Nbf, Nbf, arma::fill::zeros);
    arma::mat Pb_final(Nbf, Nbf, arma::fill::zeros);
    for (size_t k = 0; k < nsym; ++k) {
      if (!dsym[k].n_elem) continue;
      const arma::mat orb_a_ao = Sinvh_arma[k] * helfem::to_arma(final_orbs[k]);
      const arma::vec occ_a    = helfem::to_arma(final_occs[k]);
      if (restricted) {
        // orbitals[k] carries the closed-shell total density (max occ 2);
        // split evenly between alpha and beta on disk.
        const arma::mat P_block = 0.5 * (orb_a_ao * arma::diagmat(occ_a) * arma::trans(orb_a_ao));
        arma::mat Pa_tmp = Pa_final; Pa_tmp(dsym[k], dsym[k]) += P_block; Pa_final = Pa_tmp;
        arma::mat Pb_tmp = Pb_final; Pb_tmp(dsym[k], dsym[k]) += P_block; Pb_final = Pb_tmp;
      } else {
        const arma::mat orb_b_ao = Sinvh_arma[k] * helfem::to_arma(final_orbs[nsym + k]);
        const arma::vec occ_b    = helfem::to_arma(final_occs[nsym + k]);
        arma::mat Pa_tmp = Pa_final;
        Pa_tmp(dsym[k], dsym[k]) += orb_a_ao * arma::diagmat(occ_a) * arma::trans(orb_a_ao);
        Pa_final = Pa_tmp;
        arma::mat Pb_tmp = Pb_final;
        Pb_tmp(dsym[k], dsym[k]) += orb_b_ao * arma::diagmat(occ_b) * arma::trans(orb_b_ao);
        Pb_final = Pb_tmp;
      }
    }
    savechk.write("Pa", Pa_final);
    savechk.write("Pb", Pb_final);
    savechk.write("nela", nela);
    savechk.write("nelb", nelb);
    printf("Saved results to %s\n", savefile.c_str());
  }

  return 0;
}
