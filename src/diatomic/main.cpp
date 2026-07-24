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

// Diatomic driver using OpenOrbitalOptimizer. Supports:
//   - Restricted (closed-shell) DFT: one particle type, per-symmetry blocks,
//     max_occupation = 2 per orbital.
//   - Unrestricted DFT: two particle types (alpha, beta) each split into the
//     same symmetry blocks; max_occupation = 1 per orbital.
//
// Symmetry decomposition uses basis.get_sym_idx(--symmetry) with 0/1/2 for
// none / per-m / per-(m, parity). Per-block Sinvh_k is the symmetric
// orthonormalization of S restricted to that block. Fock builder assembles
// the AO Fock matrix once, then extracts and orthonormalizes per block.
//
// nela/nelb are derived from --Q and --M (spin multiplicity, 2S+1) via
// scf::parse_nela_nelb, matching the bespoke diatomic driver's CLI.
//
// HF (x_func == 0 && c_func == 0) is deferred; picking any HF method throws.

#include "../general/cmdline.h"
#include "../general/constants.h"
#include "../general/dftfuncs.h"
#include "../general/elements.h"
#include "../general/scf_helpers.h"
#include "../general/scf_driver_common.h"
#include "../general/timer.h"
#include "../general/checkpoint.h"
#include "../general/eigen_io.h"

#include "openorbitaloptimizer/scfsolver.hpp"

#include "utils.h"
#include "basis.h"
#include "dftgrid.h"
#include "dftgrid_purem.h"
#include "twodquadrature.h"
#include "../atomic/basis.h"
#include <Eigen/Eigenvalues>
#include <cfloat>
#include <climits>
#include <sstream>

using namespace helfem;

int main(int argc, char **argv) {
  cmdline::parser parser;

  parser.add<std::string>("Z1", 0, "first nuclear charge", true);
  parser.add<std::string>("Z2", 0, "second nuclear charge", true);
  parser.add<double>("Rbond", 0, "internuclear distance", true);
  parser.add<bool>("angstrom", 0, "input Rbond in angstrom instead of bohr", false, false);
  parser.add<int>("Q", 0, "charge state", false, 0);
  parser.add<int>("M", 0, "spin multiplicity (2S+1); mutually exclusive with nela/nelb", false, 0);
  parser.add<int>("nela", 0, "number of alpha electrons (leave 0 to derive from Q/M)", false, 0);
  parser.add<int>("nelb", 0, "number of beta electrons (leave 0 to derive from Q/M)", false, 0);
  parser.add<std::string>("lmax", 0, "maximum l quantum number (single int if mmax given, else comma list)", true, "");
  parser.add<int>("mmax", 0, "maximum m quantum number", false, -1);
  parser.add<double>("Rmax", 0, "practical infinity in au", false, 40.0);
  parser.add<int>("grid", 0, "type of grid: 1 linear, 2 quadratic, 3 polynomial, 4 exponential", false, 4);
  parser.add<double>("zexp", 0, "parameter in radial grid", false, 1.0);
  parser.add<int>("nelem", 0, "number of elements", true);
  parser.add<int>("nnodes", 0, "number of nodes per element", false, 15);
  parser.add<int>("nquad", 0, "radial quadrature points: DFT grid + auto-convergence seed (no longer sets integral accuracy, which now converges automatically)", false, 0);
  parser.add<std::string>("method", 0, "DFT method to use", false, "lda_x");
  parser.add<int>("ldft", 0, "theta rule for dft quadrature (0 for auto)", false, 0);
  parser.add<int>("mdft", 0, "phi rule for dft quadrature (0 for auto; unused by the pure-m path)", false, 0);
  parser.add<int>("purem", 0, "pure-m XC fast path (analytic phi): -1 auto (on when symmetry>=1), 0 off, 1 force on", false, -1);
  parser.add<double>("dftthr", 0, "density threshold for dft", false, 1e-12);
  parser.add<int>("primbas", 0, "primitive radial basis", false, 4);
  parser.add<int>("finitenuc", 0, "finite nuclear model: 0 point, 1 Gaussian, 2 spherical, 3 hollow, 4 regularized", false, 0);
  parser.add<double>("Rrms1", 0, "nucleus 1 finite rms radius", false, 0.0);
  parser.add<double>("Rrms2", 0, "nucleus 2 finite rms radius", false, 0.0);
  parser.add<int>("restricted", 0, "spin-restricted: 1 restricted, 0 unrestricted, -1 auto from nela/nelb", false, -1);
  parser.add<int>("symmetry", 0, "orbital symmetry: 0 none, 1 per-m, 2 per-(m,parity) (homonuclear only)", false, 1);
  parser.add<std::string>("x_pars", 0, "file for parameters for exchange functional", false, "");
  parser.add<std::string>("c_pars", 0, "file for parameters for correlation functional", false, "");

  // External static fields (parity with bespoke diatomic driver).
  parser.add<double>("Ez",  0, "electric dipole field",     false, 0.0);
  parser.add<double>("Qzz", 0, "electric quadrupole field", false, 0.0);
  parser.add<double>("Bz",  0, "magnetic dipole field",     false, 0.0);

  // Fock symmetry averaging over m values (parity with bespoke diatomic).
  parser.add<bool>("maverage", 0, "average Fock matrix over m values", false, false);

  // Frozen per-block occupations from occs.dat (parity with bespoke
  // diatomic's --readocc, but a bool since OOO's fixed-per-block
  // occupations apply for the whole SCF, not just N Fock builds).
  parser.add<bool>("readocc", 0, "read frozen per-block occupations from occs.dat", false, false);

  // Initial guess model potential (parity with bespoke diatomic). SAP
  // is the default because it typically converges materially faster
  // than the bare core-Hamiltonian guess. Ignored when --load supplies
  // orbitals from a checkpoint.
  parser.add<int>("iguess", 0, "initial guess: 0 core Hamiltonian, 1 GSZ, 2 SAP, 3 Thomas-Fermi", false, 2);

  // Load orbital guess from a checkpoint written by an earlier run.
  parser.add<std::string>("load", 0, "load orbital guess from checkpoint file", false, "");
  // Save basis + AO densities + Rhalf/Z1/Z2/electron counts to a
  // checkpoint (for --load and for the diatomic_dline / diatomic_dgrid
  // / diatomic_cpl analysis tools).
  parser.add<std::string>("save", 0, "save results to checkpoint file", false, "");
  // SCF convergence algorithms handed to OOO's state machine: a '+'
  // separated subset of DIIS, ODA, CG and LBFGS. OOO switches between
  // whichever are enabled and collapses gracefully on a subset.
  parser.add<std::string>("scfmethods", 0, "SCF convergence methods: '+' separated subset of DIIS, ODA, CG, LBFGS", false, "DIIS + ODA + CG");

  parser.parse_check(argc, argv);

  const int Z1        = get_Z(parser.get<std::string>("Z1"));
  const int Z2        = get_Z(parser.get<std::string>("Z2"));
        double Rbond  = parser.get<double>("Rbond");
  if (parser.get<bool>("angstrom")) {
    printf("Converting Rbond from %g angstrom to %g bohr.\n", Rbond, Rbond * ANGSTROMINBOHR);
    Rbond *= ANGSTROMINBOHR;
  }
        int Q         = parser.get<int>("Q");
        int M         = parser.get<int>("M");
        int nela      = parser.get<int>("nela");
        int nelb      = parser.get<int>("nelb");
  const std::string lmax_str = parser.get<std::string>("lmax");
        int mmax      = parser.get<int>("mmax");
  const double Rmax   = parser.get<double>("Rmax");
  const int igrid     = parser.get<int>("grid");
  const double zexp   = parser.get<double>("zexp");
  const int Nelem     = parser.get<int>("nelem");
  const int Nnodes    = parser.get<int>("nnodes");
        int Nquad     = parser.get<int>("nquad");
  const std::string method = parser.get<std::string>("method");
  const int ldft_arg  = parser.get<int>("ldft");
  const int mdft_arg  = parser.get<int>("mdft");
  const int purem_arg = parser.get<int>("purem");
  const double dftthr = parser.get<double>("dftthr");
  const int primbas   = parser.get<int>("primbas");
  const int finitenuc = parser.get<int>("finitenuc");
  const double Rrms1  = parser.get<double>("Rrms1");
  const double Rrms2  = parser.get<double>("Rrms2");
        int restr     = parser.get<int>("restricted");
        int symm      = parser.get<int>("symmetry");
  const std::string xparf = parser.get<std::string>("x_pars");
  const std::string cparf = parser.get<std::string>("c_pars");
  const double Ez     = parser.get<double>("Ez");
  const double Qzz    = parser.get<double>("Qzz");
  const double Bz     = parser.get<double>("Bz");
  const bool   maverage = parser.get<bool>("maverage");
  const bool   readocc  = parser.get<bool>("readocc");
  const int    iguess       = parser.get<int>("iguess");
  const std::string loadfile = parser.get<std::string>("load");
  const std::string savefile = parser.get<std::string>("save");

  // Derive nela/nelb from Q, M -- same convention as the bespoke diatomic
  // driver. Total nuclear charge is Z1 + Z2.
  bool restricted;
  int Ntot;
  helfem::scf_driver::derive_nela_nelb_restricted(
      nela, nelb, restr, Q, M, Z1 + Z2, restricted, Ntot);

  helfem::Vector x_pars, c_pars;
  if (xparf.size()) x_pars = scf::parse_xc_params(xparf);
  if (cparf.size()) c_pars = scf::parse_xc_params(cparf);

  int x_func, c_func;
  ::parse_xc_func(x_func, c_func, method);
  ::print_info(x_func, c_func);
  if (!is_supported(x_func) || !is_supported(c_func))
    throw std::logic_error("The specified functional is not supported in HelFEM.\n");

  double kfrac, kshort, omega;
  range_separation(x_func, omega, kfrac, kshort);
  const bool have_exx = (kfrac != 0.0 || kshort != 0.0);
  const bool have_xc  = (x_func != 0 || c_func != 0);
  if (have_exx && kshort != 0.0)
    throw std::logic_error("Range-separated hybrids not yet supported in diatomic_ooo (needs omega wiring).\n");

  auto poly = std::shared_ptr<const polynomial_basis::PolynomialBasis>(
      polynomial_basis::get_basis(primbas, Nnodes));
  if (Nquad == 0) Nquad = 5 * poly->get_nbf();
  else if (Nquad < 2 * poly->get_nbf())
    throw std::logic_error("Insufficient radial quadrature.\n");

  Eigen::VectorXi lmmax;
  if (mmax >= 0) {
    lmmax = Eigen::VectorXi::Constant(mmax + 1, std::atoi(lmax_str.c_str()));
  } else {
    std::vector<int> lmmaxv;
    std::stringstream ss(lmax_str);
    std::string tok;
    while (std::getline(ss, tok, ','))
      lmmaxv.push_back(std::atoi(tok.c_str()));
    lmmax = Eigen::Map<const Eigen::VectorXi>(lmmaxv.data(),
                                              static_cast<Eigen::Index>(lmmaxv.size()));
    mmax = static_cast<int>(lmmaxv.size()) - 1;
  }

  Eigen::VectorXi lval, mval;
  diatomic::basis::lm_to_l_m(lmmax, lval, mval);

  const double Rhalf = 0.5 * Rbond;
  const double mumax = utils::arcosh(Rmax / Rhalf);
  const helfem::Vector bval = atomic::basis::normal_grid(Nelem, mumax, igrid, zexp);

  // Homonuclear g/u symmetry is not a valid decomposition for a
  // heteronuclear molecule -- get_sym_idx(2) still splits by (m, l%2)
  // but the SCF then sees an artificially restricted trial space and
  // converges to a WRONG energy (silent in the bespoke driver too
  // until relaxation to symm=1). Match the bespoke driver's warn +
  // relax.
  if (symm == 2 && Z1 != Z2) {
    printf("Warning - asked for homonuclear symmetry for heteronuclear molecule. Relaxing to symmetry=1.\n");
    symm = 1;
  }
  // symm=2 (per-(m, parity)) also breaks in the presence of an external
  // electric or magnetic field along z -- the field mixes parity or m
  // sectors. Match the bespoke diatomic driver's relaxation to symm=1.
  if (symm == 2 && (Ez != 0.0 || Qzz != 0.0)) {
    printf("Warning - asked for full orbital symmetry in presence of electric field. Relaxing restriction.\n");
    symm = 1;
  }
  if (symm == 2 && Bz != 0.0) {
    printf("Warning - asked for full orbital symmetry in presence of magnetic field. Relaxing restriction.\n");
    symm = 1;
  }

  diatomic::basis::TwoDBasis basis(Z1, Z2, Rhalf, poly, Nquad, bval, lval, mval);
  const size_t Nbf = basis.Nbf();
  printf("Basis set: %i angular shells x %i radial = %i basis functions\n",
          (int) basis.Nang(), (int) basis.Nrad(), (int) Nbf);
  printf("Mode: %s, symmetry=%d, nela=%d nelb=%d\n",
          restricted ? "restricted" : "unrestricted", symm, nela, nelb);

  // Diatomic chemistry-layer methods now return Eigen (helfem::Matrix).
  const helfem::Matrix S    = basis.overlap();
  const helfem::Matrix T    = basis.kinetic();
  // Nuclear attraction. Point nuclei use the exact analytic
  // prolate-spheroidal matrix; a finite nuclear model is evaluated on
  // the two-dimensional quadrature grid (same path the SAP guess uses),
  // one modelpotential::ModelPotential per nucleus. finitenuc indexes
  // modelpotential::nuclear_model_t directly (0 == point), matching the
  // atomic driver's convention.
  helfem::Matrix Vnuc;
  if (finitenuc == 0) {
    Vnuc = basis.nuclear();
  } else {
    modelpotential::ModelPotential *pot1 =
        modelpotential::get_nuclear_model((modelpotential::nuclear_model_t) finitenuc, Z1, Rrms1);
    modelpotential::ModelPotential *pot2 =
        modelpotential::get_nuclear_model((modelpotential::nuclear_model_t) finitenuc, Z2, Rrms2);
    const int lquad = 4 * lmmax.maxCoeff() + 12;
    helfem::diatomic::twodquad::TwoDGrid qgrid(&basis, lquad);
    Vnuc = qgrid.model_potential(pot1, pot2);
    delete pot1;
    delete pot2;
    printf("Using finite nuclear model %d (Rrms1=%g, Rrms2=%g)\n", finitenuc, Rrms1, Rrms2);
  }

  // External static-field one-electron matrices. Diatomic has two
  // nuclei at (0, 0, +/- Rhalf); nuclear dipole/quadrupole moments are
  // non-zero and add a constant scalar Enucfield to the total energy.
  //   nucdip  = (Z2 - Z1) * Rhalf       (points along +z from Z1->Z2)
  //   nucquad = (Z1 + Z2) * Rhalf^2
  //   Enucfield = -Ez * nucdip - Qzz * nucquad / 3
  // Vel/Vmag are the electron-side matrix contributions folded into H0.
  const helfem::Matrix dip  = basis.dipole_z();
  const helfem::Matrix quad = basis.quadrupole_zz();
  const helfem::Matrix Vel  = Ez * dip + Qzz * quad / 3.0;
  const helfem::Matrix Vmag = basis.Bz_field(Bz);
  const double nucdip  = (Z2 - Z1) * Rhalf;
  const double nucquad = (Z1 + Z2) * Rhalf * Rhalf;
  const double Enucfield = -Ez * nucdip - Qzz * nucquad / 3.0;
  const bool have_efield = (Ez != 0.0 || Qzz != 0.0);
  const bool have_bfield = (Bz != 0.0);

  // For diatomic m-averaging: each outer entry groups the (+m, -m) pair
  // of BF-index sets so scf::fock_symmetry_average enforces
  // f(+m) == f(-m). At m == 0 there is no partner so the group has a
  // single member and the "average" is a no-op on that block. Matches
  // src/diatomic/main.cpp lines 318..328.
  std::vector<std::vector<std::vector<Eigen::Index>>> mavg_idx;
  if (maverage) {
    const int mmax_bf = basis.get_mval().cwiseAbs().maxCoeff();
    for (int m = 0; m <= mmax_bf; ++m) {
      std::vector<std::vector<Eigen::Index>> entry;
      entry.push_back(basis.m_indices(m));
      if (m > 0) entry.push_back(basis.m_indices(-m));
      mavg_idx.push_back(entry);
    }
  }

  // Symmetry decomposition.
  std::vector<std::vector<Eigen::Index>> dsym;
  if (symm == 0) {
    std::vector<Eigen::Index> all(Nbf);
    for (size_t i = 0; i < Nbf; ++i) all[i] = static_cast<Eigen::Index>(i);
    dsym.push_back(all);
  } else {
    dsym = basis.get_sym_idx(symm);
  }
  const size_t nsym = dsym.size();

  // Per-block Sinvh_k.
  const std::vector<helfem::Matrix> Sinvh =
      helfem::scf_driver::build_per_block_Sinvh(S, dsym);

  const int lang = ldft_arg > 0 ? ldft_arg : 4 * lval.maxCoeff() + 12;
  const int mang = mdft_arg > 0 ? mdft_arg : 4 * mmax + 12;

  // Pure-m fast path. With --symmetry >= 1 the orbitals cannot mix m, so every
  // orbital is a pure-m function and |e^{i m phi}| = 1 makes the density
  // phi-independent. The phi integral of a matrix element is then analytic,
  // 2 pi delta(m_i, m_j): V_xc is m-block diagonal and each block carries a
  // factor 2 pi. The XC quadrature therefore runs over the (mu, nu) plane
  // only, in REAL arithmetic, with no phi grid at all. The m-mixing case
  // (--symmetry=0) keeps the general 3D complex grid.
  const bool purem_xc = (purem_arg < 0) ? (symm >= 1) : (purem_arg != 0);
  if (purem_xc && symm < 1)
    throw std::logic_error("The pure-m XC path requires --symmetry >= 1: with symmetry=0 the orbitals may mix m and the density is no longer phi-independent.\n");
  auto grid   = helfem::diatomic::dftgrid::DFTGrid(&basis, lang, mang);
  auto pmgrid = helfem::diatomic::dftgrid_purem::PureMDFTGrid(&basis, lang);
  basis.compute_tei(have_exx);

  const double Enucr = Z1 * Z2 / Rbond;

  using OOO_Real = double;
  const size_t nparttype = restricted ? 1 : 2;
  OpenOrbitalOptimizer::IndexVector number_of_blocks_per_particle_type;
  Eigen::Matrix<OOO_Real, Eigen::Dynamic, 1> maximum_occupation;
  Eigen::Matrix<OOO_Real, Eigen::Dynamic, 1> number_of_particles;
  std::vector<std::string> block_descriptions;
  helfem::scf_driver::build_ooo_block_metadata<OOO_Real>(
      nsym, nparttype, restricted, Ntot, nela, nelb,
      number_of_blocks_per_particle_type, maximum_occupation,
      number_of_particles, block_descriptions);

  // Accumulate a per-block density into the full-Nbf density matrix
  // P_full through the block's basis-function scatter index dsym[k].
  auto accumulate_density = [&](helfem::Matrix & P_full, size_t k,
                                 const helfem::Matrix & orb_e,
                                 const helfem::Vector & occ_e) {
    helfem::scf_driver::accumulate_density_block<OOO_Real>(
        P_full, dsym, k, Sinvh, orb_e, occ_e);
  };

  auto orthonormalize_block =
      [&](OpenOrbitalOptimizer::FockMatrix<OOO_Real> & fock, size_t b,
          const helfem::Matrix & F_full, size_t k) {
    helfem::scf_driver::orthonormalize_fock_block<OOO_Real>(
        fock, b, dsym, k, Sinvh, F_full);
  };

  OpenOrbitalOptimizer::FockBuilder<OOO_Real, OOO_Real> fock_builder =
      [&](const OpenOrbitalOptimizer::DensityMatrix<OOO_Real, OOO_Real> & dm) {
    const auto & orbitals    = dm.first;
    const auto & occupations = dm.second;

    // Density assembly. Restricted mode has a single closed-shell channel
    // with max_occ = 2, so P comes straight from the alpha channel; no
    // Pa/Pb split, no *0.5 double-scatter.
    OpenOrbitalOptimizer::FockMatrix<OOO_Real> fock(nsym * nparttype);
    helfem::Matrix P = helfem::Matrix::Zero(Nbf, Nbf);
    helfem::Matrix Pa, Pb;
    double Exc = 0.0;
    double nelnum = 0.0, ekin_grid = 0.0;
    helfem::Matrix XCa, XCb;

    // grid.eval_Fxc takes/returns Eigen at its public boundary; densities
    // and XC potentials are Eigen throughout, so no bridging is needed.
    if (restricted) {
      for (size_t k = 0; k < nsym; ++k)
        accumulate_density(P, k, orbitals[k], occupations[k]);
      if (have_xc) {
        if (purem_xc)
          pmgrid.eval_Fxc(x_func, x_pars, c_func, c_pars, P, XCa,
                           Exc, nelnum, ekin_grid, dftthr);
        else
          grid.eval_Fxc(x_func, x_pars, c_func, c_pars, P, XCa,
                         Exc, nelnum, ekin_grid, dftthr);
      }
      if (have_exx) Pa = 0.5 * P;
    } else {
      Pa = helfem::Matrix::Zero(Nbf, Nbf);
      Pb = helfem::Matrix::Zero(Nbf, Nbf);
      for (size_t k = 0; k < nsym; ++k) {
        accumulate_density(Pa, k, orbitals[k],        occupations[k]);
        accumulate_density(Pb, k, orbitals[nsym + k], occupations[nsym + k]);
      }
      P = Pa + Pb;
      if (have_xc) {
        if (purem_xc)
          pmgrid.eval_Fxc(x_func, x_pars, c_func, c_pars, Pa, Pb,
                           XCa, XCb, Exc, nelnum, ekin_grid, nelb > 0, dftthr);
        else
          grid.eval_Fxc(x_func, x_pars, c_func, c_pars, Pa, Pb,
                         XCa, XCb, Exc, nelnum, ekin_grid, nelb > 0, dftthr);
      }
    }

    const double Ekin = (P * T).trace();
    const double Enuc = (P * Vnuc).trace();
    // External-field trace contributions. Vel/Vmag are zero when their
    // coupling is off; Enucfield is a scalar independent of the density.
    const double Eefield = have_efield ? (P * Vel).trace() + Enucfield : 0.0;
    const double Emfield = have_bfield
        ? (P * Vmag).trace() - 0.5 * Bz * (nela - nelb) : 0.0;

    const helfem::Matrix J = basis.coulomb(P);
    const double Ecoul = 0.5 * (P * J).trace();

    // Diatomic exchange kernel: pure Coulomb K (no RS split yet).
    auto exchange_fn = [&](const helfem::Matrix & P) {
      return helfem::Matrix(kfrac * basis.exchange(P));
    };
    helfem::Matrix Ka, Kb;
    double Exx = 0.0;
    helfem::scf_driver::assemble_hf_exchange(
        Ka, Kb, Exx, Pa, Pb, restricted, have_exx, exchange_fn);

    const double Etot = Ekin + Enuc + Eefield + Emfield
                       + Ecoul + Exc + Exx + Enucr;
    printf("kinetic %.10f nuclear %.10f Enucr %.10f Coulomb %.10f XC %.10f Exx %.10f",
            Ekin, Enuc, Enucr, Ecoul, Exc, Exx);
    if (have_efield) printf(" Eefield %.10f", Eefield);
    if (have_bfield) printf(" Emfield %.10f", Emfield);
    printf("  total %.10f  (nel err %.3e)\n",
            Etot, nelnum - static_cast<double>(Ntot));
    fflush(stdout);

    // Fock assembly. Restricted: one AO Fock matrix per block.
    // Unrestricted: separate alpha/beta AO Fock matrices.
    // Pre-assembled 1-electron core. Vel/Vmag are zero matrices when
    // their coupling is off so this matches T + Vnuc + J exactly in the
    // no-field case.
    const helfem::Matrix H1 = T + Vnuc + Vel + Vmag;
    auto apply_mavg = [&](helfem::Matrix & F) {
      if (maverage)
        F = scf::fock_symmetry_average(F, mavg_idx);
    };
    helfem::scf_driver::assemble_fock_blocks<OOO_Real>(
        fock, H1, J, XCa, XCb, Ka, Kb, S,
        nsym, restricted, have_xc, have_exx, have_bfield, Bz,
        apply_mavg, orthonormalize_block);
    return std::make_pair(Etot, fock);
  };

  // Initial-guess Hamiltonian per block per particle type. Fold in the
  // external-field 1e matrices so the very first Fock reflects the
  // perturbing potential. iguess selects the electron-nuclear channel:
  //   0 core Hamiltonian: bare Vnuc
  //   1..3 model potential (GSZ / SAP / Thomas-Fermi) on each centre
  //     instead of Vnuc. Matches the bespoke-diatomic guess menu; SAP
  //     is the default because it typically converges materially
  //     faster than core-H.
  helfem::Matrix Vguess;
  if (iguess == 0) {
    printf("Guess orbitals from core Hamiltonian\n");
    Vguess = Vnuc;
  } else {
    modelpotential::ModelPotential *p1 = nullptr, *p2 = nullptr;
    switch (iguess) {
    case 1:
      printf("Guess orbitals from GSZ screened nuclei\n");
      p1 = new modelpotential::GSZAtom(Z1);
      p2 = new modelpotential::GSZAtom(Z2);
      break;
    case 2:
      printf("Guess orbitals from SAP screened nuclei\n");
      p1 = new modelpotential::SAPAtom(Z1);
      p2 = new modelpotential::SAPAtom(Z2);
      break;
    case 3:
      printf("Guess orbitals from Thomas-Fermi screened nuclei\n");
      p1 = new modelpotential::TFAtom(Z1);
      p2 = new modelpotential::TFAtom(Z2);
      break;
    default:
      throw std::logic_error("Unsupported iguess value (expected 0..3).\n");
    }
    const int lquad = 4 * lmmax.maxCoeff() + 12;
    helfem::diatomic::twodquad::TwoDGrid qgrid(&basis, lquad);
    Vguess = qgrid.model_potential(p1, p2);
    delete p1;
    delete p2;
  }
  const helfem::Matrix H0 = T + Vguess + Vel + Vmag;
  const OpenOrbitalOptimizer::FockMatrix<OOO_Real> CoreH =
      helfem::scf_driver::build_coreH_from_H0<OOO_Real>(
          H0, S, dsym, Sinvh, nparttype, have_bfield, Bz);

  OpenOrbitalOptimizer::SCFSolver<OOO_Real, OOO_Real> scfsolver(
      number_of_blocks_per_particle_type, maximum_occupation,
      number_of_particles, fock_builder, block_descriptions);

  // --readocc: parse occs.dat and hand OOO a fixed per-block particle
  // count. Bespoke diatomic reads (nocca, noccb, m) rows for both
  // hetero- and homonuclear cases (symm=0/1), and adds a fourth
  // `parity in {+1,-1}` column when symm=2 (homonuclear g/u split).
  //
  // Same bool-vs-int semantic difference as atomic_ooo: OOO's frozen
  // per-block particles cannot be released mid-SCF, so we freeze for
  // the entire run instead of the first N Fock builds only.
  if (readocc) {
    if (symm == 0)
      throw std::logic_error("--readocc requires --symmetry>=1 (need per-block index).");
    const Eigen::MatrixXi occs = helfem::io::read_raw_ascii_imat("occs.dat");
    if (Z1 != Z2 && occs.cols() != 3)
      throw std::logic_error("occs.dat: heteronuclear molecule requires 3 columns (nocca, noccb, m).");
    if (Z1 == Z2 && occs.cols() != 3 && occs.cols() != 4)
      throw std::logic_error("occs.dat: homonuclear molecule requires 3 or 4 columns.");
    if (occs.cols() == 4 && symm != 2)
      throw std::logic_error("occs.dat: 4-column form (with parity) requires --symmetry=2.");

    Eigen::Matrix<OOO_Real, Eigen::Dynamic, 1> fixed_particles =
        Eigen::Matrix<OOO_Real, Eigen::Dynamic, 1>::Zero(nsym * nparttype);
    int total_a = 0, total_b = 0;
    for (Eigen::Index i = 0; i < occs.rows(); ++i) {
      const int nocca_i = static_cast<int>(occs(i, 0));
      const int noccb_i = static_cast<int>(occs(i, 1));
      std::vector<Eigen::Index> row_idx;
      if (occs.cols() == 3) {
        row_idx = basis.m_indices(occs(i, 2));
      } else {
        if (occs(i, 3) != 1 && occs(i, 3) != -1)
          throw std::logic_error("occs.dat: parity column must be +1 or -1.");
        row_idx = basis.m_indices(occs(i, 2), (occs(i, 3) == -1));
      }
      if (row_idx.empty())
        throw std::logic_error("occs.dat: row references a symmetry block with no basis functions.");
      int matched_block = -1;
      for (size_t k = 0; k < nsym; ++k) {
        if (dsym[k] == row_idx) { matched_block = static_cast<int>(k); break; }
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
  // current basis. Same semantics as the atomic --load path.
  if (loadfile.size()) {
    Checkpoint loadchk(loadfile, /*writemode=*/false);
    diatomic::basis::TwoDBasis oldbasis;
    loadchk.read(oldbasis);
    // Densities read straight into Eigen via the checkpoint's helfem::Matrix overloads.
    helfem::Matrix Pa_old, Pb_old;
    loadchk.read("Pa", Pa_old);
    loadchk.read("Pb", Pb_old);
    int nela_old = 0, nelb_old = 0;
    loadchk.read("nela", nela_old);
    loadchk.read("nelb", nelb_old);
    if (nela_old != nela || nelb_old != nelb)
      throw std::logic_error("--load: checkpoint nela/nelb do not match current run.");

    const helfem::Matrix S12         = basis.overlap(oldbasis);
    const helfem::Matrix Sinvh_full  = basis.Sinvh(/*chol*/false, /*sym*/0);
    const helfem::Matrix Pproj       = Sinvh_full * Sinvh_full.transpose() * S12;

    helfem::Matrix Pa_new = Pproj * Pa_old * Pproj.transpose();
    helfem::Matrix Pb_new = Pproj * Pb_old * Pproj.transpose();
    const double na = (Pa_new * S).trace();
    const double nb = (Pb_new * S).trace();
    if (na > 0 && nela > 0) Pa_new *= static_cast<double>(nela) / na;
    if (nb > 0 && nelb > 0) Pb_new *= static_cast<double>(nelb) / nb;

    OpenOrbitalOptimizer::Orbitals<OOO_Real>            loaded_orbs(nsym * nparttype);
    OpenOrbitalOptimizer::OrbitalOccupations<OOO_Real>  loaded_occs(nsym * nparttype);
    const double max_occ_restr = 2.0;
    const double max_occ_open  = 1.0;
    if (restricted) {
      const helfem::Matrix P_total = Pa_new + Pb_new;
      for (size_t k = 0; k < nsym; ++k)
        helfem::scf_driver::fill_block_from_density(
            k, loaded_orbs, loaded_occs, P_total, dsym[k], Sinvh[k], max_occ_restr);
    } else {
      for (size_t k = 0; k < nsym; ++k) {
        helfem::scf_driver::fill_block_from_density(
            k,        loaded_orbs, loaded_occs, Pa_new, dsym[k], Sinvh[k], max_occ_open);
        helfem::scf_driver::fill_block_from_density(
            nsym + k, loaded_orbs, loaded_occs, Pb_new, dsym[k], Sinvh[k], max_occ_open);
      }
    }
    scfsolver.initialize_with_orbitals(loaded_orbs, loaded_occs);
  } else {
    scfsolver.initialize_with_fock(CoreH);
  }

  scfsolver.set("methods", parser.get<std::string>("scfmethods"));
  scfsolver.print_citation();
  scfsolver.print_settings();
  scfsolver.run();

  if (savefile.size()) {
    Checkpoint savechk(savefile, /*writemode=*/true);
    savechk.write(basis);
    const auto final_orbs = scfsolver.get_orbitals();
    const auto final_occs = scfsolver.get_orbital_occupations();
    helfem::Matrix Pa_final, Pb_final;
    std::tie(Pa_final, Pb_final) = helfem::scf_driver::assemble_final_density<OOO_Real>(
        Nbf, restricted, dsym, Sinvh, final_orbs, final_occs);
    // Densities written straight through the checkpoint's helfem::Matrix overloads.
    savechk.write("Pa", Pa_final);
    savechk.write("Pb", Pb_final);
    savechk.write("nela", nela);
    savechk.write("nelb", nelb);
    // Extra parameters needed by density_line / density_grid / completeness.
    savechk.write("Rhalf", Rhalf);
    savechk.write("Z1", Z1);
    savechk.write("Z2", Z2);
    printf("Saved results to %s\n", savefile.c_str());
  }

  return 0;
}
