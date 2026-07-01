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
#include "../general/timer.h"

#include "openorbitaloptimizer/scfsolver.hpp"

#include "utils.h"
#include "basis.h"
#include "dftgrid.h"
#include "twodquadrature.h"
#include "../atomic/basis.h"
#include <ArmaEigen.h>
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
  parser.add<int>("nquad", 0, "number of quadrature points", false, 0);
  parser.add<std::string>("method", 0, "DFT method to use", false, "lda_x");
  parser.add<int>("ldft", 0, "theta rule for dft quadrature (0 for auto)", false, 0);
  parser.add<int>("mdft", 0, "phi rule for dft quadrature (0 for auto)", false, 0);
  parser.add<double>("dftthr", 0, "density threshold for dft", false, 1e-12);
  parser.add<int>("primbas", 0, "primitive radial basis", false, 4);
  parser.add<int>("restricted", 0, "spin-restricted: 1 restricted, 0 unrestricted, -1 auto from nela/nelb", false, -1);
  parser.add<int>("symmetry", 0, "orbital symmetry: 0 none, 1 per-m, 2 per-(m,parity) (homonuclear only)", false, 1);
  parser.add<std::string>("x_pars", 0, "file for parameters for exchange functional", false, "");
  parser.add<std::string>("c_pars", 0, "file for parameters for correlation functional", false, "");
  parser.parse_check(argc, argv);

  const int Z1        = get_Z(parser.get<std::string>("Z1"));
  const int Z2        = get_Z(parser.get<std::string>("Z2"));
  const double Rbond  = parser.get<double>("Rbond");
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
  const double dftthr = parser.get<double>("dftthr");
  const int primbas   = parser.get<int>("primbas");
        int restr     = parser.get<int>("restricted");
        int symm      = parser.get<int>("symmetry");
  const std::string xparf = parser.get<std::string>("x_pars");
  const std::string cparf = parser.get<std::string>("c_pars");

  // Derive nela/nelb from Q, M -- same convention as the bespoke diatomic
  // driver. Total nuclear charge is Z1 + Z2.
  scf::parse_nela_nelb(nela, nelb, Q, M, Z1 + Z2);
  if (restr == -1) restr = (nela == nelb) ? 1 : 0;
  const bool restricted = (restr != 0);
  if (restricted && nela != nelb)
    throw std::logic_error("Restricted mode requires nela == nelb (closed shell). "
                            "Use --restricted=0 (or leave -1 for auto) for open-shell.");
  const int Ntot = nela + nelb;

  arma::vec x_pars, c_pars;
  if (xparf.size()) x_pars = helfem::to_arma(scf::parse_xc_params(xparf));
  if (cparf.size()) c_pars = helfem::to_arma(scf::parse_xc_params(cparf));

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

  arma::ivec lmmax;
  if (mmax >= 0) {
    lmmax.ones(mmax + 1);
    lmmax *= std::atoi(lmax_str.c_str());
  } else {
    std::vector<arma::uword> lmmaxv;
    std::stringstream ss(lmax_str);
    std::string tok;
    while (std::getline(ss, tok, ','))
      lmmaxv.push_back(std::atoi(tok.c_str()));
    lmmax = arma::conv_to<arma::ivec>::from(lmmaxv);
    mmax = static_cast<int>(lmmaxv.size()) - 1;
  }

  arma::ivec lval, mval;
  diatomic::basis::lm_to_l_m(lmmax, lval, mval);

  const double Rhalf = 0.5 * Rbond;
  const double mumax = utils::arcosh(Rmax / Rhalf);
  arma::vec bval(atomic::basis::normal_grid(Nelem, mumax, igrid, zexp));

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

  diatomic::basis::TwoDBasis basis(Z1, Z2, Rhalf, poly, Nquad, bval, lval, mval);
  const size_t Nbf = basis.Nbf();
  printf("Basis set: %i angular shells x %i radial = %i basis functions\n",
          (int) basis.Nang(), (int) basis.Nrad(), (int) Nbf);
  printf("Mode: %s, symmetry=%d, nela=%d nelb=%d\n",
          restricted ? "restricted" : "unrestricted", symm, nela, nelb);

  // Diatomic chemistry-layer methods are still arma-native.
  const arma::mat S    = basis.overlap();
  const arma::mat T    = basis.kinetic();
  const arma::mat Vnuc = basis.nuclear();

  // Symmetry decomposition.
  std::vector<arma::uvec> dsym;
  if (symm == 0) {
    arma::uvec all(Nbf);
    for (size_t i = 0; i < Nbf; ++i) all(i) = i;
    dsym.push_back(all);
  } else {
    dsym = basis.get_sym_idx(symm);
  }
  const size_t nsym = dsym.size();

  // Per-block Sinvh_k.
  std::vector<arma::mat> Sinvh_arma(nsym);
  for (size_t k = 0; k < nsym; ++k) {
    if (!dsym[k].n_elem) continue;
    const arma::mat Sk = S(dsym[k], dsym[k]);
    Sinvh_arma[k] = helfem::to_arma(scf::form_Sinvh(helfem::to_eigen(Sk), /*chol*/false));
  }

  const int lang = ldft_arg > 0 ? ldft_arg : 4 * arma::max(lval) + 12;
  const int mang = mdft_arg > 0 ? mdft_arg : 4 * mmax + 12;
  auto grid = helfem::diatomic::dftgrid::DFTGrid(&basis, lang, mang);
  basis.compute_tei(have_exx);

  const double Enucr = Z1 * Z2 / Rbond;

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

  // Accumulate a per-block density into the full-Nbf density matrix
  // P_full through the block's basis-function scatter index dsym[k].
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
  // orthonormal basis via Sinvh_k^T . F_k . Sinvh_k, stash into fock[b].
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

    // Density assembly. Restricted mode has a single closed-shell channel
    // with max_occ = 2, so P comes straight from the alpha channel; no
    // Pa/Pb split, no *0.5 double-scatter.
    OpenOrbitalOptimizer::FockMatrix<OOO_Real> fock(nsym * nparttype);
    arma::mat P(Nbf, Nbf, arma::fill::zeros);
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

    const arma::mat J = basis.coulomb(P);
    const double Ecoul = 0.5 * arma::trace(P * J);

    // HF exchange. See sign-convention note in atomic/ooo.cpp: basis.exchange
    // returns a matrix that gets ADDED to the Fock and the energy
    // contribution is +0.5*trace(Pspin*Kspin) per channel.
    arma::mat Ka, Kb;
    double Exx = 0.0;
    if (have_exx) {
      Ka = kfrac * basis.exchange(Pa);
      Exx = 0.5 * arma::trace(Pa * Ka);
      if (!restricted) {
        Kb = kfrac * basis.exchange(Pb);
        Exx += 0.5 * arma::trace(Pb * Kb);
      } else {
        Exx *= 2.0;
      }
    }

    const double Etot = Ekin + Enuc + Ecoul + Exc + Exx + Enucr;
    printf("kinetic %.10f nuclear %.10f Enucr %.10f Coulomb %.10f XC %.10f Exx %.10f  total %.10f  (nel err %.3e)\n",
            Ekin, Enuc, Enucr, Ecoul, Exc, Exx, Etot, nelnum - static_cast<double>(Ntot));
    fflush(stdout);

    // Fock assembly. Restricted: one AO Fock matrix per block.
    // Unrestricted: separate alpha/beta AO Fock matrices.
    if (restricted) {
      arma::mat F_ao = T + Vnuc + J;
      if (have_xc)  F_ao += XCa;
      if (have_exx) F_ao += Ka;
      for (size_t k = 0; k < nsym; ++k)
        orthonormalize_block(fock, k, F_ao, k);
    } else {
      arma::mat Fa_ao = T + Vnuc + J;
      arma::mat Fb_ao = T + Vnuc + J;
      if (have_xc)  { Fa_ao += XCa; Fb_ao += XCb; }
      if (have_exx) { Fa_ao += Ka;  Fb_ao += Kb;  }
      for (size_t k = 0; k < nsym; ++k) {
        orthonormalize_block(fock, k,        Fa_ao, k);
        orthonormalize_block(fock, nsym + k, Fb_ao, k);
      }
    }
    return std::make_pair(Etot, fock);
  };

  // Core-H guess per block per particle type.
  const arma::mat H0 = T + Vnuc;
  OpenOrbitalOptimizer::FockMatrix<OOO_Real> CoreH(nsym * nparttype);
  for (size_t t = 0; t < nparttype; ++t) {
    for (size_t k = 0; k < nsym; ++k) {
      if (!dsym[k].n_elem) {
        CoreH[t * nsym + k] = helfem::Matrix::Zero(0, 0);
        continue;
      }
      const arma::mat H_sub = H0(dsym[k], dsym[k]);
      const arma::mat H_orth = Sinvh_arma[k].t() * H_sub * Sinvh_arma[k];
      CoreH[t * nsym + k] = helfem::to_eigen(H_orth);
    }
  }

  OpenOrbitalOptimizer::SCFSolver<OOO_Real, OOO_Real> scfsolver(
      number_of_blocks_per_particle_type, maximum_occupation,
      number_of_particles, fock_builder, block_descriptions);
  scfsolver.initialize_with_fock(CoreH);
  scfsolver.run();

  return 0;
}
