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

  arma::vec x_pars, c_pars;
  if (xparf.size()) x_pars = helfem::to_arma(scf::parse_xc_params(xparf));
  if (cparf.size()) c_pars = helfem::to_arma(scf::parse_xc_params(cparf));

  int x_func, c_func;
  ::parse_xc_func(x_func, c_func, method);
  ::print_info(x_func, c_func);
  if (!is_supported(x_func) || !is_supported(c_func))
    throw std::logic_error("The specified functional is not supported in HelFEM.\n");
  if (x_func == 0 && c_func == 0)
    throw std::logic_error("HF is not yet implemented in atomic_ooo -- pick a DFT functional.\n");

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
      igrid, zexp, Nelem0, igrid0, zexp0, Z, Zl, Zr, Rhalf, false, 0.0);

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

  // --- Symmetry decomposition. symm==0 collapses to one block containing
  //     all basis functions.
  std::vector<arma::uvec> dsym;
  if (symm == 0) {
    arma::uvec all(Nbf);
    for (size_t i = 0; i < Nbf; ++i) all(i) = i;
    dsym.push_back(all);
  } else {
    dsym = basis.get_sym_idx(symm);
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
  basis.compute_tei(false);

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

  auto scatter_add = [&](arma::mat & P_full, const arma::mat & P_k, const arma::uvec & idx) {
    P_full(idx, idx) += P_k;
  };

  OpenOrbitalOptimizer::FockBuilder<OOO_Real, OOO_Real> fock_builder =
      [&](const OpenOrbitalOptimizer::DensityMatrix<OOO_Real, OOO_Real> & dm) {
    const auto & orbitals    = dm.first;
    const auto & occupations = dm.second;

    // Assemble AO-basis alpha/beta densities from per-block orbitals.
    arma::mat Pa(Nbf, Nbf, arma::fill::zeros);
    arma::mat Pb(Nbf, Nbf, arma::fill::zeros);
    for (size_t t = 0; t < nparttype; ++t) {
      for (size_t k = 0; k < nsym; ++k) {
        if (!dsym[k].n_elem) continue;
        const size_t b = t * nsym + k;
        const arma::mat orb_k  = helfem::to_arma(orbitals[b]);
        const arma::vec occ_k  = helfem::to_arma(occupations[b]);
        const arma::mat C_k    = Sinvh_arma[k] * orb_k;
        const arma::mat P_k    = C_k * arma::diagmat(occ_k) * C_k.t();
        if (restricted) {
          const arma::mat Pha = 0.5 * P_k;
          scatter_add(Pa, Pha, dsym[k]);
          scatter_add(Pb, Pha, dsym[k]);
        } else if (t == 0) {
          scatter_add(Pa, P_k, dsym[k]);
        } else {
          scatter_add(Pb, P_k, dsym[k]);
        }
      }
    }
    const arma::mat P = Pa + Pb;

    const double Ekin = arma::trace(P * T);
    const double Enuc = arma::trace(P * Vnuc);

    const helfem::Matrix P_e = helfem::to_eigen(P);
    const arma::mat J = helfem::to_arma(basis.coulomb(P_e));
    const double Ecoul = 0.5 * arma::trace(P * J);

    double Exc = 0.0;
    double nelnum = 0.0, ekin_grid = 0.0;
    arma::mat XCa, XCb;
    if (restricted) {
      grid.eval_Fxc(x_func, x_pars, c_func, c_pars, P, XCa, Exc, nelnum, ekin_grid, dftthr);
      XCb = XCa;
    } else {
      grid.eval_Fxc(x_func, x_pars, c_func, c_pars, Pa, Pb, XCa, XCb,
                     Exc, nelnum, ekin_grid, nelb > 0, dftthr);
    }

    const double Etot = Ekin + Enuc + Ecoul + Exc;
    printf("kinetic %.10f nuclear %.10f Coulomb %.10f XC %.10f  total %.10f  (nel err %.3e)\n",
            Ekin, Enuc, Ecoul, Exc, Etot, nelnum - static_cast<double>(Ntot));
    fflush(stdout);

    const arma::mat Fa_ao = T + Vnuc + J + XCa;
    const arma::mat Fb_ao = restricted ? Fa_ao : arma::mat(T + Vnuc + J + XCb);

    OpenOrbitalOptimizer::FockMatrix<OOO_Real> fock(nsym * nparttype);
    for (size_t t = 0; t < nparttype; ++t) {
      const arma::mat & F_ao = (t == 0) ? Fa_ao : Fb_ao;
      for (size_t k = 0; k < nsym; ++k) {
        if (!dsym[k].n_elem) {
          fock[t * nsym + k] = helfem::Matrix::Zero(0, 0);
          continue;
        }
        const arma::mat Fk_sub = F_ao(dsym[k], dsym[k]);
        const arma::mat F_orth = Sinvh_arma[k].t() * Fk_sub * Sinvh_arma[k];
        fock[t * nsym + k] = helfem::to_eigen(F_orth);
      }
    }
    return std::make_pair(Etot, fock);
  };

  // Core-Hamiltonian guess per block per particle type.
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
