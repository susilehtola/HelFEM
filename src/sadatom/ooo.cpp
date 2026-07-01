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

// Sadatom driver using OpenOrbitalOptimizer. Supports:
//   - Restricted (closed-shell) DFT: one particle type, per-l blocks,
//     max_occupation = 2*(2l+1) per block.
//   - Unrestricted DFT: two particle types (alpha, beta) each split
//     into the same per-l blocks; max_occupation = 2l+1 per orbital.
//
// nela/nelb are derived from --Q and --M (spin multiplicity, 2S+1) via
// scf::parse_nela_nelb, matching the CLI convention of atomic_ooo and
// diatomic_ooo.
//
// HF (x_func == 0 && c_func == 0) is deferred; picking any HF method throws.

#include "../general/cmdline.h"
#include "../general/constants.h"
#include "../general/dftfuncs.h"
#include "../general/elements.h"
#include "../general/scf_helpers.h"

#include "openorbitaloptimizer/scfsolver.hpp"

#include "utils.h"
#include "dftgrid.h"
#include "solver.h"
#include "configurations.h"
#include "../atomic/basis.h"
#include <ArmaEigen.h>
#include <Eigen/Eigenvalues>
#include <cfloat>

using namespace helfem;

int main(int argc, char **argv) {
  cmdline::parser parser;

  parser.add<int>("grid", 0, "type of grid: 1 for linear, 2 for quadratic, 3 for polynomial, 4 for exponential", false, 4);
  parser.add<double>("zexp", 0, "parameter in radial grid", false, 2.0);
  parser.add<double>("Rmax", 0, "practical infinity", false, 40.0);
  parser.add<int>("lmax", 0, "maximum angular momentum", true);
  parser.add<int>("nelem", 0, "number of elements", true);
  parser.add<std::string>("Z", 0, "nuclear charge", true);
  parser.add<int>("Q", 0, "charge of system", false, 0);
  parser.add<int>("M", 0, "spin multiplicity (2S+1); mutually exclusive with nela/nelb", false, 0);
  parser.add<int>("nela", 0, "number of alpha electrons (leave 0 to derive from Q/M)", false, 0);
  parser.add<int>("nelb", 0, "number of beta electrons (leave 0 to derive from Q/M)", false, 0);
  parser.add<int>("restricted", 0, "spin-restricted: 1 restricted, 0 unrestricted, -1 auto from nela/nelb", false, -1);
  parser.add<int>("nnodes", 0, "number of nodes per element", false, 15);
  parser.add<int>("nquad", 0, "number of quadrature points", false, 0);
  parser.add<std::string>("method", 0, "method to use", false, "lda_x");
  parser.add<double>("dftthr", 0, "density threshold for dft", false, 1e-12);
  parser.add<int>("primbas", 0, "primitive radial basis", false, 4);
  parser.add<std::string>("x_pars", 0, "file for parameters for exchange functional", false, "");
  parser.add<std::string>("c_pars", 0, "file for parameters for correlation functional", false, "");
  parser.parse_check(argc, argv);

  const int igrid   = parser.get<int>("grid");
  const double zexp = parser.get<double>("zexp");
  const int Nelem   = parser.get<int>("nelem");
  const int Z       = get_Z(parser.get<std::string>("Z"));
        int Q       = parser.get<int>("Q");
        int M       = parser.get<int>("M");
        int nela    = parser.get<int>("nela");
        int nelb    = parser.get<int>("nelb");
        int restr   = parser.get<int>("restricted");
  const int Nnodes  = parser.get<int>("nnodes");
        int Nquad   = parser.get<int>("nquad");
  const std::string method = parser.get<std::string>("method");
  const double dftthr = parser.get<double>("dftthr");
  const int primbas   = parser.get<int>("primbas");
  const std::string xparf = parser.get<std::string>("x_pars");
  const std::string cparf = parser.get<std::string>("c_pars");
  const int lmax      = parser.get<int>("lmax");
  const double Rmax   = parser.get<double>("Rmax");

  // Derive nela/nelb from Q, M via scf::parse_nela_nelb -- same CLI as
  // atomic_ooo / diatomic_ooo (see PR #148, #147).
  scf::parse_nela_nelb(nela, nelb, Q, M, Z);
  if (restr == -1) restr = (nela == nelb) ? 1 : 0;
  const bool restricted = (restr != 0);
  if (restricted && nela != nelb)
    throw std::logic_error("Restricted mode requires nela == nelb (closed shell). "
                            "Use --restricted=0 (or leave -1 for auto) for open-shell.");
  const int Ntot = nela + nelb;

  arma::vec x_pars, c_pars;
  if (xparf.size()) {
    x_pars = helfem::to_arma(scf::parse_xc_params(xparf));
    x_pars.t().print("Exchange functional parameters");
  }
  if (cparf.size()) {
    c_pars = helfem::to_arma(scf::parse_xc_params(cparf));
    c_pars.t().print("Correlation functional parameters");
  }

  printf("Running %s %s calculation with Rmax=%e and %i elements.\n",
          restricted ? "restricted" : "unrestricted",
          method.c_str(), Rmax, Nelem);
  printf("nela=%d nelb=%d\n", nela, nelb);

  auto poly = std::shared_ptr<const polynomial_basis::PolynomialBasis>(
      polynomial_basis::get_basis(primbas, Nnodes));

  if (Nquad == 0)
    Nquad = 5 * poly->get_nbf();
  else if (Nquad < 2 * poly->get_nbf())
    throw std::logic_error("Insufficient radial quadrature.\n");
  printf("Using %i point quadrature rule.\n", Nquad);

  int x_func, c_func;
  ::parse_xc_func(x_func, c_func, method);
  ::print_info(x_func, c_func);
  if (!is_supported(x_func))
    throw std::logic_error("The specified exchange functional is not currently supported in HelFEM.\n");
  if (!is_supported(c_func))
    throw std::logic_error("The specified correlation functional is not currently supported in HelFEM.\n");
  double kfrac, kshort, omega;
  range_separation(x_func, omega, kfrac, kshort);
  const bool have_exx = (kfrac != 0.0 || kshort != 0.0);
  const bool have_xc  = (x_func != 0 || c_func != 0);
  if (have_exx && kshort != 0.0)
    throw std::logic_error("Range-separated hybrids not yet supported in sadatom_ooo (needs omega wiring).\n");

  arma::vec bval = atomic::basis::form_grid(
      modelpotential::POINT_NUCLEUS, 0.0, Nelem, Rmax, igrid, zexp,
      0, 0, 0.0, Z, 0, 0, 0.0, false, 0.0);

  const bool zeroder = false;
  auto basis = sadatom::basis::TwoDBasis(Z, modelpotential::POINT_NUCLEUS, 0.0,
                                          poly, zeroder, Nquad, bval, lmax);
  printf("Basis set has %i radial functions\n", (int) basis.Nbf());

  const helfem::Matrix S    = basis.overlap();
  const helfem::Matrix Sinvh = helfem::to_eigen(basis.Sinvh());
  const helfem::Matrix T    = basis.kinetic();
  const helfem::Matrix Tl   = basis.kinetic_l();
  const helfem::Matrix Vnuc = basis.nuclear();

  auto grid = helfem::sadatom::dftgrid::DFTGrid(&basis);
  basis.compute_tei();  // sadatom compute_tei takes no flag; exchange path is always available.

  // OOO block layout: per-l blocks.
  using OOO_Real = double;
  const size_t nblock = static_cast<size_t>(lmax + 1);
  const size_t nparttype = restricted ? 1 : 2;
  OpenOrbitalOptimizer::IndexVector number_of_blocks_per_particle_type(nparttype);
  Eigen::Matrix<OOO_Real, Eigen::Dynamic, 1> maximum_occupation(nblock * nparttype);
  Eigen::Matrix<OOO_Real, Eigen::Dynamic, 1> number_of_particles(nparttype);
  std::vector<std::string> block_descriptions(nblock * nparttype);

  for (size_t t = 0; t < nparttype; ++t) {
    number_of_blocks_per_particle_type(t) = static_cast<int>(nblock);
    number_of_particles(t) = static_cast<OOO_Real>(restricted ? Ntot : (t == 0 ? nela : nelb));
    for (size_t l = 0; l < nblock; ++l) {
      // Per-l shell capacity: restricted = 2*(2l+1), unrestricted alpha or beta = (2l+1).
      maximum_occupation(t * nblock + l) = restricted ? 2 * (2 * l + 1) : (2 * l + 1);
      std::ostringstream oss;
      if (nparttype == 2) oss << (t == 0 ? "a:" : "b:");
      oss << "l=" << l;
      block_descriptions[t * nblock + l] = oss.str();
    }
  }
  std::cout << "Max occ: " << maximum_occupation.transpose() << "\n";

  const Eigen::Index Nrad = Sinvh.rows();
  const double angfac = 4.0 * M_PI;

  // Accumulate per-l density into the full radial density Prad and into
  // the per-l cube Pl_cube for grid.eval_Fxc.
  auto accumulate_density = [&](helfem::Matrix & Prad, arma::cube & Pl_cube,
                                 size_t l, const helfem::Matrix & orb,
                                 const helfem::Vector & occ, double & Ekin_out) {
    if (occ.cwiseAbs().maxCoeff() == 0.0) return;
    const helfem::Matrix C = Sinvh * orb;
    const helfem::Matrix P_l = C * occ.asDiagonal() * C.transpose();
    Prad += P_l;
    Pl_cube.slice(l) = helfem::to_arma(P_l);
    Ekin_out += (P_l * T).trace();
    if (l > 0)
      Ekin_out += l * (l + 1) * (P_l * Tl).trace();
  };

  OpenOrbitalOptimizer::FockBuilder<OOO_Real, OOO_Real> fock_builder =
      [&](const OpenOrbitalOptimizer::DensityMatrix<OOO_Real, OOO_Real> & dm) {
    const auto & orbitals    = dm.first;
    const auto & occupations = dm.second;

    OpenOrbitalOptimizer::FockMatrix<OOO_Real> fock(nblock * nparttype);
    helfem::Matrix Prad = helfem::Matrix::Zero(Nrad, Nrad);
    double Ekin = 0.0;
    double Exc = 0.0;
    double nelnum = 0.0;
    arma::cube XCa, XCb;
    // Per-l density cubes, retained across the XC/exchange branches.
    // For unrestricted: Pal = alpha, Pbl = beta. For restricted: Pal
    // is the total density (there is no separate beta).
    arma::cube Pal(Nrad, Nrad, nblock, arma::fill::zeros);
    arma::cube Pbl;

    if (restricted) {
      for (size_t l = 0; l < nblock; ++l)
        accumulate_density(Prad, Pal, l, orbitals[l], occupations[l], Ekin);
      if (have_xc) {
        grid.eval_Fxc(x_func, x_pars, c_func, c_pars, Pal / angfac,
                       XCa, Exc, nelnum, dftthr);
        XCa /= angfac;
      }
    } else {
      Pbl.zeros(Nrad, Nrad, nblock);
      helfem::Matrix Prad_a = helfem::Matrix::Zero(Nrad, Nrad);
      helfem::Matrix Prad_b = helfem::Matrix::Zero(Nrad, Nrad);
      for (size_t l = 0; l < nblock; ++l) {
        accumulate_density(Prad_a, Pal, l, orbitals[l],           occupations[l],           Ekin);
        accumulate_density(Prad_b, Pbl, l, orbitals[nblock + l],  occupations[nblock + l],  Ekin);
      }
      Prad = Prad_a + Prad_b;
      if (have_xc) {
        grid.eval_Fxc(x_func, x_pars, c_func, c_pars, Pal / angfac, Pbl / angfac,
                       XCa, XCb, Exc, nelnum, nelb > 0, dftthr);
        XCa /= angfac;
        if (nelb > 0) XCb /= angfac;
      }
    }

    const double Enuc = (Prad * Vnuc).trace();
    const helfem::Matrix J = basis.coulomb(Prad / angfac);
    const double Ecoul = 0.5 * (Prad * J).trace();

    // HF exchange. basis.exchange takes an angular-normalized density
    // cube (Pl / ShellCapacity) and returns a cube K such that Fock_l
    // gets K.slice(l) and Exx = 0.5 * sum_l trace(K[l] * Pl[l]) with
    // Pl the UNnormalized density -- matches bespoke solver.cpp:940.
    // ShellCapacity(l) = 2*(2l+1) for restricted, (2l+1) per spin for UKS.
    arma::cube Ka, Kb;
    double Exx = 0.0;
    if (have_exx) {
      arma::cube ang_a = Pal;
      for (size_t l = 0; l < nblock; ++l)
        ang_a.slice(l) /= restricted ? 2.0 * (2 * l + 1) : (2 * l + 1);
      Ka = kfrac * basis.exchange(ang_a);
      for (size_t l = 0; l < nblock; ++l)
        Exx += 0.5 * arma::trace(Ka.slice(l) * Pal.slice(l));
      if (!restricted) {
        arma::cube ang_b = Pbl;
        for (size_t l = 0; l < nblock; ++l)
          ang_b.slice(l) /= (2 * l + 1);
        Kb = kfrac * basis.exchange(ang_b);
        for (size_t l = 0; l < nblock; ++l)
          Exx += 0.5 * arma::trace(Kb.slice(l) * Pbl.slice(l));
      }
      // No * 2 in restricted -- Pal here is the TOTAL per-l density
      // (occupations up to 2*(2l+1)), unlike atomic/diatomic where Pa
      // is the SPIN density. Matches bespoke solver.cpp:940 exactly.
    }

    const double Etot = Ekin + Enuc + Ecoul + Exc + Exx;

    printf("Ekin %.10f  Enuc %.10f  Ecoul %.10f  Exc %.10f  Exx %.10f  Etot %.10f\n",
            Ekin, Enuc, Ecoul, Exc, Exx, Etot);
    fflush(stdout);

    // Per-l Fock: F_l = T + Vnuc + J + l(l+1)*Tl + XC_l + K_l.
    auto build_fock_block = [&](size_t l, const arma::cube & XC_cube,
                                 bool add_xc, const arma::cube & K_cube,
                                 bool add_k) -> helfem::Matrix {
      helfem::Matrix Fl = T + Vnuc + J;
      if (l > 0) Fl += l * (l + 1) * Tl;
      if (add_xc)
        Fl += helfem::to_eigen(arma::mat(XC_cube.slice(l)));
      if (add_k)
        Fl += helfem::to_eigen(arma::mat(K_cube.slice(l)));
      return Sinvh.transpose() * Fl * Sinvh;
    };

    if (restricted) {
      for (size_t l = 0; l < nblock; ++l)
        fock[l] = build_fock_block(l, XCa, have_xc, Ka, have_exx);
    } else {
      for (size_t l = 0; l < nblock; ++l) {
        fock[l]          = build_fock_block(l, XCa, have_xc, Ka, have_exx);
        fock[nblock + l] = build_fock_block(l, XCb, have_xc && nelb > 0,
                                             Kb, have_exx && nelb > 0);
      }
    }
    return std::make_pair(Etot, fock);
  };

  // Core-Hamiltonian guess per l block per particle type.
  OpenOrbitalOptimizer::FockMatrix<OOO_Real> CoreH(nblock * nparttype);
  for (size_t t = 0; t < nparttype; ++t) {
    for (size_t l = 0; l < nblock; ++l) {
      helfem::Matrix Hl = T + Vnuc;
      if (l > 0) Hl += l * (l + 1) * Tl;
      CoreH[t * nblock + l] = Sinvh.transpose() * Hl * Sinvh;
    }
  }

  OpenOrbitalOptimizer::SCFSolver<OOO_Real, OOO_Real> scfsolver(
      number_of_blocks_per_particle_type, maximum_occupation,
      number_of_particles, fock_builder, block_descriptions);
  scfsolver.initialize_with_fock(CoreH);
  scfsolver.run();

  return 0;
}
