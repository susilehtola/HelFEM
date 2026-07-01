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

// Diatomic driver using OpenOrbitalOptimizer for SCF convergence.
// Completes the OOO trio (sadatom_ooo / atomic_ooo / diatomic_ooo);
// same restricted DFT scope as the other two.
//
// The diatomic Fock chain is still arma-native at the chemistry layer
// (basis.overlap / basis.coulomb / grid.eval_Fxc all return arma). The
// callback bridges to Eigen only at the OOO boundary.

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
  parser.add<std::string>("x_pars", 0, "file for parameters for exchange functional", false, "");
  parser.add<std::string>("c_pars", 0, "file for parameters for correlation functional", false, "");
  parser.parse_check(argc, argv);

  const int Z1        = get_Z(parser.get<std::string>("Z1"));
  const int Z2        = get_Z(parser.get<std::string>("Z2"));
  const double Rbond  = parser.get<double>("Rbond");
  const int Q         = parser.get<int>("Q");
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
  const std::string xparf = parser.get<std::string>("x_pars");
  const std::string cparf = parser.get<std::string>("c_pars");

  arma::vec x_pars, c_pars;
  if (xparf.size()) x_pars = helfem::to_arma(scf::parse_xc_params(xparf));
  if (cparf.size()) c_pars = helfem::to_arma(scf::parse_xc_params(cparf));

  int x_func, c_func;
  ::parse_xc_func(x_func, c_func, method);
  ::print_info(x_func, c_func);
  if (!is_supported(x_func) || !is_supported(c_func))
    throw std::logic_error("The specified functional is not supported in HelFEM.\n");
  if (x_func == 0 && c_func == 0)
    throw std::logic_error("HF is not yet implemented in diatomic_ooo -- pick a DFT functional.\n");

  auto poly = std::shared_ptr<const polynomial_basis::PolynomialBasis>(
      polynomial_basis::get_basis(primbas, Nnodes));
  if (Nquad == 0) Nquad = 5 * poly->get_nbf();
  else if (Nquad < 2 * poly->get_nbf())
    throw std::logic_error("Insufficient radial quadrature.\n");

  // Parse lmax: single int (if mmax>=0, replicated mmax+1 times) or
  // comma-separated per-m list.
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

  diatomic::basis::TwoDBasis basis(Z1, Z2, Rhalf, poly, Nquad, bval, lval, mval);
  printf("Basis set: %i angular shells x %i radial = %i basis functions\n",
          (int) basis.Nang(), (int) basis.Nrad(), (int) basis.Nbf());

  // Diatomic chemistry-layer methods are still arma.
  const arma::mat S    = basis.overlap();
  const arma::mat T    = basis.kinetic();
  const arma::mat Vnuc = basis.nuclear();
  const arma::mat Sinvh_arma = basis.Sinvh(/*chol*/false, /*sym*/0);
  const helfem::Matrix Sinvh = helfem::to_eigen(Sinvh_arma);

  const int lang = ldft_arg > 0 ? ldft_arg : 4 * arma::max(lval) + 12;
  const int mang = mdft_arg > 0 ? mdft_arg : 4 * mmax + 12;
  auto grid = helfem::diatomic::dftgrid::DFTGrid(&basis, lang, mang);

  basis.compute_tei(false);

  const double Enucr = Z1 * Z2 / Rbond;

  using OOO_Real = double;
  OpenOrbitalOptimizer::IndexVector number_of_blocks_per_particle_type(1);
  number_of_blocks_per_particle_type(0) = 1;
  Eigen::Matrix<OOO_Real, Eigen::Dynamic, 1> maximum_occupation(1);
  maximum_occupation(0) = 2.0;
  Eigen::Matrix<OOO_Real, Eigen::Dynamic, 1> number_of_particles(1);
  number_of_particles(0) = static_cast<OOO_Real>(Z1 + Z2 - Q);
  std::vector<std::string> block_descriptions = {"all"};

  OpenOrbitalOptimizer::FockBuilder<OOO_Real, OOO_Real> fock_builder =
      [&](const OpenOrbitalOptimizer::DensityMatrix<OOO_Real, OOO_Real> & dm) {
    const auto & orbitals    = dm.first;
    const auto & occupations = dm.second;

    const arma::mat orbitals_arma = helfem::to_arma(orbitals[0]);
    const arma::vec occ_arma = helfem::to_arma(occupations[0]);
    const arma::mat C = Sinvh_arma * orbitals_arma;
    const arma::mat P = C * arma::diagmat(occ_arma) * C.t();

    const double Ekin = arma::trace(P * T);
    const double Enuc = arma::trace(P * Vnuc);

    const arma::mat J = basis.coulomb(P);
    const double Ecoul = 0.5 * arma::trace(P * J);

    double Exc = 0.0;
    double nelnum = 0.0, ekin_grid = 0.0;
    arma::mat XCa;
    grid.eval_Fxc(x_func, x_pars, c_func, c_pars, P,
                   XCa, Exc, nelnum, ekin_grid, dftthr);

    const double Etot = Ekin + Enuc + Ecoul + Exc + Enucr;
    printf("kinetic   % .10f  nuclear   % .10f  Enucr % .10f\n", Ekin, Enuc, Enucr);
    printf("Coulomb   % .10f  XC        % .10f\n", Ecoul, Exc);
    printf("total     % .10f  (nel int err % .3e)\n",
            Etot, nelnum - static_cast<double>(Z1 + Z2 - Q));
    fflush(stdout);

    const arma::mat F_ao   = T + Vnuc + J + XCa;
    const arma::mat F_orth = Sinvh_arma.t() * F_ao * Sinvh_arma;

    OpenOrbitalOptimizer::FockMatrix<OOO_Real> fock(1);
    fock[0] = helfem::to_eigen(F_orth);
    return std::make_pair(Etot, fock);
  };

  const arma::mat CoreH_arma = Sinvh_arma.t() * (T + Vnuc) * Sinvh_arma;
  OpenOrbitalOptimizer::FockMatrix<OOO_Real> CoreH(1);
  CoreH[0] = helfem::to_eigen(CoreH_arma);

  OpenOrbitalOptimizer::SCFSolver<OOO_Real, OOO_Real> scfsolver(
      number_of_blocks_per_particle_type, maximum_occupation,
      number_of_particles, fock_builder, block_descriptions);
  scfsolver.initialize_with_fock(CoreH);
  scfsolver.run();

  return 0;
}
