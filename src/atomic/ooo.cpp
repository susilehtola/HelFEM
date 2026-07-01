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

// Atomic (2D angular+radial) driver using OpenOrbitalOptimizer for
// SCF convergence. Restricted DFT only in this first cut -- same
// scope as sadatom_ooo, extended to the full angular basis.
//
// HF (x_func == 0 && c_func == 0) is not supported: the Fock builder
// does not add HF exchange yet. Adding basis.exchange() at the OOO
// callback boundary is the natural follow-up.

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
  parser.add<std::string>("x_pars", 0, "file for parameters for exchange functional", false, "");
  parser.add<std::string>("c_pars", 0, "file for parameters for correlation functional", false, "");
  parser.add<bool>("zeroder", 0, "zero derivative at Rmax?", false, false);
  parser.parse_check(argc, argv);

  const int    Z          = get_Z(parser.get<std::string>("Z"));
  const int    Q          = parser.get<int>("Q");
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
  const std::string xparf = parser.get<std::string>("x_pars");
  const std::string cparf = parser.get<std::string>("c_pars");
  const bool   zeroder    = parser.get<bool>("zeroder");

  // Parse xc functional parameters; grid.eval_Fxc takes arma::vec.
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

  // Off-center nuclei: this driver is single-centre only.
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
  printf("Basis set: %i angular shells x %i radial = %i basis functions\n",
          (int) basis.Nang(), (int) basis.Nrad(), (int) basis.Nbf());

  // Phase 5: overlap / kinetic / nuclear return Eigen; Sinvh still arma.
  const helfem::Matrix S     = basis.overlap();
  const helfem::Matrix Sinvh = basis.Sinvh(false, 0);
  const helfem::Matrix T     = basis.kinetic();
  const helfem::Matrix Vnuc  = basis.nuclear();

  // DFT integration grid: auto-choose ldft / mdft if requested.
  const int ldft = ldft_arg > 0 ? ldft_arg : 4 * lmax + 12;
  const int mdft = mdft_arg > 0 ? mdft_arg : 4 * mmax + 12;
  auto grid = helfem::atomic::dftgrid::DFTGrid(&basis, ldft, mdft);

  basis.compute_tei(false);

  // OOO block layout: single block, restricted DFT, max 2 electrons per orbital.
  using OOO_Real = double;
  OpenOrbitalOptimizer::IndexVector number_of_blocks_per_particle_type(1);
  number_of_blocks_per_particle_type(0) = 1;
  Eigen::Matrix<OOO_Real, Eigen::Dynamic, 1> maximum_occupation(1);
  maximum_occupation(0) = 2.0;
  Eigen::Matrix<OOO_Real, Eigen::Dynamic, 1> number_of_particles(1);
  number_of_particles(0) = static_cast<OOO_Real>(Z - Q);
  std::vector<std::string> block_descriptions = {"all"};

  // Fock builder: DFT contribution via grid.eval_Fxc (arma::mat P /
  // XCa); Coulomb via basis.coulomb which is Eigen-native.
  OpenOrbitalOptimizer::FockBuilder<OOO_Real, OOO_Real> fock_builder =
      [&](const OpenOrbitalOptimizer::DensityMatrix<OOO_Real, OOO_Real> & dm) {
    const auto & orbitals    = dm.first;
    const auto & occupations = dm.second;

    // Rebuild the AO-basis coefficient block and density.
    const helfem::Matrix C = Sinvh * orbitals[0];
    const helfem::Matrix P = C * occupations[0].asDiagonal() * C.transpose();

    const double Ekin = (P * T).trace();
    const double Enuc = (P * Vnuc).trace();

    // Coulomb (Eigen throughout).
    const helfem::Matrix J = basis.coulomb(P);
    const double Ecoul = 0.5 * (P * J).trace();

    // XC: bridge P to arma for grid.eval_Fxc.
    double Exc = 0.0;
    double nelnum = 0.0, ekin_grid = 0.0;
    arma::mat XCa_arma;
    const arma::mat P_arma = helfem::to_arma(P);
    grid.eval_Fxc(x_func, x_pars, c_func, c_pars, P_arma,
                   XCa_arma, Exc, nelnum, ekin_grid, dftthr);
    const helfem::Matrix XCa = helfem::to_eigen(XCa_arma);

    const double Etot = Ekin + Enuc + Ecoul + Exc;
    printf("kinetic   % .10f  nuclear   % .10f\n", Ekin, Enuc);
    printf("Coulomb   % .10f  XC        % .10f\n", Ecoul, Exc);
    printf("total     % .10f  (nel int err % .3e)\n",
            Etot, nelnum - static_cast<double>(Z - Q));
    fflush(stdout);

    // Assemble Fock (AO), transform to orthonormal via Sinvh^T F Sinvh.
    const helfem::Matrix F_ao   = T + Vnuc + J + XCa;
    const helfem::Matrix F_orth = Sinvh.transpose() * F_ao * Sinvh;

    OpenOrbitalOptimizer::FockMatrix<OOO_Real> fock(1);
    fock[0] = F_orth;
    return std::make_pair(Etot, fock);
  };

  // Core Hamiltonian guess (orthonormal basis).
  OpenOrbitalOptimizer::FockMatrix<OOO_Real> CoreH(1);
  CoreH[0] = Sinvh.transpose() * (T + Vnuc) * Sinvh;

  OpenOrbitalOptimizer::SCFSolver<OOO_Real, OOO_Real> scfsolver(
      number_of_blocks_per_particle_type, maximum_occupation,
      number_of_particles, fock_builder, block_descriptions);
  scfsolver.initialize_with_fock(CoreH);
  scfsolver.run();

  return 0;
}
