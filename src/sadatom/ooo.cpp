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

// Sadatom driver using OpenOrbitalOptimizer for SCF convergence.
// Ported from the aij branch (commit 0318ab9 "Atomic calculations should
// work now with OpenOrbitalOptimizer") onto current master. The port
// adapts to:
//   - sadatom TwoDBasis constructor (dropped the taylor_order argument
//     that lived on the aij branch).
//   - Phase-5 migrated basis accessors (overlap/kinetic/nuclear/coulomb
//     now return helfem::Matrix / Eigen).
//   - OpenOrbitalOptimizer's Eigen-typed public API (was arma at the
//     time the aij snapshot was written).

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
  const int Q       = parser.get<int>("Q");
  const int Nnodes  = parser.get<int>("nnodes");
        int Nquad   = parser.get<int>("nquad");
  const std::string method = parser.get<std::string>("method");
  const double dftthr = parser.get<double>("dftthr");
  const int primbas   = parser.get<int>("primbas");
  const std::string xparf = parser.get<std::string>("x_pars");
  const std::string cparf = parser.get<std::string>("c_pars");
  const int lmax      = parser.get<int>("lmax");
  const double Rmax   = parser.get<double>("Rmax");

  // Parse xc parameters (Phase 5.15: scf::parse_xc_params returns
  // helfem::Vector; bridge to arma once because grid.eval_Fxc still
  // takes arma::vec params).
  arma::vec x_pars, c_pars;
  if (xparf.size()) {
    x_pars = helfem::to_arma(scf::parse_xc_params(xparf));
    x_pars.t().print("Exchange functional parameters");
  }
  if (cparf.size()) {
    c_pars = helfem::to_arma(scf::parse_xc_params(cparf));
    c_pars.t().print("Correlation functional parameters");
  }

  printf("Running restricted %s calculation with Rmax=%e and %i elements.\n",
          method.c_str(), Rmax, Nelem);

  auto poly = std::shared_ptr<const polynomial_basis::PolynomialBasis>(
      polynomial_basis::get_basis(primbas, Nnodes));

  if (Nquad == 0)
    Nquad = 5 * poly->get_nbf();
  else if (Nquad < 2 * poly->get_nbf())
    throw std::logic_error("Insufficient radial quadrature.\n");
  printf("Using %i point quadrature rule.\n", Nquad);

  // Functional
  int x_func, c_func;
  ::parse_xc_func(x_func, c_func, method);
  ::print_info(x_func, c_func);
  if (!is_supported(x_func))
    throw std::logic_error("The specified exchange functional is not currently supported in HelFEM.\n");
  if (!is_supported(c_func))
    throw std::logic_error("The specified correlation functional is not currently supported in HelFEM.\n");
  // Hartree-Fock (x_func == 0 && c_func == 0) is NOT supported in this
  // first-cut driver: the Fock builder below computes J via
  // basis.coulomb() but does not add HF exchange. Follow-up work needs
  // basis.exchange() and a bridge into the callback. For now, error
  // out so users don't get a silent Hartree-only answer.
  if (x_func == 0 && c_func == 0)
    throw std::logic_error("HF is not yet implemented in sadatom_ooo -- pick a DFT functional.\n");

  arma::vec bval = atomic::basis::form_grid(
      modelpotential::POINT_NUCLEUS, 0.0, Nelem, Rmax, igrid, zexp,
      0, 0, 0.0, Z, 0, 0, 0.0, false, 0.0);

  const bool zeroder = false;
  auto basis = sadatom::basis::TwoDBasis(Z, modelpotential::POINT_NUCLEUS, 0.0,
                                          poly, zeroder, Nquad, bval, lmax);
  printf("Basis set has %i radial functions\n", (int) basis.Nbf());

  // Phase 5: overlap / kinetic / kinetic_l / nuclear now return Eigen.
  // Sinvh still arma; bridge it once to Eigen up front.
  const helfem::Matrix S    = basis.overlap();
  const helfem::Matrix Sinvh = helfem::to_eigen(basis.Sinvh());
  const helfem::Matrix T    = basis.kinetic();
  const helfem::Matrix Tl   = basis.kinetic_l();
  const helfem::Matrix Vnuc = basis.nuclear();

  auto grid = helfem::sadatom::dftgrid::DFTGrid(&basis);
  basis.compute_tei();

  // OOO input: block structure (one block per l).
  using OOO_Real  = double;
  using OOO_Index = OpenOrbitalOptimizer::Index;
  OpenOrbitalOptimizer::IndexVector number_of_blocks_per_particle_type(1);
  number_of_blocks_per_particle_type(0) = lmax + 1;
  Eigen::Matrix<OOO_Real, Eigen::Dynamic, 1> maximum_occupation(lmax + 1);
  Eigen::Matrix<OOO_Real, Eigen::Dynamic, 1> number_of_particles(1);
  number_of_particles(0) = static_cast<OOO_Real>(Z - Q);
  std::vector<std::string> block_descriptions(lmax + 1);
  for (int l = 0; l <= lmax; ++l) {
    maximum_occupation(l) = 2 * (2 * l + 1);
    std::ostringstream oss; oss << "l=" << l;
    block_descriptions[l] = oss.str();
  }
  std::cout << "Max occ: " << maximum_occupation.transpose() << "\n";

  // Fock builder. Uses arma internally for the density / XC intermediate
  // buffers (grid.eval_Fxc is still arma::cube-typed and basis.compute_tei
  // caches feed basis.coulomb) and bridges at the OOO Eigen boundary.
  OpenOrbitalOptimizer::FockBuilder<OOO_Real, OOO_Real> fock_builder =
      [&](const OpenOrbitalOptimizer::DensityMatrix<OOO_Real, OOO_Real> & dm) {
    const auto & orbitals = dm.first;      // std::vector<Eigen::MatrixXd>
    const auto & occupations = dm.second;  // std::vector<Eigen::VectorXd>

    const Eigen::Index Nrad = Sinvh.rows();

    double Ekin = 0.0;
    // Total radial density (summed over l) as Eigen for coulomb().
    helfem::Matrix Prad = helfem::Matrix::Zero(Nrad, Nrad);
    // Per-l density as arma::cube for grid.eval_Fxc.
    arma::cube Pl(Nrad, Nrad, lmax + 1, arma::fill::zeros);

    for (int l = 0; l <= lmax; ++l) {
      // Skip empty blocks.
      if (occupations[l].cwiseAbs().maxCoeff() == 0.0)
        continue;

      // C in original AO basis: C = Sinvh * orbitals_l  (Eigen).
      const helfem::Matrix C = Sinvh * orbitals[l];
      // P_l = C * diag(occ_l) * C^T  (Eigen).
      const helfem::Matrix Pblock = C * occupations[l].asDiagonal() * C.transpose();

      Prad += Pblock;
      Pl.slice(l) = helfem::to_arma(Pblock);

      Ekin += (Pblock * T).trace();
      if (l > 0)
        Ekin += l * (l + 1) * (Pblock * Tl).trace();
    }

    const double Enuc = (Prad * Vnuc).trace();

    const double angfac = 4.0 * M_PI;

    // Coulomb J: basis.coulomb takes helfem::Matrix (Eigen) after Phase 5.
    const helfem::Matrix J = basis.coulomb(Prad / angfac);

    double Exc = 0.0;
    double nelnum = 0.0;
    arma::cube XC;
    if (x_func > 0 || c_func > 0) {
      grid.eval_Fxc(x_func, x_pars, c_func, c_pars, Pl / angfac,
                     XC, Exc, nelnum, dftthr);
      XC /= angfac;
    }

    const double Ecoul = 0.5 * (Prad * J).trace();
    const double Etot  = Ekin + Enuc + Ecoul + Exc;

    printf("kinetic energy         % .10f\n", Ekin);
    printf("nuclear attraction     % .10f\n", Enuc);
    printf("Coulomb repulsion      % .10f\n", Ecoul);
    printf("exchange-correlation   % .10f\n", Exc);
    printf("total energy           % .10f\n", Etot);
    fflush(stdout);

    OpenOrbitalOptimizer::FockMatrix<OOO_Real> fock(lmax + 1);
    for (int l = 0; l <= lmax; ++l) {
      helfem::Matrix Fl = T + Vnuc + J;
      if (l > 0) Fl += l * (l + 1) * Tl;
      if (x_func > 0 || c_func > 0)
        Fl += helfem::to_eigen(arma::mat(XC.slice(l)));
      // Transform to orthonormal basis: F_orth = Sinvh^T F_ao Sinvh.
      fock[l] = Sinvh.transpose() * Fl * Sinvh;
    }
    return std::make_pair(Etot, fock);
  };

  // Core Hamiltonian per l block (orthonormal basis) as initial guess.
  OpenOrbitalOptimizer::FockMatrix<OOO_Real> CoreH(lmax + 1);
  for (int l = 0; l <= lmax; ++l) {
    helfem::Matrix Hl = T + Vnuc;
    if (l > 0) Hl += l * (l + 1) * Tl;
    CoreH[l] = Sinvh.transpose() * Hl * Sinvh;
  }

  OpenOrbitalOptimizer::SCFSolver<OOO_Real, OOO_Real> scfsolver(
      number_of_blocks_per_particle_type, maximum_occupation,
      number_of_particles, fock_builder, block_descriptions);
  scfsolver.initialize_with_fock(CoreH);
  scfsolver.run();

  return 0;
}
