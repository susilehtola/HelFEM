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

// Sadatom driver (binary: gensap) -- OpenOrbitalOptimizer based.
// Thin CLI wrapper around helfem::sadatom::scf::run_atomic_scf which
// is also called from src/diatomic/twodquadrature.cpp for the
// atomic-orbital-guess sub-SCF.
//
// HF (x_func == 0 && c_func == 0) is deferred; picking any HF method throws.

#include "../general/cmdline.h"
#include "../general/dftfuncs.h"
#include "../general/elements.h"
#include "../general/scf_helpers.h"
#include "../general/scf_driver_common.h"
#include "../atomic/basis.h"
#include "scf.h"
#include <ArmaEigen.h>
#include <iostream>

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

  // Confinement potential (parity with bespoke sadatom driver).
  parser.add<int>   ("iconf",        0, "Confinement potential: 1 polynomial, 2 exponential, 3 barrier, 4 Junquera et al.", false, 0);
  parser.add<int>   ("conf_N",       0, "Exponent in confinement potential", false, 0);
  parser.add<double>("conf_R",       0, "Confinement radius",                false, 0.0);
  parser.add<double>("conf_barrier", 0, "Confinement barrier height",        false, 0.0);
  parser.add<double>("shift_conf",   0, "Where does confinement start?",     false, 0.0);
  parser.add<bool>  ("add_conf",     0, "Add element boundary at shifted confinement radius?", false, true);

  parser.add<std::string>("load", 0, "load orbital guess from checkpoint file", false, "");
  parser.add<std::string>("save", 0, "save results to checkpoint file",       false, "");

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

  const int    iconf        = parser.get<int>("iconf");
  const int    conf_N       = parser.get<int>("conf_N");
  const double conf_R       = parser.get<double>("conf_R");
  const double conf_barrier = parser.get<double>("conf_barrier");
  const double shift_conf   = parser.get<double>("shift_conf");
  const bool   add_conf     = parser.get<bool>("add_conf");

  bool restricted;
  int Ntot;
  helfem::scf_driver::derive_nela_nelb_restricted(
      nela, nelb, restr, Q, M, Z, restricted, Ntot);
  (void) Ntot;  // sadatom SCF driver does its own per-l particle counting.

  helfem::Vector x_pars, c_pars;
  if (xparf.size()) {
    x_pars = scf::parse_xc_params(xparf);
    std::cout << "Exchange functional parameters\n" << x_pars.transpose() << std::endl;
  }
  if (cparf.size()) {
    c_pars = scf::parse_xc_params(cparf);
    std::cout << "Correlation functional parameters\n" << c_pars.transpose() << std::endl;
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

  arma::vec bval = atomic::basis::form_grid(
      modelpotential::POINT_NUCLEUS, 0.0, Nelem, Rmax, igrid, zexp,
      0, 0, 0.0, Z, 0, 0, 0.0,
      iconf ? add_conf : false, shift_conf);

  sadatom::scf::AtomicSCFOptions opts;
  opts.Z            = Z;
  opts.lmax         = lmax;
  opts.poly         = poly;
  opts.Nquad        = Nquad;
  opts.bval         = bval;
  opts.nela         = nela;
  opts.nelb         = nelb;
  opts.restricted   = restricted;
  opts.x_func       = x_func;
  opts.c_func       = c_func;
  opts.x_pars       = x_pars;
  opts.c_pars       = c_pars;
  opts.dftthr       = dftthr;
  opts.iconf        = iconf;
  opts.conf_N       = conf_N;
  opts.conf_R       = conf_R;
  opts.conf_barrier = conf_barrier;
  opts.shift_conf   = shift_conf;
  opts.verbosity    = 5;
  opts.load_file    = parser.get<std::string>("load");
  opts.save_file    = parser.get<std::string>("save");

  sadatom::scf::run_atomic_scf(opts);
  return 0;
}
