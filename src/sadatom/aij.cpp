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
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */
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
#include <cfloat>

using namespace helfem;

int consistent_lmax(int njellium) {
  // Consistent pairing of lmax, given the number of jellium electrons
  const std::vector<int> magic({2, 8, 19, 36, 59, 89, 118, 163, 215, 269, 341, 425, 516, 612, 731, 859, 994, 1131, 1300, 1424, 1625, 1820, 2043, 2295, 2543, 2821, 3080, 3337, 3665, 3987, 4373, 4741, 5155, 5611, 5961, 6407, 6909, 7406, 7964, 8512, 9058, 9591, 10147, 10792, 11440, 12116, 12825, 13611, 14431, 15158});
  for(size_t i=0;i<magic.size();i++) {
    if(magic[i] > njellium)
      return (int) i;
  }

  return -1;
}

int main(int argc, char **argv) {
  cmdline::parser parser;

  // full option name, no short option, description, argument required
  parser.add<int>("grid", 0, "type of grid: 1 for linear, 2 for quadratic, 3 for polynomial, 4 for exponential", false, 4);
  parser.add<double>("zexp", 0, "parameter in radial grid", false, 2.0);
  parser.add<int>("nelem", 0, "number of elements", true);
  parser.add<int>("nuelem", 0, "number of uniform elements", true);
  parser.add<std::string>("Z", 0, "nuclear charge", true);
  parser.add<int>("Q", 0, "charge of system", false, 0);
  parser.add<int>("nnodes", 0, "number of nodes per element", false, 15);
  parser.add<int>("nquad", 0, "number of quadrature points", false, 0);
  parser.add<int>("maxit", 0, "maximum number of iterations", false, 200);
  parser.add<double>("convthr", 0, "convergence threshold", false, 1e-7);
  parser.add<std::string>("method", 0, "method to use", false, "lda_x");
  parser.add<double>("dftthr", 0, "density threshold for dft", false, 1e-12);
  parser.add<int>("restricted", 0, "spin-restricted orbitals", false, -1);
  parser.add<int>("primbas", 0, "primitive radial basis", false, 4);
  parser.add<int>("taylor_order", 0, "order of Taylor expansion near the nucleus", false, -1);
  parser.add<std::string>("x_pars", 0, "file for parameters for exchange functional", false, "");
  parser.add<std::string>("c_pars", 0, "file for parameters for correlation functional", false, "");
  parser.add<int>("njellium", 0,"number of jellium electrons", true);
  parser.add<double>("rs", 0, "Wigner-Seitz radius for jellium", true);
  parser.add<bool>("vacancy", 0, "Jellium vacancy model?", false, false);
  parser.parse_check(argc, argv);

  // Get parameters
  int igrid(parser.get<int>("grid"));
  double zexp(parser.get<double>("zexp"));
  int Nelem(parser.get<int>("nelem"));
  int Nuelem(parser.get<int>("nuelem"));

  int Z(get_Z(parser.get<std::string>("Z")));
  int Q(parser.get<int>("Q"));

  int Nnodes(parser.get<int>("nnodes"));
  int Nquad(parser.get<int>("nquad"));
  int maxit(parser.get<int>("maxit"));
  double convthr(parser.get<double>("convthr"));
  std::string method(parser.get<std::string>("method"));
  double dftthr(parser.get<double>("dftthr"));
  int restr(parser.get<int>("restricted"));
  
  int primbas(parser.get<int>("primbas"));
  int taylor_order(parser.get<int>("taylor_order"));
  std::string xparf(parser.get<std::string>("x_pars"));
  std::string cparf(parser.get<std::string>("c_pars"));

  int njellium(parser.get<int>("njellium"));
  double rs(parser.get<double>("rs"));
  bool vacancy(parser.get<bool>("vacancy"));

  // Get primitive basis
  auto poly(std::shared_ptr<const polynomial_basis::PolynomialBasis>(polynomial_basis::get_basis(primbas,Nnodes)));

  if(Nquad==0)
    // Set default value
    Nquad=5*poly->get_nbf();
  else if(Nquad<2*poly->get_nbf())
    throw std::logic_error("Insufficient radial quadrature.\n");
  // Order of quadrature rule
  printf("Using %i point quadrature rule.\n",Nquad);

  // Set default order of Taylor expansion
  if(taylor_order==-1)
    taylor_order = poly->get_nprim()-1;

  // Functional
  int x_func, c_func;
  ::parse_xc_func(x_func, c_func, method);
  ::print_info(x_func, c_func);
  if(!is_supported(x_func))
    throw std::logic_error("The specified exchange functional is not currently supported in HelFEM.\n");
  if(!is_supported(c_func))
    throw std::logic_error("The specified correlation functional is not currently supported in HelFEM.\n");

  // Determine box size
  double R = 0.0, r_inner = 0.0, r_outer = 0.0;
  if(vacancy) {
    // Inner cavity has zero charge for a radius that matches the density of the background charge.
    // Outer cavity has constant background charge. Altogether, this leads to
    R = cbrt(Z-Q+njellium)*rs;
    r_inner = cbrt(Z)*rs;
    r_outer = R - r_inner;
  } else {
    R = cbrt(njellium)*rs;
    r_inner = 0.0;
    r_outer = R;
  }

  int lmax = consistent_lmax(njellium);

  if(vacancy) {
    printf("%i jellium electrons with rs = % .3f and vacancy model leads to r_inner = % .3f r_outer = % .3f lmax = %i\n",njellium,rs,r_inner,r_outer,lmax);
  } else {
    printf("%i jellium electrons with rs = % .3f leads to R = % .3f lmax = %i\n",njellium,rs,R,lmax);
  }

  // Total number of electrons is
  arma::sword numel=Z-Q+njellium;

  // Uniform part of grid
  arma::vec bval_unif=atomic::basis::form_grid(modelpotential::POINT_NUCLEUS, 0.0, Nuelem, R, 1, 0.0, 0, 0, 0, Z, 0, 0, 0.0, false, 0.0);
  // Atomic grid
  arma::vec bval_atom=atomic::basis::form_grid(modelpotential::POINT_NUCLEUS, 0.0, Nelem, bval_unif(1), igrid, zexp, 0, 0, 0.0, Z, 0, 0, 0.0, false, 0.0);

  // Glue grids together
  arma::vec bval(bval_atom.n_elem+bval_unif.n_elem-2);
  bval.subvec(0,bval_atom.n_elem-1) = bval_atom;
  if(bval_atom(bval_atom.n_elem-1) != bval_unif(1)) {
    std::ostringstream oss;
    oss << "Grids don't coincide: difference " << bval_atom(bval_atom.n_elem-1) - bval_unif(1) << "!\n";
    throw std::logic_error(oss.str());
  }
  bval.subvec(bval_atom.n_elem,bval.n_elem-1) = bval_unif.subvec(2,bval_unif.n_elem-1);

  // Handle vacancy case
  if(vacancy) {
    arma::vec vbval(bval.n_elem+1);
    vbval.subvec(0,bval.n_elem-1)=bval;
    vbval(bval.n_elem) = r_inner;
    bval = arma::sort(vbval, "ascend");
  }
  bval.print("bval");

  bool zeroder = false;
  auto basis = sadatom::basis::TwoDBasis(Z, modelpotential::POINT_NUCLEUS, 0.0, poly, zeroder, Nquad, bval, taylor_order, lmax);

  // Form overlap matrix
  arma::mat S=basis.overlap();
  // Get half-inverse
  arma::mat Sinvh=basis.Sinvh();
  // Form kinetic energy matrix
  arma::mat T=basis.kinetic();
  // Form kinetic energy matrix
  arma::mat Tl=basis.kinetic_l();
  // Form nuclear attraction energy matrix
  arma::mat Vnuc=basis.nuclear();
  // Uniform background potential
  arma::mat Vunif=basis.background_potential(rs, r_inner, r_outer);
  // Form core Hamiltonian
  arma::mat H0=T+Vnuc+Vunif;

  // Form DFT grid
  auto grid = helfem::sadatom::dftgrid::DFTGrid(&basis);
  // Compute two-electron integrals
  basis.compute_tei();

  

  
  return 0;
}
