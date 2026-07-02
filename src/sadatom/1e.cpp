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
#include "../general/cmdline.h"
#include "../general/checkpoint.h"
#include "../general/constants.h"
#include "../general/dftfuncs.h"
#include "../general/elements.h"
#include "../general/scf_helpers.h"
#include "utils.h"
#include "dftgrid.h"
#include "solver.h"
#include "configurations.h"
#include <cfloat>

using namespace helfem;

int main(int argc, char **argv) {
  cmdline::parser parser;

  // full option name, no short option, description, argument required
  parser.add<std::string>("Z", 0, "nuclear charge", true);
  parser.add<double>("Rmax", 0, "practical infinity in au", false, 40.0);
  parser.add<int>("grid", 0, "type of grid: 1 for linear, 2 for quadratic, 3 for polynomial, 4 for exponential", false, 4);
  parser.add<int>("grid0", 0, "type of grid: 1 for linear, 2 for quadratic, 3 for polynomial, 4 for exponential", false, 4);
  parser.add<double>("zexp", 0, "parameter in radial grid", false, 2.0);
  parser.add<double>("zexp0", 0, "parameter in radial grid", false, 2.0);
  parser.add<int>("nelem", 0, "number of elements", true);
  parser.add<int>("nelem0", 0, "number of elements", false, 0);
  parser.add<int>("lmax", 0, "maximum angular momentum to include", false, 3);
  parser.add<int>("nnodes", 0, "number of nodes per element", false, 15);
  parser.add<int>("nquad", 0, "number of quadrature points", false, 0);
  parser.add<int>("primbas", 0, "primitive radial basis", false, 4);
  parser.add<int>("finitenuc", 0, "finite nuclear model", false, 0);
  parser.add<double>("Rrms", 0, "nuclear rms radius", false, 0.0);
  parser.add<std::string>("save", 0, "checkpoint to save results in", false, "1e.chk");
  parser.parse_check(argc, argv);
/*
  if(!parser.parse(argc, argv))
    throw std::logic_error("Error parsing arguments!\n");
*/

  // Get parameters
  int Z(get_Z(parser.get<std::string>("Z")));
  double Rmax(parser.get<double>("Rmax"));
  int igrid(parser.get<int>("grid"));
  int igrid0(parser.get<int>("grid0"));
  double zexp(parser.get<double>("zexp"));
  double zexp0(parser.get<double>("zexp0"));

  int finitenuc(parser.get<int>("finitenuc"));
  double Rrms(parser.get<double>("Rrms"));
  int lmax(parser.get<int>("lmax"));
  int primbas(parser.get<int>("primbas"));
  // Number of elements
  int Nelem(parser.get<int>("nelem"));
  int Nelem0(parser.get<int>("nelem0"));
  // Number of nodes
  int Nnodes(parser.get<int>("nnodes"));

  // Order of quadrature rule
  int Nquad(parser.get<int>("nquad"));

  // Open checkpoint
  std::string save(parser.get<std::string>("save"));
  Checkpoint chkpt(save,true);

  bool zeroder=false;

  // Get primitive basis
  auto poly(std::shared_ptr<const polynomial_basis::PolynomialBasis>(polynomial_basis::get_basis(primbas,Nnodes)));

  if(Nquad==0)
    // Set default value
    Nquad=5*poly->get_nbf();
  else if(Nquad<2*poly->get_nbf())
    throw std::logic_error("Insufficient radial quadrature.\n");
  // Order of quadrature rule
  printf("Using %i point quadrature rule.\n",Nquad);

  // Radial basis
  arma::vec bval=atomic::basis::form_grid((modelpotential::nuclear_model_t) finitenuc, Rrms, Nelem, Rmax, igrid, zexp, Nelem0, igrid0, zexp0, Z, 0, 0, 0.0);

  // Construct radial basis
  bool zero_func_left=true;
  bool zero_deriv_left=false;
  bool zero_func_right=true;
  polynomial_basis::FiniteElementBasis fem(poly, bval, zero_func_left, zero_deriv_left, zero_func_right, zeroder);
  atomic::basis::FEMRadialBasis radial(fem, Nquad);

  // Compute matrices. Phase 2a: radial.overlap() returns
  // helfem::Matrix (Eigen); rest of sadatom 1e is still arma -- bridge
  // at the boundary via to_arma until the chemistry layer migrates.
  arma::mat S(helfem::to_arma(radial.overlap()));
  arma::mat Sinvh=helfem::to_arma(scf::form_Sinvh(helfem::to_eigen(S),false));

  // Phase 2a: radial.kinetic / kinetic_l / nuclear return helfem::Matrix.
  arma::mat T(helfem::to_arma(radial.kinetic()));
  arma::mat Tl(helfem::to_arma(radial.kinetic_l()));
  arma::mat V(helfem::to_arma(radial.nuclear()));

  // Compute orbitals
  for(int l=0;l<=lmax;l++) {
    arma::mat H0 = Sinvh.t() * (T+Z*V+l*(l+1)*Tl) * Sinvh;
    arma::vec E;
    arma::mat C;
    arma::eig_sym(E,C,H0);
    C = Sinvh*C;

    std::ostringstream oss;
    oss << "l=" << l << " eigenvalues";
    E.print(oss.str());

    // Evaluate orbitals on grid
    std::vector<arma::mat> c(radial.Nel());
    for(size_t iel=0;iel<radial.Nel();iel++) {
      // Radial functions in element
      size_t ifirst, ilast;
      radial.get_idx(iel,ifirst,ilast);
      // Density matrix
      arma::mat Csub(C.rows(ifirst,ilast));
      arma::mat bf(helfem::to_arma(radial.get_bf(iel)));

      c[iel]=bf*Csub;
    }
    size_t Npts=c[0].n_rows;

    // Orbital values
    arma::mat Cv(radial.Nel()*Npts,C.n_cols,arma::fill::zeros);
    for(size_t iel=0;iel<radial.Nel();iel++) {
      Cv.rows(iel*Npts,(iel+1)*Npts-1)=c[iel];
    }

    oss.str("");
    oss << "orbs_" << l;
    chkpt.write(oss.str(), Cv);
    oss.str("");
    oss << "E_" << l;
    chkpt.write(oss.str(), E);
  }

  // Get the radii and quadrature weights as well
  std::vector<arma::vec> r(radial.Nel()), wr(radial.Nel());
  for(size_t iel=0;iel<radial.Nel();iel++) {
    r[iel]=helfem::to_arma(radial.get_r(iel));
    wr[iel]=helfem::to_arma(radial.get_wrad(iel));
  }
  size_t Npts=r[0].n_rows;

  // Value of radii and weights
  arma::vec radii(radial.Nel()*Npts,arma::fill::zeros);
  arma::vec weights(radial.Nel()*Npts,arma::fill::zeros);
  for(size_t iel=0;iel<radial.Nel();iel++) {
    radii.subvec(iel*Npts,(iel+1)*Npts-1)=r[iel];
    weights.subvec(iel*Npts,(iel+1)*Npts-1)=wr[iel];
  }

  chkpt.write("r",radii);
  chkpt.write("wr",weights);

  return 0;
}
