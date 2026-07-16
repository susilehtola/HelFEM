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
#include "../atomic/basis.h"
#include "Matrix.h"
#include "ArmaEigen.h"
#include <Eigen/Eigenvalues>
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

  // Radial basis. atomic::basis::form_grid still returns arma::vec
  // (atomic basis layer, migrated in a later wave); bridge its result to
  // Eigen once at the FiniteElementBasis constructor.
  const helfem::Vector bval = helfem::to_eigen(atomic::basis::form_grid(
      (modelpotential::nuclear_model_t) finitenuc, Rrms, Nelem, Rmax,
      igrid, zexp, Nelem0, igrid0, zexp0, Z, 0, 0, 0.0));

  const bool zero_func_left  = true;
  const bool zero_deriv_left = false;
  const bool zero_func_right = true;
  polynomial_basis::FiniteElementBasis fem(poly, bval,
      zero_func_left, zero_deriv_left, zero_func_right, zeroder);
  atomic::basis::FEMRadialBasis radial(fem, Nquad);

  // FE-side matrices are already helfem::Matrix; no arma bridge needed.
  const helfem::Matrix S     = radial.overlap();
  const helfem::Matrix Sinvh = scf::form_Sinvh(S, /*chol=*/false);
  const helfem::Matrix T     = radial.kinetic();
  const helfem::Matrix Tl    = radial.kinetic_l();
  const helfem::Matrix V     = radial.nuclear();

  for (int l = 0; l <= lmax; ++l) {
    const helfem::Matrix H0 =
        Sinvh.transpose() * (T + Z * V + l * (l + 1) * Tl) * Sinvh;
    Eigen::SelfAdjointEigenSolver<helfem::Matrix> es(H0);
    const helfem::Vector E = es.eigenvalues();
    helfem::Matrix C = Sinvh * es.eigenvectors();

    printf("l=%i eigenvalues\n", l);
    for (Eigen::Index i = 0; i < E.size(); ++i)
      printf("  %.15e\n", E(i));

    // Evaluate orbitals on the quadrature grid.
    const Eigen::Index Ncols = C.cols();
    const Eigen::Index Npts  = radial.get_bf(0).rows();
    helfem::Matrix Cv = helfem::Matrix::Zero(radial.Nel() * Npts, Ncols);
    for (size_t iel = 0; iel < radial.Nel(); ++iel) {
      size_t ifirst, ilast;
      radial.get_idx(iel, ifirst, ilast);
      const helfem::Matrix Csub = C.middleRows(ifirst, ilast - ifirst + 1);
      const helfem::Matrix bf   = radial.get_bf(iel);
      Cv.middleRows(iel * Npts, Npts) = bf * Csub;
    }

    std::ostringstream oss;
    oss << "orbs_" << l;
    chkpt.write(oss.str(), Cv);
    oss.str("");
    oss << "E_" << l;
    chkpt.write(oss.str(), E);
  }

  // Concatenate per-element radii and quadrature weights into flat
  // vectors and write them to the checkpoint.
  const Eigen::Index Npts = radial.get_r(0).size();
  helfem::Vector radii   = helfem::Vector::Zero(radial.Nel() * Npts);
  helfem::Vector weights = helfem::Vector::Zero(radial.Nel() * Npts);
  for (size_t iel = 0; iel < radial.Nel(); ++iel) {
    radii  .segment(iel * Npts, Npts) = radial.get_r   (iel);
    weights.segment(iel * Npts, Npts) = radial.get_wrad(iel);
  }

  chkpt.write("r",  radii);
  chkpt.write("wr", weights);

  return 0;
}
