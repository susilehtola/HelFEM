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
#include <helfem.h>
#include "../general/checkpoint.h"
#include "PolynomialBasis.h"
#include "FiniteElementBasis.h"
#include "chebyshev.h"

#include <Eigen/Eigenvalues>

using namespace helfem;

helfem::Matrix overlap(const helfem::polynomial_basis::FiniteElementBasis & fem, const helfem::Vector & x, const helfem::Vector & wx) {
  return fem.matrix_element(0, 0, x, wx, nullptr);
}

helfem::Matrix potential(const helfem::polynomial_basis::FiniteElementBasis & fem, const helfem::Vector & x, const helfem::Vector & wx, double De, double a, double re) {
  std::function<double(double)> V = [&](double r) {
    return De*std::pow(1-exp(-a*(r-re)),2);
  };
  return fem.matrix_element(0, 0, x, wx, V);
}

helfem::Matrix kinetic(const helfem::polynomial_basis::FiniteElementBasis & fem, const helfem::Vector & x, const helfem::Vector & wx) {
  return 0.5*fem.matrix_element(1, 1, x, wx, nullptr);
}

int main(int argc, char **argv) {
  // Not a --verbosity driver: opt into the library's setup reporting
  // so this tool prints exactly what it always did.
  helfem::set_verbosity(true);
  cmdline::parser parser;

  // full option name, no short option, description, argument required
  parser.add<double>("rmax", 0, "practical infinity in au", false, 40.0);
  parser.add<int>("nelem", 0, "number of elements", false, 5);
  parser.add<int>("nnodes", 0, "number of elements", false, 8);
  parser.add<int>("primbas", 0, "primitive basis", false, 5);
  parser.add<int>("nquad", 0, "primitive basis", false, -1);
  parser.add<double>("De", 0, "dissociation energy", false, 1.0);
  parser.add<double>("a", 0, "inverse length scale", false, 1.0);
  parser.add<double>("re", 0, "equilibrium bond length", false, 1.0);
  parser.add<double>("m", 0, "mass", false, 1.0);
  parser.add<bool>("deuteron", 0, "deuteron?", false, false);
  parser.add<std::string>("save", 0, "Checkpoint to save results to", false, "morse.chk");

  parser.parse_check(argc, argv);
  double rmax = parser.get<double>("rmax");
  int Nelem = parser.get<int>("nelem");
  int Nnodes = parser.get<int>("nnodes");
  int primbas = parser.get<int>("primbas");
  int Nquad = parser.get<int>("nquad");
  double De = parser.get<double>("De");
  double a = parser.get<double>("a");
  double re = parser.get<double>("re");
  bool deuteron = parser.get<bool>("deuteron");
  std::string save = parser.get<std::string>("save");

  // Get polynomial basis
  auto poly(std::shared_ptr<const helfem::polynomial_basis::PolynomialBasis>(helfem::polynomial_basis::get_basis(primbas, Nnodes)));
  if(Nquad<0)
    Nquad=5*poly->get_nbf();

  // Radial grid
  helfem::Vector r(helfem::Vector::LinSpaced(Nelem+1,0.0,rmax));

  // Finite element basis
  bool zero_func_left=true;
  bool zero_deriv_left=false;
  bool zero_func_right=true;
  bool zero_deriv_right=false;
  helfem::polynomial_basis::FiniteElementBasis fem(poly, r, zero_func_left, zero_deriv_left, zero_func_right, zero_deriv_right);

  // Quadrature rule
  helfem::Vector xq, wq;
  chebyshev::chebyshev<double>(Nquad,xq,wq);

  size_t Nbf(fem.get_nbf());
  printf("Basis set contains %i functions\n",(int) Nbf);

  // Form overlap matrix
  helfem::Matrix S(overlap(fem, xq, wq));
  // Form potential matrix
  helfem::Matrix V(potential(fem, xq, wq, De, a, re));
  // Form kinetic energy matrix
  helfem::Matrix T(kinetic(fem, xq, wq));
  if(deuteron)
    T/=2.0;

  // Form Hamiltonian
  helfem::Matrix H(T+V);

  // Form orthonormal basis
  Eigen::SelfAdjointEigenSolver<helfem::Matrix> Ses(S);
  helfem::Vector Sval(Ses.eigenvalues());
  helfem::Matrix Svec(Ses.eigenvectors());

  printf("Smallest value of overlap matrix is % e, condition number is %e\n",Sval(0),Sval(Sval.size()-1)/Sval(0));
  printf("Smallest and largest bf norms are %e and %e\n",S.diagonal().cwiseAbs().minCoeff(),S.diagonal().cwiseAbs().maxCoeff());

  // Form half-inverse
  helfem::Matrix Sinvh(Svec * Sval.cwiseInverse().cwiseSqrt().asDiagonal() * Svec.transpose());

  // Form orthonormal Hamiltonian
  helfem::Matrix Horth(Sinvh.transpose()*H*Sinvh);

  // Diagonalize Hamiltonian
  Eigen::SelfAdjointEigenSolver<helfem::Matrix> Hes(Horth);
  helfem::Vector E(Hes.eigenvalues());
  helfem::Matrix C(Hes.eigenvectors());

  // Go back to non-orthonormal basis
  C=Sinvh*C;

  for(size_t i=0;i<10;i++)
    printf("E[%i] = % .15e\n",(int) i, E(i));

  // Test orthonormality
  helfem::Matrix Smo(C.transpose()*S*C);
  Smo-=helfem::Matrix::Identity(Smo.rows(),Smo.cols());
  printf("Orbital orthonormality devation is %e\n",Smo.norm());

  // Evaluate the basis set: 0th derivative
  helfem::Matrix bfval(fem.eval_dnf(xq, 0));
  helfem::Matrix phival(bfval*C);
  helfem::Vector coords(fem.eval_coord(xq));
  helfem::Vector weights(fem.eval_weights(wq));

  // Test orbitals are still orthonormal
  helfem::Matrix Sgrid(phival.transpose()*weights.asDiagonal()*phival);
  Sgrid-=helfem::Matrix::Identity(Sgrid.rows(),Sgrid.cols());
  printf("Orbital orthonormality devation on grid is %e\n",Sgrid.norm());

  Checkpoint chkpt(save, true);
  chkpt.write("bf",bfval);
  chkpt.write("C",C);
  chkpt.write("phi",phival);
  chkpt.write("coords",coords);
  chkpt.write("weights",weights);

  return 0;
}
