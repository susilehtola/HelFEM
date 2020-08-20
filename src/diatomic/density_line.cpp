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
#include "../general/checkpoint.h"
#include "../general/constants.h"
#include "../general/timer.h"
#include "utils.h"
#include "basis.h"
#include <cfloat>
#include <climits>

using namespace helfem;

int main(int argc, char **argv) {
  cmdline::parser parser;

  // full option name, no short option, description, argument required
  parser.add<std::string>("load", 0, "load guess from checkpoint", false, "");
  parser.add<double>("x", 0, "value of x", false, 0.0);
  parser.add<double>("y", 0, "value of y", false, 0.0);
  parser.add<double>("zmin", 0, "z min", false, -5.0);
  parser.add<double>("zmax", 0, "z max", false, 5.0);
  parser.add<int>("Nz", 0, "number of points in z", false, 101);
  parser.add<std::string>("savedens", 0, "save density to file", false, "density.dat");
  parser.parse_check(argc, argv);

  // Get parameters
  std::string load(parser.get<std::string>("load"));
  double x(parser.get<double>("x"));
  double y(parser.get<double>("y"));
  double zmin(parser.get<double>("zmin"));
  double zmax(parser.get<double>("zmax"));
  std::string savedens(parser.get<std::string>("savedens"));
  int Nz(parser.get<int>("Nz"));
  
  // Load checkpoint
  Checkpoint loadchk(load,false);
  // Basis set
  diatomic::basis::TwoDBasis basis;
  loadchk.read(basis);
  // Density matrix
  arma::mat Pa, Pb;
  loadchk.read("Pa",Pa);
  loadchk.read("Pb",Pb);
  // Rhalf
  double Rhalf;
  loadchk.read("Rhalf",Rhalf);

  const arma::vec z(arma::linspace<arma::vec>(zmin,zmax,Nz));
  
  // Solve for phi angle
  const double phi(atan2(y,x));
  const double xysq(x*x+y*y);

  // Densities
  arma::mat den(Nz,4);
  den.zeros();
  for(size_t iz=0;iz<z.n_elem;iz++) {
    // Compute distances of point from the two nuclei
    double ra(sqrt(std::pow(z(iz)+Rhalf,2)+xysq));
    double rb(sqrt(std::pow(z(iz)-Rhalf,2)+xysq));

    // xi and eta are
    double xi((ra+rb)/(2*Rhalf));
    double eta((ra-rb)/(2*Rhalf));

    // Sanity check
    if(eta<-1.0)
      eta=-1.0;
    if(eta>1.0)
      eta=1.0;

    // so mu is
    double mu(utils::arcosh(xi));

    // Check range: is mu outside of basis set?
    if(mu>basis.get_mumax())
      continue;

    // Evaluate basis functions
    arma::cx_vec bf(basis.eval_bf(mu, eta, phi));

    // Evaluate density at the point
    den(iz,0)=z(iz);
    den(iz,1)=std::real(arma::as_scalar(bf.t()*Pa*bf));
    den(iz,2)=std::real(arma::as_scalar(bf.t()*Pb*bf));
    den(iz,3)=den(iz,1)+den(iz,2);
  }

  printf("Saving density to file %s\n",savedens.c_str());
  den.save(savedens,arma::raw_ascii);

  return 0;
}
