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
  parser.add<double>("mumax", 0, "mu max (negative for same as input)", false, -1.0);
  parser.add<double>("phi", 0, "value of phi angle", false, 0.0);
  parser.add<int>("Nmu", 0, "number of points in mu (negative for same as input)", false, -1);
  parser.add<int>("Nnu", 0, "number of points in nu (negative for same as input)", false, -1);
  parser.add<bool>("offset", 0, "use nu and mu offset?", false, false);
  parser.add<std::string>("savedens", 0, "save density to file", false, "density.hdf5");
  parser.parse_check(argc, argv);

  // Get parameters
  std::string load(parser.get<std::string>("load"));
  double mumax(parser.get<double>("mumax"));
  double phi(parser.get<double>("phi"));
  std::string savedens(parser.get<std::string>("savedens"));

  // Load checkpoint
  Checkpoint loadchk(load,false);
  // Basis set
  diatomic::basis::TwoDBasis basis;
  loadchk.read(basis);

  // Sanity check
  if(mumax<0 || mumax>basis.get_mumax())
    mumax=basis.get_mumax();

  int Nmu(parser.get<int>("Nmu"));
  if(Nmu<=0)
    Nmu=2*basis.Nrad();

  int Nnu(parser.get<int>("Nnu"));
  if(Nnu<=0)
    Nnu=2*arma::max(basis.get_lval());

  printf("Grid spanning mu = 0 .. %e, Nmu = %i, Nnu = %i\n",mumax,Nmu,Nnu);

  // Density matrix
  arma::mat P, Pa, Pb;
  loadchk.read("Pa",Pa);
  loadchk.read("Pb",Pb);
  loadchk.read("P",P);

  // mu array
  arma::vec mu, nu;
  bool offset(parser.get<bool>("offset"));
  if(offset) {
    double dmu=mumax/Nmu;
    double dnu=M_PI/Nnu;
    mu=arma::linspace<arma::vec>(dmu,mumax,Nmu);
    nu=arma::linspace<arma::vec>(0.5*dnu,(Nnu-0.5)*dnu,Nnu);
  } else {
    mu=arma::linspace<arma::vec>(0.0,mumax,Nmu);
    nu=arma::linspace<arma::vec>(0.0,M_PI,Nnu);
  }
  double dmudnu=(mu(1)-mu(0))*(nu(1)-nu(0));

  // Density arrays
  arma::mat dena(Nmu,Nnu);
  dena.zeros();
  arma::mat denb(Nmu,Nnu);
  denb.zeros();
  arma::mat dV(Nmu,Nnu);
  dV.zeros();
  for(size_t inu=0;inu<nu.n_elem;inu++)
    for(size_t imu=0;imu<mu.n_elem;imu++) {
      // Evaluate basis functions
      double muval(mu(imu));
      double cosnu(cos(nu(inu)));
      arma::cx_vec bf(basis.eval_bf(muval, cosnu, phi));

      // Evaluate density at the point
      dena(imu,inu)=std::real(arma::as_scalar(bf.t()*Pa*bf));
      denb(imu,inu)=std::real(arma::as_scalar(bf.t()*Pb*bf));

      // Volume element is
      double shmu(sinh(muval));
      double chmu(cosh(muval));
      double sinnu(sin(nu(inu)));
      // Get 2 pi from phi integral (assuming density is symmetric)
      dV(imu,inu)=2.0*M_PI*std::pow(basis.get_Rhalf(),3)*shmu*(chmu*chmu - cosnu*cosnu)*sinnu*dmudnu;
    }

  // Total density
  arma::mat den(dena+denb);

  printf("Norm of Pa on grid is %e\n",arma::sum(arma::sum(dena%dV)));
  printf("Norm of Pb on grid is %e\n",arma::sum(arma::sum(denb%dV)));
  printf("Norm of P on grid is %e\n",arma::sum(arma::sum(den%dV)));

  Checkpoint savechk(savedens,true);
  savechk.write("mu",mu);
  savechk.write("nu",nu);
  savechk.write("P",den);
  savechk.write("Pa",dena);
  savechk.write("Pb",denb);
  savechk.write("R",2*basis.get_Rhalf());
  printf("Saved density to file %s\n",savedens.c_str());

  return 0;
}
