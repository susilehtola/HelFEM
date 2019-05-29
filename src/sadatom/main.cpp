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
#include "../general/polynomial_basis.h"
#include "../general/utils.h"
#include "dftgrid.h"
#include "solver.h"
#include <cfloat>

arma::ivec initial_occs(int Z, int lmax) {
  // Guess occupations
  const int shell_order[]={0, 0, 1, 0, 1, 0, 2, 1, 0, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1};

  arma::ivec occs(lmax+1);
  occs.zeros();
  for(size_t i=0;i<sizeof(shell_order)/sizeof(shell_order[0]);i++) {
    int l = shell_order[i];
    if(l > lmax) {
      std::ostringstream oss;
      oss << "Insufficient lmax = " << lmax << "\n";
      throw std::logic_error(oss.str());
    }
    int nocc = std::min(Z, 2*(2*l+1));
    occs(l) += nocc;
    Z -= nocc;
    if(Z == 0)
      break;
  }

  return occs;
}

using namespace helfem;

int main(int argc, char **argv) {
  cmdline::parser parser;

  // full option name, no short option, description, argument required
  parser.add<std::string>("Z", 0, "nuclear charge", true);
  parser.add<double>("Rmax", 0, "practical infinity in au", false, 40.0);
  parser.add<int>("grid", 0, "type of grid: 1 for linear, 2 for quadratic, 3 for polynomial, 4 for exponential", false, 4);
  parser.add<double>("zexp", 0, "parameter in radial grid", false, 2.0);
  parser.add<int>("nelem", 0, "number of elements", true);
  parser.add<int>("Q", 0, "charge of system", false, 0);
  parser.add<int>("lmax", 0, "maximum angular momentum to include", false, 3);
  parser.add<int>("nnodes", 0, "number of nodes per element", false, 15);
  parser.add<int>("nquad", 0, "number of quadrature points", false, 0);
  parser.add<int>("maxit", 0, "maximum number of iterations", false, 200);
  parser.add<double>("shift", 0, "level shift for initial SCF iterations", false, 1.0);
  parser.add<double>("convthr", 0, "convergence threshold", false, 1e-7);
  parser.add<std::string>("method", 0, "method to use", false, "lda_x");
  parser.add<std::string>("pot", 0, "method to use to compute potential", false, "lda_x");
  parser.add<std::string>("occs", 0, "occupations to use", false, "auto");
  parser.add<double>("dftthr", 0, "density threshold for dft", false, 1e-12);
  parser.add<int>("restricted", 0, "spin-restricted orbitals", false, 0);
  parser.add<int>("primbas", 0, "primitive radial basis", false, 4);
  parser.add<double>("diiseps", 0, "when to start mixing in diis", false, 1e-2);
  parser.add<double>("diisthr", 0, "when to switch over fully to diis", false, 1e-3);
  parser.add<int>("diisorder", 0, "length of diis history", false, 10);
  parser.add<std::string>("occupations", 0, "occupations", false, "auto");
  if(!parser.parse(argc, argv))
    throw std::logic_error("Error parsing arguments!\n");

  // Get parameters
  double Rmax(parser.get<double>("Rmax"));
  int igrid(parser.get<int>("grid"));
  double zexp(parser.get<double>("zexp"));
  int maxit(parser.get<int>("maxit"));
  int lmax(parser.get<int>("lmax"));
  double convthr(parser.get<double>("convthr"));
  int restr(parser.get<int>("restricted"));
  int primbas(parser.get<int>("primbas"));
  // Number of elements
  int Nelem(parser.get<int>("nelem"));
  // Number of nodes
  int Nnodes(parser.get<int>("nnodes"));

  double shift(parser.get<double>("shift"));

  // Order of quadrature rule
  int Nquad(parser.get<int>("nquad"));
  double dftthr(parser.get<double>("dftthr"));

  // Nuclear charge
  int Q(parser.get<int>("Q"));
  int Z(get_Z(parser.get<std::string>("Z")));
  double diiseps=parser.get<double>("diiseps");
  double diisthr=parser.get<double>("diisthr");
  int diisorder=parser.get<int>("diisorder");

  std::string method(parser.get<std::string>("method"));
  std::string potmethod(parser.get<std::string>("pot"));
  std::string occstr(parser.get<std::string>("occs"));

  std::vector<std::string> rcalc(2);
  rcalc[0]="unrestricted";
  rcalc[1]="restricted";

  printf("Running %s %s calculation with Rmax=%e and %i elements.\n",rcalc[restr].c_str(),method.c_str(),Rmax,Nelem);

  // Get primitive basis
  polynomial_basis::PolynomialBasis *poly(polynomial_basis::get_basis(primbas,Nnodes));

  if(Nquad==0)
    // Set default value
    Nquad=5*poly->get_nbf();
  else if(Nquad<2*poly->get_nbf())
    throw std::logic_error("Insufficient radial quadrature.\n");

  // Total number of electrons is
  arma::sword numel=Z-Q;

  // Functional
  int x_func, c_func;
  ::parse_xc_func(x_func, c_func, method);
  ::print_info(x_func, c_func);

  if(is_range_separated(x_func))
    throw std::logic_error("Range separated functionals are not supported in the spherically symmetric program.\n");
  {
    bool gga, mgga_t, mgga_l;
    if(x_func>0) {
      is_gga_mgga(x_func,  gga, mgga_t, mgga_l);
      if(mgga_t || mgga_l)
        throw std::logic_error("Meta-GGA functionals are not supported in the spherically symmetric program.\n");
    }
    if(c_func>0) {
      is_gga_mgga(c_func,  gga, mgga_t, mgga_l);
      if(mgga_t || mgga_l)
        throw std::logic_error("Meta-GGA functionals are not supported in the spherically symmetric program.\n");
    }
  }

  // Fraction of exact exchange
  double kfrac(exact_exchange(x_func));
  if(kfrac!=0.0)
    throw std::logic_error("Hybrid functionals are not supported in the spherically symmetric program.\n");

  int xp_func, cp_func;
  ::parse_xc_func(xp_func, cp_func, potmethod);

  // Initialize solver
  sadatom::solver::SCFSolver solver(Z, lmax, poly, Nquad, Nelem, Rmax, igrid, zexp, x_func, c_func, maxit, shift, convthr, dftthr, diiseps, diisthr, diisorder);

  if(helfem::utils::stricmp(occstr,"auto")==0) {
    // Initialize with a sensible guess occupation
    sadatom::solver::rconf_t initial;
    initial.orbs=sadatom::solver::OrbitalChannel(true);
    solver.Initialize(initial.orbs);
    initial.orbs.SetOccs(initial_occs(Z,lmax));
    if(initial.orbs.Nel()) {
      initial.Econf=solver.Solve(initial);
    } else {
      initial.Econf=0.0;
    }

    if(restr) {
      // List of configurations
      std::vector<sadatom::solver::rconf_t> rlist;

      // Restricted calculation
      sadatom::solver::rconf_t conf(initial);
      rlist.push_back(conf);

      // Brute force search for the lowest state
      while(true) {
        // Find the lowest energy configuration
        std::sort(rlist.begin(),rlist.end());

        // Do we have an Aufbau ground state?
        conf.orbs=rlist[0].orbs;
        conf.orbs.AufbauOccupations(numel);
        while(std::find(rlist.begin(), rlist.end(), conf) == rlist.end()) {
          conf.Econf=solver.Solve(conf);
          rlist.push_back(conf);
          conf.orbs.AufbauOccupations(numel);
        }
        printf("Aufbau search finished\n");

        // Find the lowest energy configuration
        std::sort(rlist.begin(),rlist.end());

        // Generate new configurations
        std::vector<sadatom::solver::OrbitalChannel> newconfs(rlist[0].orbs.MoveElectrons());

        bool newconf=false;
        for(size_t i=0;i<newconfs.size();i++) {
          conf.orbs=newconfs[i];
          if(std::find(rlist.begin(), rlist.end(), conf) == rlist.end()) {
            newconf=true;
            conf.Econf=solver.Solve(conf);
            rlist.push_back(conf);
          }
        }
        printf("Exhaustive search finished\n");
        if(!newconf) {
          break;
        }
      }

      // Print occupations
      printf("\nMinimal energy configurations for %s\n",element_symbols[Z].c_str());
      for(size_t i=0;i<rlist.size();i++) {
        arma::ivec occs(rlist[i].orbs.Occs());
        for(size_t j=0;j<occs.n_elem;j++)
          printf(" %2i",(int) occs(j));
        printf(" % .10f",rlist[i].Econf);
        if(i>0)
          printf(" %7.2f",(rlist[i].Econf-rlist[0].Econf)*HARTREEINEV);
        printf("\n");
      }

      // Print the minimal energy configuration
      printf("\nOccupations for lowest configuration\n");
      rlist[0].orbs.Occs().t().print();
      printf("Electronic configuration is\n");
      printf("%s\n",rlist[0].orbs.Characterize().c_str());

      // Get the potential
      solver.set_func(xp_func, cp_func);
      arma::mat pot(solver.RestrictedPotential(rlist[0]));

      std::ostringstream oss;
      oss << "result_" << element_symbols[Z] << ".dat";
      pot.save(oss.str(),arma::raw_ascii);

    } else {
      // List of configurations
      std::vector<sadatom::solver::uconf_t> totlist;

      // Restricted calculation
      sadatom::solver::uconf_t conf;
      conf.orbsa=initial.orbs;
      conf.orbsb=initial.orbs;
      conf.orbsa.SetRestricted(false);
      conf.orbsb.SetRestricted(false);

      // Difference in number of electrons
      for(int dx=0; dx <= 5; dx++) {
        arma::sword numelb=numel/2 - dx;
        arma::sword numela=numel-numelb;
        if(numelb<0)
          break;

        printf("\n ************ M = %i ************\n",(int) (numela-numelb+1));

        std::vector<sadatom::solver::uconf_t> ulist;
        conf.orbsa.AufbauOccupations(numela);
        conf.orbsb.AufbauOccupations(numelb);
        conf.Econf=solver.Solve(conf);
        ulist.push_back(conf);

        // Brute force search for the lowest state
        while(true) {
          // Find the lowest energy configuration
          std::sort(ulist.begin(),ulist.end());
          // Did we find an Aufbau ground state?
          conf.orbsa=ulist[0].orbsa;
          conf.orbsb=ulist[0].orbsb;
          conf.orbsa.AufbauOccupations(numela);
          conf.orbsb.AufbauOccupations(numelb);
          while(std::find(ulist.begin(), ulist.end(), conf) == ulist.end()) {
            conf.Econf=solver.Solve(conf);
            ulist.push_back(conf);
            // Did we find the Aufbau ground state?
            conf.orbsa.AufbauOccupations(numela);
            conf.orbsb.AufbauOccupations(numelb);
          }
          printf("Aufbau search finished\n");

          // Generate new configurations
          std::vector<sadatom::solver::OrbitalChannel> newconfa(ulist[0].orbsa.MoveElectrons());
          std::vector<sadatom::solver::OrbitalChannel> newconfb(ulist[0].orbsb.MoveElectrons());

          bool newconf=false;
          for(size_t i=0;i<newconfa.size();i++) {
            for(size_t j=0;j<newconfb.size();j++) {
              conf.orbsa=newconfa[i];
              conf.orbsb=newconfb[j];
              if(std::find(ulist.begin(), ulist.end(), conf) == ulist.end()) {
                newconf=true;
                conf.Econf=solver.Solve(conf);
                ulist.push_back(conf);
              }
            }
          }
          printf("Exhaustive search finished\n");
          if(!newconf) {
            break;
          }
        }

        // Add lowest state to collection
        //totlist.push_back(ulist[0]);
        totlist.insert(totlist.end(), ulist.begin(), ulist.end());
      }

      // Sort list
      std::sort(totlist.begin(), totlist.end());

      // Print occupations
      printf("\nMinimal energy spin states for %s\n",element_symbols[Z].c_str());
      for(size_t i=0;i<totlist.size();i++) {
        printf("%2i:",(int) (totlist[i].orbsa.Nel()-totlist[i].orbsb.Nel()+1));
        arma::ivec occa(totlist[i].orbsa.Occs());
        arma::ivec occb(totlist[i].orbsb.Occs());
        for(size_t j=0;j<occa.n_elem;j++)
          printf(" %2i",(int) occa(j));
        for(size_t j=0;j<occb.n_elem;j++)
          printf(" %2i",(int) occb(j));
        printf(" % .10f",totlist[i].Econf);
        if(i>0)
          printf(" %7.2f",(totlist[i].Econf-totlist[0].Econf)*HARTREEINEV);
        printf("\n");
      }

      // Print the minimal energy configuration
      printf("\nMinimum energy state is M = %i\n",(int) (totlist[0].orbsa.Nel()-totlist[0].orbsb.Nel()+1));
      printf("Occupations for lowest configuration\n");
      totlist[0].orbsa.Occs().t().print();
      totlist[0].orbsb.Occs().t().print();
      printf("Electronic configuration is\n");
      printf("alpha: %s\n",totlist[0].orbsa.Characterize().c_str());
      printf(" beta: %s\n",totlist[0].orbsb.Characterize().c_str());

      // Get the potential
      solver.set_func(xp_func, cp_func);
      arma::mat potU(solver.UnrestrictedPotential(totlist[0]));
      arma::mat potM(solver.AveragePotential(totlist[0]));
      arma::mat potW(solver.WeightedPotential(totlist[0]));

      std::ostringstream oss;

      oss.str("");
      oss << "resultU_" << element_symbols[Z] << ".dat";
      potU.save(oss.str(),arma::raw_ascii);

      oss.str("");
      oss << "resultM_" << element_symbols[Z] << ".dat";
      potM.save(oss.str(),arma::raw_ascii);

      oss.str("");
      oss << "resultW_" << element_symbols[Z] << ".dat";
      potW.save(oss.str(),arma::raw_ascii);
    }

  } else {
    arma::ivec occs;

    std::istringstream istream(occstr);
    occs.load(istream);
    // Check length
    arma::uword expected = restr ? lmax+1 : 2*(lmax+1);
    if(occs.n_elem != expected) {
      std::ostringstream oss;
      oss << "Invalid occupations: expected length " << expected << ", got " << occs.n_elem << ".\n";
      throw std::logic_error(oss.str());
    }

    if(restr) {
      sadatom::solver::rconf_t conf;
      conf.orbs=sadatom::solver::OrbitalChannel(restr);
      solver.Initialize(conf.orbs);
      conf.orbs.SetOccs(occs);
      solver.Solve(conf);

      // Print the minimal energy configuration
      printf("Electronic configuration is\n");
      printf("%s\n",conf.orbs.Characterize().c_str());

      // Get the potential
      solver.set_func(xp_func, cp_func);
      arma::mat pot(solver.RestrictedPotential(conf));

      std::ostringstream oss;
      oss << "result_" << element_symbols[Z] << ".dat";
      pot.save(oss.str(),arma::raw_ascii);

    } else {
      sadatom::solver::uconf_t conf;
      conf.orbsa=sadatom::solver::OrbitalChannel(restr);
      conf.orbsb=sadatom::solver::OrbitalChannel(restr);
      solver.Initialize(conf.orbsa);
      solver.Initialize(conf.orbsb);
      conf.orbsa.SetOccs(occs.subvec(0,2*lmax));
      conf.orbsa.SetOccs(occs.subvec(2*lmax+1,4*lmax+1));
      solver.Solve(conf);

      printf("Electronic configuration is\n");
      printf("alpha: %s\n",conf.orbsa.Characterize().c_str());
      printf(" beta: %s\n",conf.orbsb.Characterize().c_str());

      // Get the potential
      solver.set_func(xp_func, cp_func);
      arma::mat potU(solver.UnrestrictedPotential(conf));
      arma::mat potM(solver.AveragePotential(conf));
      arma::mat potW(solver.WeightedPotential(conf));

      std::ostringstream oss;

      oss.str("");
      oss << "resultU_" << element_symbols[Z] << ".dat";
      potU.save(oss.str(),arma::raw_ascii);

      oss.str("");
      oss << "resultM_" << element_symbols[Z] << ".dat";
      potM.save(oss.str(),arma::raw_ascii);

      oss.str("");
      oss << "resultW_" << element_symbols[Z] << ".dat";
      potW.save(oss.str(),arma::raw_ascii);

    }
  }

  return 0;
}
