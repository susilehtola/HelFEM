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
#include "utils.h"
#include "dftgrid.h"
#include "solver.h"
#include "configurations.h"
#include <cfloat>
#include <cmath>

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

void hund_rule(const arma::ivec & occs, arma::ivec & occa, arma::ivec & occb) {
  occa.resize(occs.n_elem);
  occb.resize(occs.n_elem);
  // Loop over shells
  for(size_t l=0;l<occs.n_elem;l++) {
    // Number of electrons to distribute
    int numel(occs(l));
    while(numel>0) {
      // Populate shell
      int numsh = std::min(numel, (int) (2*(2*l+1)));
      int na = std::min(numsh, (int) (2*l+1));
      int nb = numsh-na;
      occa(l) += na;
      occb(l) += nb;
      numel-=numsh;
    }
  }
}

using namespace helfem;

sadatom::solver::OrbitalChannel restrict_configuration(const sadatom::solver::uconf_t & uconf) {
  sadatom::solver::OrbitalChannel helper(uconf.orbsa);
  helper.SetRestricted(true);
  helper.SetOccs(uconf.orbsa.Occs()+uconf.orbsb.Occs());

  return helper;
}

void unrestrict_occupations(const sadatom::solver::OrbitalChannel & orbs, sadatom::solver::uconf_t & conf) {
  // Update occupations
  arma::ivec occa, occb;
  hund_rule(orbs.Occs(),occa,occb);
  conf.orbsa.SetOccs(occa);
  conf.orbsb.SetOccs(occb);
}

arma::irowvec translate_occs(const arma::irowvec & occin) {
  arma::ivec occa, occb;
  hund_rule(occin.t(),occa,occb);
  arma::irowvec occs(occa.n_elem+occb.n_elem);
  occs.subvec(0,occa.n_elem-1)=occa.t();
  occs.subvec(occa.n_elem,occs.n_elem-1)=occb.t();
  occs.print("Used Hund's rules to translate occupations into");
  return occs;
}

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
  parser.add<int>("finitenuc", 0, "finite nuclear model", false, 0);
  parser.add<double>("Rrms", 0, "nuclear rms radius", false, 0.0);
  parser.add<int>("Q", 0, "charge of system", false, 0);
  parser.add<int>("lmax", 0, "maximum angular momentum to include", false, 3);
  parser.add<int>("nnodes", 0, "number of nodes per element", false, 15);
  parser.add<int>("nquad", 0, "number of quadrature points", false, 0);
  parser.add<int>("maxit", 0, "maximum number of iterations", false, 200);
  parser.add<double>("shift", 0, "level shift for initial SCF iterations", false, 1.0);
  parser.add<double>("convthr", 0, "convergence threshold", false, 1e-7);
  parser.add<std::string>("method", 0, "method to use", false, "lda_x");
  parser.add<std::string>("pot", 0, "method to use to compute potential", false, "none");
  parser.add<std::string>("occs", 0, "occupations to use", false, "auto");
  parser.add<double>("dftthr", 0, "density threshold for dft", false, 1e-12);
  parser.add<int>("iguess", 0, "guess: 0 for core, 1 for GSZ, 2 for SAP, 3 for TF", false, 2);
  parser.add<int>("restricted", 0, "spin-restricted orbitals", false, -1);
  parser.add<int>("primbas", 0, "primitive radial basis", false, 4);
  parser.add<double>("diiseps", 0, "when to start mixing in diis", false, 1e-2);
  parser.add<double>("diisthr", 0, "when to switch over fully to diis", false, 1e-3);
  parser.add<int>("diisorder", 0, "length of diis history", false, 10);
  parser.add<int>("taylor_order", 0, "order of Taylor expansion near the nucleus", false, -1);
  parser.add<bool>("saveorb", 0, "save radial orbitals to disk?", false, false);
  parser.add<bool>("savepot", 0, "save xc potential to disk?", false, false);
  parser.add<bool>("saveing", 0, "save xc ingredients to disk?", false, false);
  parser.add<bool>("zeroder", 0, "zero derivative at Rmax?", false, false);
  parser.add<std::string>("x_pars", 0, "file for parameters for exchange functional", false, "");
  parser.add<std::string>("c_pars", 0, "file for parameters for correlation functional", false, "");
  parser.add<double>("vdwthr", 0, "Density threshold for van der Waals radius", false, 0.001);
  parser.add<bool>("completeness", 0, "Compute completeness and importance profiles?", false, false);
  parser.add<int>("iconf", 0, "Confinement potential: 1 for polynomial, 2 for exponential", false, 0);
  parser.add<int>("conf_N", 0, "Exponent in polynomial confinement potential", false, 0);
  parser.add<double>("conf_R", 0, "Confinement radius", false, 0.0);
  parser.parse_check(argc, argv);
/*
  if(!parser.parse(argc, argv))
    throw std::logic_error("Error parsing arguments!\n");
*/

  // Get parameters
  double Rmax(parser.get<double>("Rmax"));
  int igrid(parser.get<int>("grid"));
  int igrid0(parser.get<int>("grid0"));
  double zexp(parser.get<double>("zexp"));
  double zexp0(parser.get<double>("zexp0"));
  int maxit(parser.get<int>("maxit"));
  int lmax(parser.get<int>("lmax"));
  double convthr(parser.get<double>("convthr"));
  int restr(parser.get<int>("restricted"));
  int primbas(parser.get<int>("primbas"));
  // Number of elements
  int Nelem(parser.get<int>("nelem"));
  int Nelem0(parser.get<int>("nelem0"));
  // Number of nodes
  int Nnodes(parser.get<int>("nnodes"));
  int taylor_order(parser.get<int>("taylor_order"));

  double shift(parser.get<double>("shift"));

  // Order of quadrature rule
  int Nquad(parser.get<int>("nquad"));
  double dftthr(parser.get<double>("dftthr"));

  int finitenuc(parser.get<int>("finitenuc"));
  double Rrms(parser.get<double>("Rrms"));

  // Nuclear charge
  int Q(parser.get<int>("Q"));
  int Z(get_Z(parser.get<std::string>("Z")));
  double diiseps=parser.get<double>("diiseps");
  double diisthr=parser.get<double>("diisthr");
  int diisorder=parser.get<int>("diisorder");
  int iguess(parser.get<int>("iguess"));

  double vdw_thr=parser.get<double>("vdwthr");

  std::string method(parser.get<std::string>("method"));
  std::string potmethod(parser.get<std::string>("pot"));
  std::string occstr(parser.get<std::string>("occs"));
  bool saveorb(parser.get<bool>("saveorb"));
  bool savepot(parser.get<bool>("savepot"));
  bool saveing(parser.get<bool>("saveing"));
  bool zeroder(parser.get<bool>("zeroder"));
  bool completeness(parser.get<bool>("completeness"));

  std::string xparf(parser.get<std::string>("x_pars"));
  std::string cparf(parser.get<std::string>("c_pars"));

  std::vector<std::string> rcalc(2);
  rcalc[0]="unrestricted";
  rcalc[1]="restricted";

  printf("Running %s %s calculation with Rmax=%e and %i elements.\n",rcalc[restr==1].c_str(),method.c_str(),Rmax,Nelem);

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

  // Total number of electrons is
  arma::sword numel=Z-Q;

  // Functional
  int x_func, c_func;
  ::parse_xc_func(x_func, c_func, method);
  ::print_info(x_func, c_func);
  if(!is_supported(x_func))
    throw std::logic_error("The specified exchange functional is not currently supported in HelFEM.\n");
  if(!is_supported(c_func))
    throw std::logic_error("The specified correlation functional is not currently supported in HelFEM.\n");

  // Potential
  int xp_func, cp_func;
  ::parse_xc_func(xp_func, cp_func, potmethod);
  {
    bool gga, mgga_t, mgga_l;
    if(x_func>0) {
      is_gga_mgga(xp_func,  gga, mgga_t, mgga_l);
      if(mgga_t || mgga_l)
        throw std::logic_error("Meta-GGA functionals are not supported in the spherically symmetric program.\n");
    }
    if(c_func>0) {
      is_gga_mgga(cp_func,  gga, mgga_t, mgga_l);
      if(mgga_t || mgga_l)
        throw std::logic_error("Meta-GGA functionals are not supported in the spherically symmetric program.\n");
    }

    double o, a, b;
    range_separation(xp_func, o, a, b);
    if(o!=0.0 || a!=0.0 || b!=0.0)
      throw std::logic_error("Optimized effective potential is not implemented in the spherically symmetric program.\n");
  }

  // Radial basis
  arma::vec bval=atomic::basis::form_grid((modelpotential::nuclear_model_t) finitenuc, Rrms, Nelem, Rmax, igrid, zexp, Nelem0, igrid0, zexp0, Z, 0, 0, 0.0);

  // Confinement parameters
  int iconf(parser.get<int>("iconf"));
  double conf_R(parser.get<double>("conf_R"));
  int conf_N(parser.get<int>("conf_N"));

  // Initialize solver
  sadatom::solver::SCFSolver solver(Z, finitenuc, Rrms, lmax, poly, zeroder, Nquad, bval, taylor_order, x_func, c_func, maxit, shift, convthr, dftthr, diiseps, diisthr, diisorder, iconf, conf_N, conf_R);

  // Set parameters if necessary
  arma::vec xpars, cpars;
  if(xparf.size()) {
    xpars = scf::parse_xc_params(xparf);
    xpars.t().print("Exchange functional parameters");
  }
  if(cparf.size()) {
    cpars = scf::parse_xc_params(cparf);
    cpars.t().print("Correlation functional parameters");
  }
  solver.set_params(xpars,cpars);

  // Final configuration (restricted case)
  helfem::sadatom::solver::rconf_t rconf;
  // Final configuration (unrestricted case)
  helfem::sadatom::solver::uconf_t uconf;

  if(helfem::utils::stricmp(occstr,"auto")==0) {
    // Initialize with a sensible guess occupation
    sadatom::solver::rconf_t initial;
    initial.orbs=sadatom::solver::OrbitalChannel(true);
    solver.Initialize(initial.orbs,iguess);
    initial.orbs.SetOccs(initial_occs(numel,lmax));
    if(initial.orbs.Nel()) {
      initial.Econf=solver.Solve(initial);
    } else {
      initial.Econf=0.0;
    }

    if(restr==1) {
      // List of configurations
      std::vector<sadatom::solver::rconf_t> rlist;

      // Restricted calculation
      sadatom::solver::rconf_t conf(initial);
      conf.Econf=solver.Solve(conf);
      if(Q!=0) {
        // Initial occupations are wrong for the state
        conf.orbs.AufbauOccupations(numel);
        conf.Econf=solver.Solve(conf);
      }
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
          printf(" %11.6f",(rlist[i].Econf-rlist[0].Econf)*HARTREEINEV);
        if(!rlist[i].converged)
          printf(" convergence failure");
        printf("\n");
      }

      // Save final configuration
      rconf=rlist[0];

    } else if(restr==-1) {
      // List of configurations
      std::vector<sadatom::solver::uconf_t> ulist;

      // Initial configuration
      arma::ivec inocc(initial_occs(numel,lmax));
      arma::ivec inocca, inoccb;
      hund_rule(inocc,inocca,inoccb);

      sadatom::solver::uconf_t conf;
      conf.orbsa=initial.orbs;
      conf.orbsb=initial.orbs;
      conf.orbsa.SetRestricted(false);
      conf.orbsb.SetRestricted(false);
      conf.orbsa.SetOccs(inocca);
      conf.orbsb.SetOccs(inoccb);
      solver.Solve(conf);
      if(Q!=0) {
        // Initial occupations are wrong for the state
        sadatom::solver::OrbitalChannel helper;
        helper=restrict_configuration(conf);
        helper.AufbauOccupations(numel);
        unrestrict_occupations(helper,conf);
        conf.Econf=solver.Solve(conf);
      }
      ulist.push_back(conf);

      // Brute force search for the lowest state
      while(true) {
        // Find the lowest energy configuration
        std::sort(ulist.begin(),ulist.end());

        // Form a helper
        sadatom::solver::OrbitalChannel helper;
        helper=restrict_configuration(ulist[0]);
        helper.AufbauOccupations(numel);
        unrestrict_occupations(helper,conf);

        // Do we have an Aufbau ground state? Test
        while(std::find(ulist.begin(), ulist.end(), conf) == ulist.end()) {
          conf.Econf=solver.Solve(conf);
          ulist.push_back(conf);

          // Update
          helper=restrict_configuration(ulist[0]);
          helper.AufbauOccupations(numel);
          unrestrict_occupations(helper,conf);
        }
        printf("Aufbau search finished\n");

        // Find the lowest energy configuration
        std::sort(ulist.begin(),ulist.end());

        // Generate new configurations
        helper=restrict_configuration(ulist[0]);
        std::vector<sadatom::solver::OrbitalChannel> newconfs(helper.MoveElectrons());

        bool newconf=false;
        for(size_t i=0;i<newconfs.size();i++) {
          unrestrict_occupations(newconfs[i],conf);
          if(std::find(ulist.begin(), ulist.end(), conf) == ulist.end()) {
            newconf=true;
            conf.Econf=solver.Solve(conf);
            ulist.push_back(conf);
          }
        }
        printf("Exhaustive search finished\n");
        if(!newconf) {
          break;
        }
      }

      // Print occupations
      printf("\nMinimal energy configurations for %s\n",element_symbols[Z].c_str());
      for(size_t i=0;i<ulist.size();i++) {
        arma::ivec occa(ulist[i].orbsa.Occs());
        arma::ivec occb(ulist[i].orbsb.Occs());
        printf("%2i:",(int) (ulist[i].orbsa.Nel()-ulist[i].orbsb.Nel()+1));
        for(size_t j=0;j<occa.n_elem;j++)
          printf(" %2i",(int) occa(j));
        for(size_t j=0;j<occb.n_elem;j++)
          printf(" %2i",(int) occb(j));
        printf(" % .10f",ulist[i].Econf);
        if(i>0)
          printf(" %11.6f",(ulist[i].Econf-ulist[0].Econf)*HARTREEINEV);
        if(!ulist[i].converged)
          printf(" convergence failure");
        printf("\n");
      }

      // Save final configuration
      uconf=ulist[0];

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
        if(!totlist[i].converged)
          printf(" convergence failure");
        printf("\n");
      }

      // Print the minimal energy configuration
      printf("\nMinimum energy state is M = %i\n",(int) (totlist[0].orbsa.Nel()-totlist[0].orbsb.Nel()+1));

      // Save final configuration
      uconf=totlist[0];
    }

  } else {
    arma::irowvec occs;

    if(helfem::utils::stricmp(occstr,"hf")==0) {
      arma::irowvec hfoccs(helfem::sadatom::get_configuration(Z).t());
      if(hfoccs.n_elem != (arma::uword) (lmax+1))
        throw std::logic_error("Run with lmax=3 for HF occupation mode.\n");
      hfoccs.print("Saito 2009 table's occupation for "+element_symbols[Z]);

      // restr=0 and restr=-1 work the same way in this case
      if(restr==-1)
        restr=0;

      if(restr) {
        occs=hfoccs;
      } else {
        // Use Hund's rule to determine occupations
        occs=translate_occs(hfoccs);
      }

    } else {
      std::istringstream istream(occstr);
      arma::irowvec inocc;
      inocc.load(istream);
      // Check length
      arma::uword expected = (std::abs(restr)==1) ? lmax+1 : 2*(lmax+1);
      if(inocc.n_elem != expected) {
        std::ostringstream oss;
        oss << "Invalid occupations: expected length " << expected << ", got " << inocc.n_elem << ".\n";
        throw std::logic_error(oss.str());
      }

      // Use Hund's rule to determine occupations
      if(restr == -1) {
        occs=translate_occs(inocc);
        if(restr==-1)
          // Switch to unrestricted mode
          restr=0;
      } else
        occs=inocc;
    }

    // Verbose operation for single configuration
    solver.set_verbose(true);
    if(restr) {
      rconf.orbs=sadatom::solver::OrbitalChannel(true);
      solver.Initialize(rconf.orbs,iguess);
      rconf.orbs.SetOccs(occs.t());
      solver.Solve(rconf);

    } else {
      uconf.orbsa=sadatom::solver::OrbitalChannel(false);
      uconf.orbsb=sadatom::solver::OrbitalChannel(false);
      solver.Initialize(uconf.orbsa,iguess);
      solver.Initialize(uconf.orbsb,iguess);
      uconf.orbsa.SetOccs(occs.subvec(0,lmax).t());
      uconf.orbsb.SetOccs(occs.subvec(lmax+1,2*lmax+1).t());
      solver.Solve(uconf);
    }
  }

  if(restr==1) {
    // Print the minimal energy configuration
    printf("\nOccupations for wanted configuration\n");
    rconf.orbs.Occs().t().print();
    printf("Electronic configuration is\n");
    printf("%s\n",rconf.orbs.Characterize().c_str());

    double nucd=solver.nuclear_density(rconf);
    double gnucd=solver.nuclear_density_gradient(rconf);
    printf("\nElectron density          at the nucleus is % e\n",nucd);
    printf("Electron density gradient at the nucleus is % e\n",gnucd);
    printf("Cusp condition is %.10f\n",-1.0/(2*Z)*gnucd/nucd);

    double rvdw(solver.vdw_radius(rconf,vdw_thr,false));
    double rvdwr2(solver.vdw_radius(rconf,vdw_thr,true));
    printf("\nEstimated vdW radius with density threshold %e is %.2f bohr = %.2f Å\n",vdw_thr,rvdw,rvdw*BOHRINANGSTROM);
    printf("vdW radius including r^2 factor is %.2f bohr = %.2f Å\n",rvdwr2,rvdwr2*BOHRINANGSTROM);

    printf("\nResult in NIST format\n");
    printf("Etot  = % 18.9f\n",rconf.Econf);
    printf("Ekin  = % 18.9f\n",rconf.Ekin);
    printf("Ecoul = % 18.9f\n",rconf.Ecoul);
    printf("Eenuc = % 18.9f\n",rconf.Epot);
    printf("Econf = % 18.9f\n",rconf.Econfinement);
    printf("Exc   = % 18.9f\n",rconf.Exc);
    rconf.orbs.Print(solver.Basis());
    (HARTREEINEV*rconf.orbs.GetGap()).t().print("HOMO-LUMO gap (eV)");

    if(completeness) {
      solver.gto_completeness_profile();
      printf("Evaluated GTO completeness profile\n");
      solver.sto_completeness_profile();
      printf("Evaluated STO completeness profile\n");
      solver.gto_importance_profile(rconf);
      printf("Evaluated GTO importance profile\n");
      solver.sto_importance_profile(rconf);
      printf("Evaluated STO importance profile\n");
    }

    // Evaluate xc ingredients
    if(saveing) {
      arma::mat ing(solver.XCIngredients(rconf));
      ing.save("xcing.dat",arma::raw_ascii);
    }
    // Evaluate the XC potential
    if(savepot) {
      arma::mat pot(solver.XCPotential(rconf));
      pot.save("xcpot.dat",arma::raw_ascii);
    }

    // Get the effective potential
    if(xp_func > 0 || cp_func > 0) {
      solver.set_func(xp_func, cp_func);
      arma::mat pot(solver.RestrictedPotential(rconf));

      std::ostringstream oss;
      oss << "result_" << element_symbols[Z] << ".dat";
      pot.save(oss.str(),arma::raw_ascii);
    }

    // Save the orbitals
    if(saveorb) {
      rconf.orbs.Save(solver.Basis(), element_symbols[Z]);
    }

    // Evaluate HF energy
    if(c_func != 0 || x_func != -1) {
      solver.set_func(-1, 0);
      printf("\nHartree-Fock energy is % .10f\n",solver.FockBuild(rconf));
    }
  } else {

    printf("Electronic configuration is\n");
    printf("alpha: %s\n",uconf.orbsa.Characterize().c_str());
    printf(" beta: %s\n",uconf.orbsb.Characterize().c_str());

    double nucd=solver.nuclear_density(uconf);
    double gnucd=solver.nuclear_density_gradient(uconf);
    printf("\nElectron density          at the nucleus is % e\n",nucd);
    printf("Electron density gradient at the nucleus is % e\n",gnucd);
    printf("Cusp condition is %.10f\n",-1.0/(2*Z)*gnucd/nucd);

    double rvdw(solver.vdw_radius(uconf,vdw_thr,false));
    double rvdwr2(solver.vdw_radius(uconf,vdw_thr,true));
    printf("\nEstimated vdW radius with density threshold %e is %.2f bohr = %.2f Å\n",vdw_thr,rvdw,rvdw*BOHRINANGSTROM);
    printf("vdW radius including r^2 factor is %.2f bohr = %.2f Å\n",rvdwr2,rvdwr2*BOHRINANGSTROM);

    printf("\nResult in NIST format\n");
    printf("Etot  = % 18.9f\n",uconf.Econf);
    printf("Ekin  = % 18.9f\n",uconf.Ekin);
    printf("Ecoul = % 18.9f\n",uconf.Ecoul);
    printf("Eenuc = % 18.9f\n",uconf.Epot);
    printf("Econf = % 18.9f\n",uconf.Econfinement);
    printf("Exc   = % 18.9f\n",uconf.Exc);
    printf("Alpha orbitals\n");
    uconf.orbsa.Print(solver.Basis());
    (HARTREEINEV*uconf.orbsa.GetGap()).t().print("Alpha HOMO-LUMO gap (eV)");
    printf("Beta  orbitals\n");
    uconf.orbsb.Print(solver.Basis());
    (HARTREEINEV*uconf.orbsb.GetGap()).t().print("Beta  HOMO-LUMO gap (eV)");

    // Evaluate xc ingredients
    if(saveing) {
      arma::mat ing(solver.XCIngredients(uconf));
      ing.save("xcing.dat",arma::raw_ascii);
    }
    // Evaluate the XC potential
    if(savepot) {
      arma::mat pot(solver.XCPotential(uconf));
      pot.save("xcpot.dat",arma::raw_ascii);
    }

    // Get the potential
    if(xp_func > 0 || cp_func > 0) {
      solver.set_func(xp_func, cp_func);
      arma::mat potU(solver.UnrestrictedPotential(uconf));
      arma::mat potM(solver.AveragePotential(uconf));
      arma::mat potW(solver.WeightedPotential(uconf));
      arma::mat potS(solver.HighSpinPotential(uconf));
      arma::mat pots(solver.LowSpinPotential(uconf));

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

      oss.str("");
      oss << "resultS_" << element_symbols[Z] << ".dat";
      potS.save(oss.str(),arma::raw_ascii);

      oss.str("");
      oss << "results_" << element_symbols[Z] << ".dat";
      pots.save(oss.str(),arma::raw_ascii);
    }

    // Save the orbitals
    if(saveorb) {
      uconf.orbsa.Save(solver.Basis(), element_symbols[Z] + "_alpha");
      uconf.orbsb.Save(solver.Basis(), element_symbols[Z] + "_beta");
    }

    // Evaluate HF energy
    if(c_func != 0 || x_func != -1) {
      solver.set_func(-1, 0);
      printf("\nHartree-Fock energy is % .10f\n",solver.FockBuild(uconf));
    }
  }

  return 0;
}
