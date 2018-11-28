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
#include "../general/diis.h"
#include "../general/dftfuncs.h"
#include "../general/elements.h"
#include "../general/gaunt.h"
#include "../general/scf_helpers.h"
#include "../general/polynomial_basis.h"
#include "basis.h"
#include "dftgrid.h"
#include "configurations.h"
#include <cfloat>

using namespace helfem;

typedef struct {
  // Primary quantum number
  int n;
  // Angular momentum
  int l;
  // Occupation
  int nocc;
  // Energy
  double E;
} occorb_t;

bool operator<(const occorb_t & lh, const occorb_t & rh) {
  return lh.E < rh.E;
}

void print(std::vector<occorb_t> occlist) {
  // Sort occupied orbitals
  std::sort(occlist.begin(),occlist.end());

  static const char shtype[]="spdf";
  printf("%2s %5s %12s\n","nl","occ","eigenvalue");
  for(size_t i=0;i<occlist.size();i++)
    printf("%i%c %5.3f % 12.6f\n",occlist[i].n,shtype[occlist[i].l],(double) occlist[i].nocc,occlist[i].E);
}

arma::mat supermat(const arma::mat & M, int lmax) {
  arma::mat S(M.n_rows*(lmax+1),M.n_cols*(lmax+1));
  S.zeros();
  for(int l=0;l<=lmax;l++) {
    S.submat(l*M.n_rows,l*M.n_cols,(l+1)*M.n_rows-1,(l+1)*M.n_cols-1)=M;
  }
  return S;
}

int main(int argc, char **argv) {
  cmdline::parser parser;

  // full option name, no short option, description, argument required
  parser.add<std::string>("Z", 0, "nuclear charge", true);
  parser.add<double>("Rmax", 0, "practical infinity in au", false, 40.0);
  parser.add<int>("grid", 0, "type of grid: 1 for linear, 2 for quadratic, 3 for polynomial, 4 for logarithmic", false, 4);
  parser.add<double>("zexp", 0, "parameter in radial grid", false, 2.0);
  parser.add<int>("nelem", 0, "number of elements", true);
  parser.add<int>("nnodes", 0, "number of nodes per element", false, 15);
  parser.add<int>("nquad", 0, "number of quadrature points", false, 0);
  parser.add<int>("maxit", 0, "maximum number of iterations", false, 50);
  parser.add<double>("convthr", 0, "convergence threshold", false, 1e-7);
  parser.add<std::string>("method", 0, "method to use", false, "HF");
  parser.add<double>("dftthr", 0, "density threshold for dft", false, 1e-12);
  parser.add<int>("restricted", 0, "spin-restricted orbitals", false, -1);
  parser.add<int>("primbas", 0, "primitive radial basis", false, 4);
  parser.add<double>("diiseps", 0, "when to start mixing in diis", false, 1e-2);
  parser.add<double>("diisthr", 0, "when to switch over fully to diis", false, 1e-3);
  parser.add<int>("diisorder", 0, "length of diis history", false, 5);
  parser.parse_check(argc, argv);

  // Get parameters
  double Rmax(parser.get<double>("Rmax"));
  int igrid(parser.get<int>("grid"));
  double zexp(parser.get<double>("zexp"));
  int maxit(parser.get<int>("maxit"));
  double convthr(parser.get<double>("convthr"));
  int restr(parser.get<int>("restricted"));
  int primbas(parser.get<int>("primbas"));
  // Number of elements
  int Nelem(parser.get<int>("nelem"));
  // Number of nodes
  int Nnodes(parser.get<int>("nnodes"));

  // Order of quadrature rule
  int Nquad(parser.get<int>("nquad"));
  double dftthr(parser.get<double>("dftthr"));

  // Nuclear charge
  int Z(get_Z(parser.get<std::string>("Z")));
  double diiseps=parser.get<double>("diiseps");
  double diisthr=parser.get<double>("diisthr");
  int diisorder=parser.get<int>("diisorder");

  std::string method(parser.get<std::string>("method"));

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

  // Get configuration
  std::vector<occ_t> conf(get_configuration(Z));

  // Determine lmax
  int lmax=0;
  for(size_t i=0;i<conf.size();i++)
    lmax=std::max(lmax,conf[i].first);

  // Count how many orbitals to occupy in each l channel
  arma::uvec locc(lmax+1);
  locc.zeros();
  for(size_t i=0;i<conf.size();i++)
    locc[conf[i].first]++;

  // Occupation of last orbital in each l channel
  arma::uvec lastocc(lmax+1);
  lastocc.zeros();
  for(size_t i=0;i<conf.size();i++)
    lastocc[conf[i].first]=conf[i].second;

  // Total number of electrons is
  int numel=0;
  for(size_t i=0;i<conf.size();i++)
    numel+=conf[i].second;

  sadatom::basis::TwoDBasis basis=sadatom::basis::TwoDBasis(Z, poly, Nquad, Nelem, Rmax, lmax, igrid, zexp);

  // Functional
  int x_func, c_func;
  ::parse_xc_func(x_func, c_func, method);
  ::print_info(x_func, c_func);

  if(is_range_separated(x_func))
    throw std::logic_error("Range separated functionals are not supported in the spherically symmetric program.\n");
  // Fraction of exact exchange
  double kfrac(exact_exchange(x_func));
  if(kfrac!=0.0)
    throw std::logic_error("Hybrid functionals are not supported in the spherically symmetric program.\n");

  // Form overlap matrix
  arma::mat S(basis.overlap());
  // Form kinetic energy matrix
  arma::mat T(basis.kinetic());
  // Form kinetic energy matrix
  arma::mat Tl(basis.kinetic_l());

  // Form DFT grid
  dftgrid::DFTGrid grid=helfem::dftgrid::DFTGrid(&basis);

  // Basis function norms
  arma::vec bfnorm(arma::pow(arma::diagvec(S),-0.5));
  
  // Get half-inverse
  arma::mat Sinvh(basis.Sinvh());
  // Form nuclear attraction energy matrix
  arma::mat Vnuc(basis.nuclear());
  // Form Hamiltonian
  arma::mat H0(T+Vnuc);

  S.save("Ssad.dat",arma::raw_ascii);
  T.save("Tsad.dat",arma::raw_ascii);
  Vnuc.save("Vsad.dat",arma::raw_ascii);
  
  // Compute two-electron integrals
  basis.compute_tei();

  // Kinetic energy l factors
  arma::vec lfac(lmax+1);
  for(int l=0;l<=lmax;l++)
    lfac(l)=l*(l+1);
  
  // Guess orbitals
  std::vector<arma::mat> C(lmax+1);
  std::vector<arma::vec> E(lmax+1);
  for(int l=0;l<=lmax;l++) {
    scf::eig_gsym(E[l],C[l],H0+lfac(l)*Tl,Sinvh);
  }

  double Ekin=0.0, Epot=0.0, Ecoul=0.0, Exc=0.0, Etot=0.0;
  double Eold=0.0;

  // S supermatrix
  bool usediis=true, useadiis=true;
  rDIIS diis(supermat(S,lmax),supermat(Sinvh,lmax),usediis,diiseps,diisthr,useadiis,true,diisorder);
  double diiserr;

  // Gaunt coefficients
  gaunt::Gaunt gaunt(0,lmax,lmax);
    
  // Density matrix
  arma::mat P;
  // l-dependent density matrices, needed for the additional kinetic energy term
  std::vector<arma::mat> Pl(lmax+1);
  // List of occupied orbitals
  std::vector<occorb_t> occlist;

  // Angular factor
  double angfac(4.0*M_PI);
  
  for(int i=1;i<=maxit;i++) {
    printf("\n**** Iteration %i ****\n\n",i);

    // Form radial density matrix
    P.zeros(C[0].n_rows,C[0].n_rows);
    occlist.clear();
    // Loop over l
    for(int l=0;l<=lmax;l++) {
      //double g=gaunt.coeff(0,0,l,0,l,0);
      double g=1.0;
      Pl[l].zeros(C[0].n_rows,C[0].n_rows);

      for(arma::uword iorb=0;iorb<locc(l);iorb++) {
        // Occupation is?
        int nocc = (iorb+1==locc(l)) ? lastocc(l) : 4*l+2;
        // Increment spherically averaged density matrix
        Pl[l] += g * nocc * C[l].col(iorb) * arma::trans(C[l].col(iorb));

        // Shell info
        occorb_t info;
        info.n=l+1+iorb;
        info.l=l;
        info.nocc=nocc;
        info.E=E[l](iorb);
        occlist.push_back(info);
      }
      // Increment full density matrix
      P+=Pl[l];
    }
    printf("Tr P = %f\n",arma::trace(P*S));
    fflush(stdout);

    // Print info
    print(occlist);

    Ekin=arma::trace(P*T);
    for(int l=0;l<=lmax;l++)
      Ekin+=lfac(l)*arma::trace(Pl[l]*Tl);
    
    Epot=arma::trace(P*Vnuc);

    // Form Coulomb matrix
    arma::mat J(basis.coulomb(P/angfac));
    Ecoul=0.5*arma::trace(P*J);
    printf("Coulomb energy %.10e\n",Ecoul);
    fflush(stdout);

    // Exchange-correlation
    Exc=0.0;
    arma::mat XC;
    double nelnum;
    grid.eval_Fxc(x_func, c_func, P/angfac, XC, Exc, nelnum, dftthr);
    // Potential needs to be divided as well
    XC/=angfac;
    printf("DFT energy %.10e\n",Exc);
    printf("Error in integrated number of electrons % e\n",nelnum-numel);
    fflush(stdout);

    // Fock matrices
    arma::mat F(H0+J+XC);

    // Update energy
    Etot=Ekin+Epot+Ecoul+Exc;
    double dE=Etot-Eold;

    printf("Total energy is % .10f\n",Etot);
    if(i>1)
      printf("Energy changed by %e\n",dE);
    Eold=Etot;
    fflush(stdout);

    /*
      S.print("S");
      T.print("T");
      Vnuc.print("Vnuc");
      Ca.print("Ca");
      Pa.print("Pa");
      J.print("J");
      Ka.print("Ka");

      arma::mat Jmo(Ca.t()*J*Ca);
      arma::mat Kmo(Ca.t()*Ka*Ca);
      Jmo.submat(0,0,10,10).print("Jmo");
      Kmo.submat(0,0,10,10).print("Kmo");


      Kmo+=Jmo;
      Kmo.print("Jmo+Kmo");

      Fa.print("Fa");
      arma::mat Fao(Sinvh.t()*Fa*Sinvh);
      Fao.print("Fao");
      Sinvh.print("Sinvh");
    */

    /*
      arma::mat Jmo(Ca.t()*J*Ca);
      arma::mat Kmo(Ca.t()*Ka*Ca);
      arma::mat Fmo(Ca.t()*Fa*Ca);
      Jmo=Jmo.submat(0,0,4,4);
      Kmo=Kmo.submat(0,0,4,4);
      Fmo=Fmo.submat(0,0,4,4);
      Jmo.print("J");
      Kmo.print("K");
      Fmo.print("F");
    */

    printf("%-21s energy: % .16f\n","Kinetic",Ekin);
    printf("%-21s energy: % .16f\n","Nuclear attraction",Epot);
    printf("%-21s energy: % .16f\n","Coulomb",Ecoul);
    printf("%-21s energy: % .16f\n","Exchange-correlation",Exc);
    printf("%-21s energy: % .16f\n","Total",Etot);
    printf("%-21s energy: % .16f\n","Virial ratio",-Etot/Ekin);
    printf("\n");

    // Since Fock operator depends on the l channel, we need to create
    // a supermatrix for DIIS.
    arma::mat Fsuper(supermat(F,lmax));
    for(int l=0;l<=lmax;l++) {
      Fsuper.submat(l*F.n_rows,l*F.n_cols,(l+1)*F.n_rows-1,(l+1)*F.n_cols-1)+=lfac(l)*Tl;
    }
    arma::mat Psuper(P.n_rows*(lmax+1),P.n_cols*(lmax+1));
    Psuper.zeros();
    for(int l=0;l<=lmax;l++) {
      Psuper.submat(l*P.n_rows,l*P.n_cols,(l+1)*P.n_rows-1,(l+1)*P.n_cols-1)=Pl[l];
    }
    // Update DIIS
    diis.update(Fsuper,Psuper,Etot,diiserr);
    printf("DIIS error is %e\n",diiserr);
    fflush(stdout);

    // Solve DIIS to get Fock update
    diis.solve_F(Fsuper);
    F=Fsuper.submat(0,0,F.n_rows-1,F.n_cols-1);

    // Have we converged? Note that DIIS error is still wrt full space, not active space.
    bool convd=(diiserr<convthr) && (std::abs(dE)<convthr);

    // Diagonalize Fock matrix to get new orbitals
    for(int l=0;l<=lmax;l++) {
      scf::eig_gsym(E[l],C[l],F+lfac(l)*Tl,Sinvh);
    }

    if(convd)
      break;
  }

  printf("%-21s energy: % .16f\n","Kinetic",Ekin);
  printf("%-21s energy: % .16f\n","Nuclear attraction",Epot);
  printf("%-21s energy: % .16f\n","Coulomb",Ecoul);
  printf("%-21s energy: % .16f\n","Exchange-correlation",Exc);
  printf("%-21s energy: % .16f\n","Total",Etot);
  printf("%-21s energy: % .16f\n","Virial ratio",-Etot/Ekin);
  printf("\n");

  // Electron density at nucleus
  printf("Electron density at nucleus % .10e\n",basis.nuclear_density(P));

  // Print info
  print(occlist);
  
  return 0;
}
