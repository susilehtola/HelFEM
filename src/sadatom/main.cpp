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
#include "../general/scf_helpers.h"
#include "../general/polynomial_basis.h"
#include "basis.h"
#include "dftgrid.h"
#include <cfloat>

// Just use spdf for everything since this is so cheap
#define LMAX 3

using namespace helfem;

typedef struct {
  // Primary quantum number
  arma::sword n;
  // Angular momentum
  arma::sword l;
  // Occupation
  double nocc;
  // Energy
  double E;
} aufbau_t;

bool operator<(const aufbau_t & lh, const aufbau_t & rh) {
  return lh.E < rh.E;
}

// Structure for finding out the ground-state occupation
typedef struct {
  // Number of electrons on s, p, d, and f shells
  arma::ivec occs;
  // Energy of configuration
  double E;
  // Orbitals
  std::vector<arma::mat> C;
  // Orbitals of configuration
  std::vector<aufbau_t> orbs;
  // Coulomb&exchange potential
  arma::mat Zeff;
  // Coulomb potential
  arma::vec Vcoul;
  // Exchange potential
  arma::vec Vexch;
  // Electron density
  arma::vec rho;
  // Weights
  arma::vec wt;
} occmap_t;

// Comparison operator for sort
bool operator<(const occmap_t & lh, const occmap_t & rh) {
  return lh.E < rh.E;
}
// Comparison
bool operator==(const arma::ivec & occs, const occmap_t & rh) {
  if(occs.n_elem != rh.occs.n_elem)
    return false;

  bool match=true;
  for(size_t j=0;j<rh.occs.n_elem;j++)
    if(occs(j) != rh.occs(j))
      match=false;

  return match;
}
// Is on list?
bool in_list(const arma::ivec & occs, const std::vector<occmap_t> & rh) {
  for(size_t i=0;i<rh.size();i++)
    if(occs == rh[i])
      return true;
  return false;
}


std::vector<aufbau_t> form_orbital_list(const std::vector<arma::vec> & E) {
  std::vector<aufbau_t> orblist;
  for(size_t l=0;l<E.size();l++)
    for(arma::uword iorb=0;iorb<E[l].n_elem;iorb++) {
      aufbau_t tmp;
      tmp.n=iorb+1;
      tmp.l=l;
      tmp.nocc=0;
      tmp.E=E[l](iorb);
      orblist.push_back(tmp);
    }

  std::sort(orblist.begin(),orblist.end());

  return orblist;
}

arma::vec find_occupations(std::vector<aufbau_t> & orblist, arma::ivec nelocc) {
  // Distribute electrons
  arma::vec occs(orblist.size());
  for(size_t io=0;io<orblist.size();io++) {
    // Occupation is?
    arma::sword nocc = 0;
    if(nelocc(orblist[io].l)) {
      nocc = std::min((arma::sword) 4*orblist[io].l+2, (arma::sword) nelocc(orblist[io].l));
      nelocc(orblist[io].l) -= nocc;
    }
    occs[io] = nocc;
  }

  return occs;
}

arma::vec focc(const arma::vec & E, double mu, double B) {
  if(!E.size())
    throw std::logic_error("Can't do Fermi occupations without orbital energies!\n");

  arma::vec focc(E.n_elem);
  for(size_t i=0;i<E.n_elem;i++)
    focc(i)=1.0/(1.0 + exp(B*(E(i)-mu)));
  return focc;
}

arma::vec get_fermi_occupations(std::vector<aufbau_t> & orblist, double mu, double B) {
  // Get orbital energies
  arma::vec E(orblist.size()), nel(orblist.size());
  for(size_t io=0;io<orblist.size();io++) {
    E(io)=orblist[io].E;
    nel(io)=4*orblist[io].l+2;
  }
  // Occupation numbers are
  return nel%focc(E,mu,B);
}

arma::vec fermi_occupations(std::vector<aufbau_t> & orblist, double T, double N) {
  // Temperature factor: 1/(kB T)
  const double B(1.0/T);

  arma::vec occ;
  double occsum;

  double Eleft(orblist[0].E);
  double Eright(orblist[orblist.size()-1].E);

  // Check that limiting values are okay
  while(arma::sum(get_fermi_occupations(orblist,Eleft,B))>N) {
    Eleft=-2*std::abs(Eleft);
  }
  while(arma::sum(get_fermi_occupations(orblist,Eright,B))<N) {
    Eright=2*std::abs(Eright);
  }

  // Iterate
  for(size_t it=1;it<100;it++) {
    double Efermi((Eleft+Eright)/2);
    occ=get_fermi_occupations(orblist,Efermi,B);
    occsum=arma::sum(occ);

    //printf("it = %i, Efermi = %e, occsum = %e\n",(int)it,Efermi,occsum);

    if(occsum>N) {
      // Chemical potential is too large, move it to the left
      Eright=Efermi;
    } else if(occsum<N) {
      // Chemical potential is too small, move it to the right
      Eleft=Efermi;
    }

    if(std::abs(occsum-N)<=10*DBL_EPSILON*N)
      break;
  }

  //printf("N = %e, sum(occ)-N = %e\n",N,occsum-N);

  // Rescale occupation numbers
  return N*occ/occsum;
}

arma::ivec aufbau_occupations(std::vector<aufbau_t> & orblist, arma::sword numel) {
  // Returned occupations
  arma::ivec nelocc;
  nelocc.zeros(LMAX+1);

  // Distribute electrons
  arma::sword noccd = 0;
  for(size_t io=0;io<orblist.size();io++) {
    // Occupation is?
    arma::sword nocc = std::min(4*orblist[io].l+2, numel-noccd);
    orblist[io].nocc = nocc;
    // Increment count
    nelocc(orblist[io].l) += nocc;

    // Bookkeeping
    noccd += nocc;
    if(orblist[io].nocc == 0)
      break;
  }

  return nelocc;
}

void print(const std::vector<aufbau_t> & orblist) {
  static const char shtype[]="spdfgh";
  printf("%3s %6s %12s\n","nl","occ","eigenvalue");
  for(size_t i=0;i<orblist.size();i++) {
    if(orblist[i].nocc == 0 && orblist[i].E > 0)
      continue;
    printf("%2i%c %6.3f % 12.6f\n",(int) (orblist[i].n+orblist[i].l),shtype[orblist[i].l],(double) orblist[i].nocc,orblist[i].E);
  }
}

void print_config(const std::vector<aufbau_t> & orblist) {
  static const char shtype[]="spdfgh";
  for(size_t i=0;i<orblist.size();i++) {
    if(orblist[i].nocc == 0)
      continue;
    printf(" %i%c^{%i}",(int) (orblist[i].n+orblist[i].l),shtype[orblist[i].l],(int) orblist[i].nocc);
  }
  printf("\n");
}

arma::mat supermat(const arma::mat & M) {
  arma::mat S(M.n_rows*(LMAX+1),M.n_cols*(LMAX+1));
  S.zeros();
  for(int l=0;l<=LMAX;l++) {
    S.submat(l*M.n_rows,l*M.n_cols,(l+1)*M.n_rows-1,(l+1)*M.n_cols-1)=M;
  }
  return S;
}

void print_orb(const sadatom::basis::TwoDBasis & basis, const std::vector<arma::mat> & C, const std::vector<aufbau_t> & orblist, const std::string & symbol) {
  const char orbtypes[]="spdfgh";

  for(size_t io=0;io<orblist.size();io++) {
    if(orblist[io].nocc==0)
      continue;

    int l = orblist[io].l;
    int n = orblist[io].n;
    // Orbital
    arma::mat Clt(arma::trans(C[l].col(n-1)));

    std::ostringstream oss;
    oss << symbol << "_" << n+l << orbtypes[l] << ".dat";
    FILE *out = fopen(oss.str().c_str(),"w");

    // Loop over elements
    for(size_t iel=0;iel<basis.get_rad_Nel();iel++) {
      arma::vec r(basis.get_r(iel));
      arma::mat bf(basis.eval_bf(iel));
      arma::uvec bf_idx(basis.bf_list(iel));
      arma::vec orbval(bf*arma::trans(Clt.cols(bf_idx)));

      for(size_t ir=0;ir<orbval.n_rows;ir++) {
        fprintf(out,"%e % e\n",r(ir),orbval(ir));
      }
    }

    fclose(out);
  }
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
  parser.add<int>("maxit", 0, "maximum number of iterations", false, 500);
  parser.add<double>("convthr", 0, "convergence threshold", false, 1e-7);
  parser.add<std::string>("method", 0, "method to use", false, "lda_x");
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

  // Total number of electrons is
  arma::sword numel=Z;

  sadatom::basis::TwoDBasis basis=sadatom::basis::TwoDBasis(Z, poly, Nquad, Nelem, Rmax, LMAX, igrid, zexp);

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

  // Compute two-electron integrals
  basis.compute_tei();

  // Kinetic energy l factors
  arma::vec lfac(LMAX+1);
  for(arma::sword l=0;l<=LMAX;l++)
    lfac(l)=l*(l+1);

  // Guess orbitals
  std::vector<arma::mat> C(LMAX+1);
  std::vector<arma::vec> E(LMAX+1);
  for(arma::sword l=0;l<=LMAX;l++) {
    scf::eig_gsym(E[l],C[l],H0+lfac(l)*Tl,Sinvh);
  }
  // Form list of orbitals
  std::vector<aufbau_t> orblist=form_orbital_list(E);
  // Set occupations
  arma::ivec curocc=aufbau_occupations(orblist,numel);

  double Ekin=0.0, Epot=0.0, Ecoul=0.0, Exc=0.0, Etot=0.0;
  double Eold=0.0;

  // S supermatrix
  bool usediis=true, useadiis=true;
  rDIIS diis(supermat(S),supermat(Sinvh),usediis,diiseps,diisthr,useadiis,true,diisorder);
  double diiserr;

  // Density matrix
  arma::mat P, Pold;
  // l-dependent density matrices, needed for the additional kinetic energy term
  std::vector<arma::mat> Pl(LMAX+1), Plold(LMAX+1);

  // Angular factor
  double angfac(4.0*M_PI);

  // Map from occupations to energy
  std::vector<occmap_t> occmap;
  size_t ioccs=0;
  do {
    // Clean old history
    diis.clear();

    for(arma::sword iscf=1;iscf<=maxit;iscf++) {
      printf("\n**** Iteration %i ****\n\n",(int) iscf);

      // Form list of orbitals
      orblist=form_orbital_list(E);

      // Form radial density matrix
      if(iscf>1) {
        Plold=Pl;
        Pold=P;
      }
      for(arma::sword l=0;l<=LMAX;l++)
        Pl[l].zeros(C[0].n_rows,C[0].n_rows);

      // Find the occupied orbitals
      arma::vec nocc(find_occupations(orblist,curocc));
      //arma::vec nocc(fermi_occupations(orblist,1e-2,numel));
      for(size_t io=0;io<orblist.size();io++) {
        if(nocc(io)) {
          // Increment spherically averaged density matrix
          const arma::vec orbcoef(C[orblist[io].l].col(orblist[io].n-1));
          Pl[orblist[io].l] += nocc(io) * orbcoef * orbcoef.t();
          // Store orbital occupation
          orblist[io].nocc = nocc(io);
        }
      }
      // Full radial density
      P.zeros(C[0].n_rows,C[0].n_rows);
      for(arma::sword l=0;l<=LMAX;l++)
        P += Pl[l];
      printf("Tr P = %f\n",arma::trace(P*S));
      fflush(stdout);

      // Print info
      print(orblist);

      Ekin=arma::trace(P*T);
      for(arma::sword l=0;l<=LMAX;l++)
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
      if(iscf>1)
        printf("Energy changed by %e\n",dE);
      Eold=Etot;
      fflush(stdout);

      // Since Fock operator depends on the l channel, we need to create
      // a supermatrix for DIIS.
      arma::mat Fsuper(supermat(F));
      for(arma::sword l=0;l<=LMAX;l++) {
        Fsuper.submat(l*F.n_rows,l*F.n_cols,(l+1)*F.n_rows-1,(l+1)*F.n_cols-1)+=lfac(l)*Tl;
      }
      arma::mat Psuper(P.n_rows*(LMAX+1),P.n_cols*(LMAX+1));
      Psuper.zeros();
      for(arma::sword l=0;l<=LMAX;l++) {
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
      for(arma::sword l=0;l<=LMAX;l++) {
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

    // Potential
    arma::vec wt(basis.quadrature_weights());
    arma::mat Zeff(basis.coulomb_screening(P));
    arma::vec vcoul(Zeff.col(1));
    arma::vec vx(basis.exchange_screening(P));
    Zeff.col(1)+=vx;
    arma::mat rho(basis.electron_density(P));

    printf("Number of electrons by quadrature: %.16f\n",arma::sum(wt%arma::square(rho.col(0))%rho.col(1)));
    printf("Coulomb energy by quadrature: %.16f\n",0.5*arma::sum(wt%vcoul%rho.col(0)%rho.col(1)));

    // Print info
    print(orblist);

    // Add to map
    occmap_t tmp;
    tmp.occs=curocc;
    tmp.E=Etot;
    tmp.orbs=orblist;
    tmp.Vcoul=vcoul;
    tmp.Vexch=vx;
    tmp.Zeff=Zeff;
    tmp.rho=rho.col(1);
    tmp.wt=wt;
    tmp.C=C;
    occmap.push_back(tmp);

    // Switch densities
    bool aufbau=false;
    if(ioccs==0) {
      aufbau=true;
      curocc=aufbau_occupations(orblist, numel);

      // Do we already have the new occupations on the list?
      aufbau=!in_list(curocc, occmap);
    }

    // Check stability
    if(!aufbau) {
      // Sort occupations in increasing energy
      std::sort(occmap.begin(),occmap.end());

      // Move an electron from shell i to shell j
      bool moved=false;
      arma::ivec refoccs(occmap[0].occs);
      for(int shell_from=0;shell_from<=LMAX;shell_from++) {
        for(int shell_to=0;shell_to<=LMAX;shell_to++) {
          // Skip identity
          if(shell_from == shell_to)
            continue;

          // Check that we have electrons we can move
          if(! refoccs(shell_from))
            continue;
          // New occupations are
          arma::ivec newocc(refoccs);
          newocc(shell_from)--;
          newocc(shell_to)++;

          // Do we have this on the list?
          if(!in_list(newocc,occmap)) {
            moved=true;
            curocc=newocc;
            break;
          }
        }

        if(moved)
          break;
      }

      if(!moved)
        break;
    }
  } while(true);

  // Sort occupations in increasing energy
  std::sort(occmap.begin(),occmap.end());

  // Print occupations
  printf("\nMinimal energy configurations for %s\n",element_symbols[Z].c_str());
  for(size_t i=0;i<occmap.size();i++) {
    for(size_t j=0;j<occmap[i].occs.n_elem;j++)
      printf(" %2i",(int) occmap[i].occs(j));
    printf(" % .10f",occmap[i].E);
    if(i>0)
      printf(" %7.2f",(occmap[i].E-occmap[0].E)*HARTREEINEV);
    printf("\n");
  }

  // Print the minimal energy configuration
  printf("\nOccupations for lowest configuration\n");
  print(occmap[0].orbs);
  printf("Electronic configuration is\n");
  print_config(occmap[0].orbs);
  print_orb(basis,occmap[0].C,occmap[0].orbs,element_symbols[Z]);

  // Assemble the potential
  arma::mat result(occmap[0].Zeff.n_rows,6);
  result.col(0)=occmap[0].Zeff.col(0);
  result.col(1)=occmap[0].rho;
  result.col(2)=occmap[0].Vcoul;
  result.col(3)=occmap[0].Vexch;
  result.col(4)=occmap[0].Zeff.col(1);
  result.col(5)=occmap[0].wt;

  std::ostringstream oss;
  oss << "result_" << element_symbols[Z] << ".dat";
  result.save(oss.str(),arma::raw_ascii);

  printf("\n");
  result.print("r, n, vC, vX, Zeff, weight");

  return 0;
}
