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
// Shell types
static const char shtype[]="spdfgh";

using namespace helfem;

typedef struct {
  int n;
  int l;
  double E;
  int nocc;
} shell_occupation_t;

bool operator<(const shell_occupation_t & lh, const shell_occupation_t & rh) {
  return lh.E < rh.E;
}

arma::mat supermat(const arma::mat & M) {
  arma::mat S(M.n_rows*(LMAX+1),M.n_cols*(LMAX+1));
  S.zeros();
  for(int l=0;l<=LMAX;l++) {
    S.submat(l*M.n_rows,l*M.n_cols,(l+1)*M.n_rows-1,(l+1)*M.n_cols-1)=M;
  }
  return S;
}

class OrbitalChannel {
protected:
  // Orbitals and orbital energies
  std::vector<arma::mat> C;
  // Orbitals and orbital energies
  std::vector<arma::vec> E;
  // Orbital occupations
  arma::ivec occs;
  // Kinetic energy l factors
  arma::vec lfac;
  // Restricted occupations?
  bool restr;

  arma::sword shell_capacity(arma::sword l) const {
    return restr ? (4*l+2) : (2*l+1);
  }

public:
  OrbitalChannel() {
  }

  OrbitalChannel(bool restr_) : restr(restr_) {
    // Orbitals and orbital energies
    C.resize(LMAX+1);
    E.resize(LMAX+1);
    // Occupation numbers
    occs.zeros(LMAX+1);
    // Kinetic energy l factors
    lfac.zeros(LMAX+1);
    for(arma::sword l=0;l<=LMAX;l++)
      lfac(l)=l*(l+1);
  }
  ~OrbitalChannel() {
  }

  bool Restricted() const {
    return restr;
  }

  bool OrbitalsInitialized() const {
    for(size_t l=0;l<C.size();l++)
      if(!C[l].n_elem)
        return false;
    return true;
  }

  bool OccupationsInitialized() const {
    return arma::sum(occs) != 0;
  }

  OrbitalChannel Unrestrict() const {
    OrbitalChannel urestr(*this);
    urestr.restr=false;
    urestr.occs /= 2;

    return urestr;
  }

  arma::sword Nel() const {
    return arma::sum(occs);
  }

  arma::ivec Occs() const {
    return occs;
  }

  std::string Characterize() {
    // Form list of shells
    std::vector<shell_occupation_t> occlist;
    for(size_t l=0;l<=LMAX;l++) {
      // Number of electrons to put in
      arma::sword numl = occs(l);
      for(size_t io=0;io<C[l].n_cols;io++) {
        arma::sword nocc = std::min(shell_capacity(l), numl);
        numl-=nocc;
        if(nocc == 0)
          break;

        shell_occupation_t sh;
        sh.n = l+io+1;
        sh.l = l;
        sh.E = E[l](io);
        sh.nocc = nocc;
        occlist.push_back(sh);
      }
    }
    std::sort(occlist.begin(),occlist.end());

    std::ostringstream oss;
    for(size_t i=0;i<occlist.size();i++) {
      if(i)
        oss << " ";
      oss << occlist[i].n << shtype[occlist[i].l] << "^{" << occlist[i].nocc << "}";
    }

    return oss.str();
  }

  bool operator==(const OrbitalChannel & rh) const {
    if(occs.n_elem != rh.occs.n_elem)
      return false;
    for(size_t i=0;i<occs.n_elem;i++) {
      if(occs(i) != rh.occs(i))
        return false;
    }
    return true;
  }

  void UpdateOrbitals(const arma::mat & F, const arma::mat & Tl, const arma::mat & Sinvh) {
    for(size_t l=0;l<=LMAX;l++)
      scf::eig_gsym(E[l],C[l],F+lfac(l)*Tl,Sinvh);
  }

  void UpdateDensity(std::vector<arma::mat> & Pl) const {
    Pl.resize(LMAX+1);
    for(size_t l=0;l<=LMAX;l++) {
      Pl[l].zeros(C[l].n_rows,C[l].n_rows);

      // Number of electrons to put in
      arma::sword numl = occs(l);
      for(size_t io=0;io<C[l].n_cols;io++) {
        arma::sword nocc = std::min(shell_capacity(l), numl);
        numl -= nocc;
        Pl[l] += nocc * C[l].col(io) * C[l].col(io).t();
        if(nocc == 0)
          break;
      }
    }
  }

  void AufbauOccupations(arma::sword numel) {
    occs.zeros();

    // Number of radial solutions
    size_t Nrad=E[0].n_elem;

    // Collect energies
    arma::vec El(E.size()*Nrad);
    for(size_t l=0;l<E.size();l++)
      El.subvec(l*Nrad,(l+1)*Nrad-1)=E[l];
    arma::ivec lval(El.n_elem);
    for(size_t l=0;l<E.size();l++)
      lval.subvec(l*Nrad,(l+1)*Nrad-1)=l*arma::ones<arma::ivec>(Nrad);

    // Sort in increasing energy
    arma::uvec idx(arma::sort_index(El,"ascend"));
    El=El(idx);
    lval=lval(idx);

    // Fill in electrons to shells
    for(size_t i=0;i<El.n_elem;i++) {
      // Shell angular momentum is
      arma::sword l=lval(i);

      // Number of electrons to occupy shell with
      arma::sword nocc = std::min(shell_capacity(l), numel);
      occs(l) += nocc;
      numel -= nocc;

      if(numel == 0)
        break;
    }
  }

  std::vector<OrbitalChannel> MoveElectrons() const {
    std::vector<OrbitalChannel> ret;
    for(int shell_from=0;shell_from<=LMAX;shell_from++) {
      // Check that we have electrons we can move
      if(!occs(shell_from))
        continue;

      for(int shell_to=0;shell_to<=LMAX;shell_to++) {
        // We include the identity, since otherwise fully
        // spin-polarized calculations don't work (empty list for beta
        // moves)

        // New channel
        OrbitalChannel newch(*this);
        newch.occs(shell_from)--;
        newch.occs(shell_to)++;

        ret.push_back(newch);
      }
    }

    return ret;
  }
};

typedef struct {
  // Orbitals
  OrbitalChannel orbs;
  // Energy of configuration
  double Econf;
} rconf_t;
bool operator==(const rconf_t & lh, const rconf_t & rh) {
  return lh.orbs == rh.orbs;
}
bool operator<(const rconf_t & lh, const rconf_t & rh) {
  return lh.Econf < rh.Econf;
}

typedef struct {
  // Orbitals
  OrbitalChannel orbsa, orbsb;
  // Energy of configuration
  double Econf;
} uconf_t;
bool operator==(const uconf_t & lh, const uconf_t & rh) {
  return (lh.orbsa == rh.orbsa) && (lh.orbsb == rh.orbsb);
}
bool operator<(const uconf_t & lh, const uconf_t & rh) {
  return lh.Econf < rh.Econf;
}

class SCFSolver {
 protected:
  sadatom::basis::TwoDBasis basis;

  arma::mat S;
  arma::mat Sinvh;

  arma::mat T;
  arma::mat Tl;
  arma::mat Vnuc;
  arma::mat H0;

  dftgrid::DFTGrid grid;
  int x_func, c_func;

  int maxit;
  double convthr;
  double dftthr;
  double diiseps;
  double diisthr;
  int diisorder;

 public:
  SCFSolver(int Z, polynomial_basis::PolynomialBasis * poly, int Nquad, int Nelem, double Rmax, int igrid, double zexp, int x_func_, int c_func_, int maxit_, double convthr_, double dftthr_, double diiseps_, double diisthr_, int diisorder_) : x_func(x_func_), c_func(c_func_), maxit(maxit_), convthr(convthr_), dftthr(dftthr_), diiseps(diiseps_), diisthr(diisthr_), diisorder(diisorder_) {
    // Form basis
    basis=sadatom::basis::TwoDBasis(Z, poly, Nquad, Nelem, Rmax, LMAX, igrid, zexp);
    // Form overlap matrix
    S=basis.overlap();
    // Get half-inverse
    Sinvh=basis.Sinvh();
    // Form kinetic energy matrix
    T=basis.kinetic();
    // Form kinetic energy matrix
    Tl=basis.kinetic_l();
    // Form nuclear attraction energy matrix
    Vnuc=basis.nuclear();
    // Form core Hamiltonian
    H0=T+Vnuc;

    // Form DFT grid
    grid=helfem::dftgrid::DFTGrid(&basis);

    // Compute two-electron integrals
    basis.compute_tei();
  }

  ~SCFSolver() {
  }

  arma::mat TotalDensity(const std::vector<arma::mat> & Pl) {
    arma::mat P(Pl[0]);
    for(size_t l=1;l<Pl.size();l++)
      P+=Pl[l];
    return P;
  }

  void Initialize(OrbitalChannel & orbs) {
    orbs.UpdateOrbitals(H0,Tl,Sinvh);
  }

  double Solve(OrbitalChannel & orbs) {
    if(!orbs.OrbitalsInitialized())
      Initialize(orbs);
    if(!orbs.Restricted())
      throw std::logic_error("Running restricted calculation with unrestricted orbitals!\n");

    double Ekin=0.0, Epot=0.0, Ecoul=0.0, Exc=0.0, Etot=0.0;
    double Eold=0.0;

    bool verbose=false;

    // S supermatrix
    bool usediis=true, useadiis=true;
    rDIIS diis(supermat(S),supermat(Sinvh),usediis,diiseps,diisthr,useadiis,verbose,diisorder);
    double diiserr;

    // Density matrix
    arma::mat P;
    // l-dependent density matrices, needed for the additional kinetic energy term
    std::vector<arma::mat> Pl;

    // Angular factor
    double angfac(4.0*M_PI);
    arma::vec lfac(LMAX+1);
    for(arma::sword l=0;l<=LMAX;l++)
      lfac(l)=l*(l+1);

    for(arma::sword iscf=1;iscf<=maxit;iscf++) {
      if(verbose) {
        printf("\n**** Iteration %i ****\n\n",(int) iscf);
      }

      // Form new density
      orbs.UpdateDensity(Pl);
      P=TotalDensity(Pl);
      if(verbose) {
        printf("Tr P = %f\n",arma::trace(P*S));
        fflush(stdout);
      }

      // Compute energy
      Ekin=arma::trace(P*T);
      for(arma::sword l=0;l<=LMAX;l++)
        Ekin+=lfac(l)*arma::trace(Pl[l]*Tl);
      Epot=arma::trace(P*Vnuc);

      // Form Coulomb matrix
      arma::mat J(basis.coulomb(P/angfac));
      Ecoul=0.5*arma::trace(P*J);
      if(verbose) {
        printf("Coulomb energy %.10e\n",Ecoul);
        fflush(stdout);
      }

      // Exchange-correlation
      Exc=0.0;
      arma::mat XC;
      double nelnum;
      grid.eval_Fxc(x_func, c_func, P/angfac, XC, Exc, nelnum, dftthr);
      // Potential needs to be divided as well
      XC/=angfac;
      if(verbose) {
        printf("DFT energy %.10e\n",Exc);
        printf("Error in integrated number of electrons % e\n",nelnum-orbs.Nel());
        fflush(stdout);
      }

      // Fock matrices
      arma::mat F(H0+J+XC);

      // Update energy
      Etot=Ekin+Epot+Ecoul+Exc;
      double dE=Etot-Eold;

      if(verbose) {
        printf("Total energy is % .10f\n",Etot);
        if(iscf>1)
          printf("Energy changed by %e\n",dE);
      }
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
      if(verbose) {
        printf("DIIS error is %e\n",diiserr);
        fflush(stdout);
      }

      // Solve DIIS to get Fock update
      diis.solve_F(Fsuper);
      F=Fsuper.submat(0,0,F.n_rows-1,F.n_cols-1);

      // Have we converged? Note that DIIS error is still wrt full space, not active space.
      bool convd=(diiserr<convthr) && (std::abs(dE)<convthr);

      // Update orbitals
      orbs.UpdateOrbitals(F,Tl,Sinvh);

      if(convd)
        break;
    }

    if(verbose) {
      printf("%-21s energy: % .16f\n","Kinetic",Ekin);
      printf("%-21s energy: % .16f\n","Nuclear attraction",Epot);
      printf("%-21s energy: % .16f\n","Coulomb",Ecoul);
      printf("%-21s energy: % .16f\n","Exchange-correlation",Exc);
      printf("%-21s energy: % .16f\n","Total",Etot);
      printf("%-21s energy: % .16f\n","Virial ratio",-Etot/Ekin);
      printf("\n");

      // Electron density at nucleus
      printf("Electron density at nucleus % .10e\n",basis.nuclear_density(P));
    } else {
      printf("Evaluated energy % .16f for configuration ",Etot);

      arma::ivec occs(orbs.Occs());
      for(size_t i=0;i<occs.size();i++)
        printf(" %i",(int) occs(i));
      printf("\n");
    }

    return Etot;
  }

  double Solve(OrbitalChannel & orbsa, OrbitalChannel & orbsb) {
    if(!orbsa.OrbitalsInitialized())
      Initialize(orbsa);
    if(!orbsb.OrbitalsInitialized())
      Initialize(orbsb);

    if(orbsa.Restricted())
      throw std::logic_error("Running unrestricted calculation with restricted orbitals!\n");
    if(orbsb.Restricted())
      throw std::logic_error("Running unrestricted calculation with restricted orbitals!\n");

    double Ekin=0.0, Epot=0.0, Ecoul=0.0, Exc=0.0, Etot=0.0;
    double Eold=0.0;

    bool verbose = false;

    // S supermatrix
    bool combine=false, usediis=true, useadiis=true;
    uDIIS diis(supermat(S),supermat(Sinvh),combine, usediis,diiseps,diisthr,useadiis,verbose,diisorder);
    double diiserr;

    // Density matrix
    arma::mat Pa, Pb, P;
    // l-dependent density matrices, needed for the additional kinetic energy term
    std::vector<arma::mat> Pal(LMAX+1), Pbl(LMAX+1), Pl(LMAX+1);

    // Angular factor
    double angfac(4.0*M_PI);
    arma::vec lfac(LMAX+1);
    for(arma::sword l=0;l<=LMAX;l++)
      lfac(l)=l*(l+1);

    for(arma::sword iscf=1;iscf<=maxit;iscf++) {
      if(verbose) {
        printf("\n**** Iteration %i ****\n\n",(int) iscf);
      }

      orbsa.UpdateDensity(Pal);
      orbsb.UpdateDensity(Pbl);
      for(size_t l=0;l<Pal.size();l++)
        Pl[l]=Pal[l]+Pbl[l];
      Pa=TotalDensity(Pal);
      Pb=TotalDensity(Pbl);
      P=Pa+Pb;

      // Energy
      Ekin=arma::trace(P*T);
      for(arma::sword l=0;l<=LMAX;l++)
        Ekin+=lfac(l)*arma::trace(Pl[l]*Tl);
      Epot=arma::trace(P*Vnuc);

      // Form Coulomb matrix
      arma::mat J(basis.coulomb(P/angfac));
      Ecoul=0.5*arma::trace(P*J);
      if(verbose) {
        printf("Coulomb energy %.10e\n",Ecoul);
        fflush(stdout);
      }

      // Exchange-correlation
      Exc=0.0;
      arma::mat XCa, XCb;
      double nelnum;
      grid.eval_Fxc(x_func, c_func, Pa/angfac, Pb/angfac, XCa, XCb, Exc, nelnum, true, dftthr);
      // Potential needs to be divided as well
      XCa/=angfac;
      XCb/=angfac;
      if(verbose) {
        printf("DFT energy %.10e\n",Exc);
        printf("Error in integrated number of electrons % e\n",nelnum-orbsa.Nel()-orbsb.Nel());
        fflush(stdout);
      }

      // Fock matrices
      arma::mat Fa(H0+J+XCa);
      arma::mat Fb(H0+J+XCb);

      // Update energy
      Etot=Ekin+Epot+Ecoul+Exc;
      double dE=Etot-Eold;

      if(verbose) {
        printf("Total energy is % .10f\n",Etot);
        if(iscf>1)
          printf("Energy changed by %e\n",dE);
      }
      Eold=Etot;
      fflush(stdout);

      // Since Fock operator depends on the l channel, we need to create
      // a supermatrix for DIIS.
      arma::mat Fasuper(supermat(Fa)), Fbsuper(supermat(Fb));
      for(arma::sword l=0;l<=LMAX;l++) {
        Fasuper.submat(l*Fa.n_rows,l*Fa.n_cols,(l+1)*Fa.n_rows-1,(l+1)*Fa.n_cols-1)+=lfac(l)*Tl;
        Fbsuper.submat(l*Fb.n_rows,l*Fb.n_cols,(l+1)*Fb.n_rows-1,(l+1)*Fb.n_cols-1)+=lfac(l)*Tl;
      }
      arma::mat Pasuper(Pa.n_rows*(LMAX+1),Pa.n_cols*(LMAX+1)), Pbsuper(Pb.n_rows*(LMAX+1),Pb.n_cols*(LMAX+1));
      Pasuper.zeros();
      Pbsuper.zeros();
      for(arma::sword l=0;l<=LMAX;l++) {
        Pasuper.submat(l*Pa.n_rows,l*Pa.n_cols,(l+1)*Pa.n_rows-1,(l+1)*Pa.n_cols-1)=Pal[l];
        Pbsuper.submat(l*Pb.n_rows,l*Pb.n_cols,(l+1)*Pb.n_rows-1,(l+1)*Pb.n_cols-1)=Pbl[l];
      }
      // Update DIIS
      diis.update(Fasuper,Fbsuper,Pasuper,Pbsuper,Etot,diiserr);
      if(verbose) {
        printf("DIIS error is %e\n",diiserr);
        fflush(stdout);
      }

      // Solve DIIS to get Fock update
      diis.solve_F(Fasuper,Fbsuper);
      Fa=Fasuper.submat(0,0,Fa.n_rows-1,Fa.n_cols-1);
      Fb=Fbsuper.submat(0,0,Fb.n_rows-1,Fb.n_cols-1);

      // Have we converged? Note that DIIS error is still wrt full space, not active space.
      bool convd=(diiserr<convthr) && (std::abs(dE)<convthr);

      // Update orbitals
      orbsa.UpdateOrbitals(Fa,Tl,Sinvh);
      orbsb.UpdateOrbitals(Fb,Tl,Sinvh);

      if(convd)
        break;
    }

    if(verbose) {
      printf("%-21s energy: % .16f\n","Kinetic",Ekin);
      printf("%-21s energy: % .16f\n","Nuclear attraction",Epot);
      printf("%-21s energy: % .16f\n","Coulomb",Ecoul);
      printf("%-21s energy: % .16f\n","Exchange-correlation",Exc);
      printf("%-21s energy: % .16f\n","Total",Etot);
      printf("%-21s energy: % .16f\n","Virial ratio",-Etot/Ekin);
      printf("\n");

      // Electron density at nucleus
      printf("Electron density at nucleus % .10e\n",basis.nuclear_density(P));
    } else {
      printf("Evaluated energy % .16f for configuration ",Etot);

      arma::ivec occa(orbsa.Occs());
      for(size_t i=0;i<occa.size();i++)
        printf(" %i",(int) occa(i));
      arma::ivec occb(orbsb.Occs());
      for(size_t i=0;i<occb.size();i++)
        printf(" %i",(int) occb(i));
      printf("\n");
    }

    return Etot;
  }

  arma::mat RestrictedPotential(OrbitalChannel & orbs) {
    if(!orbs.OrbitalsInitialized())
      throw std::logic_error("No orbitals!\n");
    if(!orbs.Restricted())
      throw std::logic_error("Running restricted calculation with unrestricted orbitals!\n");

    std::vector<arma::mat> Pl;
    orbs.UpdateDensity(Pl);
    arma::mat P=TotalDensity(Pl);

    arma::vec wt(basis.quadrature_weights());
    arma::mat Zeff(basis.coulomb_screening(P));
    arma::vec vcoul(Zeff.col(1));
    arma::vec vx(basis.exchange_screening(P));
    Zeff.col(1)+=vx;
    arma::mat rho(basis.electron_density(P));

    arma::mat result(Zeff.n_rows,6);
    result.col(0)=Zeff.col(0);
    result.col(1)=rho.col(1);
    result.col(2)=vcoul;
    result.col(3)=vx;
    result.col(4)=Zeff.col(1);
    result.col(5)=wt;

    return result;
  }

  arma::mat UnrestrictedPotential(OrbitalChannel & orbsa, OrbitalChannel & orbsb) {
    if(!orbsa.OrbitalsInitialized())
      throw std::logic_error("No orbitals!\n");
    if(!orbsb.OrbitalsInitialized())
      throw std::logic_error("No orbitals!\n");
    if(orbsa.Restricted())
      throw std::logic_error("Running unrestricted calculation with restricted orbitals!\n");
    if(orbsb.Restricted())
      throw std::logic_error("Running unrestricted calculation with restricted orbitals!\n");

    std::vector<arma::mat> Pal, Pbl;
    orbsa.UpdateDensity(Pal);
    orbsb.UpdateDensity(Pbl);
    arma::mat Pa=TotalDensity(Pal);
    arma::mat Pb=TotalDensity(Pbl);
    arma::mat P(Pa+Pb);

    arma::vec wt(basis.quadrature_weights());
    arma::mat Zeff(basis.coulomb_screening(P));
    arma::vec vcoul(Zeff.col(1));
    arma::mat vxm(basis.exchange_screening(Pa,Pb));
    // Averaged potential
    arma::vec vx=arma::mean(vxm,1);
    Zeff.col(1)+=vx;
    arma::mat rho(basis.electron_density(P));

    arma::mat result(Zeff.n_rows,6);
    result.col(0)=Zeff.col(0);
    result.col(1)=rho.col(1);
    result.col(2)=vcoul;
    result.col(3)=vx;
    result.col(4)=Zeff.col(1);
    result.col(5)=wt;

    return result;
  }

  arma::mat AveragePotential(OrbitalChannel & orbsa, OrbitalChannel & orbsb) {
    if(!orbsa.OrbitalsInitialized())
      throw std::logic_error("No orbitals!\n");
    if(!orbsb.OrbitalsInitialized())
      throw std::logic_error("No orbitals!\n");
    if(orbsa.Restricted())
      throw std::logic_error("Running unrestricted calculation with restricted orbitals!\n");
    if(orbsb.Restricted())
      throw std::logic_error("Running unrestricted calculation with restricted orbitals!\n");

    std::vector<arma::mat> Pal, Pbl;
    orbsa.UpdateDensity(Pal);
    orbsb.UpdateDensity(Pbl);
    arma::mat Pa=TotalDensity(Pal);
    arma::mat Pb=TotalDensity(Pbl);
    arma::mat P(Pa+Pb);

    arma::vec wt(basis.quadrature_weights());
    arma::mat Zeff(basis.coulomb_screening(P));
    arma::vec vcoul(Zeff.col(1));
    arma::vec vx(basis.exchange_screening(P));
    Zeff.col(1)+=vx;
    arma::mat rho(basis.electron_density(P));

    arma::mat result(Zeff.n_rows,6);
    result.col(0)=Zeff.col(0);
    result.col(1)=rho.col(1);
    result.col(2)=vcoul;
    result.col(3)=vx;
    result.col(4)=Zeff.col(1);
    result.col(5)=wt;

    return result;
  }

  arma::mat WeightedPotential(OrbitalChannel & orbsa, OrbitalChannel & orbsb) {
    if(!orbsa.OrbitalsInitialized())
      throw std::logic_error("No orbitals!\n");
    if(!orbsb.OrbitalsInitialized())
      throw std::logic_error("No orbitals!\n");
    if(orbsa.Restricted())
      throw std::logic_error("Running unrestricted calculation with restricted orbitals!\n");
    if(orbsb.Restricted())
      throw std::logic_error("Running unrestricted calculation with restricted orbitals!\n");

    std::vector<arma::mat> Pal, Pbl;
    orbsa.UpdateDensity(Pal);
    orbsb.UpdateDensity(Pbl);
    arma::mat Pa=TotalDensity(Pal);
    arma::mat Pb=TotalDensity(Pbl);
    arma::mat P(Pa+Pb);

    arma::vec wt(basis.quadrature_weights());
    arma::mat Zeff(basis.coulomb_screening(P));
    arma::vec vcoul(Zeff.col(1));
    arma::mat vxm(basis.exchange_screening(Pa,Pb));
    arma::mat rhoa(basis.electron_density(Pa));
    arma::mat rhob(basis.electron_density(Pb));
    // Averaged potential
    arma::vec vx=(vxm.col(0)%rhoa.col(1) + vxm.col(1)%rhob.col(1))/(rhoa.col(1)+rhob.col(1));
    // Set areas of small electron density to zero
    arma::vec n(rhoa.col(1)+rhob.col(1));
    vx(arma::find(n<dftthr)).zeros();
    Zeff.col(1)+=vx;

    arma::mat result(Zeff.n_rows,6);
    result.col(0)=Zeff.col(0);
    result.col(1)=rhoa.col(1)+rhob.col(1);
    result.col(2)=vcoul;
    result.col(3)=vx;
    result.col(4)=Zeff.col(1);
    result.col(5)=wt;

    return result;
  }
};

int main(int argc, char **argv) {
  cmdline::parser parser;

  // full option name, no short option, description, argument required
  parser.add<std::string>("Z", 0, "nuclear charge", true);
  parser.add<double>("Rmax", 0, "practical infinity in au", false, 40.0);
  parser.add<int>("grid", 0, "type of grid: 1 for linear, 2 for quadratic, 3 for polynomial, 4 for exponential", false, 4);
  parser.add<double>("zexp", 0, "parameter in radial grid", false, 2.0);
  parser.add<int>("nelem", 0, "number of elements", true);
  parser.add<int>("Q", 0, "charge of system", false, 0);
  parser.add<int>("nnodes", 0, "number of nodes per element", false, 15);
  parser.add<int>("nquad", 0, "number of quadrature points", false, 0);
  parser.add<int>("maxit", 0, "maximum number of iterations", false, 50);
  parser.add<double>("convthr", 0, "convergence threshold", false, 1e-7);
  parser.add<std::string>("method", 0, "method to use", false, "lda_x");
  parser.add<double>("dftthr", 0, "density threshold for dft", false, 1e-12);
  parser.add<int>("restricted", 0, "spin-restricted orbitals", false, 0);
  parser.add<int>("primbas", 0, "primitive radial basis", false, 4);
  parser.add<double>("diiseps", 0, "when to start mixing in diis", false, 1e-2);
  parser.add<double>("diisthr", 0, "when to switch over fully to diis", false, 1e-3);
  parser.add<int>("diisorder", 0, "length of diis history", false, 5);
  if(!parser.parse(argc, argv))
    throw std::logic_error("Error parsing arguments!\n");

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
  int Q(parser.get<int>("Q"));
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

  // Initialize solver
  SCFSolver solver(Z, poly, Nquad, Nelem, Rmax, igrid, zexp, x_func, c_func, maxit, convthr, dftthr, diiseps, diisthr, diisorder);

  if(restr) {
    // List of configurations
    std::vector<rconf_t> rlist;

    // Restricted calculation
    rconf_t conf;
    conf.orbs=OrbitalChannel(true);
    solver.Initialize(conf.orbs);
    conf.orbs.AufbauOccupations(numel);
    conf.Econf=solver.Solve(conf.orbs);
    rlist.push_back(conf);

    // Did we find the Aufbau ground state?
    conf.orbs.AufbauOccupations(numel);
    while(std::find(rlist.begin(), rlist.end(), conf) == rlist.end()) {
      conf.Econf=solver.Solve(conf.orbs);
      rlist.push_back(conf);
      conf.orbs.AufbauOccupations(numel);
    }
    printf("Aufbau search finished\n");

    // Brute force search for the lowest state
    while(true) {
      // Find the lowest energy configuration
      std::sort(rlist.begin(),rlist.end());
      // Generate new configurations
      std::vector<OrbitalChannel> newconfs(rlist[0].orbs.MoveElectrons());

      bool newconf=false;
      for(size_t i=0;i<newconfs.size();i++) {
        conf.orbs=newconfs[i];
        if(std::find(rlist.begin(), rlist.end(), conf) == rlist.end()) {
          newconf=true;
          conf.Econf=solver.Solve(conf.orbs);
          rlist.push_back(conf);
        }
      }
      if(!newconf) {
        printf("Exhaustive search finished\n");
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
    arma::mat pot(solver.RestrictedPotential(rlist[0].orbs));

    std::ostringstream oss;
    oss << "result_" << element_symbols[Z] << ".dat";
    pot.save(oss.str(),arma::raw_ascii);

  } else {
    // List of configurations
    std::vector<uconf_t> totlist;

    // Restricted calculation
    uconf_t conf;
    conf.orbsa=OrbitalChannel(false);
    conf.orbsb=OrbitalChannel(false);
    solver.Initialize(conf.orbsa);
    solver.Initialize(conf.orbsb);

    // Difference in number of electrons
    for(int dx=0; dx <= 5; dx++) {
      arma::sword numelb=numel/2 - dx;
      arma::sword numela=numel-numelb;
      if(numelb<0)
        break;

      printf("\n ************ M = %i ************\n",(int) (numela-numelb+1));

      std::vector<uconf_t> ulist;
      conf.orbsa.AufbauOccupations(numela);
      conf.orbsb.AufbauOccupations(numelb);
      conf.Econf=solver.Solve(conf.orbsa, conf.orbsb);
      ulist.push_back(conf);

      // Did we find the Aufbau ground state?
      conf.orbsa.AufbauOccupations(numela);
      conf.orbsb.AufbauOccupations(numelb);
      while(std::find(ulist.begin(), ulist.end(), conf) == ulist.end()) {
        conf.Econf=solver.Solve(conf.orbsa, conf.orbsb);
        ulist.push_back(conf);
        conf.orbsa.AufbauOccupations(numela);
        conf.orbsb.AufbauOccupations(numelb);
      }
      printf("Aufbau search finished\n");

      // Brute force search for the lowest state
      while(true) {
        // Find the lowest energy configuration
        std::sort(ulist.begin(),ulist.end());
        // Generate new configurations
        std::vector<OrbitalChannel> newconfa(ulist[0].orbsa.MoveElectrons());
        std::vector<OrbitalChannel> newconfb(ulist[0].orbsb.MoveElectrons());

        bool newconf=false;
        for(size_t i=0;i<newconfa.size();i++) {
          for(size_t j=0;j<newconfb.size();j++) {
            conf.orbsa=newconfa[i];
            conf.orbsb=newconfb[j];
            if(std::find(ulist.begin(), ulist.end(), conf) == ulist.end()) {
              newconf=true;
              conf.Econf=solver.Solve(conf.orbsa, conf.orbsb);
              ulist.push_back(conf);
            }
          }
        }
        if(!newconf) {
          printf("Exhaustive search finished\n");
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
    arma::mat potU(solver.UnrestrictedPotential(totlist[0].orbsa, totlist[0].orbsb));
    arma::mat potM(solver.AveragePotential(totlist[0].orbsa, totlist[0].orbsb));
    arma::mat potW(solver.WeightedPotential(totlist[0].orbsa, totlist[0].orbsb));

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

  return 0;
}
