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

  void SetRestricted(bool restr_) {
    restr=restr_;
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

  arma::sword Nel() const {
    return arma::sum(occs);
  }

  arma::ivec Occs() const {
    return occs;
  }

  void SetOccs(const arma::ivec & occs_) {
    occs=occs_;
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

  void UpdateOrbitalsDamped(const arma::mat & F, const arma::mat & Tl, const arma::mat & Sinvh, const arma::mat & S, double dampov) {
    for(size_t l=0;l<=LMAX;l++) {
      // Fock matrix
      arma::mat Fl(F+lfac(l)*Tl);

      // Count occupied shells
      arma::sword numl = occs(l);
      size_t nsh;
      for(nsh=0;nsh<C[l].n_cols;nsh++) {
        arma::sword nocc = std::min(shell_capacity(l), numl);
        numl -= nocc;
        if(nocc==0)
          break;
      }

      if(nsh) {
        // Go to MO basis
        arma::mat Fmo(C[l].t()*S*Fl*S*C[l]);
        size_t nmo=C[l].n_cols;

        arma::uvec oidx(arma::linspace<arma::uvec>(0,nsh-1,nsh));
        arma::uvec vidx(arma::linspace<arma::uvec>(nsh,nmo-1,nmo-nsh));
        // Damp OV blocks
        Fmo(oidx,vidx)*=dampov;
        Fmo(vidx,oidx)*=dampov;
        // Recreate Fock matrix
        Fl=C[l]*Fmo*C[l].t();
      }

      scf::eig_gsym(E[l],C[l],Fl,Sinvh);
    }
  }

  void UpdateOrbitalsShifted(const arma::mat & F, const arma::mat & Tl, const arma::mat & Sinvh, const arma::mat & S, double shift) {
    for(size_t l=0;l<=LMAX;l++) {
      // Fock matrix
      arma::mat Fl(F+lfac(l)*Tl);

      // Count occupied shells
      arma::sword numl = occs(l);
      size_t nsh;
      for(nsh=0;nsh<C[l].n_cols;nsh++) {
        arma::sword nocc = std::min(shell_capacity(l), numl);
        numl -= nocc;
        if(nocc==0)
          break;
      }

      arma::mat Co;
      if(nsh) {
        // Apply level shift. Occupied orbitals
        Co=(C[l].cols(0,nsh-1));
        // Shift matrix
        arma::mat shmat(-shift*S*Co*Co.t()*S);
        // Update orbitals
        scf::eig_gsym(E[l],C[l],Fl+shmat,Sinvh);
      } else {
        scf::eig_gsym(E[l],C[l],Fl,Sinvh);
      }

      /*
      if(nsh) {
        arma::mat proj(C[l].cols(0,nsh-1).t()*S*Co);
        printf("Projection of occupied am = %i subspace with shift %e is %e\n",(int) l, shift, arma::sum(arma::sum(arma::square(proj)))/nsh);
        E[l].subvec(0,nsh-1).t().print("Occupied eigenvalues");
      }
      */
    }
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
      for(int shell_to=0;shell_to<=LMAX;shell_to++) {
        // Try moving up to a whole shell at a time
        for(int nmove=1;nmove<=std::min(shell_capacity(shell_from),shell_capacity(shell_to));nmove++) {
          // Check that we have electrons we can move
          if(occs(shell_from)<nmove)
            continue;

          // We include the identity, since otherwise fully
          // spin-polarized calculations don't work (empty list for beta
          // moves)

          // New channel
          OrbitalChannel newch(*this);
          newch.occs(shell_from)-=nmove;
          newch.occs(shell_to)+=nmove;

          ret.push_back(newch);
        }
      }
    }

    return ret;
  }
};

typedef struct {
  // Orbitals
  OrbitalChannel orbs;
  // Fock matrix
  arma::mat F;
  // Densities
  std::vector<arma::mat> Pl;
  // Total energy of configuration
  double Econf;
  // Energy components
  double Ekin, Epot, Ecoul, Exc;
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
  // Fock matrices
  arma::mat Fa, Fb;
  // Densities
  std::vector<arma::mat> Pal, Pbl;
  // Total energy of configuration
  double Econf;
  // Energy components
  double Ekin, Epot, Ecoul, Exc;
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
  double shift;

  double convthr;
  double dftthr;
  double diiseps;
  double diisthr;
  int diisorder;

  bool verbose;

  arma::vec lfac;

public:
  SCFSolver(int Z, polynomial_basis::PolynomialBasis * poly, int Nquad, int Nelem, double Rmax, int igrid, double zexp, int x_func_, int c_func_, int maxit_, double shift_, double convthr_, double dftthr_, double diiseps_, double diisthr_, int diisorder_) : x_func(x_func_), c_func(c_func_), maxit(maxit_), shift(shift_), convthr(convthr_), dftthr(dftthr_), diiseps(diiseps_), diisthr(diisthr_), diisorder(diisorder_) {
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

    // Angular factor
    lfac.resize(LMAX+1);
    for(arma::sword l=0;l<=LMAX;l++)
      lfac(l)=l*(l+1);
  }

  ~SCFSolver() {
  }

  void set_func(int x_func_, int c_func_) {
    x_func=x_func_;
    c_func=c_func_;
  }

  arma::mat TotalDensity(const std::vector<arma::mat> & Pl) const {
    arma::mat P(Pl[0]);
    for(size_t l=1;l<Pl.size();l++)
      P+=Pl[l];
    return P;
  }

  void Initialize(OrbitalChannel & orbs) {
    orbs.UpdateOrbitals(H0,Tl,Sinvh);
  }

  double FockBuild(rconf_t & conf) {
    // Form density
    conf.orbs.UpdateDensity(conf.Pl);
    arma::mat P(TotalDensity(conf.Pl));
    if(verbose) {
      printf("Tr P = %f\n",arma::trace(P*S));
      fflush(stdout);
    }

    // Angular factor
    double angfac(4.0*M_PI);

    // Compute energy
    conf.Ekin=arma::trace(P*T);
    for(arma::sword l=0;l<=LMAX;l++)
      conf.Ekin+=lfac(l)*arma::trace(conf.Pl[l]*Tl);
    conf.Epot=arma::trace(P*Vnuc);

    // Form Coulomb matrix
    arma::mat J(basis.coulomb(P/angfac));
    conf.Ecoul=0.5*arma::trace(P*J);
    if(verbose) {
      printf("Coulomb energy %.10e\n",conf.Ecoul);
      fflush(stdout);
    }

    // Exchange-correlation
    conf.Exc=0.0;
    arma::mat XC;
    double nelnum;
    grid.eval_Fxc(x_func, c_func, P/angfac, XC, conf.Exc, nelnum, dftthr);
    // Potential needs to be divided as well
    XC/=angfac;
    if(verbose) {
      printf("DFT energy %.10e\n",conf.Exc);
      printf("Error in integrated number of electrons % e\n",nelnum-conf.orbs.Nel());
      fflush(stdout);
    }

    // Fock matrices
    conf.F=H0+J+XC;

    // Update energy
    conf.Econf=conf.Ekin+conf.Epot+conf.Ecoul+conf.Exc;

    return conf.Econf;
  }

  double FockBuild(uconf_t & conf) {
    // Form density
    conf.orbsa.UpdateDensity(conf.Pal);
    conf.orbsb.UpdateDensity(conf.Pbl);

    std::vector<arma::mat> Pl(conf.Pal.size());
    for(size_t l=0;l<conf.Pal.size();l++)
      Pl[l]=conf.Pal[l]+conf.Pbl[l];
    arma::mat Pa(TotalDensity(conf.Pal));
    arma::mat Pb(TotalDensity(conf.Pbl));
    arma::mat P(Pa+Pb);

    // Angular factor
    double angfac(4.0*M_PI);

    // Compute energy
    conf.Ekin=arma::trace(P*T);
    for(arma::sword l=0;l<=LMAX;l++)
      conf.Ekin+=lfac(l)*arma::trace(Pl[l]*Tl);
    conf.Epot=arma::trace(P*Vnuc);

    // Form Coulomb matrix
    arma::mat J(basis.coulomb(P/angfac));
    conf.Ecoul=0.5*arma::trace(P*J);
    if(verbose) {
      printf("Coulomb energy %.10e\n",conf.Ecoul);
      fflush(stdout);
    }

    // Exchange-correlation
    conf.Exc=0.0;
    arma::mat XCa, XCb;
    double nelnum;
    grid.eval_Fxc(x_func, c_func, Pa/angfac, Pb/angfac, XCa, XCb, conf.Exc, nelnum, true, dftthr);
    // Potential needs to be divided as well
    XCa/=angfac;
    XCb/=angfac;
    if(verbose) {
      printf("DFT energy %.10e\n",conf.Exc);
      printf("Error in integrated number of electrons % e\n",nelnum-conf.orbsa.Nel()-conf.orbsb.Nel());
      fflush(stdout);
    }

    // Fock matrices
    conf.Fa=H0+J+XCa;
    conf.Fb=H0+J+XCb;

    // Update energy
    conf.Econf=conf.Ekin+conf.Epot+conf.Ecoul+conf.Exc;

    return conf.Econf;
  }

  double Solve(rconf_t & conf) {
    if(!conf.orbs.OrbitalsInitialized())
      throw std::logic_error("Orbitals not initialized!\n");
    if(!conf.orbs.Restricted())
      throw std::logic_error("Running restricted calculation with unrestricted orbitals!\n");

    verbose=false;

    if(verbose) {
      printf("Running SCF for orbital occupations\n");
      conf.orbs.Occs().t().print();
    }

    // DIIS object. ADIIS doesn't work for (significant) fractional occupation
    bool usediis=true, useadiis=true;
    rDIIS diis(supermat(S),supermat(Sinvh),usediis,diiseps,diisthr,useadiis,verbose,diisorder);
    double diiserr;

    // Density matrix
    arma::mat P;

    double E=0.0, Eold;

    arma::sword iscf;
    for(iscf=1;iscf<=maxit;iscf++) {
      if(verbose) {
        printf("\n**** Iteration %i ****\n\n",(int) iscf);
      }

      // Form Fock matrix
      Eold=E;
      E=FockBuild(conf);

      double dE=E-Eold;
      if(verbose) {
        printf("Total energy is % .10f\n",E);
        if(iscf>1)
          printf("Energy changed by %e\n",dE);
        fflush(stdout);
      }

      // Since Fock operator depends on the l channel, we need to create
      // a supermatrix for DIIS.
      arma::mat Fsuper(supermat(conf.F));
      for(arma::sword l=0;l<=LMAX;l++) {
        Fsuper.submat(l*conf.F.n_rows,l*conf.F.n_cols,(l+1)*conf.F.n_rows-1,(l+1)*conf.F.n_cols-1)+=lfac(l)*Tl;
      }
      // Need density for DIIS as well
      P=TotalDensity(conf.Pl);
      arma::mat Psuper(P.n_rows*(LMAX+1),P.n_cols*(LMAX+1));
      Psuper.zeros();
      for(arma::sword l=0;l<=LMAX;l++) {
        Psuper.submat(l*P.n_rows,l*P.n_cols,(l+1)*P.n_rows-1,(l+1)*P.n_cols-1)=conf.Pl[l];
      }
      // Update DIIS
      diis.update(Fsuper,Psuper,E,diiserr);
      if(verbose) {
        printf("DIIS error is %e\n",diiserr);
        fflush(stdout);
      }
      // Have we converged? Note that DIIS error is still wrt full space, not active space.
      bool convd=(diiserr<convthr) && (std::abs(dE)<convthr);

      // Solve DIIS to get Fock update
      diis.solve_F(Fsuper);
      conf.F=Fsuper.submat(0,0,conf.F.n_rows-1,conf.F.n_cols-1);

      // Update orbitals and density
      if(diiserr > diisthr) {
        // Since ADIIS is unreliable, we also use a level shift.
        conf.orbs.UpdateOrbitalsShifted(conf.F,Tl,Sinvh,S,shift);
      } else {
        conf.orbs.UpdateOrbitals(conf.F,Tl,Sinvh);
      }

      if(convd)
        break;
    }
    if(iscf > maxit) {
      printf("*** Not converged; DIIS error %e ***\n",diiserr);
      fflush(stdout);
    }

    if(verbose) {
      printf("%-21s energy: % .16f\n","Kinetic",conf.Ekin);
      printf("%-21s energy: % .16f\n","Nuclear attraction",conf.Epot);
      printf("%-21s energy: % .16f\n","Coulomb",conf.Ecoul);
      printf("%-21s energy: % .16f\n","conf.Exchange-correlation",conf.Exc);
      printf("%-21s energy: % .16f\n","Total",conf.Econf);
      printf("%-21s energy: % .16f\n","Virial ratio",-conf.Econf/conf.Ekin);
      printf("\n");

      // Electron density at nucleus
      printf("Electron density at nucleus % .10e\n",basis.nuclear_density(P));
    } else {
      printf("Evaluated energy % .16f for configuration ",conf.Econf);

      arma::ivec occs(conf.orbs.Occs());
      for(size_t i=0;i<occs.size();i++)
        printf(" %i",(int) occs(i));
      printf("\n");
    }

    return E;
  }

  double Solve(uconf_t & conf) {
    if(!conf.orbsa.OrbitalsInitialized())
      throw std::logic_error("Orbitals not initialized!\n");
    if(!conf.orbsb.OrbitalsInitialized())
      throw std::logic_error("Orbitals not initialized!\n");

    if(conf.orbsa.Restricted())
      throw std::logic_error("Running unrestricted calculation with restricted orbitals!\n");
    if(conf.orbsb.Restricted())
      throw std::logic_error("Running unrestricted calculation with restricted orbitals!\n");

    double E=0.0, Eold;

    verbose = false;

    if(verbose) {
      printf("Running SCF for orbital occupations\n");
      conf.orbsa.Occs().t().print();
      conf.orbsb.Occs().t().print();
    }

    // S supermatrix
    bool combine=false, usediis=true, useadiis=true;
    uDIIS diis(supermat(S),supermat(Sinvh),combine, usediis,diiseps,diisthr,useadiis,verbose,diisorder);
    double diiserr;

    // Density matrix
    arma::mat Pa, Pb, P;

    arma::sword iscf;
    for(iscf=1;iscf<=maxit;iscf++) {
      if(verbose) {
        printf("\n**** Iteration %i ****\n\n",(int) iscf);
      }

      Eold=E;
      E=FockBuild(conf);
      double dE=E-Eold;

      if(verbose) {
        printf("Total energy is % .10f\n",E);
        if(iscf>1)
          printf("Energy changed by %e\n",dE);
        fflush(stdout);
      }

      // Since Fock operator depends on the l channel, we need to create
      // a supermatrix for DIIS.
      arma::mat Fasuper(supermat(conf.Fa)), Fbsuper(supermat(conf.Fb));
      for(arma::sword l=0;l<=LMAX;l++) {
        Fasuper.submat(l*conf.Fa.n_rows,l*conf.Fa.n_cols,(l+1)*conf.Fa.n_rows-1,(l+1)*conf.Fa.n_cols-1)+=lfac(l)*Tl;
        Fbsuper.submat(l*conf.Fb.n_rows,l*conf.Fb.n_cols,(l+1)*conf.Fb.n_rows-1,(l+1)*conf.Fb.n_cols-1)+=lfac(l)*Tl;
      }
      Pa=TotalDensity(conf.Pal);
      Pb=TotalDensity(conf.Pbl);

      arma::mat Pasuper(Pa.n_rows*(LMAX+1),Pa.n_cols*(LMAX+1)), Pbsuper(Pb.n_rows*(LMAX+1),Pb.n_cols*(LMAX+1));
      Pasuper.zeros();
      Pbsuper.zeros();
      for(arma::sword l=0;l<=LMAX;l++) {
        Pasuper.submat(l*Pa.n_rows,l*Pa.n_cols,(l+1)*Pa.n_rows-1,(l+1)*Pa.n_cols-1)=conf.Pal[l];
        Pbsuper.submat(l*Pb.n_rows,l*Pb.n_cols,(l+1)*Pb.n_rows-1,(l+1)*Pb.n_cols-1)=conf.Pbl[l];
      }
      // Update DIIS
      diis.update(Fasuper,Fbsuper,Pasuper,Pbsuper,E,diiserr);
      if(verbose) {
        printf("DIIS error is %e\n",diiserr);
        fflush(stdout);
      }

      // Have we converged? Note that DIIS error is still wrt full space, not active space.
      bool convd=(diiserr<convthr) && (std::abs(dE)<convthr);

      // Solve DIIS to get Fock update
      diis.solve_F(Fasuper,Fbsuper);
      conf.Fa=Fasuper.submat(0,0,conf.Fa.n_rows-1,conf.Fa.n_cols-1);
      conf.Fb=Fbsuper.submat(0,0,conf.Fb.n_rows-1,conf.Fb.n_cols-1);

      // Update orbitals
      // Update orbitals and density
      if(diiserr > diisthr) {
        // Since ADIIS is unreliable, we also use a level shift
        conf.orbsa.UpdateOrbitalsShifted(conf.Fa,Tl,Sinvh,S,shift);
        conf.orbsb.UpdateOrbitalsShifted(conf.Fb,Tl,Sinvh,S,shift);
      } else {
        conf.orbsa.UpdateOrbitals(conf.Fa,Tl,Sinvh);
        conf.orbsb.UpdateOrbitals(conf.Fb,Tl,Sinvh);
      }
      if(convd)
        break;
    }
    if(iscf > maxit) {
      printf("*** Not converged; DIIS error %e ***\n",diiserr);
      fflush(stdout);
    }

    if(verbose) {
      printf("%-21s energy: % .16f\n","Kinetic",conf.Ekin);
      printf("%-21s energy: % .16f\n","Nuclear attraction",conf.Epot);
      printf("%-21s energy: % .16f\n","Coulomb",conf.Ecoul);
      printf("%-21s energy: % .16f\n","conf.Exchange-correlation",conf.Exc);
      printf("%-21s energy: % .16f\n","Total",conf.Econf);
      printf("%-21s energy: % .16f\n","Virial ratio",-conf.Econf/conf.Ekin);
      printf("\n");

      // Electron density at nucleus
      printf("Electron density at nucleus % .10e\n",basis.nuclear_density(P));
    } else {
      printf("Evaluated energy % .16f for configuration ",E);

      arma::ivec occa(conf.orbsa.Occs());
      for(size_t i=0;i<occa.size();i++)
        printf(" %i",(int) occa(i));
      arma::ivec occb(conf.orbsb.Occs());
      for(size_t i=0;i<occb.size();i++)
        printf(" %i",(int) occb(i));
      printf("\n");
    }

    return E;
  }

  arma::mat RestrictedPotential(rconf_t & conf) {
    if(!conf.orbs.OrbitalsInitialized())
      throw std::logic_error("No orbitals!\n");

    arma::mat P=TotalDensity(conf.Pl);

    arma::vec wt(basis.quadrature_weights());
    arma::mat Zeff(basis.coulomb_screening(P));
    arma::vec vcoul(Zeff.col(1));
    arma::vec vxc(basis.xc_screening(P,x_func,c_func));
    Zeff.col(1)+=vxc;
    arma::mat rho(basis.electron_density(P));
    arma::vec grho(basis.electron_density_gradient(P));
    arma::vec lrho(basis.electron_density_laplacian(P));

    arma::mat result(Zeff.n_rows,8);
    result.col(0)=Zeff.col(0);
    result.col(1)=rho.col(1);
    result.col(2)=grho;
    result.col(3)=lrho;
    result.col(4)=vcoul;
    result.col(5)=vxc;
    result.col(6)=Zeff.col(1);
    result.col(7)=wt;

    return result;
  }

  arma::mat UnrestrictedPotential(uconf_t & conf) {
    if(!conf.orbsa.OrbitalsInitialized())
      throw std::logic_error("No orbitals!\n");
    if(!conf.orbsb.OrbitalsInitialized())
      throw std::logic_error("No orbitals!\n");

    arma::mat Pa=TotalDensity(conf.Pal);
    arma::mat Pb=TotalDensity(conf.Pbl);
    arma::mat P(Pa+Pb);

    arma::vec wt(basis.quadrature_weights());
    arma::mat Zeff(basis.coulomb_screening(P));
    arma::vec vcoul(Zeff.col(1));
    arma::mat vxcm(basis.xc_screening(Pa,Pb,x_func,c_func));
    // Averaged potential
    arma::vec vxc=arma::mean(vxcm,1);
    Zeff.col(1)+=vxc;
    arma::mat rho(basis.electron_density(P));
    arma::vec grho(basis.electron_density_gradient(P));
    arma::vec lrho(basis.electron_density_laplacian(P));

    arma::mat result(Zeff.n_rows,8);
    result.col(0)=Zeff.col(0);
    result.col(1)=rho.col(1);
    result.col(2)=grho;
    result.col(3)=lrho;
    result.col(4)=vcoul;
    result.col(5)=vxc;
    result.col(6)=Zeff.col(1);
    result.col(7)=wt;

    return result;
  }

  arma::mat AveragePotential(uconf_t & conf) {
    if(!conf.orbsa.OrbitalsInitialized())
      throw std::logic_error("No orbitals!\n");
    if(!conf.orbsb.OrbitalsInitialized())
      throw std::logic_error("No orbitals!\n");

    std::vector<arma::mat> Pal, Pbl;
    arma::mat Pa=TotalDensity(Pal);
    arma::mat Pb=TotalDensity(Pbl);
    arma::mat P(Pa+Pb);

    arma::vec wt(basis.quadrature_weights());
    arma::mat Zeff(basis.coulomb_screening(P));
    arma::vec vcoul(Zeff.col(1));
    arma::vec vxc(basis.xc_screening(P,x_func,c_func));
    Zeff.col(1)+=vxc;
    arma::mat rho(basis.electron_density(P));
    arma::vec grho(basis.electron_density_gradient(P));
    arma::vec lrho(basis.electron_density_laplacian(P));

    arma::mat result(Zeff.n_rows,8);
    result.col(0)=Zeff.col(0);
    result.col(1)=rho.col(1);
    result.col(2)=grho;
    result.col(3)=lrho;
    result.col(4)=vcoul;
    result.col(5)=vxc;
    result.col(6)=Zeff.col(1);
    result.col(7)=wt;

    return result;
  }

  arma::mat WeightedPotential(uconf_t & conf) {
    if(!conf.orbsa.OrbitalsInitialized())
      throw std::logic_error("No orbitals!\n");
    if(!conf.orbsb.OrbitalsInitialized())
      throw std::logic_error("No orbitals!\n");

    std::vector<arma::mat> Pal, Pbl;
    arma::mat Pa=TotalDensity(Pal);
    arma::mat Pb=TotalDensity(Pbl);
    arma::mat P(Pa+Pb);

    arma::vec wt(basis.quadrature_weights());
    arma::mat Zeff(basis.coulomb_screening(P));
    arma::vec vcoul(Zeff.col(1));
    arma::mat vxcm(basis.xc_screening(Pa,Pb,x_func,c_func));
    arma::mat rhoa(basis.electron_density(Pa));
    arma::vec grhoa(basis.electron_density_gradient(Pa));
    arma::vec lrhoa(basis.electron_density_laplacian(Pa));
    arma::mat rhob(basis.electron_density(Pb));
    arma::vec grhob(basis.electron_density_gradient(Pb));
    arma::vec lrhob(basis.electron_density_laplacian(Pb));
    // Averaged potential
    arma::vec vxc=(vxcm.col(0)%rhoa.col(1) + vxcm.col(1)%rhob.col(1))/(rhoa.col(1)+rhob.col(1));
    // Set areas of small electron density to zero
    arma::vec n(rhoa.col(1)+rhob.col(1));
    vxc(arma::find(n<dftthr)).zeros();
    Zeff.col(1)+=vxc;

    arma::mat result(Zeff.n_rows,8);
    result.col(0)=Zeff.col(0);
    result.col(1)=rhoa.col(1)+rhob.col(1);
    result.col(2)=grhoa+grhob;;
    result.col(3)=lrhoa+lrhob;
    result.col(4)=vcoul;
    result.col(5)=vxc;
    result.col(6)=Zeff.col(1);
    result.col(7)=wt;

    return result;
  }
};

arma::ivec initial_occs(int Z) {
  // Guess occupations
  const int shell_order[]={0, 0, 1, 0, 1, 0, 2, 1, 0, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1};

  arma::ivec occs(LMAX+1);
  occs.zeros();
  for(size_t i=0;i<sizeof(shell_order)/sizeof(shell_order[0]);i++) {
    int l = shell_order[i];
    int nocc = std::min(Z, 2*(2*l+1));
    occs(l) += nocc;
    Z -= nocc;
  }

  return occs;
}

void get_active(int Z, arma::ivec & frz, arma::ivec & act) {
  // Guess occupations
  arma::ivec shell_order;
  shell_order << 0 << 0 << 1 << 0 << 1 << 0 << 2 << 1 << 0 << 2 << 1 << 0 << 3 << 2 << 1 << 0 << 3 << 2 << 1;

  // Magic numbers: noble element frozen cores
  arma::uvec magicno;
  magicno << 2 << 10 << 18 << 36 << 54 << 86 << 118;
  arma::uvec nfilled;
  nfilled << 1 << 3 << 5 << 8 << 11 << 15 << 19;

  // Count how many closed shells
  arma::uvec idx(arma::find(magicno <= Z));

  // Set frozen core
  frz.zeros(LMAX+1);
  for(size_t i=0;i<nfilled(idx.n_elem);i++) {
    int l = shell_order(i);
    int nocc = std::min(Z, 2*(2*l+1));
    frz(l) += nocc;
    Z -= nocc;
  }

  // Set active space
  act.zeros(LMAX+1);
  if(idx.n_elem < nfilled.n_elem) {
    for(size_t i=nfilled(idx.n_elem)=0;i<nfilled(idx.n_elem+1);i++) {
      int l = shell_order(i);
      act(l) += 2*l+1;
    }
  }
}

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
  parser.add<int>("maxit", 0, "maximum number of iterations", false, 200);
  parser.add<double>("shift", 0, "level shift for initial SCF iterations", false, 1.0);
  parser.add<double>("convthr", 0, "convergence threshold", false, 1e-7);
  parser.add<std::string>("method", 0, "method to use", false, "lda_x");
  parser.add<std::string>("pot", 0, "method to use to compute potential", false, "lda_x");
  parser.add<double>("dftthr", 0, "density threshold for dft", false, 1e-12);
  parser.add<int>("restricted", 0, "spin-restricted orbitals", false, 0);
  parser.add<int>("primbas", 0, "primitive radial basis", false, 4);
  parser.add<double>("diiseps", 0, "when to start mixing in diis", false, 1e-2);
  parser.add<double>("diisthr", 0, "when to switch over fully to diis", false, 1e-3);
  parser.add<int>("diisorder", 0, "length of diis history", false, 10);
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
  SCFSolver solver(Z, poly, Nquad, Nelem, Rmax, igrid, zexp, x_func, c_func, maxit, shift, convthr, dftthr, diiseps, diisthr, diisorder);

  // Initialize with a sensible guess occupation
  rconf_t initial;
  initial.orbs=OrbitalChannel(true);
  solver.Initialize(initial.orbs);
  initial.orbs.SetOccs(initial_occs(Z));
  if(initial.orbs.Nel()) {
    initial.Econf=solver.Solve(initial);
  } else {
    initial.Econf=0.0;
  }

  if(restr) {
    // List of configurations
    std::vector<rconf_t> rlist;

    // Restricted calculation
    rconf_t conf(initial);
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
      std::vector<OrbitalChannel> newconfs(rlist[0].orbs.MoveElectrons());

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
    std::vector<uconf_t> totlist;

    // Restricted calculation
    uconf_t conf;
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

      std::vector<uconf_t> ulist;
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
        std::vector<OrbitalChannel> newconfa(ulist[0].orbsa.MoveElectrons());
        std::vector<OrbitalChannel> newconfb(ulist[0].orbsb.MoveElectrons());

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

  return 0;
}
