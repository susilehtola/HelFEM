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
#include "../general/diis.h"
#include "../general/dftfuncs.h"
#include "../general/elements.h"
#include "../general/timer.h"
#include "../general/scf_helpers.h"
#include "basis.h"
#include "dftgrid.h"
#include <cfloat>

using namespace helfem;

void classify_orbitals(const arma::mat & C, const arma::ivec & lvals, const arma::ivec & mvals, const std::vector<arma::uvec> & lmidx) {
  for(size_t io=0;io<C.n_cols;io++) {
    arma::vec orb(C.col(io));

    arma::vec ochar(mvals.n_elem);
    for(size_t c=0;c<mvals.n_elem;c++) {
      ochar(c)=arma::norm(orb(lmidx[c]),"fro");
    }
    ochar/=arma::sum(ochar);

    // Orbital symmetry is then
    arma::uword oidx;
    ochar.max(oidx);

    // Classification
    std::ostringstream cl;

    printf("Orbital %2i: l=%1i m=%+1i %6.2f %%\n",(int) (io+1),(int) lvals(oidx),(int) mvals(oidx),100.0*ochar(oidx));
  }
}

void normalize_matrix(arma::mat & M, const arma::vec & norm) {
  if(M.n_rows != norm.n_elem) throw std::logic_error("Incompatible dimensions!\n");
  if(M.n_cols != norm.n_elem) throw std::logic_error("Incompatible dimensions!\n");
  for(size_t i=0;i<M.n_rows;i++)
    for(size_t j=0;j<M.n_cols;j++)
      M(i,j)*=norm(i)*norm(j);
}

int main(int argc, char **argv) {
  cmdline::parser parser;

  // full option name, no short option, description, argument required
  parser.add<std::string>("Z", 0, "nuclear charge", true);
  parser.add<std::string>("Zl", 0, "left-hand nuclear charge", false, "");
  parser.add<std::string>("Zr", 0, "right-hand nuclear charge", false, "");
  parser.add<double>("Rmid", 0, "distance of nuclei from center", false, 0.0);
  parser.add<bool>("angstrom", 0, "input distances in angstrom", false, false);
  parser.add<int>("nela", 0, "number of alpha electrons", false, 0);
  parser.add<int>("nelb", 0, "number of beta  electrons", false, 0);
  parser.add<int>("Q", 0, "charge state", false, 0);
  parser.add<int>("M", 0, "spin multiplicity", false, 0);
  parser.add<int>("lmax", 0, "maximum l quantum number", true);
  parser.add<int>("mmax", 0, "maximum m quantum number", true);
  parser.add<double>("Rmax", 0, "practical infinity in au", false, 40.0);
  parser.add<int>("grid", 0, "type of grid: 1 for linear, 2 for quadratic, 3 for polynomial, 4 for exponential", false, 4);
  parser.add<int>("grid0", 0, "type of grid: 1 for linear, 2 for quadratic, 3 for polynomial, 4 for exponential", false, 4);
  parser.add<double>("zexp", 0, "parameter in radial grid", false, 2.0);
  parser.add<double>("zexp0", 0, "parameter in radial grid", false, 2.0);
  parser.add<int>("nelem", 0, "number of elements", true);
  parser.add<int>("nelem0", 0, "number of elements between center and off-center nuclei", false, 0);
  parser.add<int>("nnodes", 0, "number of nodes per element", false, 15);
  parser.add<int>("nquad", 0, "number of quadrature points", false, 0);
  parser.add<int>("maxit", 0, "maximum number of iterations", false, 50);
  parser.add<double>("convthr", 0, "convergence threshold", false, 1e-7);
  parser.add<double>("Ez", 0, "electric dipole field", false, 0.0);
  parser.add<double>("Qzz", 0, "electric quadrupole field", false, 0.0);
  parser.add<double>("Bz", 0, "magnetic dipole field", false, 0.0);
  parser.add<bool>("diag", 0, "exact diagonalization", false, 1);
  parser.add<std::string>("method", 0, "method to use", false, "HF");
  parser.add<int>("ldft", 0, "theta rule for dft quadrature (0 for auto)", false, 0);
  parser.add<int>("mdft", 0, "phi rule for dft quadrature (0 for auto)", false, 0);
  parser.add<double>("dftthr", 0, "density threshold for dft", false, 1e-12);
  parser.add<int>("restricted", 0, "spin-restricted orbitals", false, -1);
  parser.add<int>("symmetry", 0, "force orbital symmetry", false, 1);
  parser.add<int>("primbas", 0, "primitive radial basis", false, 4);
  parser.add<double>("diiseps", 0, "when to start mixing in diis", false, 1e-2);
  parser.add<double>("diisthr", 0, "when to switch over fully to diis", false, 1e-3);
  parser.add<int>("diisorder", 0, "length of diis history", false, 5);
  parser.add<int>("readocc", 0, "read occupations from file, use until nth build", false, 0);
  parser.add<double>("perturb", 0, "randomly perturb initial guess", false, 0.0);
  parser.add<int>("seed", 0, "seed for random perturbation", false, 0);
  parser.add<int>("iguess", 0, "guess: 0 for core, 1 for GSZ, 2 for SAP, 3 for TF", false, 2);
  parser.add<int>("finitenuc", 0, "finite nuclear model", false, 0);
  parser.add<double>("Rrms", 0, "finite nuclear rms radius", false, 0.0);
  parser.add<std::string>("load", 0, "load guess from checkpoint", false, "");
  parser.add<std::string>("save", 0, "save calculation to checkpoint", false, "helfem.chk");
  parser.add<std::string>("x_pars", 0, "file for parameters for exchange functional", false, "");
  parser.add<std::string>("c_pars", 0, "file for parameters for correlation functional", false, "");
  parser.add<bool>("maverage", 0, "average Fock matrix over m values", false, false);
  parser.add<double>("dampfock", 0, "damping factor for off-diagonal elents", false, 0.7);
  parser.add<double>("dampthr", 0, "damping threshold", false, 0.1);
  parser.add<bool>("zeroder", 0, "zero derivative at Rmax?", false, false);
  parser.add<int>("taylor_order", 0, "order of Taylor expansion near the nucleus", false, -1);
  parser.add<int>("iconf", 0, "Confinement potential: 1 for polynomial, 2 for exponential, 3 for barrier, 4 for Junquera et al.", false, 0);
  parser.add<int>("conf_N", 0, "Exponent in confinement potential", false, 0);
  parser.add<double>("conf_R", 0, "Confinement radius", false, 0.0);
  parser.add<double>("conf_barrier", 0, "Confinement barrier height", false, 0.0);
  parser.add<double>("shift_conf", 0, "Where does confinement start?", false, 0.0);
  parser.add<bool>("add_conf", 0, "Add element boundary at shifted potential radius R?", false, true);
  parser.parse_check(argc, argv);

  // Get parameters
  double Rmax(parser.get<double>("Rmax"));
  int igrid(parser.get<int>("grid"));
  int igrid0(parser.get<int>("grid0"));
  double zexp(parser.get<double>("zexp"));
  double zexp0(parser.get<double>("zexp0"));
  double Ez(parser.get<double>("Ez"));
  double Qzz(parser.get<double>("Qzz"));
  double Bz(parser.get<double>("Bz"));

  int maxit(parser.get<int>("maxit"));
  double convthr(parser.get<double>("convthr"));

  bool diag(parser.get<bool>("diag"));
  int restr(parser.get<int>("restricted"));
  int symm(parser.get<int>("symmetry"));
  int iguess(parser.get<int>("iguess"));
  bool maverage(parser.get<bool>("maverage"));

  int primbas(parser.get<int>("primbas"));
  // Number of elements
  int Nelem0(parser.get<int>("nelem0"));
  int Nelem(parser.get<int>("nelem"));
  // Number of nodes
  int Nnodes(parser.get<int>("nnodes"));
  int taylor_order(parser.get<int>("taylor_order"));

  // Order of quadrature rule
  int Nquad(parser.get<int>("nquad"));
  // Angular grid
  int lmax(parser.get<int>("lmax"));
  int mmax(parser.get<int>("mmax"));

  // DFT angular grid
  int ldft(parser.get<int>("ldft"));
  int mdft(parser.get<int>("mdft"));
  double dftthr(parser.get<double>("dftthr"));

  int finitenuc(parser.get<int>("finitenuc"));
  double Rrms(parser.get<double>("Rrms"));

  // Nuclear charge
  int Z(get_Z(parser.get<std::string>("Z")));
  int Zl(get_Z(parser.get<std::string>("Zl")));
  int Zr(get_Z(parser.get<std::string>("Zr")));
  double Rhalf(parser.get<double>("Rmid"));
  // Number of occupied states
  int nela(parser.get<int>("nela"));
  int nelb(parser.get<int>("nelb"));
  int Q(parser.get<int>("Q"));
  int M(parser.get<int>("M"));

  double diiseps=parser.get<double>("diiseps");
  double diisthr=parser.get<double>("diisthr");
  int diisorder=parser.get<int>("diisorder");

  std::string method(parser.get<std::string>("method"));

  double perturb=parser.get<double>("perturb");
  int seed=parser.get<int>("seed");

  std::string save(parser.get<std::string>("save"));
  std::string load(parser.get<std::string>("load"));

  std::string xparf(parser.get<std::string>("x_pars"));
  std::string cparf(parser.get<std::string>("c_pars"));

  double dampfock(parser.get<double>("dampfock"));
  double dampthr(parser.get<double>("dampthr"));

  bool zeroder(parser.get<bool>("zeroder"));

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

  // Open checkpoint in save mode
  Checkpoint chkpt(save,true);

  // Read occupations from file?
  int readocc=parser.get<int>("readocc");
  if(readocc<0)
    readocc=INT_MAX;
  arma::imat occs;
  if(readocc) {
    occs.load("occs.dat",arma::raw_ascii);
    if(symm == 2 && occs.n_cols != 4) {
      throw std::logic_error("Must have four columns in occupation data to use full atomic symmetry.\n");
    }
    if(symm == 1 && occs.n_cols != 3) {
      throw std::logic_error("Must have three columns in occupation data to use axial symmetry.\n");
    }
  }

  if(parser.get<bool>("angstrom")) {
    // Convert to atomic units
    Rhalf*=ANGSTROMINBOHR;
  }

  scf::parse_nela_nelb(nela,nelb,Q,M,Z+Zl+Zr);
  if(restr==-1) {
    // If number of electrons differs then unrestrict
    restr=(nela==nelb);
  }
  chkpt.write("nela",nela);
  chkpt.write("nelb",nelb);

  std::vector<std::string> rcalc(2);
  rcalc[0]="unrestricted";
  rcalc[1]="restricted";

  printf("Running %s %s calculation with Rmax=%e and %i elements.\n",rcalc[restr].c_str(),method.c_str(),Rmax,Nelem);

  // Get primitive basis
  auto poly(std::shared_ptr<const polynomial_basis::PolynomialBasis>(polynomial_basis::get_basis(primbas,Nnodes)));

  if(Nquad==0)
    // Set default value
    Nquad=5*poly->get_nbf();
  else if(Nquad<2*poly->get_nbf())
    throw std::logic_error("Insufficient radial quadrature.\n");

  // Set default order of Taylor expansion
  if(taylor_order==-1)
    taylor_order = poly->get_nprim()-1;

  printf("Using %i point quadrature rule.\n",Nquad);
  printf("Angular grid spanning from l=0..%i, m=%i..%i.\n",lmax,-mmax,mmax);

  // Confinement parameters
  bool add_conf(parser.get<bool>("add_conf"));
  int conf_N(parser.get<int>("conf_N"));
  double conf_R(parser.get<double>("conf_R"));
  double conf_barrier(parser.get<double>("conf_barrier"));
  int iconf(parser.get<int>("iconf"));
  double shift_conf(parser.get<double>("shift_conf"));

  // Construct the angular basis
  arma::ivec lval, mval;
  atomic::basis::angular_basis(lmax,mmax,lval,mval);
  // and the radial one
  arma::vec bval=atomic::basis::form_grid((modelpotential::nuclear_model_t) finitenuc, Rrms, Nelem, Rmax, igrid, zexp, Nelem0, igrid0, zexp0, Z, Zl, Zr, Rhalf, add_conf, shift_conf);

  atomic::basis::TwoDBasis basis;
  basis=atomic::basis::TwoDBasis(Z, (modelpotential::nuclear_model_t) finitenuc, Rrms, poly, zeroder, Nquad, bval, taylor_order, lval, mval, Zl, Zr, Rhalf);
  chkpt.write(basis);
  printf("Basis set consists of %i angular shells composed of %i radial functions, totaling %i basis functions\n",(int) basis.Nang(), (int) basis.Nrad(), (int) basis.Nbf());
  printf("%ith order Taylor series used to evaluate basis functions for r <= %e, error %e\n",taylor_order, basis.get_small_r_taylor_cutoff(), basis.get_taylor_diff());

  printf("One-electron matrix requires %s\n",scf::memory_size(basis.mem_1el()).c_str());
  printf("Auxiliary one-electron integrals require %s\n",scf::memory_size(basis.mem_1el_aux()).c_str());
  printf("Auxiliary two-electron integrals require %s\n",scf::memory_size(basis.mem_2el_aux()).c_str());

  double Enucr=(Rhalf>0) ? Z*(Zl+Zr)/Rhalf + Zl*Zr/(2*Rhalf) : 0.0;
  printf("Central nuclear charge is %i\n",Z);
  printf("Left- and right-hand nuclear charges are %i and %i at distance % .3f from center\n",Zl,Zr,Rhalf);
  printf("Nuclear repulsion energy is %e\n",Enucr);
  printf("Number of electrons is %i %i\n",nela,nelb);

  // Symmetry indices
  std::vector<arma::uvec> dsym;
  if(symm==2 && (Ez!=0.0 || Qzz!=0.0)) {
    printf("Warning - asked for full orbital symmetry in presence of electric field. Relaxing restriction.\n");
    symm=1;
  }
  if(symm==2 && Bz!=0.0) {
    printf("Warning - asked for full orbital symmetry in presence of magnetic field. Relaxing restriction.\n");
    symm=1;
  }
  if(symm)
    dsym=basis.get_sym_idx(symm);

  arma::ivec lvals, mvals;
  lvals=basis.get_l();
  mvals=basis.get_m();
  std::vector<arma::uvec> lmidx(lvals.n_elem);
  for(size_t i=0;i<lmidx.size();i++)
    lmidx[i]=basis.lm_indices(lvals(i),mvals(i));

  // For m averaging
  std::vector< std::vector<arma::uvec> > l_idx(arma::max(lvals)+1);
  for(int l=0; l< (int) l_idx.size(); l++)
    for(int m=-l;m<=l;m++)
      l_idx[l].push_back(basis.lm_indices(l,m));

  // Forced occupations?
  arma::ivec occnuma, occnumb;
  std::vector<arma::uvec> occsym;
  if(readocc) {
    // Number of occupied alpha orbitals is first column
    occnuma=occs.col(0);
    // Number of occupied beta orbitals is second column
    occnumb=occs.col(1);
    // l and m values are third and fourth column
    if(symm==1)
      for(size_t i=0;i<occs.n_rows;i++)
        occsym.push_back(basis.m_indices(occs(i,2)));
    else if(symm==2)
      for(size_t i=0;i<occs.n_rows;i++)
        occsym.push_back(basis.lm_indices(occs(i,2),occs(i,3)));
    else
      throw std::logic_error("Not implemented!\n");

    // Check consistency of values
    if(arma::sum(occnuma) != nela) {
      std::ostringstream oss;
      oss << "Specified alpha occupations don't match wanted spin state.\n";
      oss << "Occupying " << arma::sum(occnuma) << " orbitals but should have " << nela << " orbitals.\n";
      throw std::logic_error(oss.str());
    }
    if(arma::sum(occnumb) != nelb) {
      std::ostringstream oss;
      oss << "Specified alpha occupations don't match wanted spin state.\n";
      oss << "Occupying " << arma::sum(occnumb) << " orbitals but should have " << nelb << " orbitals.\n";
      throw std::logic_error(oss.str());
    }
  }

  // Functional
  int x_func, c_func;
  ::parse_xc_func(x_func, c_func, method);
  ::print_info(x_func, c_func);
  if(!is_supported(x_func))
    throw std::logic_error("The specified exchange functional is not currently supported in HelFEM.\n");
  if(!is_supported(c_func))
    throw std::logic_error("The specified correlation functional is not currently supported in HelFEM.\n");

  bool dft=(x_func>0 || c_func>0);

  bool erfc, yukawa;
  is_range_separated(x_func, erfc, yukawa);
  // Fraction of exact exchange
  double kfrac, kshort, omega;
  range_separation(x_func, omega, kfrac, kshort);
  if(omega!=0.0) {
    printf("\nUsing range-separated exchange with range-separation constant omega = % .3f.\n",omega);
    printf("Using % .3f %% short-range and % .3f %% long-range exchange.\n",(kfrac+kshort)*100,kfrac*100);
    if(yukawa) {
      printf("Using the Yukawa kernel for range separation.\n");
    } else {
      printf("Using the error function kernel for range separation.\n");
    }
  } else if(kfrac!=0.0)
    printf("\nUsing hybrid exchange with % .3f %% of exact exchange.\n",kfrac*100);
  else
    printf("\nA pure exchange functional used, no exact exchange.\n");

  Timer timer;

  // Form overlap matrix
  arma::mat S(basis.overlap());
  chkpt.write("S",S);
  // Form kinetic energy matrix
  arma::mat T(basis.kinetic());
  chkpt.write("T",T);

  // Form DFT grid
  helfem::atomic::dftgrid::DFTGrid grid;
  if(dft) {
    // These would appear to give reasonably converged values
    if(ldft==0)
      // Default value: we have 2*lmax from the bra and ket and 2 from
      // the volume element, and allow for 2*lmax from the
      // density/potential. Add in 10 more for a bit more accuracy.
      ldft=4*lmax+10;
    if(ldft<2*lmax)
      throw std::logic_error("Increase ldft to guarantee accuracy of quadrature!\n");

    if(mdft==0)
      // Default value: we have 2*mmax from the bra and ket, and allow
      // for 2*mmax from the density/potential. Add in 5 to make
      // sure quadrature is still accurate for mmax=0
      mdft=4*mmax+5;
    if(mdft<2*mmax)
      throw std::logic_error("Increase mdft to guarantee accuracy of quadrature!\n");

    // Form grid
    grid=helfem::atomic::dftgrid::DFTGrid(&basis,ldft,mdft);

    // Basis function norms
    arma::vec bfnorm(arma::pow(arma::diagvec(S),-0.5));

    // Check accuracy of grid
    double Sthr=1e-10;
    double Tthr=1e-8;
    bool inacc=false;
    {
      arma::mat Sdft(grid.eval_overlap());
      Sdft-=S;
      normalize_matrix(Sdft,bfnorm);

      double Serr(arma::norm(Sdft,"fro"));
      printf("Error in overlap matrix evaluated through xc grid is %e\n",Serr);
      fflush(stdout);
      if(Serr>=Sthr)
        inacc=true;
    }
    {
      arma::mat Tdft(grid.eval_kinetic());
      // Compute relative error
      for(size_t j=0;j<Tdft.n_cols;j++)
        for(size_t i=0;i<Tdft.n_rows;i++)
          Tdft(i,j)=std::abs(Tdft(i,j)-T(i,j))/(1+std::abs(T(i,j)));

      double Terr(arma::norm(Tdft,"fro"));
      printf("Relative error in kinetic matrix evaluated through xc grid is %e\n",Terr);
      fflush(stdout);
      if(Terr>=Tthr)
        inacc=true;
    }
    if(inacc)
      printf("Warning - possibly inaccurate quadrature!\n");
    printf("\n");
  }

  // Get half-inverse
  timer.set();
  arma::mat Sinvh(basis.Sinvh(!diag,symm));
  chkpt.write("Sinvh",Sinvh);
  printf("Half-inverse formed in %.6f\n",timer.get());
  {
    arma::mat Smo(Sinvh.t()*S*Sinvh);
    Smo-=arma::eye<arma::mat>(Smo.n_rows,Smo.n_cols);
    printf("Orbital orthonormality deviation is %e\n",arma::norm(Smo,"fro"));
  }
  arma::mat Sh(basis.Shalf(!diag,symm));
  chkpt.write("Sh",Sh);
  printf("Half-overlap formed in %.6f\n",timer.get());
  {
    arma::mat Smo(Sh.t()*Sinvh);
    Smo-=arma::eye<arma::mat>(Smo.n_rows,Smo.n_cols);
    printf("Half-overlap error is %e\n",arma::norm(Smo,"fro"));
  }

  // Form nuclear attraction energy matrix
  Timer tnuc;
  if(Zl!=0 || Zr !=0)
    printf("Computing nuclear attraction integrals\n");
  arma::mat Vnuc=basis.nuclear();
  chkpt.write("Vuc",Vnuc);
  if(Zl!=0 || Zr !=0)
    printf("Done in %.6f\n",tnuc.get());

  // Confinement potential
  arma::mat Vconf(basis.Nbf(),basis.Nbf(),arma::fill::zeros);
  if(iconf) {
    printf("Computing confinement potential\n");
    Vconf=basis.confinement(conf_N, conf_R, iconf, conf_barrier, shift_conf);
  }
  chkpt.write("Vconf",Vconf);

  // Dipole coupling
  arma::mat dip(basis.dipole_z());
  chkpt.write("dip",dip);
  // Quadrupole coupling
  arma::mat quad(basis.quadrupole_zz());
  chkpt.write("quad",quad);

  // Electric field coupling (minus sign cancels one from charge)
  arma::mat Vel(Ez*dip + Qzz*quad/3.0);
  chkpt.write("Vel",Vel);
  // Magnetic field coupling
  arma::mat Vmag(basis.Bz_field(Bz));
  chkpt.write("Vmag",Vmag);
  // Form Hamiltonian
  arma::mat H0(T+Vnuc+Vel+Vmag+Vconf);
  chkpt.write("H0",H0);

  printf("One-electron matrices formed in %.6f\n",timer.get());

  // Occupied and virtual orbitals
  arma::mat Caocc, Cbocc, Cavirt, Cbvirt;
  arma::vec Ea, Eb;
  // Number of eigenenergies to print
  arma::uword nena(std::min((arma::uword) nela+4,Sinvh.n_cols));
  arma::uword nenb(std::min((arma::uword) nelb+4,Sinvh.n_cols));

  // Guess orbitals
  timer.set();
  {
    arma::mat Ca, Cb;
    if(load.size()) {
      printf("Guess orbitals from checkpoint\n");

      // Load checkpoint
      Checkpoint loadchk(load,false);
      // Old basis set
      atomic::basis::TwoDBasis oldbasis;
      loadchk.read(oldbasis);

      arma::mat oldSinvh;
      loadchk.read("Sinvh",oldSinvh);

      // Interbasis overlap
      arma::mat S12(basis.overlap(oldbasis));

      switch(iguess) {
      case(0):
        printf("Guess orbitals from Fock matrix projection\n");
	{
	  // Convert to orthonormal basis
	  S12=arma::trans(Sinvh)*S12*oldSinvh;
	  // Helper
	  arma::mat SSinvh(S*Sinvh);

	  // Fock matrix
	  arma::mat F;

	  // Load Fock matrix
	  loadchk.read("Fa",F);
	  // Project onto the old orthogonal basis
	  F=arma::trans(oldSinvh)*F*oldSinvh;
	  // Project onto the new basis
	  F=S12*F*arma::trans(S12);
	  // Go back to original basis
	  F=SSinvh*F*arma::trans(SSinvh);
	  // Diagonalize
	  if(symm)
	    scf::eig_gsym_sub(Ea,Ca,F,Sinvh,dsym);
	  else
	    scf::eig_gsym(Ea,Ca,F,Sinvh);

	  // Load Fock matrix
	  loadchk.read("Fb",F);
	  // Project onto the old orthogonal basis
	  F=arma::trans(oldSinvh)*F*oldSinvh;
	  // Project onto the new basis
	  F=S12*F*arma::trans(S12);
	  // Go back to original basis
	  F=SSinvh*F*arma::trans(SSinvh);
	  // Diagonalize
	  if(symm)
	    scf::eig_gsym_sub(Eb,Cb,F,Sinvh,dsym);
	  else
	    scf::eig_gsym(Eb,Cb,F,Sinvh);
	}
        break;

      case(1):
      default:
        // Project lowest orbitals
        printf("Guess orbitals from previous calculation\n");
      {
	// Projector
	arma::mat P((Sinvh*arma::trans(Sinvh))*S12);

	// Orbitals
	arma::mat C;

	// Alpha orbitals
	loadchk.read("Ca",C);
	// Project onto new basis: C1 = S11^-1 S12 C2
	Ca=P*C;

	// Beta orbitals
	loadchk.read("Cb",C);
	Cb=P*C;

	// Run Gram-Schmidt to make sure orbitals are orthonormal
	for(int ia=0;ia<nela;ia++) {
	  for(int ja=0;ja<ia;ja++)
	    Ca.col(ia)-= Ca.col(ja)*(arma::trans(Ca.col(ja))*S*Ca.col(ia));
	  Ca.col(ia) /= sqrt(arma::as_scalar(arma::trans(Ca.col(ia))*S*Ca.col(ia)));
	}

	for(int ib=0;ib<nelb;ib++) {
	  for(int jb=0;jb<ib;jb++)
	    Cb.col(ib) -= Cb.col(jb)*(arma::trans(Cb.col(jb))*S*Cb.col(ib));
	  Cb.col(ib) /= sqrt(arma::as_scalar(arma::trans(Cb.col(ib))*S*Cb.col(ib)));
	}

	// Read in orbital energies
	loadchk.read("Ea",Ea);
	if(Ea.n_elem<Ca.n_cols)
	  Ea=Ea.subvec(0,Ca.n_cols-1);
	loadchk.read("Eb",Eb);
	if(Eb.n_elem<Cb.n_cols)
	  Eb=Eb.subvec(0,Cb.n_cols-1);
      }
      break;
      }
    } else {
      modelpotential::ModelPotential * model;
      switch(iguess) {
      case(0):
        // Use core guess
        printf("Guess orbitals from core Hamiltonian\n");
        model = new modelpotential::PointNucleus(Z);
        break;

      case(1):
        // Use GSZ guess
        printf("Guess orbitals from GSZ screened nucleus\n");
        model = new modelpotential::GSZAtom(Z);
        break;

      case(2):
        // Use SAP guess
        printf("Guess orbitals from SAP screened nucleus\n");
        model = new modelpotential::SAPAtom(Z);
        break;

      case(3):
        // Use Thomas-Fermi guess
        printf("Guess orbitals from Thomas-Fermi nucleus\n");
        model = new modelpotential::TFAtom(Z);
        break;

      default:
        throw std::logic_error("Unsupported guess\n");
      }

      // Form guess Hamiltonian
      arma::mat Hguess(T+Vel+Vmag+basis.model_potential(model));
      // and free memory
      delete model;

      // Diagonalize the hamiltonian
      if(symm)
        scf::eig_gsym_sub(Ea,Ca,Hguess,Sinvh,dsym);
      else
        scf::eig_gsym(Ea,Ca,Hguess,Sinvh);

      // Beta guess is the same as the alpha guess
      Cb=Ca;
      Eb=Ea;

      // Enforce occupation according to specified symmetry
      if(readocc) {
	scf::enforce_occupations(Ca,Ea,S,occnuma,occsym);
	if(restr && nela==nelb)
	  Cb=Ca;
	else
	  scf::enforce_occupations(Cb,Eb,S,occnumb,occsym);
      }
    }

    // Perturb guess
    if(perturb) {
      // Generate norb x norb rotation matrix
      arma::arma_rng::set_seed(seed);
      Ca*=scf::perturbation_matrix(Ca.n_cols,perturb);
      if(restr && nela==nelb) {
        Cb=Ca;
      } else {
        Cb*=scf::perturbation_matrix(Cb.n_cols,perturb);
      }
      printf("Guess orbitals perturbed by %e\n",perturb);
    }

    // Alpha orbitals
    Caocc=Ca.cols(0,nela-1);
    if(Ca.n_cols>(size_t) nela)
      Cavirt=Ca.cols(nela,Ca.n_cols-1);

    // Beta guess
    if(nelb)
      Cbocc=Cb.cols(0,nelb-1);
    if(Cb.n_cols>(size_t) nelb)
      Cbvirt=Cb.cols(nelb,Cb.n_cols-1);

    Ea.subvec(0,nena-1).t().print("Alpha orbital energies");
    Eb.subvec(0,nenb-1).t().print("Beta  orbital energies");

    printf("\n");
    printf("Alpha orbital symmetries\n");
    classify_orbitals(Caocc,lvals,mvals,lmidx);
    if(nelb>0) {
      printf("\n");
      printf("Beta orbital symmetries\n");
      classify_orbitals(Cbocc,lvals,mvals,lmidx);
    }
    printf("\n");
  }
  printf("Initial guess performed in %.6f\n",timer.get());

  printf("Computing two-electron integrals\n");
  fflush(stdout);
  timer.set();
  basis.compute_tei(kfrac!=0.0);
  if(yukawa)
    basis.compute_yukawa(omega);
  else if(erfc)
    basis.compute_erfc(omega);
  printf("Done in %.6f\n",timer.get());

  double Ekin=0.0, Epot=0.0, Ecoul=0.0, Exx=0.0, Exc=0.0, Eefield=0.0, Emfield=0.0, Econf=0.0, Etot=0.0;
  double Eold=0.0;

  bool usediis=true, useadiis=true, diiscomb=false;
  uDIIS diis(S,Sinvh,diiscomb,usediis,diiseps,diisthr,useadiis,true,diisorder);
  double diiserr;

  // Density matrices
  arma::mat P, Pa, Pb;

  for(int i=1;i<=maxit;i++) {
    printf("\n**** Iteration %i ****\n\n",i);

    // Form density matrix
    Pa=scf::form_density(Caocc,nela);
    Pb=scf::form_density(Cbocc,nelb);
    if(Pb.n_rows == 0)
      Pb.zeros(Pa.n_rows,Pa.n_cols);
    P=Pa+Pb;

    chkpt.write("P",P);
    chkpt.write("Pa",Pa);
    chkpt.write("Pb",Pb);

    printf("Tr Pa = %f\n",arma::trace(Pa*S));
    if(nelb)
      printf("Tr Pb = %f\n",arma::trace(Pb*S));
    fflush(stdout);

    Ekin=arma::trace(P*T);
    Epot=arma::trace(P*Vnuc);
    Eefield=arma::trace(P*Vel);
    Emfield=arma::trace(P*Vmag)-Bz/2.0*(nela-nelb);
    Econf=arma::trace(P*Vconf);

    // Form Coulomb matrix
    timer.set();
    arma::mat J(basis.coulomb(P));
    double tJ(timer.get());
    Ecoul=0.5*arma::trace(P*J);
    printf("Coulomb energy %.10e % .6f\n",Ecoul,tJ);
    fflush(stdout);

    chkpt.write("J",J);

    // Form exchange matrix
    timer.set();
    arma::mat Ka, Kb;
    if(kfrac!=0.0 || kshort!=0.0) {
      Ka.zeros(Caocc.n_rows,Caocc.n_rows);
      Kb.zeros(Caocc.n_rows,Caocc.n_rows);
      if(kfrac!=0.0)
        Ka+=kfrac*basis.exchange(Pa);
      if(omega!=0.0)
        Ka+=kshort*basis.rs_exchange(Pa);

      if(nelb) {
        if(restr && nela==nelb) {
          Kb=Ka;
        } else {
          if(kfrac!=0.0)
            Kb+=kfrac*basis.exchange(Pb);
          if(omega!=0.0)
            Kb+=kshort*basis.rs_exchange(Pb);
        }
      }

      double tK(timer.get());
      Exx=0.5*arma::trace(Pa*Ka);
      if(Kb.n_rows == Pb.n_rows && Kb.n_cols == Pb.n_cols)
        Exx+=0.5*arma::trace(Pb*Kb);
      printf("Exchange energy %.10e % .6f\n",Exx,tK);
    } else {
      Exx=0.0;
    }
    fflush(stdout);

    chkpt.write("Ka",Ka);
    chkpt.write("Kb",Kb);

    // Exchange-correlation
    Exc=0.0;
    arma::mat XCa, XCb;
    if(dft) {
      timer.set();
      double nelnum;
      double ekin;
      if(restr && nela==nelb) {
        grid.eval_Fxc(x_func, xpars, c_func, cpars, P, XCa, Exc, nelnum, ekin, dftthr);
        XCb=XCa;
      } else {
        grid.eval_Fxc(x_func, xpars, c_func, cpars, Pa, Pb, XCa, XCb, Exc, nelnum, ekin, nelb>0, dftthr);
      }
      double txc(timer.get());
      printf("DFT energy %.10e % .6f\n",Exc,txc);
      printf("Error in integrated number of electrons % e\n",nelnum-nela-nelb);
      if(ekin!=0.0)
        printf("Error in integral of kinetic energy density % e\n",ekin-Ekin);
    }
    fflush(stdout);
    chkpt.write("XCa",XCb);
    chkpt.write("XCb",XCb);

    // Fock matrices
    arma::mat Fa(H0+J);
    arma::mat Fb(H0+J);
    if(Ka.n_rows == Fa.n_rows) {
      Fa+=Ka;
    }
    if(Kb.n_rows == Fb.n_rows) {
      Fb+=Kb;
    }
    if(dft) {
      Fa+=XCa;
      if(nelb>0) {
        Fb+=XCb;
      }
    }
    if(Bz!=0.0) {
      // Add in the B*Sz term
      Fa-=Bz*S/2.0;
      Fb+=Bz*S/2.0;
    }

    // m averaging?
    if(maverage) {
      Fa=scf::fock_symmetry_average(Fa,l_idx);
      Fb=scf::fock_symmetry_average(Fb,l_idx);
    }
    // Enforce symmetry of Fock matrix
    if(symm) {
      Fa=scf::enforce_fock_symmetry(Fa,dsym);
      Fb=scf::enforce_fock_symmetry(Fb,dsym);
    }

    // ROHF update to Fock matrix
    if(restr && nela!=nelb)
      scf::ROHF_update(Fa,Fb,P,Sh,Sinvh,nela,nelb);

    chkpt.write("Fa",Fa);
    chkpt.write("Fb",Fb);

    // Update energy
    Etot=Ekin+Epot+Eefield+Emfield+Ecoul+Exx+Exc+Enucr+Econf;
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

    // Update DIIS
    timer.set();
    diis.update(Fa,Fb,Pa,Pb,Etot,diiserr);
    printf("DIIS error is %e, update done in %.6f\n",diiserr,timer.get());
    fflush(stdout);

    // Solve DIIS to get Fock update
    timer.set();
    diis.solve_F(Fa,Fb);
    printf("DIIS solution done in %.6f\n",timer.get());
    fflush(stdout);

    // Have we converged? Note that DIIS error is still wrt full space, not active space.
    bool convd=(diiserr<convthr) && (std::abs(dE)<convthr);

    // Damping?
    if(dampfock != 1.0 && diiserr >= dampthr) {
      printf("Damping off-diagonal elements of Fock matrix by % .3f\n",dampfock);
      if(nela && Fa.n_rows > (size_t) nela) {
        arma::mat Ca(arma::join_rows(Caocc, Cavirt));
        arma::mat focka_mo(Ca.t()*Fa*Ca);
        focka_mo.submat(0,nela,nela-1,focka_mo.n_rows-1) *= dampfock;
        focka_mo.submat(nela,0,focka_mo.n_rows-1,nela-1) *= dampfock;
        Fa = S*Ca*focka_mo*Ca.t()*S;
      }
      if(nelb && Fb.n_rows > (size_t) nelb) {
        arma::mat Cb(arma::join_rows(Cbocc, Cbvirt));
        arma::mat fockb_mo(Cb.t()*Fb*Cb);
        fockb_mo.submat(0,nelb,nelb-1,fockb_mo.n_rows-1) *= dampfock;
        fockb_mo.submat(nelb,0,fockb_mo.n_rows-1,nelb-1) *= dampfock;
        Fb = S*Cb*fockb_mo*Cb.t()*S;
      }
    }

    // Diagonalize Fock matrix to get new orbitals
    timer.set();
    arma::mat Ca, Cb;
    if(symm)
      scf::eig_gsym_sub(Ea,Ca,Fa,Sinvh,dsym);
    else
      scf::eig_gsym(Ea,Ca,Fa,Sinvh);
    // Enforce occupation according to specified symmetry
    if(i<readocc) {
      scf::enforce_occupations(Ca,Ea,S,occnuma,occsym);
    }

    if(restr && nela==nelb) {
      Eb=Ea;
      Cb=Ca;
    } else {
      if(symm)
        scf::eig_gsym_sub(Eb,Cb,Fb,Sinvh,dsym);
      else
        scf::eig_gsym(Eb,Cb,Fb,Sinvh);
    }
    // Enforce occupation according to specified symmetry
    if(i<readocc) {
      scf::enforce_occupations(Cb,Eb,S,occnumb,occsym);
    }

    chkpt.write("Ca",Ca);
    chkpt.write("Cb",Cb);
    chkpt.write("Ea",Ea);
    chkpt.write("Eb",Eb);

    Caocc=Ca.cols(0,nela-1);
    if(Ca.n_cols>(size_t) nela)
      Cavirt=Ca.cols(nela,Ca.n_cols-1);
    if(nelb>0)
      Cbocc=Cb.cols(0,nelb-1);
    if(Cb.n_cols>(size_t) nelb)
      Cbvirt=Cb.cols(nelb,Cb.n_cols-1);
    if(symm)
      printf("Subspace diagonalization done in %.6f\n",timer.get());
    else
      printf("Full diagonalization done in %.6f\n",timer.get());

    if(Ea.n_elem>(size_t)nela)
      printf("Alpha HOMO-LUMO gap is % .3f eV\n",(Ea(nela)-Ea(nela-1))*HARTREEINEV);
    if(nelb && Eb.n_elem>(size_t)nelb)
      printf("Beta  HOMO-LUMO gap is % .3f eV\n",(Eb(nelb)-Eb(nelb-1))*HARTREEINEV);
    fflush(stdout);

    printf("\n");
    printf("Alpha orbital symmetries\n");
    classify_orbitals(Caocc,lvals,mvals,lmidx);
    if(nelb>0) {
      printf("\n");
      printf("Beta orbital symmetries\n");
      classify_orbitals(Cbocc,lvals,mvals,lmidx);
    }
    printf("\n");

    if(convd)
      break;
  }

  printf("%-21s energy: % .16f\n","Kinetic",Ekin);
  printf("%-21s energy: % .16f\n","Nuclear attraction",Epot);
  printf("%-21s energy: % .16f\n","Nuclear repulsion",Enucr);
  printf("%-21s energy: % .16f\n","Coulomb",Ecoul);
  printf("%-21s energy: % .16f\n","Exact exchange",Exx);
  printf("%-21s energy: % .16f\n","Exchange-correlation",Exc);
  printf("%-21s energy: % .16f\n","Electric field",Eefield);
  printf("%-21s energy: % .16f\n","Magnetic field",Emfield);
  printf("%-21s energy: % .16f\n", "Confinement potential",Econf);
  printf("%-21s energy: % .16f\n","Total",Etot);
  printf("%-21s energy: % .16f\n","Virial ratio",-Etot/Ekin);

  printf("\n");
  printf("Electronic dipole     moment % .16e\n",-arma::trace(dip*P));
  printf("Electronic quadrupole moment % .16e\n",-arma::trace(quad*P));

  // Electron density at nucleus
  if(Z!=0) {
    double nanuc=basis.nuclear_density(Pa)(0);
    double nbnuc=basis.nuclear_density(Pb)(0);
    double nnuc=basis.nuclear_density(P)(0);

    double dnanuc=basis.nuclear_density_gradient(Pa)(0);
    double dnbnuc=basis.nuclear_density_gradient(Pb)(0);
    double dnnuc=basis.nuclear_density_gradient(P)(0);

    printf("Electron density          at nucleus % .10e % .10e % .10e\n",nanuc,nbnuc,nnuc);
    printf("Electron density gradient at nucleus % .10e % .10e % .10e\n",dnanuc,dnbnuc,dnnuc);
    printf("Cusp condition is %.10f\n",-1.0/(2*Z)*dnnuc/nnuc);
  }

  // Calculate <r^2> matrix
  arma::mat rinvmat(basis.radial_integral(-1));
  arma::mat rmat(basis.radial_integral(1));
  arma::mat rsqmat(basis.radial_integral(2));
  arma::mat rcbmat(basis.radial_integral(3));
  // rms sizes
  arma::vec rinva(arma::ones<arma::vec>(Caocc.n_cols)/arma::diagvec(arma::trans(Caocc)*rinvmat*Caocc));
  arma::vec ra(arma::diagvec(arma::trans(Caocc)*rmat*Caocc));
  arma::vec rmsa(arma::sqrt(arma::diagvec(arma::trans(Caocc)*rsqmat*Caocc)));
  arma::vec rcba(arma::pow(arma::diagvec(arma::trans(Caocc)*rcbmat*Caocc),1.0/3.0));

  arma::vec rinvb, rb, rmsb, rcbb;
  if(nelb) {
    rinvb=arma::ones<arma::vec>(Cbocc.n_cols)/arma::diagvec(arma::trans(Cbocc)*rinvmat*Cbocc);
    rb=arma::diagvec(arma::trans(Cbocc)*rmat*Cbocc);
    rmsb=arma::sqrt(arma::diagvec(arma::trans(Cbocc)*rsqmat*Cbocc));
    rcbb=arma::pow(arma::diagvec(arma::trans(Cbocc)*rcbmat*Cbocc),1.0/3.0);
  }

  printf("\nOccupied orbital analysis:\n");
  printf("Alpha orbitals\n");
  printf("%2s %13s %12s %12s %12s %12s\n","io","energy","1/<r^-1>","<r>","sqrt(<r^2>)","cbrt(<r^3>)");
  for(int io=0;io<nela;io++) {
    printf("%2i % e %e %e %e %e\n",(int) io+1, Ea(io), rinva(io), ra(io), rmsa(io), rcba(io));
  }
  printf("Beta orbitals\n");
  for(int io=0;io<nelb;io++) {
    printf("%2i % e %e %e %e %e\n",(int) io+1, Eb(io), rinvb(io), rb(io), rmsb(io), rcbb(io));
  }

  /*
  // Test orthonormality
  arma::mat Smo(Ca.t()*S*Ca);
  Smo-=arma::eye<arma::mat>(Smo.n_rows,Smo.n_cols);
  printf("Alpha orthonormality deviation is %e\n",arma::norm(Smo,"fro"));
  Smo=(Cb.t()*S*Cb);
  Smo-=arma::eye<arma::mat>(Smo.n_rows,Smo.n_cols);
  printf("Beta orthonormality deviation is %e\n",arma::norm(Smo,"fro"));
  */

  return 0;
}
