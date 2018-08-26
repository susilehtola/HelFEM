#include "../general/cmdline.h"
#include "../general/constants.h"
#include "../general/diis.h"
#include "../general/dftfuncs.h"
#include "../general/timer.h"
#include "../general/elements.h"
#include "../general/scf_helpers.h"
#include "basis.h"
#include "dftgrid.h"
#include <cfloat>

//#define SPARSE

using namespace helfem;

int main(int argc, char **argv) {
  cmdline::parser parser;

  // full option name, no short option, description, argument required
  parser.add<std::string>("Z1", 0, "first nuclear charge", true);
  parser.add<std::string>("Z2", 0, "second nuclear charge", true);
  parser.add<double>("Rbond", 0, "internuclear distance", true);
  parser.add<bool>("angstrom", 0, "input distances in angstrom", false, false);
  parser.add<int>("nela", 0, "number of alpha electrons", false, 0);
  parser.add<int>("nelb", 0, "number of beta  electrons", false, 0);
  parser.add<int>("Q", 0, "charge state", false, 0);
  parser.add<int>("M", 0, "spin multiplicity", false, 0);
  parser.add<int>("lmax", 0, "maximum l quantum number", true);
  parser.add<int>("mmax", 0, "maximum m quantum number", true);
  parser.add<int>("lpad", 0, "padding for max l for more accurate Qlm recursion", false, 10);
  parser.add<double>("Rmax", 0, "practical infinity in au", false, 40.0);
  parser.add<int>("grid", 0, "type of grid: 1 for linear, 2 for quadratic, 3 for polynomial, 4 for logarithmic", false, 4);
  parser.add<double>("zexp", 0, "parameter in radial grid", false, 2.0);
  parser.add<int>("nelem", 0, "number of elements", true);
  parser.add<int>("nnodes", 0, "number of nodes per element", false, 6);
  parser.add<int>("der_order", 0, "level of derivative continuity", false, 0);
  parser.add<int>("nquad", 0, "number of quadrature points", false, 50);
  parser.add<int>("maxit", 0, "maximum number of iterations", false, 50);
  parser.add<double>("convthr", 0, "convergence threshold", false, 1e-7);
  parser.add<double>("Ez", 0, "electric dipole field", false, 0.0);
  parser.add<double>("Qzz", 0, "electric quadrupole field", false, 0.0);
  parser.add<std::string>("method", 0, "method to use", false, "HF");
  parser.add<int>("ldft", 0, "theta rule for dft quadrature (0 for auto)", false, 0);
  parser.add<int>("mdft", 0, "phi rule for dft quadrature (0 for auto)", false, 0);
  parser.add<int>("restricted", 0, "spin-restricted orbitals", false, -1);
  parser.add<double>("dftthr", 0, "density threshold for dft", false, 1e-12);
  parser.parse_check(argc, argv);

  // Get parameters
  double Rmax(parser.get<double>("Rmax"));
  int igrid(parser.get<int>("grid"));
  double zexp(parser.get<double>("zexp"));
  double Ez(parser.get<double>("Ez"));
  double Qzz(parser.get<double>("Qzz"));

  int maxit(parser.get<int>("maxit"));
  double convthr(parser.get<double>("convthr"));

  bool diag(parser.get<bool>("diag"));
  int restr(parser.get<int>("restricted"));

  // Number of elements
  int Nelem(parser.get<int>("nelem"));
  // Number of nodes
  int Nnodes(parser.get<int>("nnodes"));
  // Derivative order
  int der_order(parser.get<int>("der_order"));
  // Order of quadrature rule
  int Nquad(parser.get<int>("nquad"));
  // Angular grid
  int lmax(parser.get<int>("lmax"));
  int mmax(parser.get<int>("mmax"));
  int lpad(parser.get<int>("lpad"));

  // DFT angular grid
  int ldft(parser.get<int>("ldft"));
  int mdft(parser.get<int>("mdft"));
  double dftthr(parser.get<double>("dftthr"));

  // Nuclear charge
  int Z1(get_Z(parser.get<std::string>("Z1")));
  int Z2(get_Z(parser.get<std::string>("Z2")));
  double Rbond(parser.get<double>("Rbond"));
  // Number of occupied states
  int nela(parser.get<int>("nela"));
  int nelb(parser.get<int>("nelb"));
  int Q(parser.get<int>("Q"));
  int M(parser.get<int>("M"));

  std::string method(parser.get<std::string>("method"));

  if(parser.get<bool>("angstrom")) {
    // Convert to atomic units
    Rbond*=ANGSTROMINBOHR;
  }

  scf::parse_nela_nelb(nela,nelb,Q,M,Z1+Z2);
  if(restr==-1) {
    // If number of electrons differs then unrestrict
    restr=(nela==nelb);
  }

  std::vector<std::string> rcalc(2);
  rcalc[0]="unrestricted";
  rcalc[1]="restricted";

  printf("Running %s %s calculation with Rmax=%e and %i elements.\n",rcalc[restr].c_str(),method.c_str(),Rmax,Nelem);
  printf("Using %i point quadrature rule.\n",Nquad);
  printf("Basis set composed of %i nodes with %i:th derivative continuity.\n",Nnodes,der_order);
  printf("This means using primitive polynomials of order %i.\n",Nnodes*(der_order+1)-1);

  printf("Angular grid spanning from l=0..%i, m=%i..%i.\n",lmax,-mmax,mmax);

  diatomic::basis::TwoDBasis basis(Z1, Z2, Rbond, Nnodes, der_order, Nquad, Nelem, Rmax, lmax, mmax, igrid, zexp, lpad);
  printf("Basis set consists of %i angular shells composed of %i radial functions, totaling %i basis functions\n",(int) basis.Nang(), (int) basis.Nrad(), (int) basis.Nbf());

  printf("One-electron matrix requires %s\n",scf::memory_size(basis.mem_1el()).c_str());
  printf("Auxiliary one-electron integrals require %s\n",scf::memory_size(basis.mem_1el_aux()).c_str());
  printf("Auxiliary two-electron integrals require %s\n",scf::memory_size(basis.mem_2el_aux()).c_str());

  double Enucr=Z1*Z2/Rbond;
  printf("Left- and right-hand nuclear charges are %i and %i at distance % .3f\n",Z1,Z2,Rbond);
  printf("Nuclear repulsion energy is %e\n",Enucr);
  printf("Number of electrons is %i %i\n",nela,nelb);

  // Functional
  int x_func, c_func;
  ::parse_xc_func(x_func, c_func, method);
  ::print_info(x_func, c_func);

  bool dft=(x_func>0 || c_func>0);
  if(is_range_separated(x_func))
    throw std::logic_error("Range separated functionals are not supported.\n");
  // Fraction of exact exchange
  double kfrac(exact_exchange(x_func));

  helfem::dftgrid::DFTGrid grid;
  if(dft) {
    if(ldft==0)
      // Default value: we have 2*lmax from the bra and ket and 2 from
      // the volume element, and allow for 2*lmax from the
      // density/potential. Add in 3 more for a bit more accuracy.
      ldft=4*lmax+5;
    if(ldft<2*lmax+2)
      throw std::logic_error("Increase ldft to guarantee accuracy of quadrature!\n");

    if(mdft==0)
      // Default value: we have 2*mmax from the bra and ket, and allow
      // for 2*mmax from the density/potential. Add in a 3 to make
      // sure quadrature is still accurate for mmax=0
      mdft=4*mmax+3;
    if(mdft<2*mmax)
      throw std::logic_error("Increase mdft to guarantee accuracy of quadrature!\n");

    grid=helfem::dftgrid::DFTGrid(&basis,ldft,mdft);
  }

  Timer timer;

  // Form overlap matrix
  arma::mat S(basis.overlap());
  // Get half-inverse
  timer.set();
  arma::mat Sinvh(basis.Sinvh(!diag));
  printf("Half-inverse formed in %.6f\n",timer.get());
  {
    arma::mat Smo(Sinvh.t()*S*Sinvh);
    Smo-=arma::eye<arma::mat>(Smo.n_rows,Smo.n_cols);
    printf("Orbital orthonormality deviation is %e\n",arma::norm(Smo,"fro"));
  }
  arma::mat Sh(basis.Shalf(!diag));
  printf("Half-overlap formed in %.6f\n",timer.get());
  {
    arma::mat Smo(Sh.t()*Sinvh);
    Smo-=arma::eye<arma::mat>(Smo.n_rows,Smo.n_cols);
    printf("Half-overlap error is %e\n",arma::norm(Smo,"fro"));
  }

  if(dft) {
    arma::mat Snum(grid.eval_overlap());
    // Convert to orthonormalized basis since norms of basis functions vary over a huge scale
    Snum=arma::trans(Sinvh)*(Snum-S)*Sinvh;
    double Serr(arma::norm(Snum,"fro"));
    printf("Error in overlap matrix evaluated through xc grid is %e\n",Serr);
    fflush(stdout);
    if(Serr>=1e-10)
      throw std::logic_error("Increase size of DFT grid!\n");
  }

  // Form nuclear attraction energy matrix
  Timer tnuc;
  arma::mat Vnuc(basis.nuclear());

  // Dipole coupling
  arma::mat dip(basis.dipole_z());
  // Quadrupole coupling
  arma::mat quad(basis.quadrupole_zz());

  // Electric field coupling (minus sign cancels one from charge)
  arma::mat Vel(Ez*dip + Qzz*quad/3.0);
  // Form kinetic energy matrix
  arma::mat T(basis.kinetic());

  // Form Hamiltonian
  arma::mat H0(T+Vnuc+Vel);

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
    // Use core guess
    arma::mat C;
    scf::eig_gsym(Ea,C,H0,Sinvh);
    Caocc=C.cols(0,nela-1);
    if(C.n_cols>(size_t) nela)
      Cavirt=C.cols(nela,C.n_cols-1);

    // Beta guess
    if(nelb)
      Cbocc=Caocc.cols(0,nelb-1);
    Cbvirt = (nelb<nela) ? arma::join_rows(Caocc.cols(nelb,nela-1),Cavirt) : Cavirt;
    Eb=Ea;

    Ea.subvec(0,nena-1).t().print("Alpha orbital energies");
    Eb.subvec(0,nenb-1).t().print("Beta  orbital energies");
  }
  printf("Initial guess performed in %.6f\n",timer.get());

  printf("Computing two-electron integrals\n");
  fflush(stdout);
  timer.set();
  basis.compute_tei(kfrac!=0.0);
  printf("Done in %.6f\n",timer.get());

  double Ekin, Epot, Ecoul, Exx, Exc, Efield, Etot;
  double Eold=0.0;

  double diiseps=1e-2, diisthr=1e-3;
  bool usediis=true, useadiis=true;
  bool diis_c1=false;
  int diisorder=5;
  uDIIS diis(S,Sinvh,usediis,diis_c1,diiseps,diisthr,useadiis,true,diisorder);
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

    printf("Tr Pa = %f\n",arma::trace(Pa*S));
    if(nelb)
      printf("Tr Pb = %f\n",arma::trace(Pb*S));
    fflush(stdout);

    Ekin=arma::trace(P*T);
    Epot=arma::trace(P*Vnuc);
    Efield=arma::trace(P*Vel);

    // Form Coulomb matrix
    timer.set();
    arma::mat J(basis.coulomb(P));
    double tJ(timer.get());
    Ecoul=0.5*arma::trace(P*J);
    printf("Coulomb energy %.10e % .6f\n",Ecoul,tJ);
    fflush(stdout);

    // Form exchange matrix
    timer.set();
    arma::mat Ka, Kb;
    if(kfrac!=0.0) {
      Ka=kfrac*basis.exchange(Pa);

      if(nelb) {
        if(restr && nela==nelb)
          Kb=Ka;
        else
          Kb=kfrac*basis.exchange(Pb);
      } else
        Kb.zeros(Cbocc.n_rows,Cbocc.n_rows);
      double tK(timer.get());
      Exx=0.5*arma::trace(Pa*Ka);
      if(Kb.n_rows == Pb.n_rows && Kb.n_cols == Pb.n_cols)
        Exx+=0.5*arma::trace(Pb*Kb);
      printf("Exchange energy %.10e % .6f\n",Exx,tK);
    } else {
      Exx=0.0;
    }
    fflush(stdout);

    // Exchange-correlation
    Exc=0.0;
    arma::mat XCa, XCb;
    if(dft) {
      timer.set();
      double nelnum;
      double ekin;
      if(restr && nela==nelb) {
        grid.eval_Fxc(x_func, c_func, P, XCa, Exc, nelnum, ekin, dftthr);
        XCb=XCa;
      } else {
        grid.eval_Fxc(x_func, c_func, Pa, Pb, XCa, XCb, Exc, nelnum, ekin, nelb>0, dftthr);
      }
      double txc(timer.get());
      printf("DFT energy %.10e % .6f\n",Exc,txc);
      printf("Error in integrated number of electrons % e\n",nelnum-nela-nelb);
      if(ekin!=0.0)
        printf("Error in integral of kinetic energy density % e\n",ekin-Ekin);
    }
    fflush(stdout);

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

    // ROHF update to Fock matrix
    if(restr && nela!=nelb)
      scf::ROHF_update(Fa,Fb,P,Sh,Sinvh,nela,nelb);

    // Update energy
    Etot=Ekin+Epot+Efield+Ecoul+Exx+Exc+Enucr;
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

    // Diagonalize Fock matrix to get new orbitals
    timer.set();
    arma::mat Ca, Cb;
    scf::eig_gsym(Ea,Ca,Fa,Sinvh);
    if(restr && nela==nelb) {
      Eb=Ea;
      Cb=Ca;
    } else {
      scf::eig_gsym(Eb,Cb,Fb,Sinvh);
    }
    Caocc=Ca.cols(0,nela-1);
    if(Ca.n_cols>(size_t) nela)
      Cavirt=Ca.cols(nela,Ca.n_cols-1);
    if(nelb>0)
      Cbocc=Cb.cols(0,nelb-1);
    if(Cb.n_cols>(size_t) nelb)
      Cbvirt=Cb.cols(nelb,Cb.n_cols-1);
    printf("Full diagonalization done in %.6f\n",timer.get());

    if(Ea.n_elem>(size_t)nela)
      printf("Alpha HOMO-LUMO gap is % .3f eV\n",(Ea(nela)-Ea(nela-1))*HARTREEINEV);
    if(nelb && Eb.n_elem>(size_t)nelb)
      printf("Beta  HOMO-LUMO gap is % .3f eV\n",(Eb(nelb)-Eb(nelb-1))*HARTREEINEV);
    fflush(stdout);

    if(convd)
      break;
  }

  printf("%-21s energy: % .16f\n","Kinetic",Ekin);
  printf("%-21s energy: % .16f\n","Nuclear attraction",Epot);
  printf("%-21s energy: % .16f\n","Nuclear repulsion",Enucr);
  printf("%-21s energy: % .16f\n","Coulomb",Ecoul);
  printf("%-21s energy: % .16f\n","Exact exchange",Exx);
  printf("%-21s energy: % .16f\n","Exchange-correlation",Exc);
  printf("%-21s energy: % .16f\n","Electric field",Efield);
  printf("%-21s energy: % .16f\n","Total",Etot);
  printf("%-21s energy: % .16f\n","Virial ratio",Etot/Ekin);

  double eldip=arma::trace(dip*P);
  double nucdip=(-Z1+Z2)*Rbond/2.0;
  double elquad=arma::trace(quad*P);
  double nucquad=(-Z1+Z2)*Rbond*Rbond/4.0;

  printf("\n");
  printf("Electronic dipole     moment % .16e\n",-eldip);
  printf("Nuclear    dipole     moment % .16e\n",nucdip);
  printf("Total      dipole     moment % .16e\n",-eldip+nucdip);
  printf("Electronic quadrupole moment % .16e\n",-elquad);
  printf("Nuclear    quadrupole moment % .16e\n",nucquad);
  printf("Total      quadrupole moment % .16e\n",-elquad+nucquad);

  printf("\n");
  printf("Nuclear electron densities\n");
  arma::vec nucdena(basis.nuclear_density(Pa));
  arma::vec nucdenb(basis.nuclear_density(Pb));
  for(size_t i=0;i<nucdena.size();i++)
    printf(" % .10e % .10e % .10e\n",nucdena(i),nucdenb(i),nucdena(i)+nucdenb(i));

  /*
  // Calculate <r^2> matrix
  arma::mat rmat(basis.radial_integral(1));
  arma::mat rsqmat(basis.radial_integral(2));
  // rms sizes
  arma::vec ra(arma::sqrt(arma::diagvec(arma::trans(Caocc)*rmat*Caocc)));
  arma::vec rmsa(arma::sqrt(arma::diagvec(arma::trans(Caocc)*rsqmat*Caocc)));
  arma::vec rb, rmsb;
  if(nelb) {
    rb=arma::sqrt(arma::diagvec(arma::trans(Cbocc)*rmat*Cbocc));
    rmsb=arma::sqrt(arma::diagvec(arma::trans(Cbocc)*rsqmat*Cbocc));
  }

  printf("\nOccupied orbital analysis:\n");
  printf("%2s %13s %12s %12s %13s %12s %12s\n","io","energy","<r>","sqrt(<r^2>)","energy","<r>","sqrt(<r^2>)");
  for(int io=0;io<nelb;io++) {
    printf("%2i % e %e %e % e %e %e\n",(int) io+1, Ea(io), ra(io), rmsa(io), Eb(io), rb(io), rmsb(io));
  }
  for(int io=nelb;io<nela;io++) {
    printf("%2i % e %e %e\n",(int) io+1, Ea(io), ra(io), rmsa(io));
  }
  printf("\n");
  */

  Ea.subvec(0,nena-1).t().print("Alpha orbital energies");
  Eb.subvec(0,nenb-1).t().print("Beta  orbital energies");


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
