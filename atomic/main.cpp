#include "../general/cmdline.h"
#include "../general/diis.h"
#include "basis.h"
#include <boost/timer/timer.hpp>
#include <cfloat>
#include <SymEigsSolver.h>  // Also includes <MatOp/DenseGenMatProd.h>

//#define SPARSE

using namespace helfem;

arma::mat form_density(const arma::mat & C, size_t nocc) {
  if(C.n_cols<nocc)
    throw std::logic_error("Not enough orbitals!\n");
  else if(nocc>0)
    return C.cols(0,nocc-1)*arma::trans(C.cols(0,nocc-1));
  else // nocc=0
    return arma::zeros<arma::mat>(C.n_rows,C.n_rows);
}

void eig_gsym(arma::vec & E, arma::mat & C, const arma::mat & F, const arma::mat & Sinvh, size_t neig) {
  // Form matrix in orthonormal basis
  arma::mat Forth(Sinvh.t()*F*Sinvh);

#ifndef SPARSE
  printf("Dense diagonalization.\n");
  if(!arma::eig_sym(E,C,Forth))
    throw std::logic_error("Eigendecomposition failed!\n");

  // Drop the virtuals
  if(neig>E.n_elem) {
    E=E.subvec(0,neig-1);
    C=C.cols(0,neig-1);
  }
#else
  printf("Sparse diagonalization.\n");
  // Construct matrix operation object using the wrapper class DenseGenMatProd
  DenseGenMatProd<double> op(Forth);

  // Construct eigen solver object
  SymEigsSolver< double, SMALLEST_ALGE, DenseGenMatProd<double> > eigs(&op, neig, neig+20);

  // Initialize and compute
  eigs.init();
  int nconv = eigs.compute(10000,1e-6);

  // Retrieve results
  if(nconv == neig) {
    E = eigs.eigenvalues();
    C = eigs.eigenvectors();

    // Values are returned in largest magnitude order, so they need to
    // be resorted
    arma::uvec Eord(arma::sort_index(E));
    E=E(Eord);
    C=C.cols(Eord);

  } else {
    std::ostringstream oss;
    oss << "Eigendecomposition failed: only " << nconv << " eigenvalues converged!\n";
    throw std::logic_error(oss.str());
  }

  if(E.n_elem != neig) {
    std::ostringstream oss;
    oss << "Not enough eigenvalues: asked for " << neig << ", got " << E.n_elem << "!\n";
    throw std::logic_error(oss.str());
  }
  if(C.n_cols != neig) {
    std::ostringstream oss;
    oss << "Not enough eigenvectors: asked for " << neig << ", got " << C.n_cols << "!\n";
    throw std::logic_error(oss.str());
  }
#endif

  // Return to non-orthonormal basis
  C=Sinvh*C;
}

int main(int argc, char **argv) {
  cmdline::parser parser;

  // full option name, no short option, description, argument required
  parser.add<int>("Z", 0, "nuclear charge");
  parser.add<int>("Zl", 0, "left-hand nuclear charge");
  parser.add<int>("Zr", 0, "right-hand nuclear charge");
  parser.add<double>("Rmid", 0, "distance of nuclei from center");
  parser.add<int>("nela", 0, "number of alpha electrons");
  parser.add<int>("nelb", 0, "number of beta  electrons");
  parser.add<int>("lmax", 0, "maximum l quantum number");
  parser.add<int>("mmax", 0, "maximum m quantum number");
  parser.add<double>("Rmax", 0, "practical infinity");
  parser.add<int>("grid", 0, "type of grid: 1 for linear, 2 for quadratic, 3 for polynomial, 4 for logarithmic");
  parser.add<double>("zexp", 0, "parameter in radial grid");
  parser.add<int>("nelem0", 0, "number of elements between center and off-center nuclei");
  parser.add<int>("nelem", 0, "number of elements");
  parser.add<int>("nnodes", 0, "number of nodes per element");
  parser.add<int>("der_order", 0, "level of derivative continuity");
  parser.add<int>("nquad", 0, "number of quadrature points");
  parser.add<double>("Ez", 0, "electric field");
  parser.parse_check(argc, argv);

  // Get parameters
  double Rmax(parser.get<double>("Rmax"));
  int igrid(parser.get<int>("grid"));
  double zexp(parser.get<double>("zexp"));
  double Ez(parser.get<double>("Ez"));
  // Number of elements
  int Nelem0(parser.get<int>("nelem0"));
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

  // Nuclear charge
  int Z(parser.get<int>("Z"));
  int Zl(parser.get<int>("Zl"));
  int Zr(parser.get<int>("Zr"));
  double Rhalf(parser.get<double>("Rmid"));
  // Number of occupied states
  int nela(parser.get<int>("nela"));
  int nelb(parser.get<int>("nelb"));

  printf("Running calculation with Rmax=%e and %i elements.\n",Rmax,Nelem);
  printf("Using %i point quadrature rule.\n",Nquad);
  printf("Basis set composed of %i nodes with %i:th derivative continuity.\n",Nnodes,der_order);
  printf("This means using primitive polynomials of order %i.\n",Nnodes*(der_order+1)-1);

  printf("Angular grid spanning from l=0..%i, m=%i..%i.\n",lmax,-mmax,mmax);

  basis::TwoDBasis basis;
  if((Zl!=0 || Zr!=0) && Rhalf!=0.0)
    basis=basis::TwoDBasis(Z, Nnodes, der_order, Nquad, Nelem0, Nelem, Rmax, lmax, mmax, igrid, zexp, Zl, Zr, Rhalf);
  else
    basis=basis::TwoDBasis(Z, Nnodes, der_order, Nquad, Nelem, Rmax, lmax, mmax, igrid, zexp);
  printf("Basis set contains %i functions\n",(int) basis.Nbf());

  double Enucr=Z*(Zl+Zr)/Rhalf + Zl*Zr/(2*Rhalf);
  printf("Central nuclear charge is %i\n",Z);
  printf("Left- and right-hand nuclear charges are %i and %i at distance % .3f from center\n",Zl,Zr,Rhalf);
  printf("Nuclear repulsion energy is %e\n",Enucr);
  printf("Number of electrons is %i %i\n",nela,nelb);

  boost::timer::cpu_timer timer;

  // Form overlap matrix
  arma::mat S(basis.overlap());
  // Form nuclear attraction energy matrix
  arma::mat Vnuc(basis.nuclear());
  // Form electric field coupling matrix
  arma::mat Vel(basis.electric(Ez));
  // Form kinetic energy matrix
  arma::mat T(basis.kinetic());

  // Form Hamiltonian
  arma::mat H0(T+Vnuc+Vel);

  printf("One-electron matrices formed in %.6f\n",timer.elapsed().wall*1e-9);

  // Get half-inverse
  timer.start();
  arma::mat Sinvh(basis.Sinvh());
  printf("Half-inverse formed in %.6f\n",timer.elapsed().wall*1e-9);

  // Number of states to find
  int nstates(std::min((arma::uword) nela+20,Sinvh.n_cols-1));

  // Diagonalize Hamiltonian
  timer.start();
  arma::vec Ea, Eb;
  arma::mat Ca, Cb;
  eig_gsym(Ea,Ca,H0,Sinvh,nstates);
  Eb=Ea;
  Cb=Ca;
  printf("Diagonalization done in %.6f\n",timer.elapsed().wall*1e-9);

  arma::uword nena(std::min((arma::uword) nela+4,Ea.n_elem-1));
  arma::uword nenb(std::min((arma::uword) nelb+4,Eb.n_elem-1));
  Ea.subvec(0,nena).t().print("Alpha orbital energies");
  Eb.subvec(0,nenb).t().print("Beta  orbital energies");

  printf("Computing two-electron integrals\n");
  fflush(stdout);
  timer.start();
  basis.compute_tei();
  printf("Done in %.6f\n",timer.elapsed().wall*1e-9);

  double Ekin, Epot, Ecoul, Exx, Efield, Etot;
  double Eold=0.0;

  double diiseps=1e-2, diisthr=1e-3;
  bool usediis=true, useadiis=true;
  bool diis_c1=false;
  int diisorder=5;
  uDIIS diis(S,Sinvh,usediis,diis_c1,diiseps,diisthr,useadiis,true,diisorder);
  double diiserr;

  for(int i=0;i<1000;i++) {
    printf("\n**** Iteration %i ****\n\n",i);

    // Form density matrix
    arma::mat Pa(form_density(Ca,nela));
    arma::mat Pb(form_density(Cb,nelb));
    arma::mat P(Pa+Pb);

    // Calculate <r^2>
    //printf("<r^2> is %e\n",arma::trace(basis.radial_integral(2)*Pnew));

    printf("Tr Pa = %f\n",arma::trace(Pa*S));
    printf("Tr Pb = %f\n",arma::trace(Pb*S));

    Ekin=arma::trace(P*T);
    Epot=arma::trace(P*Vnuc);
    Efield=arma::trace(P*Vel);

    // Form Coulomb matrix
    timer.start();
    arma::mat J(basis.coulomb(P));
    double tJ(timer.elapsed().wall*1e-9);
    Ecoul=0.5*arma::trace(P*J);
    printf("Coulomb energy %.10e % .6f\n",Ecoul,tJ);

    // Form exchange matrix
    timer.start();
    arma::mat Ka(basis.exchange(Pa));
    arma::mat Kb;
    if(nelb)
      Kb=basis.exchange(Pb);
    else
      Kb.zeros(Cb.n_rows,Cb.n_rows);
    double tK(timer.elapsed().wall*1e-9);
    Exx=0.5*(arma::trace(Pa*Ka)+arma::trace(Pb*Kb));
    printf("Exchange energy %.10e % .6f\n",Exx,tK);

    // Fock matrices
    arma::mat Fa(H0+J+Ka);
    arma::mat Fb(H0+J+Kb);
    Etot=Ekin+Epot+Efield+Ecoul+Exx+Enucr;

    if(i>0)
      printf("Energy changed by %e\n",Etot-Eold);
    Eold=Etot;

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
    timer.start();
    diis.update(Fa,Fb,Pa,Pb,Etot,diiserr);
    printf("Diis error is %e\n",diiserr);
    fflush(stdout);
    // Solve DIIS to get Fock update
    diis.solve_F(Fa,Fb);
    printf("DIIS update and solution done in %.6f\n",timer.elapsed().wall*1e-9);

    // Diagonalize Fock matrix to get new orbitals
    timer.start();
    eig_gsym(Ea,Ca,Fa,Sinvh,nstates);
    eig_gsym(Eb,Cb,Fb,Sinvh,nstates);
    printf("Diagonalization done in %.6f\n",timer.elapsed().wall*1e-9);

    double convthr(1e-8);
    if(i>0 && diiserr<convthr && std::abs(Etot-Eold)<convthr)
      break;
  }

  printf("%-21s energy: % .16f\n","Kinetic",Ekin);
  printf("%-21s energy: % .16f\n","Nuclear attraction",Epot);
  printf("%-21s energy: % .16f\n","Nuclear repulsion",Enucr);
  printf("%-21s energy: % .16f\n","Coulomb",Ecoul);
  printf("%-21s energy: % .16f\n","Exchange",Exx);
  printf("%-21s energy: % .16f\n","Electric field",Efield);
  printf("%-21s energy: % .16f\n","Total",Etot);
  Ea.subvec(0,nena).t().print("Alpha orbital energies");
  Eb.subvec(0,nenb).t().print("Beta  orbital energies");

  // Test orthonormality
  arma::mat Smo(Ca.t()*S*Ca);
  Smo-=arma::eye<arma::mat>(Smo.n_rows,Smo.n_cols);
  printf("Alpha orthonormality deviation is %e\n",arma::norm(Smo,"fro"));
  Smo=(Cb.t()*S*Cb);
  Smo-=arma::eye<arma::mat>(Smo.n_rows,Smo.n_cols);
  printf("Beta orthonormality deviation is %e\n",arma::norm(Smo,"fro"));

  return 0;
}
