#include "../general/cmdline.h"
#include "basis.h"
#include <boost/timer/timer.hpp>

using namespace helfem;

arma::mat form_density(const arma::mat & C, size_t nocc) {
  if(C.n_cols<nocc)
    throw std::logic_error("Not enough orbitals!\n");

  if(nocc>0)
    return C.cols(0,nocc-1)*arma::trans(C.cols(0,nocc-1));
  else
    return arma::zeros<arma::mat>(C.n_rows,C.n_rows);
}

void eig_gsym(arma::vec & E, arma::mat & C, const arma::mat & F, const arma::mat & Sinvh) {
  arma::mat Forth(Sinvh.t()*F*Sinvh);
  arma::eig_sym(E,C,Forth);
  C=Sinvh*C;
}

int main(int argc, char **argv) {
  cmdline::parser parser;

  // full option name, no short option, description, argument required
  parser.add<int>("Z", 0, "nuclear charge");
  parser.add<int>("nela", 0, "number of alpha electrons");
  parser.add<int>("nelb", 0, "number of beta  electrons");
  parser.add<int>("lmax", 0, "maximum l quantum number");
  parser.add<int>("mmax", 0, "maximum m quantum number");
  parser.add<double>("Rmax", 0, "practical infinity");
  parser.add<int>("nelem", 0, "number of elements");
  parser.add<int>("nnodes", 0, "number of nodes per element");
  parser.add<int>("der_order", 0, "level of derivative continuity");
  parser.add<int>("nquad", 0, "number of quadrature points");
  parser.add<double>("damp", 0, "damping factor");
  parser.parse_check(argc, argv);

  // Get parameters
  double Rmax(parser.get<double>("Rmax"));
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

  // Nuclear charge
  int Z(parser.get<int>("Z"));
  // Number of occupied states
  int nela(parser.get<int>("nela"));
  int nelb(parser.get<int>("nelb"));
  // Damping factor
  double damp(parser.get<double>("damp"));

  printf("Running calculation with Rmax=%e and %i elements.\n",Rmax,Nelem);
  printf("Using %i point quadrature rule.\n",Nquad);
  printf("Basis set composed of %i nodes with %i:th derivative continuity.\n",Nnodes,der_order);
  printf("This means using primitive polynomials of order %i.\n",Nnodes*(der_order+1)-1);

  printf("Angular grid spanning from l=0..%i, m=%i..%i.\n",lmax,-mmax,mmax);

  basis::TwoDBasis basis(Z, Nnodes, der_order, Nquad, Nelem, Rmax, lmax, mmax);
  printf("Basis set contains %i functions\n",(int) basis.Nbf());

  printf("Nuclear charge is %i\n",Z);
  printf("Number of electrons is %i %i\n",nela,nelb);

  boost::timer::cpu_timer timer;

  // Form overlap matrix
  arma::mat S(basis.overlap());
  // Form nuclear attraction energy matrix
  arma::mat Vnuc(basis.nuclear());
  // Form kinetic energy matrix
  arma::mat T(basis.kinetic());

  // Form Hamiltonian
  arma::mat H0(T+Vnuc);

  printf("One-electron matrices formed in %.6f\n",timer.elapsed().wall*1e-9);

  // Get half-inverse
  timer.start();
  arma::mat Sinvh(basis.Sinvh());
  printf("Half-inverse formed in %.6f\n",timer.elapsed().wall*1e-9);

  // Diagonalize Hamiltonian
  arma::vec Ea, Eb;
  arma::mat Ca, Cb;
  eig_gsym(Ea,Ca,H0,Sinvh);
  Eb=Ea;
  Cb=Ca;

  arma::uword nena(std::min((arma::uword) nela+4,Ea.n_elem-1));
  arma::uword nenb(std::min((arma::uword) nelb+4,Eb.n_elem-1));
  Ea.subvec(0,nena).t().print("Alpha orbital energies");
  Eb.subvec(0,nenb).t().print("Beta  orbital energies");

  printf("Computing two-electron integrals\n");
  fflush(stdout);
  timer.start();
  basis.compute_tei();
  printf("Done in %.6f\n",timer.elapsed().wall*1e-9);

  arma::mat Paold, Pbold;
  double Ekin, Epot, Ecoul, Exx, Etot;
  double Eold=0.0;
  for(int i=0;i<1000;i++) {
    printf("\n**** Iteration %i ****\n\n",i);

    // Form density matrix
    arma::mat Panew(form_density(Ca,nela));
    arma::mat Pbnew(form_density(Cb,nelb));

    // Calculate <r^2>
    //printf("<r^2> is %e\n",arma::trace(basis.radial_integral(4)*Pnew));

    // Damp update
    arma::mat Pa, Pb;
    if(i==0) {
      Pa=Panew;
      Pb=Pbnew;
    } else {
      Pa=(1-damp)*Paold+damp*Panew;
      Pb=(1-damp)*Pbold+damp*Pbnew;
    }
    Paold=Pa;
    Pbold=Pb;
    arma::mat P(Pa+Pb);

    printf("Tr Pa = %f\n",arma::trace(Pa*S));
    printf("Tr Pb = %f\n",arma::trace(Pb*S));

    // Form Coulomb matrix
    arma::mat J(basis.coulomb(P));
    // Form exchange matrix
    arma::mat Ka(basis.exchange(Pa));
    arma::mat Kb(basis.exchange(Pb));
    // Fock matrices
    arma::mat Fa(H0+J+Ka);
    arma::mat Fb(H0+J+Kb);

    Ekin=arma::trace(P*T);
    Epot=arma::trace(P*Vnuc);
    Ecoul=0.5*arma::trace(P*J);
    Exx=0.5*(arma::trace(Pa*Ka)+arma::trace(Pb*Kb));
    Etot=Ekin+Epot+Ecoul+Exx;

    printf("Coulomb energy %.10f\n",Ecoul);
    printf("Exchange energy %.10f\n",Exx);

    if(i>0)
      printf("Energy changed by %e\n",Etot-Eold);
    Eold=Etot;

    /*
    arma::mat Jmo(Ca.t()*J*Ca);
    arma::mat Kmo(Ca.t()*Ka*Ca);
    Jmo.submat(0,0,10,10).print("Jmo");
    Kmo.submat(0,0,10,10).print("Kmo");

    S.print("S");
    T.print("T");
    Vnuc.print("Vnuc");
    Ca.print("Ca");
    Pa.print("Pa");
    J.print("J");
    Ka.print("Ka");

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

    // Diagonalize Fock matrix to get new orbitals
    eig_gsym(Ea,Ca,Fa,Sinvh);
    eig_gsym(Eb,Cb,Fb,Sinvh);

    // Check commutator
    arma::mat comma(arma::trans(Sinvh)*(Fa*Panew*S-S*Panew*Fa)*Sinvh);
    arma::mat commb(arma::trans(Sinvh)*(Fb*Pbnew*S-S*Pbnew*Fb)*Sinvh);
    double diiserra(arma::max(arma::max(arma::abs(comma))));
    double diiserrb(arma::max(arma::max(arma::abs(commb))));

    printf("Diis error is %e %e\n",diiserra,diiserrb);
    fflush(stdout);

    if(std::max(diiserra,diiserrb)<1e-6)
      break;
  }

  printf("%-21s energy: % .16e\n","Kinetic",Ekin);
  printf("%-21s energy: % .16e\n","Nuclear attraction",Epot);
  printf("%-21s energy: % .16e\n","Coulomb",Ecoul);
  printf("%-21s energy: % .16e\n","Exchange",Exx);
  printf("%-21s energy: % .16e\n","Total",Etot);
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
