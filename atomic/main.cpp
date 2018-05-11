#include "basis.h"
#include <boost/timer/timer.hpp>

using namespace helfem;

int main(int argc, char **argv) {
  if(argc!=9) {
    printf("Usage: %s Rmax Nel Nnode Nder Nquad Z lmax mmax\n",argv[0]);
    return 1;
  }

  // Maximum R
  double Rmax=atof(argv[1]);
  // Number of elements
  int Nelem=atoi(argv[2]);

  // Number of nodes
  int Nnodes=atoi(argv[3]);
  // Derivative order
  int der_order=atoi(argv[4]);

  // Order of quadrature rule
  int Nquad=atoi(argv[5]);

  // Nuclear charge
  int Z(atoi(argv[6]));

  // Angular grid
  int lmax(atoi(argv[7]));
  int mmax(atoi(argv[8]));


  printf("Running calculation with Rmax=%e and %i elements.\n",Rmax,Nelem);
  printf("Using %i point quadrature rule.\n",Nquad);
  printf("Basis set composed of %i nodes with %i:th derivative continuity.\n",Nnodes,der_order);
  printf("This means using primitive polynomials of order %i.\n",Nnodes*(der_order+1)-1);

  printf("Angular grid spanning from l=0..%i, m=%i..%i.\n",lmax,-mmax,mmax);

  basis::TwoDBasis basis(Z, Nnodes, der_order, Nquad, Nelem, Rmax, lmax, mmax);
  printf("Basis set contains %i functions\n",(int) basis.Nbf());

  boost::timer::cpu_timer timer;

  // Form overlap matrix
  arma::mat S(basis.overlap());
  // Form nuclear attraction energy matrix
  arma::mat V(basis.nuclear());
  // Form kinetic energy matrix
  arma::mat T(basis.kinetic());

  // Form Hamiltonian
  arma::mat H(T+V);

  printf("One-electron matrices formed in %.6f\n",timer.elapsed().wall*1e-9);

  // Get half-inverse
  timer.start();
  arma::mat Sinvh(basis.Sinvh());
  printf("Half-inverse formed in %.6f\n",timer.elapsed().wall*1e-9);

  // Form orthonormal Hamiltonian
  arma::mat Horth(arma::trans(Sinvh)*H*Sinvh);

  // Diagonalize Hamiltonian
  arma::vec E;
  arma::mat C;
  arma::eig_sym(E,C,Horth);

  printf("First eigenvalues in Rydberg\n");
  for(size_t i=0;i<std::min((arma::uword) 8,E.n_elem);i++)
    printf("%i % 12.8f % 12.8f\n",(int) i+1,2*E(i),2*(E(i)+0.5/std::pow(i+1,2)));

  // Go back to non-orthonormal basis
  C=Sinvh*C;

  // Form density matrix
  arma::mat P(C.col(0)*arma::trans(C.col(0)));
  printf("Density contains %f electrons\n",arma::trace(P*S));

  printf("Computing two-electron integrals\n");
  fflush(stdout);
  timer.start();
  basis.compute_tei();
  printf("Done in %.6f\n",timer.elapsed().wall*1e-9);

  // Form Coulomb matrix
  printf("Forming Coulomb matrix\n");
  fflush(stdout);
  timer.start();
  arma::mat J(basis.coulomb(P));
  printf("Done in %.6f\n",timer.elapsed().wall*1e-9);

  printf("Self-interaction Coulomb is %e\n",arma::trace(P*J));
  fflush(stdout);

  // Form exchange matrix
  printf("Forming exchange matrix\n");
  fflush(stdout);
  timer.start();
  arma::mat K(basis.exchange(P));
  printf("Done in %.6f\n",timer.elapsed().wall*1e-9);
  printf("Self-interaction exchange is %e\n",arma::trace(P*K));
  fflush(stdout);

  arma::mat Jmo(C.t()*J*C);
  Jmo.print("J_MO");
  arma::mat Kmo(C.t()*K*C);
  Kmo.print("K_MO");

  // Test orthonormality
  arma::mat Smo(C.t()*S*C);
  Smo-=arma::eye<arma::mat>(Smo.n_rows,Smo.n_cols);
  printf("Orbital orthonormality devation is %e\n",arma::norm(Smo,"fro"));

  return 0;
}
