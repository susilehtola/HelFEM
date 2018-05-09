#include "../general/polynomial.h"
#include "../general/chebyshev.h"
#include "quadrature.h"

using namespace helfem;

size_t get_Nbf(const arma::vec & r, const arma::mat & bf, int noverlap) {
  // Number of elements is
  size_t Nelem=r.n_elem-1;
  // Count the number of basis functions
  size_t Nbf=bf.n_cols*Nelem;
  // .. minus the number of functions that overlap between elements
  Nbf-=(Nelem-1)*noverlap;

  return Nbf;
}

void get_functions(size_t iel, const arma::mat & bf, int noverlap, size_t & ifirst, size_t & ilast) {
  // The first function will be
  ifirst=iel*(bf.n_cols-noverlap);
  // and the last one
  ilast=ifirst+bf.n_cols-1;
}

arma::mat overlap(const arma::vec & r, const arma::vec & x, const arma::vec & wx, const arma::mat & bf, int noverlap) {  
  // Build overlap matrix
  size_t Nbf(get_Nbf(r,bf,noverlap));
  arma::mat S(Nbf,Nbf);
  S.zeros();

  // Loop over elements
  for(size_t iel=0;iel<r.n_elem-1;iel++) {
    // Get the primitive overlap matrix
    arma::mat Sel(quadrature::radial_integral(r(iel),r(iel+1),2,x,wx,bf));

    // Where are we in the matrix?
    size_t ifirst, ilast;
    get_functions(iel,bf,noverlap,ifirst,ilast);
    S.submat(ifirst,ifirst,ilast,ilast)+=Sel;

    //printf("Element %i: functions %i - %i\n",(int) iel,(int) ifirst, (int) ilast);
  }

  return S;
}

arma::mat nuclear(const arma::vec & r, const arma::vec & x, const arma::vec & wx, const arma::mat & bf, int noverlap, int Z) {
  // Build nuclear attraction matrix
  size_t Nbf(get_Nbf(r,bf,noverlap));
  arma::mat V(Nbf,Nbf);
  V.zeros();

  // Loop over elements
  for(size_t iel=0;iel<r.n_elem-1;iel++) {
    // Get the primitive overlap matrix
    arma::mat Vel(quadrature::radial_integral(r(iel),r(iel+1),1,x,wx,bf));

    // Where are we in the matrix?
    size_t ifirst, ilast;
    get_functions(iel,bf,noverlap,ifirst,ilast);
    V.submat(ifirst,ifirst,ilast,ilast)-=Z*Vel;
  }

  return V;
}

arma::mat kinetic(const arma::vec & r, const arma::vec & x, const arma::vec & wx, const arma::mat & bf, const arma::mat & dbf, int noverlap, int l) {
  // Build kinetic energy matrix
  size_t Nbf(get_Nbf(r,bf,noverlap));
  arma::mat T(Nbf,Nbf);
  T.zeros();

  // Loop over elements
  for(size_t iel=0;iel<r.n_elem-1;iel++) {
    // Get the primitive overlap matrix
    arma::mat Tel(quadrature::derivative_integral(r(iel),r(iel+1),x,wx,dbf));

    // Where are we in the matrix?
    size_t ifirst, ilast;
    get_functions(iel,bf,noverlap,ifirst,ilast);
    T.submat(ifirst,ifirst,ilast,ilast)+=Tel;
  }

  // Put in the l(l+1) terms
  int lfac(l*(l+1));
  if(lfac>0) {
    for(size_t iel=0;iel<r.n_elem-1;iel++) {
      // Get the primitive overlap matrix
      arma::mat Lel(quadrature::radial_integral(r(iel),r(iel+1),0,x,wx,bf));
      
      // Where are we in the matrix?
      size_t ifirst, ilast;
      get_functions(iel,bf,noverlap,ifirst,ilast);
      T.submat(ifirst,ifirst,ilast,ilast)+=lfac*Lel;
    }
  }

  // Put in the remaining factor 1/2
  return 0.5*T;
}

arma::mat remove_edges(const arma::mat & M, int noverlap) {
  // Drop the last noverlap rows and columns
  return M.submat(0,0,M.n_rows-noverlap,M.n_cols-noverlap);
}

int main(int argc, char **argv) {
  if(argc!=6) {
    printf("Usage: %s Rmax Nel Nnode Nder Nquad\n",argv[0]);
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
  // which means the number of overlapping functions is
  int noverlap=der_order+1;

  // Order of quadrature rule
  int Nquad=atoi(argv[5]);

  printf("Running calculation with Rmax=%e and %i elements.\n",Rmax,Nelem);
  printf("Using %i point quadrature rule.\n",Nquad);
  printf("Basis set composed of %i nodes with %i:th derivative continuity.\n",Nnodes,der_order);
  printf("This means using primitive polynomials of order %i.\n",Nnodes*(der_order+1)-1);
  
  // Nuclear charge
  int Z=1;
  // l quantum number
  int l=0;

  // Radial grid
#if 0
  arma::vec r(arma::linspace<arma::vec>(0,Rmax,Nelem+1));
#else
  arma::vec r(arma::exp(arma::linspace<arma::vec>(0,log(Rmax),Nelem+1))-arma::ones<arma::vec>(Nelem+1));
#endif
  
  // Quadrature rule
  arma::vec x, wx;
  chebyshev::chebyshev(Nquad,x,wx);

  // Basis polynomials
  arma::mat bf_C(polynomial::hermite_coeffs(Nnodes,der_order));
  // First and second derivative
  arma::mat dbf_C(polynomial::derivative_coeffs(bf_C,1));
  bf_C.print("Basis expansion");
  dbf_C.print("Basis derivative expansion");

  // Evaluate polynomials at quadrature points
  arma::mat bf(polynomial::polyval(bf_C,x));
  arma::mat dbf(polynomial::polyval(dbf_C,x));

  bf.print("Basis polynomial");
  dbf.print("Basis polynomial derivative");

  size_t Nbf(get_Nbf(r,bf,noverlap));
  printf("Basis set contains %i functions\n",(int) Nbf);
  
  // Form overlap matrix
  arma::mat S(overlap(r,x,wx,bf,noverlap));
  // Form nuclear attraction energy matrix
  arma::mat V(nuclear(r,x,wx,bf,noverlap,Z));
  // Form kinetic energy matrix
  arma::mat T(kinetic(r,x,wx,bf,dbf,noverlap,l));

  // Form Hamiltonian
  arma::mat H(T+V);

  // Get rid of the spurious trail elements
  S=remove_edges(S,noverlap);
  T=remove_edges(T,noverlap);
  V=remove_edges(V,noverlap);
  H=remove_edges(H,noverlap);

  //S.print("Overlap");
  //T.print("Kinetic");
  //V.print("Nuclear");
  //H.print("Hamiltonian");

  // Form orthonormal basis
  arma::vec Sval;
  arma::mat Svec;
  arma::eig_sym(Sval,Svec,S);

  //Sval.print("S eigenvalues");
  printf("Smallest value of overlap matrix is % e, condition number is %e\n",Sval(0),Sval(Sval.n_elem-1)/Sval(0));
  printf("Smallest and largest bf norms are %e and %e\n",arma::min(arma::abs(arma::diagvec(S))),arma::max(arma::abs(arma::diagvec(S))));
  
  // Form half-inverse
  arma::mat Sinvh(Svec * arma::diagmat(arma::pow(Sval, -0.5)) * arma::trans(Svec));

  // Form orthonormal Hamiltonian
  arma::mat Horth(arma::trans(Sinvh)*H*Sinvh);

  // Diagonalize Hamiltonian
  arma::vec E;
  arma::mat C;
  arma::eig_sym(E,C,Horth);

  printf("First eigenvalues in Rydberg\n");
  for(size_t i=0;i<8;i++)
    printf("%i % 12.8f % 12.8f\n",(int) i+1,2*E(i),2*(E(i)+0.5/std::pow(i+1,2)));

  // Go back to non-orthonormal basis
  C=Sinvh*C;

  // Test orthonormality
  arma::mat Smo(C.t()*S*C);
  Smo-=arma::eye<arma::mat>(Smo.n_rows,Smo.n_cols);
  printf("Orbital orthonormality devation is %e\n",arma::norm(Smo,"fro"));

  return 0;
}
