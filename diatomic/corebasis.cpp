#include "../general/cmdline.h"
#include "../general/constants.h"
#include "../general/timer.h"
#include "../general/elements.h"
#include "../general/scf_helpers.h"
#include "basis.h"
#include "dftgrid.h"
#include <cfloat>

using namespace helfem;

void eval(int Z1, int Z2, double Rbond, const polynomial_basis::PolynomialBasis * poly, int Nquad, int Nelem, double Rmax, const arma::ivec & lmmax, int igrid, double zexp, double Ez, double Qzz, int norb, double & E, arma::uword & nang, arma::uword & nrad, arma::vec & Eval) {

  int lpad=0;
  int symm=1;
  diatomic::basis::TwoDBasis basis(Z1, Z2, Rbond, poly, Nquad, Nelem, Rmax, lmmax, igrid, zexp, lpad, false);

  bool diag=true;
  // Symmetry indices
  std::vector<arma::uvec> dsym=basis.get_sym_idx(symm);

  // Form overlap matrix
  arma::mat S(basis.overlap());
  // Form kinetic energy matrix
  arma::mat T(basis.kinetic());
  // Get half-inverse
  arma::mat Sinvh(basis.Sinvh(!diag,symm));
  // Form nuclear attraction energy matrix
  arma::mat Vnuc(basis.nuclear());
  // Form Hamiltonian
  arma::mat H0(T+Vnuc);
  if(Ez!=0.0)
    H0+=Ez*basis.dipole_z();
  // Quadrupole coupling
  if(Qzz!=0.0)
    H0+=Qzz*basis.quadrupole_zz()/3.0;

  // Use core guess
  arma::vec Ev;
  arma::mat C;
  scf::eig_gsym_sub(Ev,C,H0,Sinvh,dsym);
  //scf::eig_gsym(Ev,C,H0,Sinvh);

  // Sanity check
  norb=std::min((int) Ev.n_elem,norb);
  Eval=Ev.subvec(0,norb-1);
  E=arma::sum(Eval);
  nang=basis.Nang();
  nrad=basis.Nrad();
}

int main(int argc, char **argv) {
  cmdline::parser parser;

  // full option name, no short option, description, argument required
  parser.add<std::string>("Z1", 0, "first nuclear charge", true);
  parser.add<std::string>("Z2", 0, "second nuclear charge", true);
  parser.add<double>("Rbond", 0, "internuclear distance", true);
  parser.add<bool>("angstrom", 0, "input distances in angstrom", false, false);
  parser.add<double>("Rmax", 0, "practical infinity in au", false, 40.0);
  parser.add<int>("grid", 0, "type of grid: 1 for linear, 2 for quadratic, 3 for polynomial, 4 for logarithmic", false, 1);
  parser.add<double>("zexp", 0, "parameter in radial grid", false, 1.0);
  parser.add<int>("nnodes", 0, "number of nodes per element", false, 15);
  parser.add<int>("primbas", 0, "primitive radial basis", false, 4);
  parser.add<int>("nquad", 0, "number of quadrature points", false, 0);
  parser.add<double>("Ez", 0, "electric dipole field", false, 0.0);
  parser.add<double>("Qzz", 0, "electric quadrupole field", false, 0.0);
  parser.add<int>("thresh", 0, "convergence threshold, 10 corresponds to 1e-10", false, 10);
  parser.add<int>("nadd", 0, "number of funcs to add", false, 2);
  parser.parse_check(argc, argv);

  // Get parameters
  double Rmax(parser.get<double>("Rmax"));
  int igrid(parser.get<int>("grid"));
  double zexp(parser.get<double>("zexp"));
  double Ez(parser.get<double>("Ez"));
  double Qzz(parser.get<double>("Qzz"));
  int primbas(parser.get<int>("primbas"));
  // Number of nodes
  int Nnodes(parser.get<int>("nnodes"));
  // Order of quadrature rule
  int Nquad(parser.get<int>("nquad"));
  int nadd(parser.get<int>("nadd"));
  int thresh(parser.get<int>("thresh"));

  if(nadd%2)
    printf("WARNING - Adding an odd number of functions at a time does not give a balanced description of gerade/ungerade orbitals and may give wrong results.\n");

  // Nuclear charge
  int Z1(get_Z(parser.get<std::string>("Z1")));
  int Z2(get_Z(parser.get<std::string>("Z2")));
  double Rbond(parser.get<double>("Rbond"));

  if(parser.get<bool>("angstrom")) {
    // Convert to atomic units
    Rbond*=ANGSTROMINBOHR;
  }

  // Determine orbitals
  std::vector<int> norbs(4);
  num_orbs(Z1, Z2, norbs[0], norbs[1], norbs[2], norbs[3]);
  for(size_t m=norbs.size()-1;m<norbs.size();m--)
    if(norbs[m]==0)
      norbs.erase(norbs.begin()+m);

  printf("Determining basis set for %s-%s at distance %e with Rmax=%e.\n",element_symbols[Z1].c_str(),element_symbols[Z2].c_str(),Rbond,Rmax);

  // Get primitive basis
  polynomial_basis::PolynomialBasis *poly(polynomial_basis::get_basis(primbas,Nnodes));

  if(Nquad==0)
    // Set default value
    Nquad=5*Nnodes;
  else if(Nquad<2*Nnodes)
    throw std::logic_error("Insufficient radial quadrature.\n");

  printf("Using %i point quadrature rule.\n",Nquad);

  // Determine number of elements
  int Nelem=1;

  // Final angular grid
  arma::ivec lmgrid(norbs.size());

  // Loop over threshold
  int ithr=0;
  int othr=-1;
  std::vector<bool> init(norbs.size(),true);

  while(ithr<=thresh) {
    double thr=1.0/std::pow(10.0,ithr);

    if(ithr!=othr) {
      printf("**** thr = %e ****\n",thr);
      othr=ithr;
    }
    bool cvd=true;

    // Loop over orbital types
    for(size_t m=norbs.size()-1;m<norbs.size();m--) {
      // Angular grid in test calculation
      arma::ivec lmmax;
      lmmax.ones(m+1);
      lmmax*=-1;
      // Start off value
      if(init[m]) {
        if(m<norbs.size()-1) {
          // Safe assumption: n(sigma) > n(pi) > n(delta) > n(phi)
          lmmax(m)=lmgrid(m+1);
        } else {
          // Start off with one function
          lmmax(m)=m;
        }
        init[m]=false;
      } else {
        // Use previous value
        lmmax(m)=lmgrid(m);
      }

      // Initial estimate
      double E;
      arma::vec Eval;
      arma::uword Nrad, Nang;
      eval(Z1, Z2, Rbond, poly, Nquad, Nelem, Rmax, lmmax, igrid, zexp, Ez, Qzz, norbs[m], E, Nrad, Nang, Eval);

      Eval.t().print("Initial eigenvalues");

      printf("Initial energy is %e\n",E);

      int iiter=0;
      double dE;
      while(true) {
        iiter++;

        double Ea, Er;
        arma::vec Eva, Evr;
        arma::uword Nra, Naa;
        arma::uword Nrr, Nar;

        arma::ivec lmtr(lmmax);
        lmtr(m)+=nadd;

        printf("m=%i iteration %i\n",(int) m,iiter);

        eval(Z1, Z2, Rbond, poly, Nquad, Nelem, Rmax, lmtr, igrid, zexp, Ez, Qzz, norbs[m], Ea, Nra, Naa, Eva);
        double dEa=Ea-E;
        printf("Addition of %i partial waves decreases energy by %e\n",nadd,dEa);

        eval(Z1, Z2, Rbond, poly, Nquad, Nelem+nadd, Rmax, lmmax, igrid, zexp, Ez, Qzz, norbs[m], Er, Nrr, Nar, Evr);
        double dEr=Er-E;
        printf("Addition of %i radial elements decreases energy by %e\n",nadd,dEr);

        dE=std::min(dEa,dEr);
        if(std::abs(dE)<thr)
          break;

        // Angular loop is not converged
        cvd=false;

        if(dEa==dE) {
          lmmax=lmtr;
          E=Ea;
          Eval=Eva;
          Nrad=Nra;
          Nang=Naa;
          printf("Basis set has now %i partial waves\n",lmmax(m));
        } else {
          Nelem+=nadd;
          E=Er;
          Eval=Evr;
          Nrad=Nrr;
          Nang=Nar;
          printf("Basis set has now %i radial elements\n",Nelem);
        }
        Eval.t().print("Current eigenvalues");
        printf("\n");
      }
      printf("m=%i is converged with %i elements and %i partial waves\n\n",(int) m,(int) Nelem,(int) lmmax(m));
      lmgrid(m)=lmmax(m);
    }

    if(cvd) {
      printf("\n");
      printf("An estimated accuracy of %e is achieved with\n",thr);
      printf("--grid=%i --zexp=%e --primbas=%i --nnodes=%i --nelem=%i --Rmax=%e --lmax=", igrid, zexp, primbas, Nnodes, (int) Nelem, Rmax);
      for(size_t m=0;m<lmgrid.n_elem;m++) {
        if(m)
          printf(",");
        printf("%i",(int) lmgrid(m));
      }
      printf("\n\n\n");
      ithr++;
    }
  }

  return 0;
}
