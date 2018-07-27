#include "basis.h"

using namespace helfem;

int main(int argc, char **argv) {
  int Z=1;
  int Nnodes=10;
  int der_order=0;
  int Nelem=5;
  double Rmax=50;
  int lmax=0;
  int mmax=0;
  int igrid=1;
  double zexp=2.0;

  arma::mat Sold, Vold, Told;
  std::vector<arma::mat> pteiold;

  for(int nquad=10;nquad<=1e5;nquad*=2) {  
    atomic::basis::TwoDBasis basis(Z, Nnodes, der_order, nquad, Nelem, Rmax, lmax, mmax, igrid, zexp);

    // Form overlap matrix
    arma::mat S(basis.overlap());
    // Form nuclear attraction energy matrix
    arma::mat Vnuc(basis.nuclear());
    // Form kinetic energy matrix
    arma::mat T(basis.kinetic());

    basis.compute_tei(false);
    std::vector<arma::mat> ptei(basis.get_prim_tei());

    for(int iel=0;iel<Nelem;iel++) {
      size_t idx=iel*Nelem + iel;

      std::ostringstream fname;
      fname << "ptei_" << iel << "_" << iel << "_" << nquad << ".dat";
      ptei[idx].save(fname.str(),arma::raw_ascii);

      //ptei[idx].print(fname.str());
    }
    
    if(Sold.n_elem) {
      Sold-=S;
      Vold-=Vnuc;
      Told-=T;
      for(size_t i=0;i<pteiold.size();i++)
        pteiold[i]-=ptei[i];

      printf("nquad = %i\n",nquad);
      printf("dS = %e\n",arma::norm(Sold,"fro"));
      printf("dV = %e\n",arma::norm(Vold,"fro"));
      printf("dT = %e\n",arma::norm(Told,"fro"));

      arma::vec dnorm(pteiold.size());
      for(size_t i=0;i<pteiold.size();i++)
        dnorm(i)=arma::norm(pteiold[i],"fro");

      arma::uword maxind;
      double dmax=dnorm.max(maxind);
      printf("deri[%i] max = %e\n",(int) maxind, dmax);
      
      for(int iel=0;iel<Nelem;iel++) {
        size_t idx=iel*Nelem + iel;
        printf("deri[%i] = %e\n",(int) idx, dnorm(idx));
      }
    }
    Sold=S;
    Vold=Vnuc;
    Told=T;
    pteiold=ptei;
  }

  return 0;
}
