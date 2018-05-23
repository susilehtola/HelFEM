#include "legendre_pq.h"
#include "legendre.h"

#include <armadillo>

int main(void) {
  arma::vec x;
  x << 5.0 << 2.0 << 1.000001;

  int lmax=4;
  
  for(size_t i=0;i<x.n_elem;i++) {
    arma::mat lm(lmax+1,lmax+1);
    lm.zeros();

    for(int l=0;l<=lmax;l++)
      for(int m=0;m<=l;m++)
        lm(l,m)=::legendre::legendreP_prolate(l,m,x(i));

    printf("\n\tPlm x=%.6f\n",x(i));
    for(size_t r=0;r<lm.n_rows;r++) {
      for(size_t c=0;c<lm.n_cols;c++)
        printf(" % .12e",lm(r,c));
      printf("\n");
    }
  }
  for(size_t i=0;i<x.n_elem;i++) {
    arma::mat lm(lmax+1,lmax+1);
    lm.zeros();

    for(int l=0;l<=lmax;l++)
      for(int m=0;m<=lmax;m++)
        lm(l,m)=::legendre::legendreQ_prolate(l,m,x(i));

    printf("\n\tQlm x=%.6f\n",x(i));
    for(size_t r=0;r<lm.n_rows;r++) {
      for(size_t c=0;c<lm.n_cols;c++)
        printf(" % .12e",lm(r,c));
      printf("\n");
    }
  }

  x=arma::linspace<arma::vec>(-0.999,0.999,1999);
  x.save("x.dat",arma::raw_ascii);
  for(int l=0;l<=lmax;l++)
    for(int m=0;m<=std::min(l,::legendre::legendrePQ_max_m());m++) {
      printf("l=%i, m=%i\n",l,m);
      arma::vec Plm(helfem::legendre::legendreP(l,m,x));
      arma::vec Qlm(helfem::legendre::legendreQ(l,m,x));

      std::ostringstream oss;
      oss << l << "_" << m;
      Plm.save("P_" + oss.str() + ".dat",arma::raw_ascii);
      Qlm.save("Q_" + oss.str() + ".dat",arma::raw_ascii);
    }

  x=arma::linspace<arma::vec>(1.1,40,389);
  x.save("xpro.dat",arma::raw_ascii);
  for(int l=0;l<=lmax;l++)
    for(int m=0;m<=std::min(l,::legendre::legendrePQ_max_m());m++) {
      arma::vec Plm(helfem::legendre::legendreP_prolate(l,m,x));
      arma::vec Qlm(helfem::legendre::legendreQ_prolate(l,m,x));

      std::ostringstream oss;
      oss << l << "_" << m;
      Plm.save("Ppro_" + oss.str() + ".dat",arma::raw_ascii);
      Qlm.save("Qpro_" + oss.str() + ".dat",arma::raw_ascii);
    }

  return 0;
}
