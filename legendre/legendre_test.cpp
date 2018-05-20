#include "legendre_pq.h"

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
  
  return 0;
}
