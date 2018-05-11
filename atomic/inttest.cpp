#include "quadrature.h"
#include "../general/polynomial.h"
#include "../general/chebyshev.h"

using namespace helfem;

void run(double R, int n_quad) {
  // Basis functions on [0, R]: x/R, (R-x)/R.

  // Get primitive polynomial representation for LIP
  arma::mat bf_C=polynomial::hermite_coeffs(2, 0);

  // Get quadrature rule
  arma::vec xq, wq;
  chebyshev::chebyshev(n_quad,xq,wq);
  // Evaluate polynomials at quadrature points
  arma::mat bf=polynomial::polyval(bf_C,xq);

  // r values are
  arma::vec r(0.5*R*arma::ones<arma::vec>(xq.n_elem)+0.5*R*xq);

  // Get inner integral by quadrature
  arma::mat teiinner(quadrature::twoe_inner_integral(0,R,xq,wq,bf,0));

  // Test against analytical integrals
  arma::mat teiishould(r.n_elem,4);
  teiishould.col(0)=arma::pow(r,5)/(5.0*std::pow(R,2))-arma::pow(r,4)/(2.0*R)+arma::pow(r,3)/3.0;
  teiishould.col(1)=-arma::pow(r,5)/(5.0*std::pow(R,2))+arma::pow(r,4)/(4.0*R);
  teiishould.col(2)=teiishould.col(1);
  teiishould.col(3)=arma::pow(r,5)/(5.0*std::pow(R,2));

  teiinner.save("teii_q.dat",arma::raw_ascii);
  teiishould.save("teii.dat",arma::raw_ascii);

  arma::mat teiq(quadrature::twoe_integral(0,R,xq,wq,bf,0));

  arma::mat tei(4,4);
  // Maple gives the following integrals for L=0, in units of R^5

  // 1111
  tei(0,0) = 1.0/1008.0;
  // 1112
  tei(0,1) = 1.0/1440.0;
  // 1121
  tei(0,2) = tei(0,1);
  // 1122
  tei(0,3) = 1.0/1260.0;

  // 1211
  tei(1,0) = 1.0/560.0;
  // 1212
  tei(1,1) = 17.0/10080.0;
  // 1221
  tei(1,2) = tei(1,1);
  // 1222
  tei(1,3) = 1.0/360.0;

  // 2111
  tei(2,0) = tei(1,0);
  // 2112
  tei(2,1) = tei(1,1);
  // 2121
  tei(2,2) = tei(1,2);
  // 2122
  tei(2,3) = tei(1,3);

  // 2211
  tei(3,0) = 37.0/5040.0;
  // 2212
  tei(3,1) = 13.0/1440.0;
  // 2221
  tei(3,2) = tei(3,1);
  // 2222
  tei(3,3) = 1.0/45.0;

  // Symmetrization and coefficient
  tei=4.0*M_PI*(tei+tei.t())*std::pow(R,5);

  tei.print("Analytical");
  teiq.print("Quadrature");
  teiq-=tei;
  teiq.print("Difference");
}

int main(int argc, char **argv) {
  if(argc!=3) {
    printf("Usage: %s nquad R\n",argv[0]);
    return 1;
  }

  int nquad(atoi(argv[1]));
  double R(atof(argv[2]));
  run(R,nquad);
  return 0;
}
