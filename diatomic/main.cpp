#include "../general/polynomial.h"

using namespace helfem;

int main(void) {
  arma::vec x(arma::linspace<arma::vec>(-.5,.5,3));
  x.print("x");
  
  arma::vec p;
  p.clear();
  p << 1;
  arma::vec f0=polynomial::polyval(p,x);
  f0.print("f0");

  p.clear();
  p << 0 << 1;
  arma::vec fx=polynomial::polyval(p,x);
  fx.print("fx");

  p.clear();
  p << 0 << 0 << 1;
  arma::vec fx2=polynomial::polyval(p,x);
  fx2.print("fx^2");
  
  arma::vec c(arma::ones<arma::vec>(5));
  c.print("c");
  for(int o=0;o<3;o++) {
    arma::vec d(polynomial::derivative_coeffs(c,o));
    d.print("d");
  }
  
  arma::mat l(polynomial::hermite_coeffs(2,0));
  l.print("L");
  arma::mat h21(polynomial::hermite_coeffs(2,1));
  h21.print("H21");
  arma::mat h22(polynomial::hermite_coeffs(2,2));
  h22.print("H22");
  return 0;
}
