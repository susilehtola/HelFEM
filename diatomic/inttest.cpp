#include "quadrature.h"
#include "../general/polynomial.h"
#include "../general/chebyshev.h"
#include "../general/gaunt.h"

using namespace helfem;

void run(double Rhalf, double mumax, int n_quad) {
  int Z1=1;
  int Z2=0;
  
  // Basis functions on [0, mumax]: mu/mumax, (mumax-mu)/mumax.
  double mumin=0.0;

  // Get primitive polynomial representation for LIP
  arma::mat bf_C=polynomial::hermite_coeffs(2, 0);
  // Derivatives
  arma::mat df_C=polynomial::derivative_coeffs(bf_C, 1);

  // Get quadrature rule
  arma::vec xq, wq;
  chebyshev::chebyshev(n_quad,xq,wq);

  gaunt::Gaunt gaunt(0,2,0);
  
  // Overlap matrix
  arma::mat R12(diatomic::quadrature::radial_integral(mumin,mumax,1,2,xq,wq,polynomial::polyval(bf_C,xq)));
  arma::mat R10(diatomic::quadrature::radial_integral(mumin,mumax,1,0,xq,wq,polynomial::polyval(bf_C,xq)));
  // Overlap matrix by quadrature is then
  arma::mat Squad(std::pow(Rhalf,3)*(R12-R10*gaunt.cosine2_coupling(0,0,0,0)));

  // Analytical overlap matrix is
  arma::mat Sanal(2,2);
  Sanal(0,0) = 2 * pow(cosh(mumax), 3) - 6 * cosh(mumax) + 4.0;
  Sanal(0,1) = 3 * mumax * sinh(mumax) * pow(cosh(mumax), 2) - 2 * pow(cosh(mumax), 3) - 3 * mumax * sinh(mumax) + 6 * cosh(mumax) - 4;
  Sanal(1,0) = Sanal(0,1);
  Sanal(1,1) = (9 * pow(mumax, 2) + 2) * pow(cosh(mumax), 3) - 6 * mumax * sinh(mumax) * pow(cosh(mumax), 2) + (-0.9e1 * pow(mumax, 2) - 6) * cosh(mumax) + 6 * mumax * sinh(mumax) + 4;
  Sanal*=std::pow(Rhalf,3)/(27*mumax*mumax);

  Squad.print("Quadrature overlap");
  Sanal.print("Analytical overlap");
  arma::mat dS(Squad-Sanal);
  dS.print("Overlap error");

  // Kinetic energy matrix
  double mulen(mumax/2);
  arma::mat Tquad(diatomic::quadrature::radial_integral(mumin,mumax,1,0,xq,wq,polynomial::polyval(df_C,xq))/(mulen*mulen)*Rhalf/2.0);
  arma::mat Tanal(2,2);
  Tanal(0,0) = 0.3e1 / 0.2e1 * cosh(mumax) - 0.3e1 / 0.2e1;
  Tanal(0,1) = -0.3e1 / 0.2e1 * cosh(mumax) + 0.3e1 / 0.2e1;
  Tanal(1,0) = Tanal(0,1);
  Tanal(1,1) = 0.3e1 / 0.2e1 * cosh(mumax) - 0.3e1 / 0.2e1;
  Tanal/=3*mumax*mumax/Rhalf;

  Tquad.print("Quadrature kinetic");
  Tanal.print("Analytical kinetic");
  arma::mat dT(Tquad-Tanal);
  dT.print("Kinetic error");

  arma::mat Vquad(-2*Rhalf*Rhalf*(Z1+Z2)*diatomic::quadrature::radial_integral(mumin,mumax,1,1,xq,wq,polynomial::polyval(bf_C,xq)));
  arma::mat Vanal(2,2);
  Vanal(0,0) = (-pow(cosh(mumax), 0.2e1) + pow(mumax, 0.2e1) + 0.1e1) * (Z1 + Z2) / 2.0;
  Vanal(0,1) = sinh(mumax) * (Z1 + Z2) * (-cosh(mumax) * mumax + sinh(mumax)) / 2.0;
  Vanal(1,0) = Vanal(0,1);
  Vanal(1,1) = (Z1 + Z2) * (-0.2e1 * pow(cosh(mumax), 0.2e1) * pow(mumax, 0.2e1) + 0.2e1 * sinh(mumax) * cosh(mumax) * mumax - pow(cosh(mumax), 0.2e1) + pow(mumax, 0.2e1) + 0.1e1) / 2.0;
  Vanal/=2*mumax*mumax/(Rhalf*Rhalf);

  Vquad.print("Quadrature nuclear");
  Vanal.print("Analytical nuclear");
  arma::mat dV(Vquad-Vanal);
  dV.print("Nuclear error");

  // Form eigenvectors
  arma::vec Sval;
  arma::mat Svec;
  arma::eig_sym(Sval,Svec,Sanal);
  arma::mat H0(Tanal+Vanal);
  
  arma::mat Sorth(Svec*arma::diagmat(arma::pow(Sval,-0.5))*arma::trans(Svec));
  arma::mat Horth(arma::trans(Sorth)*H0*Sorth);

  arma::vec Hval;
  arma::mat Hvec;
  arma::eig_sym (Hval, Hvec, Horth);
  Hval.print("Eigenvalues");
  

  return;

  // Get inner integral by quadrature
  arma::mat teiinner(diatomic::quadrature::twoe_inner_integral(mumin,mumax,xq,wq,bf_C,0));

  // Test against analytical integrals. r values are
  arma::vec mu(0.5*mumax*arma::ones<arma::vec>(xq.n_elem)+0.5*mumax*xq);

  /*
  // The inner integral should give the following
  arma::mat teiishould(r.n_elem,4);
  teiishould.col(0)=(arma::ones<arma::vec>(r.n_elem) - r/R + 1.0/3.0*arma::square(r)/(R*R));
  teiishould.col(1)=r%(-2.0*r + 3*R*arma::ones<arma::vec>(r.n_elem))/(6.0*R*R);
  teiishould.col(2)=teiishould.col(1);
  teiishould.col(3)=arma::square(r)/(3.0*R*R);

  r.save("r.dat",arma::raw_ascii);
  teiinner.save("teii_q.dat",arma::raw_ascii);
  teiishould.save("teii.dat",arma::raw_ascii);

  teiishould-=teiinner;
  printf("Error in inner integral is %e\n",arma::norm(teiishould,"fro"));

  arma::mat teiq(diatomic::quadrature::twoe_integral(mumin,mumax,xq,wq,bf_C,0));

  arma::mat tei(4,4);
  // Maple gives the following integrals for L=0, in units of R

  // 1111
  tei(0,0) = 47.0/180.0;
  // 1112
  tei(0,1) = 11/360.0;
  // 1121
  tei(0,2) = tei(0,1);
  // 1122
  tei(0,3) = 1.0/90.0;

  // 1211
  tei(1,0) = 1.0/10.0;
  // 1212
  tei(1,1) = 1.0/40.0;
  // 1221
  tei(1,2) = tei(1,1);
  // 1222
  tei(1,3) = 1.0/60.0;

  // 2111
  tei(2,0) = tei(1,0);
  // 2112
  tei(2,1) = tei(1,1);
  // 2121
  tei(2,2) = tei(1,2);
  // 2122
  tei(2,3) = tei(1,3);

  // 2211
  tei(3,0) = 3.0/20.0;
  // 2212
  tei(3,1) = 7.0/120.0;
  // 2221
  tei(3,2) = tei(3,1);
  // 2222
  tei(3,3) = 1.0/15.0;

  // Symmetrization and coefficient
  tei=4.0*M_PI*(tei+tei.t())*R;

  tei.print("Analytical");
  teiq.print("Quadrature");
  teiq-=tei;
  teiq.print("Difference");
  */
}

int main(int argc, char **argv) {
  if(argc!=4) {
    printf("Usage: %s nquad Rhalf mumax\n",argv[0]);
    return 1;
  }

  int nquad(atoi(argv[1]));
  double Rhalf(atof(argv[2]));
  double mumax(atof(argv[3]));

  printf("nquad = %i, Rhalf = %e, mumax = % e\n",nquad,Rhalf,mumax);
  
  run(Rhalf,mumax,nquad);
  return 0;
}
