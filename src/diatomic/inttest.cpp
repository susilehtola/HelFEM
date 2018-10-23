#include "quadrature.h"
#include "../general/polynomial.h"
#include "../general/chebyshev.h"
#include "../general/gaunt.h"
#include "../general/utils.h"

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

  arma::mat bf(polynomial::polyval(bf_C,xq));

  gaunt::Gaunt gaunt(0,2,0);

  // Overlap matrix
  arma::mat R12(diatomic::quadrature::radial_integral(mumin,mumax,1,2,xq,wq,bf));
  arma::mat R10(diatomic::quadrature::radial_integral(mumin,mumax,1,0,xq,wq,bf));
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

  arma::mat Vquad(-Rhalf*Rhalf*(Z1+Z2)*diatomic::quadrature::radial_integral(mumin,mumax,1,1,xq,wq,bf));
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

#if 0
  arma::mat teiq00(diatomic::quadrature::twoe_integral(mumin,mumax,0,0,xq,wq,bf_C,0,0));

  arma::mat tei00(4,4);
  // Maple gives the following integrals for L=0, in units of Rh^5

  // 1111
  tei00(0,0) = 1.3923429112033219754;
  // 1112
  tei00(0,1) = .78308409502963405351;
  // 1121
  tei00(0,2) = tei00(0,1);
  // 1122
  tei00(0,3) = .85873392147578315336;

  // 1211
  tei00(1,0) = 1.8875428732911552488;
  // 1212
  tei00(1,1) = 1.6829582741348087289;
  // 1221
  tei00(1,2) = tei00(1,1);
  // 1222
  tei00(1,3) = 2.9982552875981985777;

  // 2111
  tei00(2,0) = tei00(1,0);
  // 2112
  tei00(2,1) = tei00(1,1);
  // 2121
  tei00(2,2) = tei00(1,2);
  // 2122
  tei00(2,3) = tei00(1,3);

  // 2211
  tei00(3,0) = 6.3746987745315668278;
  // 2212
  tei00(3,1) = 8.5090203888416728384;
  // 2221
  tei00(3,2) = tei00(3,1);
  // 2222
  tei00(3,3) = 25.378321835379063539;

  // Symmetrization and coefficient
  tei00=(tei00+tei00.t());

  tei00.print("Analytical 00");
  teiq00.print("Quadrature 00");
  teiq00-=tei00;
  teiq00.print("Difference 00");


  // Cross-element integral: Plm is smaller mu, Qlm is larger mu
  arma::mat Plmq(diatomic::quadrature::Plm_radial_integral(mumin,mumax,0,xq,wq,bf,0,0));

  arma::mat Plma(2,2);
  Plma(0,0)=4.8567958819830275555;
  Plma(0,1)=8.9838462335747242395;
  Plma(1,0)=Plma(0,1);
  Plma(1,1)=50.385460175655368409;

  Plma.print("Analytical Plm integral");
  Plmq.print("Quadrature Plm integral");
  Plma=Plmq-Plma;
  Plma.print("Difference");

  arma::mat Plmq2(diatomic::quadrature::Plm_radial_integral(mumin,mumax,2,xq,wq,bf,0,0));
  arma::mat Plma2(2,2);
  Plma2(0,0)=1211.8772998862055871;
  Plma2(0,1)=7872.1033397310303422;
  Plma2(1,0)=Plma2(0,1);
  Plma2(1,1)=119271.19236080028659;

  Plma2.print("Analytical Plm 2 integral");
  Plmq2.print("Quadrature Plm 2 integral");
  Plma2=Plmq2-Plma2;
  Plma2.print("Difference");

  arma::mat Qlmq(diatomic::quadrature::Qlm_radial_integral(mumax,2*mumax,0,xq,wq,bf,0,0));
  arma::mat Qlma(2,2);
  Qlma(0,0)=1.6666542573040936436;
  Qlma(0,1)=.83333212258300176401;
  Qlma(1,0)=Qlma(0,1);
  Qlma(1,1)=1.6666663648383271959;

  std::cout.precision(14);
  std::cout.setf(std::ios::scientific);
  Qlma.raw_print(std::cout,"Analytical Qlm integral");
  Qlmq.raw_print(std::cout,"Quadrature Qlm integral");
  Qlma=Qlmq-Qlma;
  Qlma.raw_print(std::cout,"Difference");

  arma::mat Qlmq2(diatomic::quadrature::Qlm_radial_integral(mumax,2*mumax,2,xq,wq,bf,0,0));
  arma::mat Qlma2(2,2);
  Qlma2(0,0)=1.2095545080441514832e6;
  Qlma2(0,1)=4.8519826288623907719e6;
  Qlma2(1,0)=Qlma2(0,1);
  Qlma2(1,1)=4.9729378018894519212e7;

  Qlma2.print("Analytical Qlm 2 integral");
  Qlmq2.print("Quadrature Qlm 2 integral");
  Qlma2=Qlmq2-Qlma2;
  Qlma2.print("Difference");


  arma::mat cteiq00(utils::product_tei(Qlmq,Plmq));

  arma::mat ctei00(4,4);
  // 1111
  ctei00(0,0) = 8.0945995335640032329;
  // 1112
  ctei00(0,1) = 14.972965572152661017;
  // 1121
  ctei00(0,2) = ctei00(0,1);
  // 1122
  ctei00(0,3) = 83.975141707981885693;

  // 1211
  ctei00(1,0) = 4.0473240212852984875;
  // 1212
  ctei00(1,1) = 7.4865276507841307983;
  // 1221
  ctei00(1,2) = ctei00(1,1);
  // 1222
  ctei00(1,3) = 41.987822475500193061;

  // 2111
  ctei00(2,0) = ctei00(1,0);
  // 2112
  ctei00(2,1) = ctei00(1,1);
  // 2121
  ctei00(2,2) = ctei00(1,2);
  // 2122
  ctei00(2,3) = ctei00(1,3);

  // 2211
  ctei00(3,0) = 8.0946583373864097182;
  // 2212
  ctei00(3,1) = 14.973074344378482992;
  // 2221
  ctei00(3,2) = ctei00(3,1);
  // 2222
  ctei00(3,3) = 83.975751751665835726;

  ctei00.print("Analytical 00");
  cteiq00.print("Quadrature 00");
  cteiq00-=ctei00;
  cteiq00.print("Difference 00");
#endif
}

int main(int argc, char **argv) {
  if(argc!=3) {
    printf("Usage: %s nquad Rhalf\n",argv[0]);
    return 1;
  }

  int nquad(atoi(argv[1]));
  double Rhalf(atof(argv[2]));
  double mumax(5.0);

  printf("nquad = %i, Rhalf = %e, mumax = % e\n",nquad,Rhalf,mumax);

  run(Rhalf,mumax,nquad);
  return 0;
}
