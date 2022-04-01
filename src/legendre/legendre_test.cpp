#include "Legendre_Wrapper.h"
#include "../general/spherical_harmonics.h"
#include <armadillo>

void get_coord(double Rh, double mu, double eta, double phi, double & x, double & y, double & z) {
  x=Rh*sinh(mu)*sin(eta)*cos(phi);
  y=Rh*sinh(mu)*sin(eta)*sin(phi);
  z=Rh*cosh(mu)*cos(eta);

  printf("% .16e, % .16e, % .16e\n",x,y,z);
}

arma::mat get_Plm(int lmax, int mmax, double xi) {
  arma::mat Plm(lmax+1,mmax+1);
  calc_Plm_arr(Plm.memptr(),lmax,mmax,xi);
  return Plm;
}

arma::mat get_Qlm(int lmax, int mmax, double xi) {
  arma::mat Qlm(lmax+1,mmax+1);
  calc_Qlm_arr(Qlm.memptr(),lmax,mmax,xi);
  return Qlm;
}

double factorial_ratio(int pmax, int pmin) {
  // Check consistency of arguments
  if(pmax < pmin)
    return 1.0/factorial_ratio(pmin, pmax);

  // Calculate ratio
  double r=1.0;
  for(int p=pmax;p>pmin;p--)
    r*=p;

  return r;
}

int main(void) {
  int Lmax=50;

  double thr=1e-10;
  double eps;

  double Rh=0.2;

  double deta=0.3;
  double eta1=deta*M_PI;
  double eta2=(1.0-deta)*M_PI;

  double dphi=0.125;
  double phi1=dphi*M_PI;
  double phi2=(2.0-dphi)*M_PI;

  double mu1=0.1;
  double mu2=2.4;

  double x1, y1, z1;
  get_coord(Rh, mu1, eta1, phi1, x1, y1, z1);
  double x2, y2, z2;
  get_coord(Rh, mu2, eta2, phi2, x2, y2, z2);

  printf("eta1 = % e, cos(eta1) = % e\n",eta1, cos(eta1));
  printf("eta2 = % e, cos(eta2) = % e\n",eta2, cos(eta2));
  printf("mu1 = % e, cosh(mu1) = % e\n",mu1, cosh(mu1));
  printf("mu2 = % e, cosh(mu2) = % e\n",mu2, cosh(mu2));

  double r=sqrt(std::pow(x1-x2,2) + std::pow(y1-y2,2) + std::pow(z1-z2,2));

  // Exact energy is
  double Eex=1.0/r;

  double mumin=std::min(mu1, mu2);
  double mumax=std::max(mu1, mu2);

  // Neumann expansion
  std::complex<double> Eneu(0.0,0.0);

  int lpad=0;
  arma::mat Plm(get_Plm(Lmax+lpad,Lmax+lpad,cosh(mumin)));
  arma::mat Qlm(get_Qlm(Lmax+lpad,Lmax+lpad,cosh(mumax)));

  printf("%21s % .16f\n","Exact energy",Eex);
  for(int L=0;L<=Lmax;L++) {
    for(int M=-L;M<=L;M++) {
      double leg = Plm(L, std::abs(M)) * Qlm(L, std::abs(M));
      std::complex<double> sph = spherical_harmonics(L, M, cos(eta1), phi1) * std::conj(spherical_harmonics(L, M, cos(eta2), phi2) / factorial_ratio(L+std::abs(M),L-std::abs(M)));
      Eneu += std::pow(-1.0,M) * leg * sph * (4.0*M_PI/Rh);

#if 0
      printf("Legendre % e\n",leg);
      printf("Spherical harmonic % e\n",sph);
      printf("Partial sum % e\n",std::real(Eneu));
#endif
    }
    printf("%15s L=%3i % .16f\n","Neumann expn",L,std::real(Eneu));
  }
  printf("%21s % .16f\n","Exact energy",Eex);

  return 0.0;
}
