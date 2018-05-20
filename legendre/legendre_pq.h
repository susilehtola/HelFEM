#ifndef LEGENDRE_PQ_H
#define LEGENDRE_PQ_H

namespace legendre {
  /* Returns the maximum l value supported by the library */
  int legendrePQ_max_l();
  /* Returns the maximum m value supported by the library */
  int legendrePQ_max_m();
  /* Computes associated Legendre function of the first kind P_l^m(x), |x|<=1 */
  double legendreP(int l, int m, double x);
  /* Computes associated Legendre function of the second kind Q_l^m(x), |x|<=1 */
  double legendreQ(int l, int m, double x);
  /* Computes associated Legendre function of the first kind P_l^m(x), |x|>1 */
  double legendreP_prolate(int l, int m, double x);
  /* Computes associated Legendre function of the second kind Q_l^m(x), |x|>1 */
  double legendreQ_prolate(int l, int m, double x);
}

#endif
