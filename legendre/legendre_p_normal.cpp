/* Generated on Maple 2017.1, X86 64 LINUX, Jun 19 2017, Build ID 1238644 using 2018-05-20 */

#include "legendre_pq.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

double legendreP(int u, int v, double x) {
  double y=0.0;
  switch(u) {
  case(0):
    switch (v) {
    case(0):
      y = 0.1e1;
      break;
    default:
      fprintf(stderr,"%s(%i,%i) not implemented!\n",__FUNCTION__,u,v);
      exit(1);
      break;
    }
    break;
  case(1):
    switch (v) {
    case(0):
      y = x;
      break;
    case(1):
      y = sqrt(x - 0.1e1) * sqrt(x + 0.1e1);
      break;
    default:
      fprintf(stderr,"%s(%i,%i) not implemented!\n",__FUNCTION__,u,v);
      exit(1);
      break;
    }
    break;
  case(2):
    switch (v) {
    case(0):
      y = -0.1e1 / 0.2e1 + 0.3e1 / 0.2e1 * x * x;
      break;
    case(1):
      y = 0.3e1 * sqrt(x - 0.1e1) * sqrt(x + 0.1e1) * x;
      break;
    case(2):
      y = 0.3e1 * (x - 0.1e1) * (x + 0.1e1);
      break;
    default:
      fprintf(stderr,"%s(%i,%i) not implemented!\n",__FUNCTION__,u,v);
      exit(1);
      break;
    }
    break;
  case(3):
    switch (v) {
    case(0):
      y = 0.5e1 / 0.2e1 * pow(x, 0.3e1) - 0.3e1 / 0.2e1 * x;
      break;
    case(1):
      y = 0.3e1 / 0.2e1 * sqrt(x - 0.1e1) * sqrt(x + 0.1e1) * (0.5e1 * x * x - 0.1e1);
      break;
    case(2):
      y = 0.15e2 * (x - 0.1e1) * (x + 0.1e1) * x;
      break;
    case(3):
      y = 0.15e2 * pow(x - 0.1e1, 0.3e1 / 0.2e1) * pow(x + 0.1e1, 0.3e1 / 0.2e1);
      break;
    default:
      fprintf(stderr,"%s(%i,%i) not implemented!\n",__FUNCTION__,u,v);
      exit(1);
      break;
    }
    break;
  case(4):
    switch (v) {
    case(0):
      y = 0.3e1 / 0.8e1 + 0.35e2 / 0.8e1 * pow(x, 0.4e1) - 0.15e2 / 0.4e1 * x * x;
      break;
    case(1):
      y = (0.35e2 * pow(x, 0.3e1) - 0.15e2 * x) * sqrt(x + 0.1e1) * sqrt(x - 0.1e1) / 0.2e1;
      break;
    case(2):
      y = 0.105e3 / 0.2e1 * (x * x - 0.1e1 / 0.7e1) * (x - 0.1e1) * (x + 0.1e1);
      break;
    case(3):
      y = 0.105e3 * pow(x - 0.1e1, 0.3e1 / 0.2e1) * pow(x + 0.1e1, 0.3e1 / 0.2e1) * x;
      break;
    case(4):
      y = 0.105e3 * pow(x - 0.1e1, 0.2e1) * pow(x + 0.1e1, 0.2e1);
      break;
    default:
      fprintf(stderr,"%s(%i,%i) not implemented!\n",__FUNCTION__,u,v);
      exit(1);
      break;
    }
    break;
  case(5):
    switch (v) {
    case(0):
      y = 0.63e2 / 0.8e1 * pow(x, 0.5e1) - 0.35e2 / 0.4e1 * pow(x, 0.3e1) + 0.15e2 / 0.8e1 * x;
      break;
    case(1):
      y = 0.15e2 / 0.8e1 * sqrt(x - 0.1e1) * sqrt(x + 0.1e1) * (0.21e2 * pow(x, 0.4e1) - 0.14e2 * x * x + 0.1e1);
      break;
    case(2):
      y = 0.315e3 / 0.2e1 * (x * x - 0.1e1 / 0.3e1) * (x - 0.1e1) * (x + 0.1e1) * x;
      break;
    case(3):
      y = 0.105e3 / 0.2e1 * pow(x - 0.1e1, 0.3e1 / 0.2e1) * pow(x + 0.1e1, 0.3e1 / 0.2e1) * (0.9e1 * x * x - 0.1e1);
      break;
    case(4):
      y = 0.945e3 * pow(x - 0.1e1, 0.2e1) * pow(x + 0.1e1, 0.2e1) * x;
      break;
    case(5):
      y = 0.945e3 * pow(x - 0.1e1, 0.5e1 / 0.2e1) * pow(x + 0.1e1, 0.5e1 / 0.2e1);
      break;
    default:
      fprintf(stderr,"%s(%i,%i) not implemented!\n",__FUNCTION__,u,v);
      exit(1);
      break;
    }
    break;
  case(6):
    switch (v) {
    case(0):
      y = -0.5e1 / 0.16e2 + 0.231e3 / 0.16e2 * pow(x, 0.6e1) - 0.315e3 / 0.16e2 * pow(x, 0.4e1) + 0.105e3 / 0.16e2 * x * x;
      break;
    case(1):
      y = (0.693e3 * pow(x, 0.5e1) - 0.630e3 * pow(x, 0.3e1) + 0.105e3 * x) * sqrt(x + 0.1e1) * sqrt(x - 0.1e1) / 0.8e1;
      break;
    case(2):
      y = 0.3465e4 / 0.8e1 * pow(x, 0.6e1) - 0.5355e4 / 0.8e1 * pow(x, 0.4e1) + 0.1995e4 / 0.8e1 * x * x - 0.105e3 / 0.8e1;
      break;
    case(3):
      y = (0.3465e4 * pow(x, 0.3e1) - 0.945e3 * x) * pow(x + 0.1e1, 0.3e1 / 0.2e1) * pow(x - 0.1e1, 0.3e1 / 0.2e1) / 0.2e1;
      break;
    case(4):
      y = 0.945e3 / 0.2e1 * pow(x - 0.1e1, 0.2e1) * pow(x + 0.1e1, 0.2e1) * (0.11e2 * x * x - 0.1e1);
      break;
    case(5):
      y = 0.10395e5 * pow(x - 0.1e1, 0.5e1 / 0.2e1) * pow(x + 0.1e1, 0.5e1 / 0.2e1) * x;
      break;
    case(6):
      y = 0.10395e5 * pow(x - 0.1e1, 0.3e1) * pow(x + 0.1e1, 0.3e1);
      break;
    default:
      fprintf(stderr,"%s(%i,%i) not implemented!\n",__FUNCTION__,u,v);
      exit(1);
      break;
    }
    break;
  case(7):
    switch (v) {
    case(0):
      y = 0.429e3 / 0.16e2 * pow(x, 0.7e1) - 0.693e3 / 0.16e2 * pow(x, 0.5e1) + 0.315e3 / 0.16e2 * pow(x, 0.3e1) - 0.35e2 / 0.16e2 * x;
      break;
    case(1):
      y = 0.7e1 / 0.16e2 * sqrt(x - 0.1e1) * sqrt(x + 0.1e1) * (0.429e3 * pow(x, 0.6e1) - 0.495e3 * pow(x, 0.4e1) + 0.135e3 * x * x - 0.5e1);
      break;
    case(2):
      y = 0.9009e4 / 0.8e1 * pow(x, 0.7e1) - 0.15939e5 / 0.8e1 * pow(x, 0.5e1) + 0.7875e4 / 0.8e1 * pow(x, 0.3e1) - 0.945e3 / 0.8e1 * x;
      break;
    case(3):
      y = 0.315e3 / 0.8e1 * pow(x - 0.1e1, 0.3e1 / 0.2e1) * pow(x + 0.1e1, 0.3e1 / 0.2e1) * (0.143e3 * pow(x, 0.4e1) - 0.66e2 * x * x + 0.3e1);
      break;
    case(4):
      y = 0.45045e5 / 0.2e1 * pow(x - 0.1e1, 0.2e1) * (x * x - 0.3e1 / 0.13e2) * pow(x + 0.1e1, 0.2e1) * x;
      break;
    case(5):
      y = 0.10395e5 / 0.2e1 * pow(x - 0.1e1, 0.5e1 / 0.2e1) * pow(x + 0.1e1, 0.5e1 / 0.2e1) * (0.13e2 * x * x - 0.1e1);
      break;
    case(6):
      y = 0.135135e6 * pow(x - 0.1e1, 0.3e1) * pow(x + 0.1e1, 0.3e1) * x;
      break;
    default:
      fprintf(stderr,"%s(%i,%i) not implemented!\n",__FUNCTION__,u,v);
      exit(1);
      break;
    }
    break;
  case(8):
    switch (v) {
    case(0):
      y = 0.35e2 / 0.128e3 + 0.6435e4 / 0.128e3 * pow(x, 0.8e1) - 0.3003e4 / 0.32e2 * pow(x, 0.6e1) + 0.3465e4 / 0.64e2 * pow(x, 0.4e1) - 0.315e3 / 0.32e2 * x * x;
      break;
    case(1):
      y = (0.6435e4 * pow(x, 0.7e1) - 0.9009e4 * pow(x, 0.5e1) + 0.3465e4 * pow(x, 0.3e1) - 0.315e3 * x) * sqrt(x + 0.1e1) * sqrt(x - 0.1e1) / 0.16e2;
      break;
    case(2):
      y = 0.45045e5 / 0.16e2 * pow(x, 0.8e1) - 0.45045e5 / 0.8e1 * pow(x, 0.6e1) + 0.3465e4 * pow(x, 0.4e1) - 0.5355e4 / 0.8e1 * x * x + 0.315e3 / 0.16e2;
      break;
    case(3):
      y = 0.3465e4 / 0.8e1 * pow(x - 0.1e1, 0.3e1 / 0.2e1) * pow(x + 0.1e1, 0.3e1 / 0.2e1) * x * (0.39e2 * pow(x, 0.4e1) - 0.26e2 * x * x + 0.3e1);
      break;
    case(4):
      y = 0.10395e5 / 0.8e1 * pow(x - 0.1e1, 0.2e1) * pow(x + 0.1e1, 0.2e1) * (0.65e2 * pow(x, 0.4e1) - 0.26e2 * x * x + 0.1e1);
      break;
    case(5):
      y = 0.675675e6 / 0.2e1 * pow(x - 0.1e1, 0.5e1 / 0.2e1) * pow(x + 0.1e1, 0.5e1 / 0.2e1) * (x * x - 0.1e1 / 0.5e1) * x;
      break;
    case(6):
      y = 0.2027025e7 / 0.2e1 * pow(x - 0.1e1, 0.3e1) * pow(x + 0.1e1, 0.3e1) * (x * x - 0.1e1 / 0.15e2);
      break;
    default:
      fprintf(stderr,"%s(%i,%i) not implemented!\n",__FUNCTION__,u,v);
      exit(1);
      break;
    }
    break;
  case(9):
    switch (v) {
    case(0):
      y = 0.12155e5 / 0.128e3 * pow(x, 0.9e1) - 0.6435e4 / 0.32e2 * pow(x, 0.7e1) + 0.9009e4 / 0.64e2 * pow(x, 0.5e1) - 0.1155e4 / 0.32e2 * pow(x, 0.3e1) + 0.315e3 / 0.128e3 * x;
      break;
    case(1):
      y = 0.45e2 / 0.128e3 * sqrt(x - 0.1e1) * sqrt(x + 0.1e1) * (0.2431e4 * pow(x, 0.8e1) - 0.4004e4 * pow(x, 0.6e1) + 0.2002e4 * pow(x, 0.4e1) - 0.308e3 * x * x + 0.7e1);
      break;
    case(2):
      y = 0.495e3 / 0.16e2 * (x - 0.1e1) * (x + 0.1e1) * x * (0.221e3 * pow(x, 0.6e1) - 0.273e3 * pow(x, 0.4e1) + 0.91e2 * x * x - 0.7e1);
      break;
    case(3):
      y = 0.3465e4 / 0.16e2 * pow(x - 0.1e1, 0.3e1 / 0.2e1) * pow(x + 0.1e1, 0.3e1 / 0.2e1) * (0.221e3 * pow(x, 0.6e1) - 0.195e3 * pow(x, 0.4e1) + 0.39e2 * x * x - 0.1e1);
      break;
    case(4):
      y = 0.135135e6 / 0.8e1 * pow(x - 0.1e1, 0.2e1) * pow(x + 0.1e1, 0.2e1) * x * (0.17e2 * pow(x, 0.4e1) - 0.10e2 * x * x + 0.1e1);
      break;
    case(5):
      y = 0.135135e6 / 0.8e1 * pow(x - 0.1e1, 0.5e1 / 0.2e1) * pow(x + 0.1e1, 0.5e1 / 0.2e1) * (0.85e2 * pow(x, 0.4e1) - 0.30e2 * x * x + 0.1e1);
      break;
    case(6):
      y = 0.11486475e8 / 0.2e1 * (x * x - 0.3e1 / 0.17e2) * pow(x - 0.1e1, 0.3e1) * pow(x + 0.1e1, 0.3e1) * x;
      break;
    default:
      fprintf(stderr,"%s(%i,%i) not implemented!\n",__FUNCTION__,u,v);
      exit(1);
      break;
    }
    break;
  case(10):
    switch (v) {
    case(0):
      y = -0.63e2 / 0.256e3 + 0.46189e5 / 0.256e3 * pow(x, 0.10e2) - 0.109395e6 / 0.256e3 * pow(x, 0.8e1) + 0.45045e5 / 0.128e3 * pow(x, 0.6e1) - 0.15015e5 / 0.128e3 * pow(x, 0.4e1) + 0.3465e4 / 0.256e3 * x * x;
      break;
    case(1):
      y = 0.55e2 / 0.128e3 * sqrt(x - 0.1e1) * sqrt(x + 0.1e1) * x * (0.4199e4 * pow(x, 0.8e1) - 0.7956e4 * pow(x, 0.6e1) + 0.4914e4 * pow(x, 0.4e1) - 0.1092e4 * x * x + 0.63e2);
      break;
    case(2):
      y = 0.2078505e7 / 0.128e3 * pow(x, 0.10e2) - 0.5141565e7 / 0.128e3 * pow(x, 0.8e1) + 0.2207205e7 / 0.64e2 * pow(x, 0.6e1) - 0.765765e6 / 0.64e2 * pow(x, 0.4e1) + 0.183645e6 / 0.128e3 * x * x - 0.3465e4 / 0.128e3;
      break;
    case(3):
      y = 0.6435e4 / 0.16e2 * pow(x - 0.1e1, 0.3e1 / 0.2e1) * pow(x + 0.1e1, 0.3e1 / 0.2e1) * x * (0.323e3 * pow(x, 0.6e1) - 0.357e3 * pow(x, 0.4e1) + 0.105e3 * x * x - 0.7e1);
      break;
    case(4):
      y = 0.45045e5 / 0.16e2 * pow(x - 0.1e1, 0.2e1) * pow(x + 0.1e1, 0.2e1) * (0.323e3 * pow(x, 0.6e1) - 0.255e3 * pow(x, 0.4e1) + 0.45e2 * x * x - 0.1e1);
      break;
    case(5):
      y = 0.135135e6 / 0.8e1 * pow(x - 0.1e1, 0.5e1 / 0.2e1) * pow(x + 0.1e1, 0.5e1 / 0.2e1) * x * (0.323e3 * pow(x, 0.4e1) - 0.170e3 * x * x + 0.15e2);
      break;
    case(6):
      y = 0.675675e6 / 0.8e1 * pow(x - 0.1e1, 0.3e1) * pow(x + 0.1e1, 0.3e1) * (0.323e3 * pow(x, 0.4e1) - 0.102e3 * x * x + 0.3e1);
      break;
    default:
      fprintf(stderr,"%s(%i,%i) not implemented!\n",__FUNCTION__,u,v);
      exit(1);
      break;
    }
    break;
  case(11):
    switch (v) {
    case(0):
      y = 0.88179e5 / 0.256e3 * pow(x, 0.11e2) - 0.230945e6 / 0.256e3 * pow(x, 0.9e1) + 0.109395e6 / 0.128e3 * pow(x, 0.7e1) - 0.45045e5 / 0.128e3 * pow(x, 0.5e1) + 0.15015e5 / 0.256e3 * pow(x, 0.3e1) - 0.693e3 / 0.256e3 * x;
      break;
    case(1):
      y = 0.33e2 / 0.256e3 * sqrt(x - 0.1e1) * sqrt(x + 0.1e1) * (0.29393e5 * pow(x, 0.10e2) - 0.62985e5 * pow(x, 0.8e1) + 0.46410e5 * pow(x, 0.6e1) - 0.13650e5 * pow(x, 0.4e1) + 0.1365e4 * x * x - 0.21e2);
      break;
    case(2):
      y = 0.2145e4 / 0.128e3 * (x - 0.1e1) * (x + 0.1e1) * x * (0.2261e4 * pow(x, 0.8e1) - 0.3876e4 * pow(x, 0.6e1) + 0.2142e4 * pow(x, 0.4e1) - 0.420e3 * x * x + 0.21e2);
      break;
    case(3):
      y = 0.45045e5 / 0.128e3 * pow(x - 0.1e1, 0.3e1 / 0.2e1) * pow(x + 0.1e1, 0.3e1 / 0.2e1) * (0.969e3 * pow(x, 0.8e1) - 0.1292e4 * pow(x, 0.6e1) + 0.510e3 * pow(x, 0.4e1) - 0.60e2 * x * x + 0.1e1);
      break;
    case(4):
      y = 0.43648605e8 / 0.16e2 * (pow(x, 0.6e1) - pow(x, 0.4e1) + 0.5e1 / 0.19e2 * x * x - 0.5e1 / 0.323e3) * pow(x - 0.1e1, 0.2e1) * pow(x + 0.1e1, 0.2e1) * x;
      break;
    case(5):
      y = 0.135135e6 / 0.16e2 * pow(x - 0.1e1, 0.5e1 / 0.2e1) * pow(x + 0.1e1, 0.5e1 / 0.2e1) * (0.2261e4 * pow(x, 0.6e1) - 0.1615e4 * pow(x, 0.4e1) + 0.255e3 * x * x - 0.5e1);
      break;
    case(6):
      y = 0.2297295e7 / 0.8e1 * pow(x - 0.1e1, 0.3e1) * pow(x + 0.1e1, 0.3e1) * x * (0.399e3 * pow(x, 0.4e1) - 0.190e3 * x * x + 0.15e2);
      break;
    default:
      fprintf(stderr,"%s(%i,%i) not implemented!\n",__FUNCTION__,u,v);
      exit(1);
      break;
    }
    break;
  case(12):
    switch (v) {
    case(0):
      y = 0.231e3 / 0.1024e4 + 0.676039e6 / 0.1024e4 * pow(x, 0.12e2) - 0.969969e6 / 0.512e3 * pow(x, 0.10e2) + 0.2078505e7 / 0.1024e4 * pow(x, 0.8e1) - 0.255255e6 / 0.256e3 * pow(x, 0.6e1) + 0.225225e6 / 0.1024e4 * pow(x, 0.4e1) - 0.9009e4 / 0.512e3 * x * x;
      break;
    case(1):
      y = 0.39e2 / 0.256e3 * sqrt(x - 0.1e1) * sqrt(x + 0.1e1) * x * (0.52003e5 * pow(x, 0.10e2) - 0.124355e6 * pow(x, 0.8e1) + 0.106590e6 * pow(x, 0.6e1) - 0.39270e5 * pow(x, 0.4e1) + 0.5775e4 * x * x - 0.231e3);
      break;
    case(2):
      y = 0.22309287e8 / 0.256e3 * pow(x, 0.12e2) - 0.16489473e8 / 0.64e2 * pow(x, 0.10e2) + 0.72747675e8 / 0.256e3 * pow(x, 0.8e1) - 0.2297295e7 / 0.16e2 * pow(x, 0.6e1) + 0.8333325e7 / 0.256e3 * pow(x, 0.4e1) - 0.171171e6 / 0.64e2 * x * x + 0.9009e4 / 0.256e3;
      break;
    case(3):
      y = 0.15015e5 / 0.128e3 * pow(x - 0.1e1, 0.3e1 / 0.2e1) * pow(x + 0.1e1, 0.3e1 / 0.2e1) * x * (0.7429e4 * pow(x, 0.8e1) - 0.11628e5 * pow(x, 0.6e1) + 0.5814e4 * pow(x, 0.4e1) - 0.1020e4 * x * x + 0.45e2);
      break;
    case(4):
      y = 0.135135e6 / 0.128e3 * pow(x - 0.1e1, 0.2e1) * pow(x + 0.1e1, 0.2e1) * (0.7429e4 * pow(x, 0.8e1) - 0.9044e4 * pow(x, 0.6e1) + 0.3230e4 * pow(x, 0.4e1) - 0.340e3 * x * x + 0.5e1);
      break;
    case(5):
      y = 0.2297295e7 / 0.16e2 * pow(x - 0.1e1, 0.5e1 / 0.2e1) * pow(x + 0.1e1, 0.5e1 / 0.2e1) * x * (0.437e3 * pow(x, 0.6e1) - 0.399e3 * pow(x, 0.4e1) + 0.95e2 * x * x - 0.5e1);
      break;
    case(6):
      y = 0.2297295e7 / 0.16e2 * pow(x - 0.1e1, 0.3e1) * pow(x + 0.1e1, 0.3e1) * (0.3059e4 * pow(x, 0.6e1) - 0.1995e4 * pow(x, 0.4e1) + 0.285e3 * x * x - 0.5e1);
      break;
    default:
      fprintf(stderr,"%s(%i,%i) not implemented!\n",__FUNCTION__,u,v);
      exit(1);
      break;
    }
    break;
  case(13):
    switch (v) {
    case(0):
      y = 0.1300075e7 / 0.1024e4 * pow(x, 0.13e2) - 0.2028117e7 / 0.512e3 * pow(x, 0.11e2) + 0.4849845e7 / 0.1024e4 * pow(x, 0.9e1) - 0.692835e6 / 0.256e3 * pow(x, 0.7e1) + 0.765765e6 / 0.1024e4 * pow(x, 0.5e1) - 0.45045e5 / 0.512e3 * pow(x, 0.3e1) + 0.3003e4 / 0.1024e4 * x;
      break;
    case(1):
      y = 0.91e2 / 0.1024e4 * sqrt(x - 0.1e1) * sqrt(x + 0.1e1) * (0.185725e6 * pow(x, 0.12e2) - 0.490314e6 * pow(x, 0.10e2) + 0.479655e6 * pow(x, 0.8e1) - 0.213180e6 * pow(x, 0.6e1) + 0.42075e5 * pow(x, 0.4e1) - 0.2970e4 * x * x + 0.33e2);
      break;
    case(2):
      y = 0.1365e4 / 0.256e3 * (x - 0.1e1) * (x + 0.1e1) * x * (0.37145e5 * pow(x, 0.10e2) - 0.81719e5 * pow(x, 0.8e1) + 0.63954e5 * pow(x, 0.6e1) - 0.21318e5 * pow(x, 0.4e1) + 0.2805e4 * x * x - 0.99e2);
      break;
    case(3):
      y = 0.15015e5 / 0.256e3 * pow(x - 0.1e1, 0.3e1 / 0.2e1) * pow(x + 0.1e1, 0.3e1 / 0.2e1) * (0.37145e5 * pow(x, 0.10e2) - 0.66861e5 * pow(x, 0.8e1) + 0.40698e5 * pow(x, 0.6e1) - 0.9690e4 * pow(x, 0.4e1) + 0.765e3 * x * x - 0.9e1);
      break;
    case(4):
      y = 0.255255e6 / 0.128e3 * pow(x - 0.1e1, 0.2e1) * pow(x + 0.1e1, 0.2e1) * x * (0.10925e5 * pow(x, 0.8e1) - 0.15732e5 * pow(x, 0.6e1) + 0.7182e4 * pow(x, 0.4e1) - 0.1140e4 * x * x + 0.45e2);
      break;
    case(5):
      y = 0.2297295e7 / 0.128e3 * pow(x - 0.1e1, 0.5e1 / 0.2e1) * pow(x + 0.1e1, 0.5e1 / 0.2e1) * (0.10925e5 * pow(x, 0.8e1) - 0.12236e5 * pow(x, 0.6e1) + 0.3990e4 * pow(x, 0.4e1) - 0.380e3 * x * x + 0.5e1);
      break;
    case(6):
      y = 0.43648605e8 / 0.16e2 * pow(x - 0.1e1, 0.3e1) * pow(x + 0.1e1, 0.3e1) * x * (0.575e3 * pow(x, 0.6e1) - 0.483e3 * pow(x, 0.4e1) + 0.105e3 * x * x - 0.5e1);
      break;
    default:
      fprintf(stderr,"%s(%i,%i) not implemented!\n",__FUNCTION__,u,v);
      exit(1);
      break;
    }
    break;
  case(14):
    switch (v) {
    case(0):
      y = -0.429e3 / 0.2048e4 + 0.5014575e7 / 0.2048e4 * pow(x, 0.14e2) - 0.16900975e8 / 0.2048e4 * pow(x, 0.12e2) + 0.22309287e8 / 0.2048e4 * pow(x, 0.10e2) - 0.14549535e8 / 0.2048e4 * pow(x, 0.8e1) + 0.4849845e7 / 0.2048e4 * pow(x, 0.6e1) - 0.765765e6 / 0.2048e4 * pow(x, 0.4e1) + 0.45045e5 / 0.2048e4 * x * x;
      break;
    case(1):
      y = 0.105e3 / 0.1024e4 * sqrt(x - 0.1e1) * sqrt(x + 0.1e1) * x * (0.334305e6 * pow(x, 0.12e2) - 0.965770e6 * pow(x, 0.10e2) + 0.1062347e7 * pow(x, 0.8e1) - 0.554268e6 * pow(x, 0.6e1) + 0.138567e6 * pow(x, 0.4e1) - 0.14586e5 * x * x + 0.429e3);
      break;
    case(2):
      y = 0.456326325e9 / 0.1024e4 * pow(x, 0.14e2) - 0.1571790675e10 / 0.1024e4 * pow(x, 0.12e2) + 0.2119382265e10 / 0.1024e4 * pow(x, 0.10e2) - 0.1411304895e10 / 0.1024e4 * pow(x, 0.8e1) + 0.480134655e9 / 0.1024e4 * pow(x, 0.6e1) - 0.77342265e8 / 0.1024e4 * pow(x, 0.4e1) + 0.4639635e7 / 0.1024e4 * x * x - 0.45045e5 / 0.1024e4;
      break;
    case(3):
      y = 0.23205e5 / 0.256e3 * pow(x - 0.1e1, 0.3e1 / 0.2e1) * pow(x + 0.1e1, 0.3e1 / 0.2e1) * x * (0.58995e5 * pow(x, 0.10e2) - 0.120175e6 * pow(x, 0.8e1) + 0.86526e5 * pow(x, 0.6e1) - 0.26334e5 * pow(x, 0.4e1) + 0.3135e4 * x * x - 0.99e2);
      break;
    case(4):
      y = 0.2297295e7 / 0.256e3 * pow(x - 0.1e1, 0.2e1) * pow(x + 0.1e1, 0.2e1) * (0.6555e4 * pow(x, 0.10e2) - 0.10925e5 * pow(x, 0.8e1) + 0.6118e4 * pow(x, 0.6e1) - 0.1330e4 * pow(x, 0.4e1) + 0.95e2 * x * x - 0.1e1);
      break;
    case(5):
      y = 0.43648605e8 / 0.128e3 * pow(x - 0.1e1, 0.5e1 / 0.2e1) * pow(x + 0.1e1, 0.5e1 / 0.2e1) * x * (0.1725e4 * pow(x, 0.8e1) - 0.2300e4 * pow(x, 0.6e1) + 0.966e3 * pow(x, 0.4e1) - 0.140e3 * x * x + 0.5e1);
      break;
    case(6):
      y = 0.218243025e9 / 0.128e3 * pow(x - 0.1e1, 0.3e1) * pow(x + 0.1e1, 0.3e1) * (0.3105e4 * pow(x, 0.8e1) - 0.3220e4 * pow(x, 0.6e1) + 0.966e3 * pow(x, 0.4e1) - 0.84e2 * x * x + 0.1e1);
      break;
    default:
      fprintf(stderr,"%s(%i,%i) not implemented!\n",__FUNCTION__,u,v);
      exit(1);
      break;
    }
    break;
  case(15):
    switch (v) {
    case(0):
      y = 0.9694845e7 / 0.2048e4 * pow(x, 0.15e2) - 0.35102025e8 / 0.2048e4 * pow(x, 0.13e2) + 0.50702925e8 / 0.2048e4 * pow(x, 0.11e2) - 0.37182145e8 / 0.2048e4 * pow(x, 0.9e1) + 0.14549535e8 / 0.2048e4 * pow(x, 0.7e1) - 0.2909907e7 / 0.2048e4 * pow(x, 0.5e1) + 0.255255e6 / 0.2048e4 * pow(x, 0.3e1) - 0.6435e4 / 0.2048e4 * x;
      break;
    case(1):
      y = 0.15e2 / 0.2048e4 * sqrt(x - 0.1e1) * sqrt(x + 0.1e1) * (0.9694845e7 * pow(x, 0.14e2) - 0.30421755e8 * pow(x, 0.12e2) + 0.37182145e8 * pow(x, 0.10e2) - 0.22309287e8 * pow(x, 0.8e1) + 0.6789783e7 * pow(x, 0.6e1) - 0.969969e6 * pow(x, 0.4e1) + 0.51051e5 * x * x - 0.429e3);
      break;
    case(2):
      y = 0.1785e4 / 0.1024e4 * (x - 0.1e1) * (x + 0.1e1) * x * (0.570285e6 * pow(x, 0.12e2) - 0.1533870e7 * pow(x, 0.10e2) + 0.1562275e7 * pow(x, 0.8e1) - 0.749892e6 * pow(x, 0.6e1) + 0.171171e6 * pow(x, 0.4e1) - 0.16302e5 * x * x + 0.429e3);
      break;
    case(3):
      y = 0.69615e5 / 0.1024e4 * pow(x - 0.1e1, 0.3e1 / 0.2e1) * pow(x + 0.1e1, 0.3e1 / 0.2e1) * (0.190095e6 * pow(x, 0.12e2) - 0.432630e6 * pow(x, 0.10e2) + 0.360525e6 * pow(x, 0.8e1) - 0.134596e6 * pow(x, 0.6e1) + 0.21945e5 * pow(x, 0.4e1) - 0.1254e4 * x * x + 0.11e2);
      break;
    case(4):
      y = 0.3968055e7 / 0.256e3 * pow(x - 0.1e1, 0.2e1) * pow(x + 0.1e1, 0.2e1) * x * (0.10005e5 * pow(x, 0.10e2) - 0.18975e5 * pow(x, 0.8e1) + 0.12650e5 * pow(x, 0.6e1) - 0.3542e4 * pow(x, 0.4e1) + 0.385e3 * x * x - 0.11e2);
      break;
    case(5):
      y = 0.43648605e8 / 0.256e3 * pow(x - 0.1e1, 0.5e1 / 0.2e1) * pow(x + 0.1e1, 0.5e1 / 0.2e1) * (0.10005e5 * pow(x, 0.10e2) - 0.15525e5 * pow(x, 0.8e1) + 0.8050e4 * pow(x, 0.6e1) - 0.1610e4 * pow(x, 0.4e1) + 0.105e3 * x * x - 0.1e1);
      break;
    case(6):
      y = 0.218243025e9 / 0.128e3 * pow(x - 0.1e1, 0.3e1) * pow(x + 0.1e1, 0.3e1) * x * (0.10005e5 * pow(x, 0.8e1) - 0.12420e5 * pow(x, 0.6e1) + 0.4830e4 * pow(x, 0.4e1) - 0.644e3 * x * x + 0.21e2);
      break;
    default:
      fprintf(stderr,"%s(%i,%i) not implemented!\n",__FUNCTION__,u,v);
      exit(1);
      break;
    }
    break;
  case(16):
    switch (v) {
    case(0):
      y = 0.6435e4 / 0.32768e5 + 0.300540195e9 / 0.32768e5 * pow(x, 0.16e2) - 0.145422675e9 / 0.4096e4 * pow(x, 0.14e2) + 0.456326325e9 / 0.8192e4 * pow(x, 0.12e2) - 0.185910725e9 / 0.4096e4 * pow(x, 0.10e2) + 0.334639305e9 / 0.16384e5 * pow(x, 0.8e1) - 0.20369349e8 / 0.4096e4 * pow(x, 0.6e1) + 0.4849845e7 / 0.8192e4 * pow(x, 0.4e1) - 0.109395e6 / 0.4096e4 * x * x;
      break;
    case(1):
      y = 0.17e2 / 0.2048e4 * sqrt(x - 0.1e1) * sqrt(x + 0.1e1) * x * (0.17678835e8 * pow(x, 0.14e2) - 0.59879925e8 * pow(x, 0.12e2) + 0.80528175e8 * pow(x, 0.10e2) - 0.54679625e8 * pow(x, 0.8e1) + 0.19684665e8 * pow(x, 0.6e1) - 0.3594591e7 * pow(x, 0.4e1) + 0.285285e6 * x * x - 0.6435e4);
      break;
    case(2):
      y = 0.4508102925e10 / 0.2048e4 * pow(x, 0.16e2) - 0.8870783175e10 / 0.1024e4 * pow(x, 0.14e2) + 0.14146116075e11 / 0.1024e4 * pow(x, 0.12e2) - 0.11712375675e11 / 0.1024e4 * pow(x, 0.10e2) + 0.334639305e9 / 0.64e2 * pow(x, 0.8e1) - 0.1324007685e10 / 0.1024e4 * pow(x, 0.6e1) + 0.160044885e9 / 0.1024e4 * pow(x, 0.4e1) - 0.7329465e7 / 0.1024e4 * x * x + 0.109395e6 / 0.2048e4;
      break;
    case(3):
      y = 0.101745e6 / 0.1024e4 * pow(x - 0.1e1, 0.3e1 / 0.2e1) * pow(x + 0.1e1, 0.3e1 / 0.2e1) * x * (0.310155e6 * pow(x, 0.12e2) - 0.780390e6 * pow(x, 0.10e2) + 0.740025e6 * pow(x, 0.8e1) - 0.328900e6 * pow(x, 0.6e1) + 0.69069e5 * pow(x, 0.4e1) - 0.6006e4 * x * x + 0.143e3);
      break;
    case(4):
      y = 0.1322685e7 / 0.1024e4 * pow(x - 0.1e1, 0.2e1) * pow(x + 0.1e1, 0.2e1) * (0.310155e6 * pow(x, 0.12e2) - 0.660330e6 * pow(x, 0.10e2) + 0.512325e6 * pow(x, 0.8e1) - 0.177100e6 * pow(x, 0.6e1) + 0.26565e5 * pow(x, 0.4e1) - 0.1386e4 * x * x + 0.11e2);
      break;
    case(5):
      y = 0.3968055e7 / 0.256e3 * pow(x - 0.1e1, 0.5e1 / 0.2e1) * pow(x + 0.1e1, 0.5e1 / 0.2e1) * x * (0.310155e6 * pow(x, 0.10e2) - 0.550275e6 * pow(x, 0.8e1) + 0.341550e6 * pow(x, 0.6e1) - 0.88550e5 * pow(x, 0.4e1) + 0.8855e4 * x * x - 0.231e3);
      break;
    case(6):
      y = 0.43648605e8 / 0.256e3 * pow(x - 0.1e1, 0.3e1) * pow(x + 0.1e1, 0.3e1) * (0.310155e6 * pow(x, 0.10e2) - 0.450225e6 * pow(x, 0.8e1) + 0.217350e6 * pow(x, 0.6e1) - 0.40250e5 * pow(x, 0.4e1) + 0.2415e4 * x * x - 0.21e2);
      break;
    default:
      fprintf(stderr,"%s(%i,%i) not implemented!\n",__FUNCTION__,u,v);
      exit(1);
      break;
    }
    break;
  case(17):
    switch (v) {
    case(0):
      y = 0.583401555e9 / 0.32768e5 * pow(x, 0.17e2) - 0.300540195e9 / 0.4096e4 * pow(x, 0.15e2) + 0.1017958725e10 / 0.8192e4 * pow(x, 0.13e2) - 0.456326325e9 / 0.4096e4 * pow(x, 0.11e2) + 0.929553625e9 / 0.16384e5 * pow(x, 0.9e1) - 0.66927861e8 / 0.4096e4 * pow(x, 0.7e1) + 0.20369349e8 / 0.8192e4 * pow(x, 0.5e1) - 0.692835e6 / 0.4096e4 * pow(x, 0.3e1) + 0.109395e6 / 0.32768e5 * x;
      break;
    case(1):
      y = 0.153e3 / 0.32768e5 * sqrt(x - 0.1e1) * sqrt(x + 0.1e1) * (0.64822395e8 * pow(x, 0.16e2) - 0.235717800e9 * pow(x, 0.14e2) + 0.345972900e9 * pow(x, 0.12e2) - 0.262462200e9 * pow(x, 0.10e2) + 0.109359250e9 * pow(x, 0.8e1) - 0.24496472e8 * pow(x, 0.6e1) + 0.2662660e7 * pow(x, 0.4e1) - 0.108680e6 * x * x + 0.715e3);
      break;
    case(2):
      y = 0.2907e4 / 0.2048e4 * (x - 0.1e1) * (x + 0.1e1) * x * (0.3411705e7 * pow(x, 0.14e2) - 0.10855425e8 * pow(x, 0.12e2) + 0.13656825e8 * pow(x, 0.10e2) - 0.8633625e7 * pow(x, 0.8e1) + 0.2877875e7 * pow(x, 0.6e1) - 0.483483e6 * pow(x, 0.4e1) + 0.35035e5 * x * x - 0.715e3);
      break;
    case(3):
      y = 0.14535e5 / 0.2048e4 * pow(x - 0.1e1, 0.3e1 / 0.2e1) * pow(x + 0.1e1, 0.3e1 / 0.2e1) * (0.10235115e8 * pow(x, 0.14e2) - 0.28224105e8 * pow(x, 0.12e2) + 0.30045015e8 * pow(x, 0.10e2) - 0.15540525e8 * pow(x, 0.8e1) + 0.4029025e7 * pow(x, 0.6e1) - 0.483483e6 * pow(x, 0.4e1) + 0.21021e5 * x * x - 0.143e3);
      break;
    case(4):
      y = 0.305235e6 / 0.1024e4 * pow(x - 0.1e1, 0.2e1) * pow(x + 0.1e1, 0.2e1) * x * (0.3411705e7 * pow(x, 0.12e2) - 0.8064030e7 * pow(x, 0.10e2) + 0.7153575e7 * pow(x, 0.8e1) - 0.2960100e7 * pow(x, 0.6e1) + 0.575575e6 * pow(x, 0.4e1) - 0.46046e5 * x * x + 0.1001e4);
      break;
    case(5):
      y = 0.43648605e8 / 0.1024e4 * pow(x - 0.1e1, 0.5e1 / 0.2e1) * pow(x + 0.1e1, 0.5e1 / 0.2e1) * (0.310155e6 * pow(x, 0.12e2) - 0.620310e6 * pow(x, 0.10e2) + 0.450225e6 * pow(x, 0.8e1) - 0.144900e6 * pow(x, 0.6e1) + 0.20125e5 * pow(x, 0.4e1) - 0.966e3 * x * x + 0.7e1);
      break;
    case(6):
      y = 0.1003917915e10 / 0.256e3 * pow(x - 0.1e1, 0.3e1) * pow(x + 0.1e1, 0.3e1) * x * (0.40455e5 * pow(x, 0.10e2) - 0.67425e5 * pow(x, 0.8e1) + 0.39150e5 * pow(x, 0.6e1) - 0.9450e4 * pow(x, 0.4e1) + 0.875e3 * x * x - 0.21e2);
      break;
    default:
      fprintf(stderr,"%s(%i,%i) not implemented!\n",__FUNCTION__,u,v);
      exit(1);
      break;
    }
    break;
  case(18):
    switch (v) {
    case(0):
      y = -0.12155e5 / 0.65536e5 + 0.2268783825e10 / 0.65536e5 * pow(x, 0.18e2) - 0.9917826435e10 / 0.65536e5 * pow(x, 0.16e2) + 0.4508102925e10 / 0.16384e5 * pow(x, 0.14e2) - 0.4411154475e10 / 0.16384e5 * pow(x, 0.12e2) + 0.5019589575e10 / 0.32768e5 * pow(x, 0.10e2) - 0.1673196525e10 / 0.32768e5 * pow(x, 0.8e1) + 0.156165009e9 / 0.16384e5 * pow(x, 0.6e1) - 0.14549535e8 / 0.16384e5 * pow(x, 0.4e1) + 0.2078505e7 / 0.65536e5 * x * x;
      break;
    case(1):
      y = 0.171e3 / 0.32768e5 * sqrt(x - 0.1e1) * sqrt(x + 0.1e1) * x * (0.119409675e9 * pow(x, 0.16e2) - 0.463991880e9 * pow(x, 0.14e2) + 0.738168900e9 * pow(x, 0.12e2) - 0.619109400e9 * pow(x, 0.10e2) + 0.293543250e9 * pow(x, 0.8e1) - 0.78278200e8 * pow(x, 0.6e1) + 0.10958948e8 * pow(x, 0.4e1) - 0.680680e6 * x * x + 0.12155e5);
      break;
    case(2):
      y = 0.14535e5 / 0.32768e5 * (x - 0.1e1) * (x + 0.1e1) * (0.23881935e8 * pow(x, 0.16e2) - 0.81880920e8 * pow(x, 0.14e2) + 0.112896420e9 * pow(x, 0.12e2) - 0.80120040e8 * pow(x, 0.10e2) + 0.31081050e8 * pow(x, 0.8e1) - 0.6446440e7 * pow(x, 0.6e1) + 0.644644e6 * pow(x, 0.4e1) - 0.24024e5 * x * x + 0.143e3);
      break;
    case(3):
      y = 0.347123925225e12 / 0.2048e4 * pow(x - 0.1e1, 0.3e1 / 0.2e1) * (pow(x, 0.14e2) - 0.3e1 * pow(x, 0.12e2) + 0.39e2 / 0.11e2 * pow(x, 0.10e2) - 0.65e2 / 0.31e2 * pow(x, 0.8e1) + 0.585e3 / 0.899e3 * pow(x, 0.6e1) - 0.91e2 / 0.899e3 * pow(x, 0.4e1) + 0.91e2 / 0.13485e5 * x * x - 0.13e2 / 0.103385e6) * pow(x + 0.1e1, 0.3e1 / 0.2e1) * x;
      break;
    case(4):
      y = 0.3357585e7 / 0.2048e4 * pow(x - 0.1e1, 0.2e1) * pow(x + 0.1e1, 0.2e1) * (0.1550775e7 * pow(x, 0.14e2) - 0.4032015e7 * pow(x, 0.12e2) + 0.4032015e7 * pow(x, 0.10e2) - 0.1950975e7 * pow(x, 0.8e1) + 0.470925e6 * pow(x, 0.6e1) - 0.52325e5 * pow(x, 0.4e1) + 0.2093e4 * x * x - 0.13e2);
      break;
    case(5):
      y = 0.77224455e8 / 0.1024e4 * pow(x - 0.1e1, 0.5e1 / 0.2e1) * pow(x + 0.1e1, 0.5e1 / 0.2e1) * x * (0.471975e6 * pow(x, 0.12e2) - 0.1051830e7 * pow(x, 0.10e2) + 0.876525e6 * pow(x, 0.8e1) - 0.339300e6 * pow(x, 0.6e1) + 0.61425e5 * pow(x, 0.4e1) - 0.4550e4 * x * x + 0.91e2);
      break;
    case(6):
      y = 0.1003917915e10 / 0.1024e4 * pow(x - 0.1e1, 0.3e1) * pow(x + 0.1e1, 0.3e1) * (0.471975e6 * pow(x, 0.12e2) - 0.890010e6 * pow(x, 0.10e2) + 0.606825e6 * pow(x, 0.8e1) - 0.182700e6 * pow(x, 0.6e1) + 0.23625e5 * pow(x, 0.4e1) - 0.1050e4 * x * x + 0.7e1);
      break;
    default:
      fprintf(stderr,"%s(%i,%i) not implemented!\n",__FUNCTION__,u,v);
      exit(1);
      break;
    }
    break;
  case(19):
    switch (v) {
    case(0):
      y = 0.4418157975e10 / 0.65536e5 * pow(x, 0.19e2) - 0.20419054425e11 / 0.65536e5 * pow(x, 0.17e2) + 0.9917826435e10 / 0.16384e5 * pow(x, 0.15e2) - 0.10518906825e11 / 0.16384e5 * pow(x, 0.13e2) + 0.13233463425e11 / 0.32768e5 * pow(x, 0.11e2) - 0.5019589575e10 / 0.32768e5 * pow(x, 0.9e1) + 0.557732175e9 / 0.16384e5 * pow(x, 0.7e1) - 0.66927861e8 / 0.16384e5 * pow(x, 0.5e1) + 0.14549535e8 / 0.65536e5 * pow(x, 0.3e1) - 0.230945e6 / 0.65536e5 * x;
      break;
    case(1):
      y = 0.95e2 / 0.65536e5 * sqrt(x - 0.1e1) * sqrt(x + 0.1e1) * (0.883631595e9 * pow(x, 0.18e2) - 0.3653936055e10 * pow(x, 0.16e2) + 0.6263890380e10 * pow(x, 0.14e2) - 0.5757717420e10 * pow(x, 0.12e2) + 0.3064591530e10 * pow(x, 0.10e2) - 0.951080130e9 * pow(x, 0.8e1) + 0.164384220e9 * pow(x, 0.6e1) - 0.14090076e8 * pow(x, 0.4e1) + 0.459459e6 * x * x - 0.2431e4);
      break;
    case(2):
      y = 0.5985e4 / 0.32768e5 * (x - 0.1e1) * (x + 0.1e1) * x * (0.126233085e9 * pow(x, 0.16e2) - 0.463991880e9 * pow(x, 0.14e2) + 0.695987820e9 * pow(x, 0.12e2) - 0.548354040e9 * pow(x, 0.10e2) + 0.243221550e9 * pow(x, 0.8e1) - 0.60386040e8 * pow(x, 0.6e1) + 0.7827820e7 * pow(x, 0.4e1) - 0.447304e6 * x * x + 0.7293e4);
      break;
    case(3):
      y = 0.1119195e7 / 0.32768e5 * pow(x - 0.1e1, 0.3e1 / 0.2e1) * pow(x + 0.1e1, 0.3e1 / 0.2e1) * (0.11475735e8 * pow(x, 0.16e2) - 0.37218600e8 * pow(x, 0.14e2) + 0.48384180e8 * pow(x, 0.12e2) - 0.32256120e8 * pow(x, 0.10e2) + 0.11705850e8 * pow(x, 0.8e1) - 0.2260440e7 * pow(x, 0.6e1) + 0.209300e6 * pow(x, 0.4e1) - 0.7176e4 * x * x + 0.39e2);
      break;
    case(4):
      y = 0.25741485e8 / 0.2048e4 * pow(x - 0.1e1, 0.2e1) * pow(x + 0.1e1, 0.2e1) * x * (0.498945e6 * pow(x, 0.14e2) - 0.1415925e7 * pow(x, 0.12e2) + 0.1577745e7 * pow(x, 0.10e2) - 0.876525e6 * pow(x, 0.8e1) + 0.254475e6 * pow(x, 0.6e1) - 0.36855e5 * pow(x, 0.4e1) + 0.2275e4 * x * x - 0.39e2);
      break;
    case(5):
      y = 0.77224455e8 / 0.2048e4 * pow(x - 0.1e1, 0.5e1 / 0.2e1) * pow(x + 0.1e1, 0.5e1 / 0.2e1) * (0.2494725e7 * pow(x, 0.14e2) - 0.6135675e7 * pow(x, 0.12e2) + 0.5785065e7 * pow(x, 0.10e2) - 0.2629575e7 * pow(x, 0.8e1) + 0.593775e6 * pow(x, 0.6e1) - 0.61425e5 * pow(x, 0.4e1) + 0.2275e4 * x * x - 0.13e2);
      break;
    case(6):
      y = 0.1930611375e10 / 0.1024e4 * pow(x - 0.1e1, 0.3e1) * pow(x + 0.1e1, 0.3e1) * x * (0.698523e6 * pow(x, 0.12e2) - 0.1472562e7 * pow(x, 0.10e2) + 0.1157013e7 * pow(x, 0.8e1) - 0.420732e6 * pow(x, 0.6e1) + 0.71253e5 * pow(x, 0.4e1) - 0.4914e4 * x * x + 0.91e2);
      break;
    default:
      fprintf(stderr,"%s(%i,%i) not implemented!\n",__FUNCTION__,u,v);
      exit(1);
      break;
    }
    break;
  case(20):
    switch (v) {
    case(0):
      y = 0.46189e5 / 0.262144e6 + 0.34461632205e11 / 0.262144e6 * pow(x, 0.20e2) - 0.83945001525e11 / 0.131072e6 * pow(x, 0.18e2) + 0.347123925225e12 / 0.262144e6 * pow(x, 0.16e2) - 0.49589132175e11 / 0.32768e5 * pow(x, 0.14e2) + 0.136745788725e12 / 0.131072e6 * pow(x, 0.12e2) - 0.29113619535e11 / 0.65536e5 * pow(x, 0.10e2) + 0.15058768725e11 / 0.131072e6 * pow(x, 0.8e1) - 0.557732175e9 / 0.32768e5 * pow(x, 0.6e1) + 0.334639305e9 / 0.262144e6 * pow(x, 0.4e1) - 0.4849845e7 / 0.131072e6 * x * x;
      break;
    case(1):
      y = 0.172308161025e12 / 0.65536e5 * (pow(x, 0.18e2) - 0.57e2 / 0.13e2 * pow(x, 0.16e2) + 0.3876e4 / 0.481e3 * pow(x, 0.14e2) - 0.3876e4 / 0.481e3 * pow(x, 0.12e2) + 0.1938e4 / 0.407e3 * pow(x, 0.10e2) - 0.1938e4 / 0.1147e4 * pow(x, 0.8e1) + 0.11628e5 / 0.33263e5 * pow(x, 0.6e1) - 0.1292e4 / 0.33263e5 * pow(x, 0.4e1) + 0.323e3 / 0.166315e6 * x * x - 0.323e3 / 0.11475735e8) * sqrt(x - 0.1e1) * sqrt(x + 0.1e1) * x;
      break;
    case(2):
      y = 0.21945e5 / 0.65536e5 * (x - 0.1e1) * (x + 0.1e1) * (0.149184555e9 * pow(x, 0.18e2) - 0.585262485e9 * pow(x, 0.16e2) + 0.949074300e9 * pow(x, 0.14e2) - 0.822531060e9 * pow(x, 0.12e2) + 0.411265530e9 * pow(x, 0.10e2) - 0.119399670e9 * pow(x, 0.8e1) + 0.19213740e8 * pow(x, 0.6e1) - 0.1524900e7 * pow(x, 0.4e1) + 0.45747e5 * x * x - 0.221e3);
      break;
    case(3):
      y = 0.1514205e7 / 0.32768e5 * pow(x - 0.1e1, 0.3e1 / 0.2e1) * pow(x + 0.1e1, 0.3e1 / 0.2e1) * x * (0.19458855e8 * pow(x, 0.16e2) - 0.67856520e8 * pow(x, 0.14e2) + 0.96282900e8 * pow(x, 0.12e2) - 0.71524440e8 * pow(x, 0.10e2) + 0.29801850e8 * pow(x, 0.8e1) - 0.6921720e7 * pow(x, 0.6e1) + 0.835380e6 * pow(x, 0.4e1) - 0.44200e5 * x * x + 0.663e3);
      break;
    case(4):
      y = 0.77224455e8 / 0.32768e5 * pow(x - 0.1e1, 0.2e1) * pow(x + 0.1e1, 0.2e1) * (0.6486285e7 * pow(x, 0.16e2) - 0.19957800e8 * pow(x, 0.14e2) + 0.24542700e8 * pow(x, 0.12e2) - 0.15426840e8 * pow(x, 0.10e2) + 0.5259150e7 * pow(x, 0.8e1) - 0.950040e6 * pow(x, 0.6e1) + 0.81900e5 * pow(x, 0.4e1) - 0.2600e4 * x * x + 0.13e2);
      break;
    case(5):
      y = 0.386122275e9 / 0.2048e4 * pow(x - 0.1e1, 0.5e1 / 0.2e1) * pow(x + 0.1e1, 0.5e1 / 0.2e1) * x * (0.1297257e7 * pow(x, 0.14e2) - 0.3492615e7 * pow(x, 0.12e2) + 0.3681405e7 * pow(x, 0.10e2) - 0.1928355e7 * pow(x, 0.8e1) + 0.525915e6 * pow(x, 0.6e1) - 0.71253e5 * pow(x, 0.4e1) + 0.4095e4 * x * x - 0.65e2);
      break;
    case(6):
      y = 0.25097947875e11 / 0.2048e4 * pow(x - 0.1e1, 0.3e1) * pow(x + 0.1e1, 0.3e1) * (0.299367e6 * pow(x, 0.14e2) - 0.698523e6 * pow(x, 0.12e2) + 0.623007e6 * pow(x, 0.10e2) - 0.267003e6 * pow(x, 0.8e1) + 0.56637e5 * pow(x, 0.6e1) - 0.5481e4 * pow(x, 0.4e1) + 0.189e3 * x * x - 0.1e1);
      break;
    default:
      fprintf(stderr,"%s(%i,%i) not implemented!\n",__FUNCTION__,u,v);
      exit(1);
      break;
    }
    break;
  default:
    fprintf(stderr,"%s(%i,%i) not implemented!\n",__FUNCTION__,u,v);
    exit(1);
    break;
  }
  return y;
}
