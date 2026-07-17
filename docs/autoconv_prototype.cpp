// Prototype: auto-converging FEM matrix-element evaluator.
//
// Demonstrates <d^k B | f(x) | d^l B> quadrature that refines its own Gauss-
// Lobatto order until the value is stable to eps(T), across double / long
// double / _Float128, on the three integrand regimes that matter for HelFEM:
//   A) polynomial       (overlap / kinetic / POINT-nucleus after the r^2 measure)
//   B) smooth non-poly  (finite nuclei, model potentials, erfc/Yukawa)
//   C) a kink at x0     (the radial 2e Green's function r<^L / r>^(L+1) at r=r')
//
// Uses lib1dfem's templated Gauss-Lobatto rule -- the real machinery.
#include <lib1dfem/types.h>
#include <lib1dfem/lobatto.h>
#include <functional>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

using helfem::lib1dfem::Vec;

// One Gauss-Lobatto panel of order n on [a,b].
template <typename T>
T panel(T a, T b, int n, const std::function<T(T)> & g) {
  Vec<T> x, w;
  helfem::lib1dfem::lobatto::lobatto_compute<T>(n, x, w);   // nodes/weights on [-1,1]
  const T half = (b - a) / T(2), mid = (a + b) / T(2);
  T I = T(0);
  for (int k = 0; k < n; ++k) I += w(k) * g(mid + half * x(k));
  return I * half;
}

// Auto-converge the order on a single panel until |I_n - I_{n-1}| <= tol*|I_n|.
// Returns {value, order, converged}.
template <typename T>
struct Res { T val; int order; bool ok; };

template <typename T>
Res<T> integrate(T a, T b, const std::function<T(T)> & g, int nmax = 400) {
  const T tol = T(8) * std::numeric_limits<T>::epsilon();
  T prev = T(0);
  bool have = false;
  for (int n = 2; n <= nmax; ++n) {
    T I = panel<T>(a, b, n, g);
    if (have && std::abs(I - prev) <= tol * (std::abs(I) + tol))
      return {I, n, true};
    prev = I; have = true;
  }
  return {prev, nmax, false};
}

// Subdivision-aware: same, but split at the breakpoints (e.g. the kink) so each
// sub-panel sees a smooth integrand. Sums the per-panel auto-converged values.
template <typename T>
Res<T> integrate_split(T a, T b, const std::vector<T> & brk,
                       const std::function<T(T)> & g) {
  std::vector<T> pts; pts.push_back(a);
  for (T p : brk) if (p > a && p < b) pts.push_back(p);
  pts.push_back(b);
  T tot = T(0); int maxord = 0; bool ok = true;
  for (size_t i = 0; i + 1 < pts.size(); ++i) {
    Res<T> r = integrate<T>(pts[i], pts[i + 1], g);
    tot += r.val; maxord = std::max(maxord, r.order); ok = ok && r.ok;
  }
  return {tot, maxord, ok};
}

template <typename T>
const char * name();
template <> const char * name<double>()      { return "double     "; }
template <> const char * name<long double>() { return "long double"; }
template <> const char * name<_Float128>()   { return "_Float128  "; }

template <typename T>
void run() {
  const int P = std::numeric_limits<T>::max_digits10;
  // Case A: g = x^2 on [0,1]  (polynomial; exact ref 1/3).
  {
    auto g = [](T x){ return x * x; };
    Res<T> r = integrate<T>(T(0), T(1), g);
    T err = std::abs(r.val - T(1) / T(3));
    printf("  A poly x^2      %s  n=%3d  err=%.*Le\n", name<T>(), r.order, 3, (long double)err);
  }
  // Case B: g = exp(-x) on [0,1]  (smooth non-poly; ref 1 - 1/e at precision T).
  {
    auto g = [](T x){ return std::exp(-x); };
    Res<T> r = integrate<T>(T(0), T(1), g);
    T err = std::abs(r.val - (T(1) - std::exp(T(-1))));
    printf("  B exp(-x)       %s  n=%3d  err=%.*Le\n", name<T>(), r.order, 3, (long double)err);
  }
  // Case C: g = |x - 1/2| on [0,1]  (kink; ref 1/4). Naive vs split-at-kink.
  {
    T x0 = T(1) / T(2);
    auto g = [x0](T x){ return std::abs(x - x0); };
    Res<T> naive = integrate<T>(T(0), T(1), g);
    Res<T> split = integrate_split<T>(T(0), T(1), {x0}, g);
    T en = std::abs(naive.val - T(1) / T(4));
    T es = std::abs(split.val - T(1) / T(4));
    printf("  C kink |x-.5|   %s  naive: n=%3d err=%.*Le %s | split@kink: n=%3d err=%.*Le\n",
           name<T>(), naive.order, 3, (long double)en, naive.ok ? "conv" : "STALL",
           split.order, 3, (long double)es);
  }
  (void)P;
}

int main() {
  printf("auto-converging matrix element (tol = 8*eps(T)); n = Gauss-Lobatto order reached\n\n");
  run<double>();
  run<long double>();
  run<_Float128>();
  return 0;
}
