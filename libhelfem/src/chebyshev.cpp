/*
 *                This source code is part of
 *
 *                          HelFEM
 *                             -
 * Finite element methods for electronic structure calculations on small systems
 *
 * Written by Susi Lehtola, 2018-
 * Copyright (c) 2018- Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */
#include "chebyshev.h"

namespace helfem {
  namespace chebyshev {
    // Modified Gauss-Chebyshev quadrature of the second kind for calculating
    // \int_{-1}^{1} f(x) dx
    void chebyshev(int n, arma::vec &x, arma::vec &w) {
      // Resize vectors to correct size
      x.zeros(n);
      w.zeros(n);

      // 1/(n+1)
      double oonpp = 1.0 / (n + 1.0);

      // cos ( i*pi / (n+1))
      double cosine;
      // sin ( i*pi / (n+1))
      double sine;
      double sinesq;

      // Fill elements
      for (int i = 1; i <= n; i++) {
        // Compute value of sine and cosine
        sine = sin(i * M_PI * oonpp);
        sinesq = sine * sine;
        cosine = cos(i * M_PI * oonpp);

        // Weight is
        w(i - 1) = 16.0 / 3.0 / (n + 1.0) * sinesq * sinesq;

        // Node is
        x(i - 1) = 1.0 - 2.0 * i * oonpp +
                   M_2_PI * (1.0 + 2.0 / 3.0 * sinesq) * cosine * sine;
      }

      // Reverse order
      x = reverse(x);
      w = reverse(w);
    }
  } // namespace chebyshev
} // namespace helfem
