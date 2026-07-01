/*
 *                This source code is part of
 *
 *                          HelFEM
 *                             -
 * Finite element methods for electronic structure calculations on small systems
 *
 * Written by Susi Lehtola, 2015-
 * Copyright (c) 2015- Susi Lehtola
 *
 * Originally part of the ERKALE package.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * See the LICENSE file at the root of this source distribution
 * for the full license text.
 */

// Phase 5.17: LBFGS migrated arma -> Eigen. No external callers
// outside diis.cpp, so no bridging is needed.

#include "lbfgs.h"

LBFGS::LBFGS(size_t nmax_) : nmax(nmax_) {}
LBFGS::~LBFGS() {}

void LBFGS::update(const helfem::Vector & x, const helfem::Vector & g) {
  xk.push_back(x);
  gk.push_back(g);
  if (xk.size() > nmax) {
    xk.erase(xk.begin());
    gk.erase(gk.begin());
  }
}

helfem::Vector LBFGS::apply_diagonal_hessian(const helfem::Vector & q) const {
  if (xk.size() >= 2) {
    const helfem::Vector s = xk.back() - xk[xk.size() - 2];
    const helfem::Vector y = gk.back() - gk[gk.size() - 2];
    return (s.dot(y) / y.dot(y)) * q;
  }
  return q;
}

helfem::Vector LBFGS::solve() const {
  // Algorithm 9.1 in Nocedal.
  const size_t k = gk.size() - 1;
  helfem::Vector q = gk[k];

  std::vector<helfem::Vector> sk(k), yk(k);
  for (size_t i = 0; i < k; ++i) sk[i] = xk[i + 1] - xk[i];
  for (size_t i = 0; i < k; ++i) yk[i] = gk[i + 1] - gk[i];

  std::vector<double> alphai(k);
  for (size_t i = k - 1; i < k; --i) {  // relies on size_t wraparound
    alphai[i] = sk[i].dot(q) / yk[i].dot(sk[i]);
    q -= alphai[i] * yk[i];
  }

  helfem::Vector r = apply_diagonal_hessian(q);
  for (size_t i = 0; i < k; ++i) {
    const double beta = yk[i].dot(r) / yk[i].dot(sk[i]);
    r += sk[i] * (alphai[i] - beta);
  }
  return r;
}

void LBFGS::clear() {
  xk.clear();
  gk.clear();
}
