/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Susi Lehtola
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * See the LICENSE file at the root of this source distribution
 * for the full license text.
 */

// Phase 5.17: internal state and math migrated arma -> Eigen. The
// public API (constructors, update, solve_F, solve_P) still takes
// arma::mat / arma::mat& for SCF-driver compat; bridging happens at
// the class boundary.

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <ArmaEigen.h>
#include <Eigen/SVD>
#include <Eigen/QR>
#include "diis.h"
#include "lbfgs.h"

// Maximum allowed absolute weight for a Fock matrix
#define MAXWEIGHT 10.0
// Trigger cooloff if energy rises more than
#define COOLTHR 0.1

bool operator<(const diis_pol_entry_t & lhs, const diis_pol_entry_t & rhs) {
  return lhs.E < rhs.E;
}
bool operator<(const diis_unpol_entry_t & lhs, const diis_unpol_entry_t & rhs) {
  return lhs.E < rhs.E;
}

namespace {
  // Vectorise a matrix in column-major order into an Eigen Vector.
  inline helfem::Vector vectorise(const helfem::Matrix & M) {
    helfem::Vector v(M.size());
    std::memcpy(v.data(), M.data(), sizeof(double) * static_cast<size_t>(M.size()));
    return v;
  }
}

DIIS::DIIS(const arma::mat & S_, const arma::mat & Sinvh_, bool usediis_, double diiseps_, double diisthr_, bool useadiis_, bool verbose_, size_t imax_) {
  S = helfem::to_eigen(S_);
  Sinvh = helfem::to_eigen(Sinvh_);
  usediis  = usediis_;
  useadiis = useadiis_;
  verbose  = verbose_;
  imax     = imax_;
  diiseps  = diiseps_;
  diisthr  = diisthr_;
  cooloff  = 0;
}

rDIIS::rDIIS(const arma::mat & S_, const arma::mat & Sinvh_, bool usediis_, double diiseps_, double diisthr_, bool useadiis_, bool verbose_, size_t imax_) : DIIS(S_, Sinvh_, usediis_, diiseps_, diisthr_, useadiis_, verbose_, imax_) {}

uDIIS::uDIIS(const arma::mat & S_, const arma::mat & Sinvh_, bool combine_, bool usediis_, double diiseps_, double diisthr_, bool useadiis_, bool verbose_, size_t imax_) : DIIS(S_, Sinvh_, usediis_, diiseps_, diisthr_, useadiis_, verbose_, imax_), combine(combine_) {}

DIIS::~DIIS() {}
rDIIS::~rDIIS() {}
uDIIS::~uDIIS() {}

void rDIIS::clear() { stack.clear(); }
void uDIIS::clear() { stack.clear(); }
void rDIIS::erase_last() { stack.erase(stack.begin()); }
void uDIIS::erase_last() { stack.erase(stack.begin()); }

void rDIIS::update(const arma::mat & F, const arma::mat & P, double E, double & error) {
  diis_unpol_entry_t hlp;
  hlp.F = helfem::to_eigen(F);
  hlp.P = helfem::to_eigen(P);
  hlp.E = E;

  // errmat = F P S; FPS - SPF; then to orthonormal basis.
  helfem::Matrix errmat = hlp.F * hlp.P * S;
  errmat -= errmat.transpose().eval();
  errmat = Sinvh.transpose() * errmat * Sinvh;
  hlp.err = vectorise(errmat);

  error = errmat.cwiseAbs().maxCoeff();

  if (stack.size() == imax) erase_last();
  stack.push_back(hlp);

  PiF_update();
}

void rDIIS::PiF_update() {
  const helfem::Matrix & Fn = stack.back().F;
  const helfem::Matrix & Pn = stack.back().P;

  PiF = helfem::Vector::Zero(stack.size());
  for (size_t i = 0; i < stack.size(); ++i)
    PiF(i) = ((stack[i].P - Pn) * Fn).trace();

  PiFj = helfem::Matrix::Zero(stack.size(), stack.size());
  for (size_t i = 0; i < stack.size(); ++i)
    for (size_t j = 0; j < stack.size(); ++j)
      PiFj(i, j) = ((stack[i].P - Pn) * (stack[j].F - Fn)).trace();
}

void uDIIS::update(const arma::mat & Fa, const arma::mat & Fb, const arma::mat & Pa, const arma::mat & Pb, double E, double & error) {
  diis_pol_entry_t hlp;
  hlp.Fa = helfem::to_eigen(Fa);
  hlp.Fb = helfem::to_eigen(Fb);
  hlp.Pa = helfem::to_eigen(Pa);
  hlp.Pb = helfem::to_eigen(Pb);
  hlp.E  = E;

  helfem::Matrix errmata = hlp.Fa * hlp.Pa * S;
  helfem::Matrix errmatb = hlp.Fb * hlp.Pb * S;
  errmata -= errmata.transpose().eval();
  errmatb -= errmatb.transpose().eval();
  errmata = Sinvh.transpose() * errmata * Sinvh;
  errmatb = Sinvh.transpose() * errmatb * Sinvh;

  if (combine) {
    hlp.err = vectorise(helfem::Matrix(errmata + errmatb));
  } else {
    const helfem::Vector va = vectorise(errmata);
    const helfem::Vector vb = vectorise(errmatb);
    hlp.err.resize(va.size() + vb.size());
    hlp.err.head(va.size()) = va;
    hlp.err.tail(vb.size()) = vb;
  }

  error = hlp.err.cwiseAbs().maxCoeff();

  if (stack.size() == imax) erase_last();
  stack.push_back(hlp);

  PiF_update();
}

void uDIIS::PiF_update() {
  const helfem::Matrix & Fan = stack.back().Fa;
  const helfem::Matrix & Fbn = stack.back().Fb;
  const helfem::Matrix & Pan = stack.back().Pa;
  const helfem::Matrix & Pbn = stack.back().Pb;

  PiF = helfem::Vector::Zero(stack.size());
  for (size_t i = 0; i < stack.size(); ++i)
    PiF(i) = ((stack[i].Pa - Pan) * Fan).trace()
            + ((stack[i].Pb - Pbn) * Fbn).trace();

  PiFj = helfem::Matrix::Zero(stack.size(), stack.size());
  for (size_t i = 0; i < stack.size(); ++i)
    for (size_t j = 0; j < stack.size(); ++j)
      PiFj(i, j) = ((stack[i].Pa - Pan) * (stack[j].Fa - Fan)).trace()
                 + ((stack[i].Pb - Pbn) * (stack[j].Fb - Fbn)).trace();
}

helfem::Vector rDIIS::get_energies() const {
  helfem::Vector E(stack.size());
  for (size_t i = 0; i < stack.size(); ++i) E(i) = stack[i].E;
  return E;
}

helfem::Matrix rDIIS::get_diis_error() const {
  helfem::Matrix err(stack[0].err.size(), stack.size());
  for (size_t i = 0; i < stack.size(); ++i) err.col(i) = stack[i].err;
  return err;
}

helfem::Vector uDIIS::get_energies() const {
  helfem::Vector E(stack.size());
  for (size_t i = 0; i < stack.size(); ++i) E(i) = stack[i].E;
  return E;
}
helfem::Matrix uDIIS::get_diis_error() const {
  helfem::Matrix err(stack[0].err.size(), stack.size());
  for (size_t i = 0; i < stack.size(); ++i) err.col(i) = stack[i].err;
  return err;
}

helfem::Vector DIIS::get_w() {
  const helfem::Matrix de = get_diis_error();
  const double err = de.col(de.cols() - 1).cwiseAbs().maxCoeff();

  helfem::Vector w;

  if (useadiis && !usediis) {
    w = get_w_adiis();
    if (verbose) {
      printf("ADIIS weights\n");
      std::cout << w.transpose() << "\n";
    }
  } else if (!useadiis && usediis) {
    if (err > diisthr)
      throw std::runtime_error("DIIS error too large for only DIIS to converge wave function.\n");
    w = get_w_diis();
    if (verbose) {
      printf("DIIS weights\n");
      std::cout << w.transpose() << "\n";
    }
  } else if (useadiis && usediis) {
    double diisw;
    if (diiseps == diisthr) {
      diisw = (err <= diisthr) ? 1.0 : 0.0;
    } else {
      diisw = std::clamp(1.0 - (err - diisthr) / (diiseps - diisthr), 0.0, 1.0);
    }
    double adiisw = 1.0 - diisw;

    if (cooloff > 0) {
      diisw = 0.0;
      cooloff--;
    } else {
      const helfem::Vector E = get_energies();
      if (E.size() > 1 && E(E.size() - 1) - E(E.size() - 2) > COOLTHR) {
        cooloff = 2;
        diisw = 0.0;
      }
    }

    w = helfem::Vector::Zero(de.cols());

    helfem::Vector wd, wa;
    if (diisw != 0.0) {
      wd = get_w_diis();
      w += diisw * wd;
    }
    if (adiisw != 0.0) {
      wa = get_w_adiis();
      w += adiisw * wa;
    }

    if (verbose) {
      if (adiisw != 0.0) { printf("ADIIS weights\n"); std::cout << wa.transpose() << "\n"; }
      if (diisw  != 0.0) { printf("CDIIS weights\n"); std::cout << wd.transpose() << "\n"; }
      if (adiisw != 0.0 && diisw != 0.0) { printf(" DIIS weights\n"); std::cout << w.transpose() << "\n"; }
    }
  } else {
    throw std::runtime_error("Nor DIIS or ADIIS has been turned on.\n");
  }
  return w;
}

helfem::Vector DIIS::get_w_diis() const {
  return get_w_diis_wrk(get_diis_error());
}

helfem::Vector DIIS::get_w_diis_wrk(const helfem::Matrix & errs) const {
  const Eigen::Index N = errs.cols();

  helfem::Matrix B = helfem::Matrix::Zero(N, N);
  for (Eigen::Index i = 0; i < N; ++i)
    for (Eigen::Index j = 0; j < N; ++j)
      B(i, j) = errs.col(i).dot(errs.col(j));

  // Solve B w = 1 via SVD; then renormalise so sum(w) = 1.
  helfem::Vector rh = helfem::Vector::Ones(N);
  Eigen::JacobiSVD<helfem::Matrix> svd(B, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const helfem::Vector sval = svd.singularValues();
  const helfem::Matrix U = svd.matrixU();
  const helfem::Matrix V = svd.matrixV();

  helfem::Vector sol = helfem::Vector::Zero(N);
  for (Eigen::Index i = 0; i < N; ++i)
    if (sval(i) != 0.0)
      sol += (U.col(i).dot(rh) / sval(i)) * V.col(i);

  if (sol.sum() == 0.0)
    sol = helfem::Vector::Ones(N);
  sol /= sol.sum();
  return sol;
}

void rDIIS::solve_F(arma::mat & F) {
  helfem::Vector sol;
  while (true) {
    sol = get_w();
    if (std::abs(sol(sol.size() - 1)) <= std::sqrt(DBL_EPSILON)) {
      if (verbose) printf("Weight on last matrix too small, reducing to %i matrices.\n", (int) stack.size() - 1);
      erase_last();
      PiF_update();
    } else {
      break;
    }
  }

  helfem::Matrix Fout = helfem::Matrix::Zero(stack[0].F.rows(), stack[0].F.cols());
  for (size_t i = 0; i < stack.size(); ++i)
    Fout += sol(i) * stack[i].F;
  F = helfem::to_arma(Fout);
}

void uDIIS::solve_F(arma::mat & Fa, arma::mat & Fb) {
  helfem::Vector sol;
  while (true) {
    sol = get_w();
    if (std::abs(sol(sol.size() - 1)) <= std::sqrt(DBL_EPSILON)) {
      if (verbose) printf("Weight on last matrix too small, reducing to %i matrices.\n", (int) stack.size() - 1);
      erase_last();
      PiF_update();
    } else {
      break;
    }
  }

  helfem::Matrix Fao = helfem::Matrix::Zero(stack[0].Fa.rows(), stack[0].Fa.cols());
  helfem::Matrix Fbo = helfem::Matrix::Zero(stack[0].Fb.rows(), stack[0].Fb.cols());
  for (size_t i = 0; i < stack.size(); ++i) {
    Fao += sol(i) * stack[i].Fa;
    Fbo += sol(i) * stack[i].Fb;
  }
  Fa = helfem::to_arma(Fao);
  Fb = helfem::to_arma(Fbo);
}

void rDIIS::solve_P(arma::mat & P) {
  helfem::Vector sol;
  while (true) {
    sol = get_w();
    if (std::abs(sol(sol.size() - 1)) <= std::sqrt(DBL_EPSILON)) {
      if (verbose) printf("Weight on last matrix too small, reducing to %i matrices.\n", (int) stack.size() - 1);
      erase_last();
      PiF_update();
    } else {
      break;
    }
  }

  helfem::Matrix Pout = helfem::Matrix::Zero(stack[0].P.rows(), stack[0].P.cols());
  for (size_t i = 0; i < stack.size(); ++i)
    Pout += sol(i) * stack[i].P;
  P = helfem::to_arma(Pout);
}

void uDIIS::solve_P(arma::mat & Pa, arma::mat & Pb) {
  helfem::Vector sol;
  while (true) {
    sol = get_w();
    if (std::abs(sol(sol.size() - 1)) <= std::sqrt(DBL_EPSILON)) {
      if (verbose) printf("Weight on last matrix too small, reducing to %i matrices.\n", (int) stack.size() - 1);
      erase_last();
      PiF_update();
    } else {
      break;
    }
  }

  helfem::Matrix Pao = helfem::Matrix::Zero(stack[0].Pa.rows(), stack[0].Pa.cols());
  helfem::Matrix Pbo = helfem::Matrix::Zero(stack[0].Pb.rows(), stack[0].Pb.cols());
  for (size_t i = 0; i < stack.size(); ++i) {
    Pao += sol(i) * stack[i].Pa;
    Pbo += sol(i) * stack[i].Pb;
  }
  Pa = helfem::to_arma(Pao);
  Pb = helfem::to_arma(Pbo);
}

// -------- ADIIS: parabolic backtracking line search on x -> c(x) = x.^2 / dot(x,x) --------

static void find_minE(const std::vector< std::pair<double,double> > & steps, double & Emin, size_t & imin) {
  Emin = steps[0].second;
  imin = 0;
  for (size_t i = 1; i < steps.size(); ++i)
    if (steps[i].second < Emin) {
      Emin = steps[i].second;
      imin = i;
    }
}

static helfem::Vector compute_c(const helfem::Vector & x) {
  return x.array().square().matrix() / x.dot(x);
}

helfem::Vector DIIS::get_w_adiis() const {
  const Eigen::Index N = PiF.size();

  if (N == 1) return helfem::Vector::Ones(1);

  helfem::Vector x = helfem::Vector::Ones(N) / static_cast<double>(N);

  LBFGS bfgs;
  double steplen = 0.01;
  const double fac = 2.0;

  for (size_t iiter = 0; iiter < 1000; ++iiter) {
    const helfem::Vector g = get_dEdx_adiis(x);
    if (g.norm() <= 1e-7) break;

    bfgs.update(x, g);
    const helfem::Vector sd = -bfgs.solve();

    std::vector< std::pair<double, double> > steps;
    steps.push_back({steplen / fac, get_E_adiis(x + sd * (steplen / fac))});
    steps.push_back({steplen,       get_E_adiis(x + sd * steplen)});

    double Emin;
    size_t imin;
    while (true) {
      std::sort(steps.begin(), steps.end());
      find_minE(steps, Emin, imin);
      if (imin == 0 || imin == steps.size() - 1) {
        std::pair<double, double> p;
        if (imin == 0) {
          p.first = steps[imin].first / fac;
          if (steps[imin].first < DBL_EPSILON) break;
        } else {
          p.first = steps[imin].first * fac;
        }
        p.second = get_E_adiis(x + sd * p.first);
        steps.push_back(p);
      } else {
        break;
      }
    }

    if (imin != 0 && imin != steps.size() - 1) {
      // Parabolic interpolation A b = y.
      helfem::Matrix A(3, 3);
      helfem::Vector y(3);
      for (size_t i = 0; i < 3; ++i) {
        A(i, 0) = 1.0;
        A(i, 1) = steps[imin + i - 1].first;
        A(i, 2) = std::pow(A(i, 1), 2);
        y(i)    = steps[imin + i - 1].second;
      }
      Eigen::ColPivHouseholderQR<helfem::Matrix> qr(A);
      const helfem::Vector b = qr.solve(y);
      if (qr.info() == Eigen::Success && b(2) > std::sqrt(DBL_EPSILON)) {
        const double x0 = -b(1) / (2 * b(2));
        if (A(0, 1) < x0 && x0 < A(2, 1)) {
          steps.push_back({x0, get_E_adiis(x + sd * x0)});
          find_minE(steps, Emin, imin);
        }
      }
    }

    if (steps[imin].first < DBL_EPSILON) break;
    x += steps[imin].first * sd;
    steplen = steps[imin].first;
  }

  return compute_c(x);
}

double DIIS::get_E_adiis(const helfem::Vector & x) const {
  if (x.size() != PiF.size())
    throw std::domain_error("Incorrect number of parameters.\n");
  const helfem::Vector c = compute_c(x);
  return 2.0 * c.dot(PiF) + (c.transpose() * PiFj * c)(0);
}

helfem::Vector DIIS::get_dEdx_adiis(const helfem::Vector & x) const {
  const helfem::Vector c = compute_c(x);
  const helfem::Vector dEdc = 2.0 * PiF + PiFj * c + PiFj.transpose() * c;

  // Jacobian jac(i, j) = dc_i / dx_j, then dEdx = jac^T dEdc.
  const double xnorm = x.dot(x);
  helfem::Matrix jac(c.size(), c.size());
  for (Eigen::Index i = 0; i < c.size(); ++i) {
    for (Eigen::Index j = 0; j < c.size(); ++j)
      jac(i, j) = -c(i) * 2.0 * x(j) / xnorm;
    jac(i, i) += 2.0 * x(i) / xnorm;
  }
  return jac.transpose() * dEdc;
}
