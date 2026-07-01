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


#ifndef ERKALE_DIIS
#define ERKALE_DIIS

// Phase 5.17: internal state migrated to Eigen. Public API (constructors,
// update, solve_F, solve_P) still takes arma::mat & for compat with
// SCF drivers; bridging happens at the class boundary.
#include <armadillo>
#include <Matrix.h>
#include <vector>

/// Spin-polarized entry
typedef struct {
  helfem::Matrix Pa;
  helfem::Matrix Fa;
  helfem::Matrix Pb;
  helfem::Matrix Fb;
  double E;
  /// DIIS error vector (vectorised error matrix)
  helfem::Vector err;
} diis_pol_entry_t;

/// Spin-unpolarized entry
typedef struct {
  helfem::Matrix P;
  helfem::Matrix F;
  double E;
  helfem::Vector err;
} diis_unpol_entry_t;

/// Helper for sort
bool operator<(const diis_pol_entry_t & lhs, const diis_pol_entry_t & rhs);
/// Helper for sort
bool operator<(const diis_unpol_entry_t & lhs, const diis_unpol_entry_t & rhs);

/**
 * \class DIIS
 *
 * \brief DIIS - Direct Inversion in the Iterative Subspace and ADIIS
 *
 * This class contains the DIIS and ADIIS convergence accelerators.
 *
 *
 * The original DIIS (C1-DIIS) is based on the articles
 *
 * P. Pulay, "Convergence acceleration of iterative sequences. The
 * case of SCF iteration", Chem. Phys. Lett. 73 (1980), pp. 393 - 398
 *
 * and
 *
 * P. Pulay, "Improved SCF Convergence Acceleration", J. Comp. Chem. 3
 * (1982), pp. 556 - 560.
 *
 * Using C1-DIIS is, however, not recommended. An improved method that
 * yields better convergence was suggested in
 *
 * H. Sellers, "The C2-DIIS convergence acceleration algorithm",
 * Int. J. Quant. Chem. 45 (1993), pp. 31 - 41
 *
 * which used to be the default method in ERKALE. However, C2-DIIS is
 * still problematic when linear dependencies exist in the basis. The
 * current implementation employs a singular value decomposition as
 * suggested in
 *
 * Hans Henrik B. Sørensen, and Ole Østerby, "On One-Point Iterations
 * and DIIS", AIP Conference Proceedings 1168 (2009), pp. 468 - 472.
 *
 *
 * The ADIIS algorithm is described in
 *
 * X. Hu and W. Yang, "Accelerating self-consistent field convergence
 * with the augmented Roothaan–Hall energy function",
 * J. Chem. Phys. 132 (2010), 054109.
 *
 * \author Susi Lehtola
 * \date 2011/05/08 19:32
 */

class DIIS {
 protected:
  helfem::Matrix S;
  helfem::Matrix Sinvh;

  bool usediis;
  bool useadiis;
  bool verbose;

  double diiseps;
  double diisthr;
  int cooloff;

  size_t imax;
  virtual helfem::Vector get_energies() const = 0;
  virtual helfem::Matrix get_diis_error() const = 0;
  virtual void erase_last() = 0;

  helfem::Vector PiF;
  helfem::Matrix PiFj;

  helfem::Vector get_w();
  helfem::Vector get_w_diis() const;
  helfem::Vector get_w_diis_wrk(const helfem::Matrix & err) const;
  helfem::Vector get_w_adiis() const;

 public:
  // Public API still arma-typed for compat with SCF drivers.
  DIIS(const arma::mat & S, const arma::mat & Sinvh, bool usediis, double diiseps, double diisthr, bool useadiis, bool verbose, size_t imax);
  virtual ~DIIS();

  virtual void clear() = 0;

  double get_E_adiis(const helfem::Vector & x) const;
  helfem::Vector get_dEdx_adiis(const helfem::Vector & x) const;
};

/// Spin-restricted DIIS
class rDIIS: protected DIIS {
  std::vector<diis_unpol_entry_t> stack;

  helfem::Vector get_energies() const;
  helfem::Matrix get_diis_error() const;
  void erase_last();
  void PiF_update();

 public:
  rDIIS(const arma::mat & S, const arma::mat & Sinvh, bool usediis, double diiseps, double diisthr, bool useadiis, bool verbose, size_t imax);
  ~rDIIS();

  // Public API still arma-typed.
  void update(const arma::mat & F, const arma::mat & P, double E, double & error);
  void solve_F(arma::mat & F);
  void solve_P(arma::mat & P);
  void clear();
};

/// Spin-unrestricted DIIS
class uDIIS: protected DIIS {
  std::vector<diis_pol_entry_t> stack;

  helfem::Vector get_energies() const;
  helfem::Matrix get_diis_error() const;
  void erase_last();
  void PiF_update();

  bool combine;

 public:
  uDIIS(const arma::mat & S, const arma::mat & Sinvh, bool combine, bool usediis, double diiseps, double diisthr, bool useadiis, bool verbose, size_t imax);
  ~uDIIS();

  void update(const arma::mat & Fa, const arma::mat & Fb, const arma::mat & Pa, const arma::mat & Pb, double E, double & error);
  void solve_F(arma::mat & Fa, arma::mat & Fb);
  void solve_P(arma::mat & Pa, arma::mat & Pb);
  void clear();
};

#endif
