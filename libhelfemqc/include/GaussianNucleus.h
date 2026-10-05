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
 * SPDX-License-Identifier: BSD-3-Clause
 * See the LICENSE file at the root of this source distribution
 * for the full license text.
 */
#ifndef MODELPOTENTIAL_GAUSSIANNUCLEUS_H
#define MODELPOTENTIAL_GAUSSIANNUCLEUS_H

#include <helfem/ModelPotential.h>
#include <vector>

namespace helfem {
  namespace modelpotential {
    /// Gaussian nucleus
    template <typename T>
    class GaussianNucleusT : public ModelPotentialT<T> {
      /// Charge
      int Z;
      /// Size
      T mu_;

      /// Cutoff for Taylor series
      T Rcut;
      /// Quadrature split radii, set with mu_; see breakpoints()
      std::vector<T> splits_;
    public:
      /// Constructor
      GaussianNucleusT(int Z, T Rrms);
      /// Destructor
      ~GaussianNucleusT();
      /// Potential
      T V(T r) const override;
      /// The potential has no kink, but it has a scale: -Z erf(mu r)/r turns
      /// from constant to Coulomb around r ~ 1/mu, which for a real nucleus
      /// (1/mu ~ 1e-4 bohr) is a thousand times smaller than the first
      /// element. Plain order-refinement then resolves it only by brute
      /// order and hits the cap (Ne, Visscher-Dyall Rrms: n=512, relative
      /// change still 2e-12). Splitting at 2^k/mu, doubling until
      /// erfc(2^k) < eps(T), gives panels matched to the turnover, and
      /// beyond the last one V = -Z/r to working precision, so the integrand
      /// is polynomial there again -- {1, 2, 4, 8}/mu in double.
      std::vector<T> breakpoints(T a, T b) const override;
      /// Get mu_
      T mu() const;
      /// Set mu_
      void set_mu(T mu_);
    };

    /// The double instantiation, which every existing caller uses.
    using GaussianNucleus = GaussianNucleusT<double>;
  }
}

#endif
