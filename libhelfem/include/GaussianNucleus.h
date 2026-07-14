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

#include <ModelPotential.h>

namespace helfem {
  namespace modelpotential {
    /// Gaussian nucleus
    template <typename T>
    class GaussianNucleusT : public ModelPotentialT<T> {
      /// Charge
      int Z;
      /// Size
      T mu;

      /// Cutoff for Taylor series
      T Rcut;
    public:
      /// Constructor
      GaussianNucleusT(int Z, T Rrms);
      /// Destructor
      ~GaussianNucleusT();
      /// Potential
      T V(T r) const override;
      /// Get mu
      T get_mu() const;
      /// Set mu
      void set_mu(T mu);
    };

    /// The double instantiation, which every existing caller uses.
    using GaussianNucleus = GaussianNucleusT<double>;
  }
}

#endif
