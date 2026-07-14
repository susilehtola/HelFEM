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
#ifndef MODELPOTENTIAL_REGULARIZEDNUCLEUS_H
#define MODELPOTENTIAL_REGULARIZEDNUCLEUS_H

#include <ModelPotential.h>

namespace helfem {
  namespace modelpotential {
    /** Regularized nucleus, i.e. Gygi's Analytic Norm-Conserving
        Regularized Potential from doi:10.1021/acs.jctc.2c01191.
     */
    template <typename T>
    class RegularizedNucleusT : public ModelPotentialT<T> {
      /// Charge
      int Z;
      /// Size parameters
      T a, b;
    public:
      /// Constructor
      RegularizedNucleusT(int Z, T a);
      /// Destructor
      ~RegularizedNucleusT();
      /// Potential
      T V(T r) const override;
      /// Get a
      T get_a() const;
      /// Get b
      T get_b() const;
      /// Set mu
      void set_a(T a);
    };

    /// The double instantiation, which every existing caller uses.
    using RegularizedNucleus = RegularizedNucleusT<double>;
  }
}

#endif
