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
#ifndef MODELPOTENTIAL_HOLLOWNUCLEUS_H
#define MODELPOTENTIAL_HOLLOWNUCLEUS_H

#include <ModelPotential.h>

namespace helfem {
  namespace modelpotential {
    /// Thin hollow nucleus
    template <typename T>
    class HollowNucleusT : public ModelPotentialT<T> {
      /// Charge
      int Z;
      /// Size
      T R_;
    public:
      /// Constructor
      HollowNucleusT(int Z, T R_);
      /// Destructor
      ~HollowNucleusT();
      /// Potential
      T V(T r) const override;
      /// All the charge sits on the shell r = R_, so the potential is
      /// constant inside and -Z/r outside, with a kink at R_.
      std::vector<T> breakpoints(T a, T b) const override {
        if (R_ > a && R_ < b) return std::vector<T>{R_};
        return std::vector<T>();
      }
      /// Get R_
      T R() const;
      /// Set R_
      void set_R(T R_);
    };

    /// The double instantiation, which every existing caller uses.
    using HollowNucleus = HollowNucleusT<double>;
  }
}

#endif
