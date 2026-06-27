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
#ifndef MODELPOTENTIAL_RADIALPOTENTIAL_H
#define MODELPOTENTIAL_RADIALPOTENTIAL_H

#include <ModelPotential.h>

namespace helfem {
  namespace modelpotential {
    /// Simple r^n radial potential
    class RadialPotential : public ModelPotential {
      /// Exponent
      int n;
    public:
      /// Constructor
      RadialPotential(int n);
      /// Destructor
      ~RadialPotential();
      /// Potential
      double V(double r) const override;
    };
  }
}

#endif
