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
#ifndef MODELPOTENTIAL_MODELPOTENTIAL_H
#define MODELPOTENTIAL_MODELPOTENTIAL_H

#include <armadillo>

namespace helfem {
  namespace modelpotential {
    /// Model potential
    class ModelPotential {
    public:
      /// Constructor
      ModelPotential();
      /// Destructor
      virtual ~ModelPotential();

      /// Potential
      virtual double V(double r) const=0;
      /// Potential
      arma::vec V(const arma::vec & r) const;
    };
  }
}

#endif
