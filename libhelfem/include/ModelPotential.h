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
      virtual double V(double r) const = 0;
      /// Potential
      arma::vec V(const arma::vec &r) const;
    };
  } // namespace modelpotential
} // namespace helfem

#endif
