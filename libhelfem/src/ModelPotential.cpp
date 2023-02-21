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
#include <ModelPotential.h>

namespace helfem {
  namespace modelpotential {
    ModelPotential::ModelPotential() {
    }

    ModelPotential::~ModelPotential() {
    }

    arma::vec ModelPotential::V(const arma::vec & r) const {
      arma::vec pot(r.n_elem);
      for(size_t i=0;i<r.n_elem;i++)
        pot(i)=V(r(i));
      return pot;
    }
  }
}
