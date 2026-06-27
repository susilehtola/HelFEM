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
