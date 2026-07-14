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
#include "PointNucleus.h"
#include <cmath>

namespace helfem {
  namespace modelpotential {
    template <typename T>
    PointNucleusT<T>::PointNucleusT(int Z_) : Z(Z_) {
    }

    template <typename T>
    PointNucleusT<T>::~PointNucleusT() {
    }

    template <typename T>
    T PointNucleusT<T>::V(T R) const {
      return -Z/R;
    }

    template class PointNucleusT<double>;
    template class PointNucleusT<long double>;
  }
}
