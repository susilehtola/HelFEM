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
#include "RadialPotential.h"
#include <cmath>

namespace helfem {
  namespace modelpotential {
    template <typename T>
    RadialPotentialT<T>::RadialPotentialT(int n_) : n(n_) {
    }

    template <typename T>
    RadialPotentialT<T>::~RadialPotentialT() {
    }

    template <typename T>
    T RadialPotentialT<T>::V(T R) const {
      return std::pow(R,n);
    }

    template class RadialPotentialT<double>;
    template class RadialPotentialT<long double>;
#ifdef HELFEM_HAVE_FLOAT128
    template class RadialPotentialT<_Float128>;
#endif
  }
}
