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
#include "HollowNucleus.h"
#include <cmath>

namespace helfem {
  namespace modelpotential {
    template <typename T>
    HollowNucleusT<T>::HollowNucleusT(int Z_, T R_) : Z(Z_), R(R_) {
    }

    template <typename T>
    HollowNucleusT<T>::~HollowNucleusT() {
    }

    template <typename T>
    T HollowNucleusT<T>::V(T r) const {
      if(r>=R) {
        return -Z/r;
      } else {
        return -Z/R;
      }
    }

    template <typename T>
    T HollowNucleusT<T>::get_R() const {
      return R;
    }

    template <typename T>
    void HollowNucleusT<T>::set_R(T R_) {
      R=R_;
    }

    template class HollowNucleusT<double>;
    template class HollowNucleusT<long double>;
  }
}
