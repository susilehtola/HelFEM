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
#include "SphericalNucleus.h"
#include <cmath>

namespace helfem {
  namespace modelpotential {
    template <typename T>
    SphericalNucleusT<T>::SphericalNucleusT(int Z_, T Rrms) : Z(Z_) {
      // Eqn (4) in Visscher-Dyall 1997
      R0_ = std::sqrt(T(5)/T(3))*Rrms;
    }

    template <typename T>
    SphericalNucleusT<T>::~SphericalNucleusT() {
    }

    template <typename T>
    T SphericalNucleusT<T>::V(T r) const {
      if(r>=R0_) {
        // See full charge, eqn (7a) in Visscher-Dyall 1997
        return -Z/r;
      } else {
        // See only charge inside, eqn (7b) in Visscher-Dyall 1997
        return -Z/(T(2)*R0_)*(T(3)-std::pow(r/R0_,2));
      }
    }

    template <typename T>
    T SphericalNucleusT<T>::R0() const {
      return R0_;
    }

    template <typename T>
    void SphericalNucleusT<T>::set_R0(T R0) {
      R0_=R0;
    }

    template class SphericalNucleusT<double>;
    template class SphericalNucleusT<long double>;
#ifdef HELFEM_HAVE_FLOAT128
    template class SphericalNucleusT<_Float128>;
#endif
  }
}
