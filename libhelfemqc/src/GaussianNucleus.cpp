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
#include "GaussianNucleus.h"
#include "utils.h"
#include <cmath>
#include <limits>

namespace helfem {
  namespace modelpotential {
    template <typename T>
    GaussianNucleusT<T>::GaussianNucleusT(int Z_, T Rrms) : Z(Z_) {
      // Eqn (11) in Visscher-Dyall 1997
      set_mu(std::sqrt(T(3)/T(2))/Rrms);
    }

    template <typename T>
    GaussianNucleusT<T>::~GaussianNucleusT() {
    }

    template <typename T>
    T GaussianNucleusT<T>::V(T R) const {
      // Taylor series for erf(mu*r)/r from Maple
      if(R <= Rcut) {
        T mur2 = std::pow(mu*R,2);
        return -Z*utils::two_over_sqrtpi<T>()*mu*( T(1) + (T(-1)/T(3) + (T(1)/T(10) - T(1)/T(42)*mur2)*mur2)*mur2);
      } else {
        return -Z*std::erf(mu*R)/R;
      }
    }

    template <typename T>
    T GaussianNucleusT<T>::get_mu() const {
      return mu;
    }

    template <typename T>
    void GaussianNucleusT<T>::set_mu(T mu_) {
      // Set value
      mu=mu_;
      // Update Taylor series cutoff: sixth-order term is epsilon
      Rcut = std::pow(T(42)*std::numeric_limits<T>::epsilon(), T(1)/T(6))/mu;
    }

    template class GaussianNucleusT<double>;
    template class GaussianNucleusT<long double>;
#ifdef HELFEM_HAVE_FLOAT128
    template class GaussianNucleusT<_Float128>;
#endif
  }
}
