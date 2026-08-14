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
      // Taylor series for erf(mu_*r)/r from Maple
      if(R <= Rcut) {
        T mur2 = std::pow(mu_*R,2);
        return -Z*utils::two_over_sqrtpi<T>()*mu_*( T(1) + (T(-1)/T(3) + (T(1)/T(10) - T(1)/T(42)*mur2)*mur2)*mur2);
      } else {
        return -Z*std::erf(mu_*R)/R;
      }
    }

    template <typename T>
    T GaussianNucleusT<T>::mu() const {
      return mu_;
    }

    template <typename T>
    void GaussianNucleusT<T>::set_mu(T mu) {
      // Set value
      mu_=mu;
      // Update Taylor series cutoff: sixth-order term is epsilon
      Rcut = std::pow(T(42)*std::numeric_limits<T>::epsilon(), T(1)/T(6))/mu_;
    }

    template class GaussianNucleusT<double>;
    template class GaussianNucleusT<long double>;
#ifdef HELFEM_HAVE_FLOAT128
    template class GaussianNucleusT<_Float128>;
#endif
  }
}
