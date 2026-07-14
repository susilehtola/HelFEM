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
    template <typename T>
    ModelPotentialT<T>::ModelPotentialT() {
    }

    template <typename T>
    ModelPotentialT<T>::~ModelPotentialT() {
    }

    template class ModelPotentialT<double>;
    template class ModelPotentialT<long double>;
#ifdef HELFEM_HAVE_FLOAT128
    template class ModelPotentialT<_Float128>;
#endif
  }
}
