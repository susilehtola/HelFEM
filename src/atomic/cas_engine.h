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
#ifndef HELFEM_ATOMIC_CAS_ENGINE_H
#define HELFEM_ATOMIC_CAS_ENGINE_H

#include "TwoDBasis.h"
#include "../general/cas_integrals.h"

namespace helfem {
  namespace atomic {
    namespace basis {

      /// Adapts the atomic TwoDBasis to cas::JKProvider.
      ///
      /// Everything the CAS machinery needs from a geometry is J, K and hcore,
      /// so this is the whole atomic side of it -- the integral algebra lives
      /// once in src/general/cas_integrals.cpp and is shared with diatomic.
      ///
      /// The basis must have had compute_tei() called on it; J and K are not
      /// available before that.
      class AtomicCASEngine : public helfem::cas::JKProvider {
       public:
        explicit AtomicCASEngine(const TwoDBasis & basis)
            : basis_(basis), hcore_(basis.kinetic() + basis.nuclear()) {}

        helfem::Matrix coulomb(const helfem::Matrix & P) const override {
          return basis_.coulomb(P);
        }

        /// Passed straight through: TwoDBasis::exchange already returns the
        /// signed contribution added to a spin channel's Fock matrix, which is
        /// exactly cas::JKProvider's contract.
        helfem::Matrix exchange(const helfem::Matrix & P) const override {
          return basis_.exchange(P);
        }

        const helfem::Matrix & hcore() const override { return hcore_; }

       private:
        const TwoDBasis & basis_;
        helfem::Matrix hcore_;
      };

    } // namespace basis
  } // namespace atomic
} // namespace helfem

#endif // HELFEM_ATOMIC_CAS_ENGINE_H
