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
#ifndef HELFEM_DIATOMIC_CAS_ENGINE_H
#define HELFEM_DIATOMIC_CAS_ENGINE_H

#include "basis.h"
#include "../general/cas_integrals.h"

#include <stdexcept>

namespace helfem {
  namespace diatomic {
    namespace basis {

      /// Adapts the diatomic TwoDBasis to cas::JKProvider.
      ///
      /// As small as the atomic one, and for the same reason: the CAS
      /// machinery only ever asks a geometry for J, K and hcore, so all the
      /// algebra lives once in src/general/cas_integrals.cpp.
      ///
      /// The one thing this has to police that the atomic engine does not is
      /// TwoDBasis::absm_symmetric. With it set, exchange() builds only the
      /// m >= 0 half of K and mirrors the rest, which is exact for a density
      /// symmetric under m -> -m and wrong otherwise. A CAS forms PAIR
      /// densities C_t C_u^T between individual active orbitals, and one
      /// between the +m and -m partners of a pi shell carries dm = 2 -- it is
      /// not m-symmetric, so the mirrored half would be silently wrong. The
      /// flag is therefore refused rather than worked around.
      ///
      /// Note what this does NOT restrict. --symmetry only decides how the SCF
      /// blocks its orthonormalization and whether the driver sets that flag;
      /// the AO basis itself is identical for every --symmetry value, and
      /// coulomb() never consults the flag at all. So the CAS integrals are
      /// independent of --symmetry, and a symmetry=3 reference is usable here
      /// provided the flag is cleared first (see the class comment in
      /// cas_integrals.h for what symmetry=3 then means for the ACTIVE SPACE,
      /// which is a separate question from the integrals).
      class DiatomicCASEngine : public helfem::cas::JKProvider {
       public:
        explicit DiatomicCASEngine(const TwoDBasis & basis)
            : basis_(basis), hcore_(basis.kinetic() + basis.nuclear()) {
          if (basis_.is_absm_symmetric())
            throw std::logic_error(
                "DiatomicCASEngine: the basis has absm_symmetric set, which "
                "makes exchange() assume a density symmetric under m -> -m. A "
                "CAS forms pair densities between individual active orbitals, "
                "which are not; call set_absm_symmetric(false) first.\n");
        }

        helfem::Matrix coulomb(const helfem::Matrix & P) const override {
          return basis_.coulomb(P);
        }

        /// Passed straight through, exactly as in the atomic engine:
        /// TwoDBasis::exchange returns the signed contribution added to a spin
        /// channel's Fock matrix, which is cas::JKProvider's contract.
        helfem::Matrix exchange(const helfem::Matrix & P) const override {
          return basis_.exchange(P);
        }

        const helfem::Matrix & hcore() const override { return hcore_; }

       private:
        const TwoDBasis & basis_;
        helfem::Matrix hcore_;
      };

    } // namespace basis
  } // namespace diatomic
} // namespace helfem

#endif // HELFEM_DIATOMIC_CAS_ENGINE_H
