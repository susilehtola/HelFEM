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
#include "orbital_table.h"
#include "../general/constants.h"

#include <cmath>
#include <cstdio>
#include <sstream>

namespace helfem {
  namespace sadatom {

    std::vector<helfem::Vector> orbital_energies(const helfem::Cube & orbitals_orth,
                                                 const helfem::Cube & fock_orth) {
      std::vector<helfem::Vector> E(orbitals_orth.size());
      for (size_t b = 0; b < orbitals_orth.size(); b++) {
        const helfem::Matrix & C = orbitals_orth[b];
        const helfem::Matrix Fmo = C.transpose() * fock_orth[b] * C;
        E[b] = Fmo.diagonal();
      }
      return E;
    }

    void print_orbital_table(const basis::TwoDBasis & basis,
                             const helfem::Cube & orbs_ao,
                             const std::vector<helfem::Vector> & occs,
                             const std::vector<helfem::Vector> & energies,
                             size_t bstart, size_t bend,
                             const std::string & label,
                             double occthr) {
      static const char shtype[] = "spdfgh";
      const std::vector< std::pair<int, helfem::Matrix> > rmat(basis.Rmatrices());

      printf("\n%s orbitals\n", label.c_str());
      printf("%3s %8s %16s", "nl", "occ", "E");
      for (size_t ir = 0; ir < rmat.size(); ir++) {
        std::ostringstream oss;
        oss << "<r>(" << rmat[ir].first << ")";
        printf(" %12s", oss.str().c_str());
      }
      printf(" %12s\n", "r(max)");

      for (size_t b = bstart; b < bend; b++) {
        const int l = (int) (b - bstart);
        const helfem::Vector & occ_l = occs[b];
        const helfem::Vector & E_l = energies[b];
        for (Eigen::Index io = 0; io < occ_l.size(); io++) {
          // Skip the virtuals: the table is about the occupied shells.
          if (std::abs(occ_l(io)) < occthr) continue;
          const helfem::Vector orb = orbs_ao[b].col(io);
          const helfem::Matrix P = orb * orb.transpose();
          const int n = (int) io + l + 1;
          const char ltag = (l < 6) ? shtype[l] : '?';
          printf("%2i%c % 8.4f % 16.9f", n, ltag, occ_l(io), E_l(io));
          for (size_t ir = 0; ir < rmat.size(); ir++)
            printf(" %12e", std::pow((P * rmat[ir].second).trace(), 1.0 / rmat[ir].first));
          printf(" %12e\n", basis.electron_density_maximum_radius(P));
        }
      }

      // Per-l gap between the lowest unoccupied and the highest occupied
      // orbital of the channel, as the bespoke sadatom solver printed
      // from OrbitalChannel::GetGap. With no occupied orbital in the
      // channel the gap is the orbital energy itself.
      printf("%s HOMO-LUMO gap (eV) per l:", label.c_str());
      for (size_t b = bstart; b < bend; b++) {
        const helfem::Vector & occ_l = occs[b];
        const helfem::Vector & E_l = energies[b];
        bool have = false;
        double gap = 0.0;
        for (Eigen::Index io = 0; io < occ_l.size() && io < E_l.size(); io++) {
          if (std::abs(occ_l(io)) >= occthr) continue;
          gap = (io == 0) ? E_l(io) : E_l(io) - E_l(io - 1);
          have = true;
          break;
        }
        if (have)
          printf("  %.4f", gap * HARTREEINEV);
        else
          printf("  %s", "n/a");
      }
      printf("\n");
    }

  } // namespace sadatom
} // namespace helfem
