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
#ifndef SADATOM_ORBITAL_TABLE_H
#define SADATOM_ORBITAL_TABLE_H

#include "basis.h"
#include <string>

namespace helfem {
  namespace sadatom {

    /// An orbital counts as occupied above this occupation. Matches the
    /// aij --savethr default, so the two drivers draw the same line.
    const double default_occupation_threshold = 1e-6;

    /// Orbital energies eps_i = <i|F|i> of a converged solution, one
    /// vector per block.
    ///
    /// Both arguments live in the orthonormal basis: orbitals_orth is
    /// what OpenOrbitalOptimizer hands back, fock_orth the Fock matrix
    /// built at the converged density.
    ///
    /// The diagonal form is deliberate. A converged first-order solution
    /// has orbitals that diagonalize the Fock matrix, so this is the
    /// usual eigenvalue; but the second-order phase returns orbitals
    /// that are stationary without being canonical, and there a fresh
    /// diagonalization produces a different set of vectors whose index
    /// no longer labels the same orbital as the occupation vector's.
    /// The diagonal stays aligned column-by-column with the occupations,
    /// and it is the Janak derivative dE/dn_i -- the quantity the
    /// occupation optimization is about.
    std::vector<helfem::Vector> orbital_energies(const helfem::Cube & orbitals_orth,
                                                 const helfem::Cube & fock_orth);

    /// Print the per-shell summary of one spin channel: occupation,
    /// orbital energy, the radial extents <r^n>^(1/n) for the n the
    /// basis tabulates, and the radius at which the shell density peaks,
    /// followed by the per-l gap between the lowest unoccupied and the
    /// highest occupied orbital of each l.
    ///
    /// The block range [bstart, bend) is one channel's worth of l
    /// blocks, so block b carries angular momentum l = b - bstart.
    /// orbs_ao holds the orbitals in the non-orthonormal (AO) basis;
    /// occs and energies are indexed by absolute block, like orbs_ao.
    void print_orbital_table(const basis::TwoDBasis & basis,
                             const helfem::Cube & orbs_ao,
                             const std::vector<helfem::Vector> & occs,
                             const std::vector<helfem::Vector> & energies,
                             size_t bstart, size_t bend,
                             const std::string & label,
                             double occthr = default_occupation_threshold);

  } // namespace sadatom
} // namespace helfem

#endif // SADATOM_ORBITAL_TABLE_H
