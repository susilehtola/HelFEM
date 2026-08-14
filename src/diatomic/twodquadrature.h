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

#ifndef DIATOMIC_2DQUAD_H
#define DIATOMIC_2DQUAD_H

#include "basis.h"
#include "../general/model_potential.h"
#include "../sadatom/basis.h"

namespace helfem {
  namespace diatomic {
    namespace twodquad {
      /// Where to place probe?
      typedef enum {
                    // Probe left atom
                    PROBE_LEFT,
                    // Probe midbond
                    PROBE_MIDDLE,
                    // Probe right atom
                    PROBE_RIGHT
      } probe_t;

      /// Worker class
      class TwoDGridWorker {
      protected:
        /// Basis set
        const helfem::diatomic::basis::TwoDBasis *basp;

        /// Angular grid
        helfem::Vector cth, wang;
        /// Radial grid
        helfem::Vector r;
        /// Radial weight
        helfem::Vector wrad;
        /// Total quadrature weight
        helfem::Vector wtot;

        /// Value of m
        int m;
        /// List of basis functions in element
        std::vector<Eigen::Index> bf_ind;
        /// Values of important functions in grid points, Nbf * Ngrid
        helfem::Matrix bf;

        /// Value of integrand, Ngrid
        helfem::Matrix itg;

      public:
        /// Dummy constructor
        TwoDGridWorker();
        /// Constructor
        TwoDGridWorker(const helfem::diatomic::basis::TwoDBasis * basp, int lang);
        /// Destructor
        ~TwoDGridWorker();

        /// Compute basis functions on grid points
        void compute_bf(size_t iel, size_t irad, int m);

        /// Compute model potential
        void model_potential(const modelpotential::ModelPotential * p1, const modelpotential::ModelPotential * p2);

        /// Compute AO projection
        void ao_projection(const std::function<helfem::Vector(double r)> & compute_ao, probe_t p);
        /// Compute GTO projection
        void gto(int l, const helfem::Vector & expn, probe_t p);
        /// Compute STO projection
        void sto(int l, const helfem::Vector & expn, probe_t p);
        /// Multiply in the Legendre polynomial
        void multiply_Plm(int l, int m, probe_t p);

        /// Evaluate potential energy matrix elements
        void eval_pot(helfem::Matrix & V) const;
        /// Evaluate basis set projection
        void eval_proj(helfem::Matrix & S) const;
        /// Evaluate projection's overlap
        void eval_proj_overlap(helfem::Matrix & S) const;
      };

      /// Wrapper routine
      class TwoDGrid {
      private:
        /// Pointer to basis set
        const helfem::diatomic::basis::TwoDBasis * basp;
        /// Angular rule
        int lang;

        /// Left-hand and right-hand atomic basis sets
        sadatom::basis::TwoDBasis lh_basis, rh_basis;
        /// Left-hand and right-hand atomic orbitals
        helfem::Cube lh_orbs, rh_orbs;
        /// Occupations
        Eigen::VectorXi lh_occs, rh_occs;

      public:
        /// Dummy constructor
        TwoDGrid();
        /// Constructor
        TwoDGrid(const helfem::diatomic::basis::TwoDBasis * basp, int lang);
        /// Destructor
        ~TwoDGrid();

        /// Compute model potential matrix
        helfem::Matrix model_potential(const modelpotential::ModelPotential * p1, const modelpotential::ModelPotential * p2);

        /// Compute GTO projection
        /// Project the tabulated atomic orbitals of (Z, l) onto the
        /// diatomic basis, for the atom sitting at probe p. Rows are the
        /// stored orbitals of that l, columns the diatomic basis
        /// functions -- the same layout gto_projection returns.
        ///
        /// This is the GTO/STO projection with the trial function
        /// replaced by the database's radial orbitals, which is all a
        /// projected initial guess needs: ao_projection already takes an
        /// arbitrary radial function.
        helfem::Matrix atomdb_projection(int Z, int l, int m, probe_t p);
        /// Overlap of those same tabulated orbitals with each other on
        /// this grid, for normalizing the projection.
        helfem::Matrix atomdb_overlap(int Z, int l, int m, probe_t p);

        helfem::Matrix gto_projection(int l, int m, const helfem::Vector & expn, probe_t p);
        /// Compute GTO projection
        helfem::Matrix gto_overlap(int l, int m, const helfem::Vector & expn, probe_t p);
        /// Compute STO projection
        helfem::Matrix sto_projection(int l, int m, const helfem::Vector & expn, probe_t p);

        /// AO projection <bf|AO> on a panel-graded quadrature that
        /// covers BOTH factors exactly: panels never cross a mu-element
        /// boundary (the basis is polynomial on every panel) and are
        /// bisected until the AO's radial and angular arguments are
        /// narrow on each panel it can reach (the AO is low-degree
        /// there). Spectrally accurate at every exponent, no rule
        /// switch. Layout matches the grid path: (nexp x Nbf).
        /// One panelisation sweep per probe serves every l and every
        /// exponent: the panels, radial rows and shell harmonics are
        /// l-independent, so computing the whole l-range at once is
        /// what makes the tool's scan affordable. Returns one
        /// (nexp x Nbf) matrix per l in [lmin, lmax_ao].
        ///
        /// Values below Y ~ 1e-14 in the derived profiles are noise
        /// limited (that is eps^2 of the plateau) for ANY double
        /// precision rule; this diagnostic makes no claims there.
        std::vector<helfem::Matrix> graded_projections(int lmin, int lmax_ao, int m,
                                                       const helfem::Vector & expn,
                                                       probe_t p, bool sto_probe) const;
        /// Compute STO overlap
        helfem::Matrix sto_overlap(int l, int m, const helfem::Vector & expn, probe_t p);

        /// Compute atomic orbital projection
        helfem::Matrix atomic_projection(int l, int m, probe_t p);

        /// Compute atoms
        void compute_atoms(int Zl, int Zr);
      };
    }
  }
}

#endif
