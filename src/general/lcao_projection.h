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
#ifndef LCAO_PROJECTION_H
#define LCAO_PROJECTION_H

// Projection of analytic radial AOs (GTO / STO) onto a converged finite
// element solution. lcao.h owns the evaluators; this owns the projection
// half, and is kept separate so lcao.h stays free of the radial-basis
// headers that only the projection needs.

#include "lcao.h"
#include <RadialBasis.h>
#include <functional>

namespace helfem {
  namespace lcao {

    /// Radial part R(r) of a trial AO of angular momentum l and exponent
    /// ex. helfem::lcao::radial_GTO and radial_STO have this shape.
    using RadialAO = std::function<double(double r, int l, double ex)>;

    /// Box radius beyond which a GTO of this exponent is negligible:
    /// alpha*rmax^2 = 50 puts the Gaussian factor below 1e-21, which the
    /// r^(l+1) prefactor cannot bring back to significance for any sane l.
    double gto_rmax(double alpha);
    /// Same for an STO: zeta*rmax = 60.
    double sto_rmax(double zeta);

    /// Function-specific FE expansion of a trial radial AO.
    ///
    /// The AO is expanded as u(r) = r*R(r) on its OWN exponent-scaled
    /// radial grid: LIP coefficients are nodal values, so the expansion
    /// needs no quadrature, and on the AO's own scale the interpolation
    /// is spectrally accurate. Every integral against a solution basis
    /// then goes through the auto-converging cross-basis overlap, which
    /// quadratures each element-pair intersection separately, so neither
    /// basis's rule has to resolve the other's scale.
    ///
    /// Evaluating the AO at the SOLUTION basis's quadrature nodes instead
    /// misintegrates both ends of an exponent sweep: a tight AO
    /// (alpha ~ 1e10, support ~1e-5 au) lives entirely between the first
    /// element's quadrature nodes, and a diffuse one extends past the last
    /// element boundary, which also breaks its normalization.
    struct AOBasis {
      /// Exponent-scaled radial basis for the trial AO
      helfem::atomic::basis::FEMRadialBasis rad;
      /// Radius of each basis function's interpolation node
      helfem::Vector rnode;
      /// Overlap in this basis, used to normalize the expansion
      helfem::Matrix Sao;
    };

    /// Build an AO basis spanning [0, rmax].
    ///
    /// primbas MUST be 4 (LIP). The coefficient construction in
    /// ao_coefficients() reads the AO at the interpolation nodes and uses
    /// those values directly as expansion coefficients, which is valid
    /// only because LIP is cardinal at its nodes. A HIP basis carries
    /// function values AND derivatives as its degrees of freedom, and a
    /// Legendre basis is modal, so neither admits that construction
    /// without also supplying the derivative. The restriction is enforced
    /// rather than assumed: anything else throws.
    AOBasis make_ao_basis(double rmax, int primbas = 4, int nnodes = 15,
                          int nelem = 10);

    /// Nodal expansion coefficients of u_AO = r*R_AO(r), normalized so
    /// that c' Sao c = 1. The normalization is numerical rather than
    /// analytic so that interpolation and box-truncation error cannot
    /// push a completeness profile past 1.
    helfem::Vector ao_coefficients(const AOBasis &ao, int l, double ex,
                                   const RadialAO &eval_ao);

    /// Cross-overlap b_i = <u_i | u_AO> between the solution basis
    /// functions and one normalized trial AO.
    ///
    /// This is the primitive: everything below is a thin consumer of it.
    /// Sx is the cross-basis overlap solrad.overlap(ao.rad), which depends
    /// only on the two grids -- not on l, not on the evaluator -- so a
    /// caller sweeping l or driving an optimizer builds it once alongside
    /// the AO basis and passes it in.
    helfem::Vector project_ao(const AOBasis &ao, const helfem::Matrix &Sx,
                              int l, double ex, const RadialAO &eval_ao);

    /// How completely the solution basis reproduces the AO: the norm of
    /// its projection onto the orthonormalized FE space. 1 = fully
    /// represented.
    double completeness(const helfem::Matrix &Sinvh, const helfem::Vector &b);

    /// How important the exponent is for the converged density: the norm
    /// of the AO's projection onto the occupied orbitals. Cocc holds the
    /// occupied columns for the relevant angular momentum.
    double importance(const helfem::Matrix &Cocc, const helfem::Vector &b);

    /// Completeness profile Y(alpha, l) and importance profile I(alpha, l)
    /// over an exponent grid: a loop over project_ao.
    ///
    /// Column 0 is the exponent; columns 1..lmax+1 are the per-l profiles.
    /// Both profiles come from the same cross-basis overlap, which depends
    /// only on the grids, so they are computed together: one AO basis and
    /// one cross overlap per exponent, shared by every l channel and both
    /// profiles.
    ///
    /// C[l] holds the solution orbitals of angular momentum l and occs(l)
    /// its electron count; the occupied column count follows as
    /// ceil(occs(l) / (2*(2l+1))).
    void ao_profiles(const helfem::atomic::basis::FEMRadialBasis &solrad,
                     const helfem::Matrix &Sinvh, const helfem::Cube &C,
                     const Eigen::VectorXi &occs, int lmax,
                     const helfem::Vector &expn, const RadialAO &eval_ao,
                     const std::function<double(double ex)> &ao_rmax,
                     helfem::Matrix &completeness_out,
                     helfem::Matrix &importance_out);

  } // namespace lcao
} // namespace helfem

#endif
