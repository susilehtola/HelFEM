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

#include "scf.h"
#include "dftgrid.h"
#include "../general/dftfuncs.h"
#include "../general/scf_helpers.h"

#include "openorbitaloptimizer/scfsolver.hpp"

#include <ArmaEigen.h>
#include <Eigen/Eigenvalues>

namespace helfem {
  namespace sadatom {
    namespace scf {

      AtomicSCFResult run_atomic_scf(const AtomicSCFOptions & opts) {
        using OOO_Real = double;

        const int lmax   = opts.lmax;
        const int Ntot   = opts.nela + opts.nelb;
        const bool restricted = opts.restricted;
        // Restricted here is "spherically averaged, spin-averaged"
        // sadatom: the SCF sees a single density channel and Ntot may
        // be odd. nela != nelb is allowed -- the SCF only ever
        // consults Ntot for restricted runs.

        double kfrac, kshort, omega;
        range_separation(opts.x_func, omega, kfrac, kshort);
        const bool have_exx = (kfrac != 0.0 || kshort != 0.0);
        const bool have_xc  = (opts.x_func != 0 || opts.c_func != 0);
        bool rs_erfc = false, rs_yukawa = false;
        if (kshort != 0.0)
          is_range_separated(opts.x_func, rs_erfc, rs_yukawa);

        sadatom::basis::TwoDBasis basis(opts.Z, opts.finitenuc, opts.Rrms,
                                         opts.poly, opts.zeroder, opts.Nquad,
                                         opts.bval, lmax);

        const helfem::Matrix S    = basis.overlap();
        const helfem::Matrix Sinvh = basis.Sinvh();
        const helfem::Matrix T    = basis.kinetic();
        const helfem::Matrix Tl   = basis.kinetic_l();
        const helfem::Matrix Vnuc = basis.nuclear();

        helfem::Matrix Vconf = helfem::Matrix::Zero(basis.Nbf(), basis.Nbf());
        if (opts.iconf) {
          Vconf = basis.confinement(opts.conf_N, opts.conf_R, opts.iconf,
                                     opts.conf_barrier, opts.shift_conf);
        }
        const bool have_conf = (opts.iconf != 0);

        auto grid = helfem::sadatom::dftgrid::DFTGrid(&basis);
        basis.compute_tei();
        if (rs_yukawa) basis.compute_yukawa(omega);
        else if (rs_erfc) basis.compute_erfc(omega);

        const size_t nblock = static_cast<size_t>(lmax + 1);
        const size_t nparttype = restricted ? 1 : 2;
        OpenOrbitalOptimizer::IndexVector number_of_blocks_per_particle_type(nparttype);
        Eigen::Matrix<OOO_Real, Eigen::Dynamic, 1> maximum_occupation(nblock * nparttype);
        Eigen::Matrix<OOO_Real, Eigen::Dynamic, 1> number_of_particles(nparttype);
        std::vector<std::string> block_descriptions(nblock * nparttype);

        for (size_t t = 0; t < nparttype; ++t) {
          number_of_blocks_per_particle_type(t) = static_cast<int>(nblock);
          number_of_particles(t) = static_cast<OOO_Real>(
              restricted ? Ntot : (t == 0 ? opts.nela : opts.nelb));
          for (size_t l = 0; l < nblock; ++l) {
            maximum_occupation(t * nblock + l) =
                restricted ? 2 * (2 * l + 1) : (2 * l + 1);
            std::ostringstream oss;
            if (nparttype == 2) oss << (t == 0 ? "a:" : "b:");
            oss << "l=" << l;
            block_descriptions[t * nblock + l] = oss.str();
          }
        }

        const Eigen::Index Nrad = Sinvh.rows();
        const double angfac = 4.0 * M_PI;

        auto accumulate_density = [&](helfem::Matrix & Prad, arma::cube & Pl_cube,
                                       size_t l, const helfem::Matrix & orb,
                                       const helfem::Vector & occ, double & Ekin_out) {
          if (occ.cwiseAbs().maxCoeff() == 0.0) return;
          const helfem::Matrix C = Sinvh * orb;
          const helfem::Matrix P_l = C * occ.asDiagonal() * C.transpose();
          Prad += P_l;
          Pl_cube.slice(l) = helfem::to_arma(P_l);
          Ekin_out += (P_l * T).trace();
          if (l > 0)
            Ekin_out += l * (l + 1) * (P_l * Tl).trace();
        };

        OpenOrbitalOptimizer::FockBuilder<OOO_Real, OOO_Real> fock_builder =
            [&](const OpenOrbitalOptimizer::DensityMatrix<OOO_Real, OOO_Real> & dm) {
          const auto & orbitals    = dm.first;
          const auto & occupations = dm.second;

          OpenOrbitalOptimizer::FockMatrix<OOO_Real> fock(nblock * nparttype);
          helfem::Matrix Prad = helfem::Matrix::Zero(Nrad, Nrad);
          double Ekin = 0.0;
          double Exc = 0.0;
          double nelnum = 0.0;
          arma::cube XCa, XCb;
          arma::cube Pal(Nrad, Nrad, nblock, arma::fill::zeros);
          arma::cube Pbl;

          if (restricted) {
            for (size_t l = 0; l < nblock; ++l)
              accumulate_density(Prad, Pal, l, orbitals[l], occupations[l], Ekin);
            if (have_xc) {
              grid.eval_Fxc(opts.x_func, opts.x_pars, opts.c_func, opts.c_pars,
                             Pal / angfac, XCa, Exc, nelnum, opts.dftthr);
              XCa /= angfac;
            }
          } else {
            Pbl.zeros(Nrad, Nrad, nblock);
            helfem::Matrix Prad_a = helfem::Matrix::Zero(Nrad, Nrad);
            helfem::Matrix Prad_b = helfem::Matrix::Zero(Nrad, Nrad);
            for (size_t l = 0; l < nblock; ++l) {
              accumulate_density(Prad_a, Pal, l, orbitals[l], occupations[l], Ekin);
              accumulate_density(Prad_b, Pbl, l, orbitals[nblock + l], occupations[nblock + l], Ekin);
            }
            Prad = Prad_a + Prad_b;
            if (have_xc) {
              grid.eval_Fxc(opts.x_func, opts.x_pars, opts.c_func, opts.c_pars,
                             Pal / angfac, Pbl / angfac, XCa, XCb,
                             Exc, nelnum, opts.nelb > 0, opts.dftthr);
              XCa /= angfac;
              if (opts.nelb > 0) XCb /= angfac;
            }
          }

          const double Enuc = (Prad * Vnuc).trace();
          const double Econf = have_conf ? (Prad * Vconf).trace() : 0.0;
          const helfem::Matrix J = basis.coulomb(Prad / angfac);
          const double Ecoul = 0.5 * (Prad * J).trace();

          arma::cube Ka, Kb;
          double Exx = 0.0;
          if (have_exx) {
            arma::cube ang_a = Pal;
            for (size_t l = 0; l < nblock; ++l)
              ang_a.slice(l) /= restricted ? 2.0 * (2 * l + 1) : (2 * l + 1);
            Ka.zeros(Nrad, Nrad, nblock);
            if (kfrac  != 0.0) Ka += kfrac  * basis.exchange(ang_a);
            if (kshort != 0.0) Ka += kshort * basis.rs_exchange(ang_a);
            for (size_t l = 0; l < nblock; ++l)
              Exx += 0.5 * arma::trace(Ka.slice(l) * Pal.slice(l));
            if (!restricted) {
              arma::cube ang_b = Pbl;
              for (size_t l = 0; l < nblock; ++l)
                ang_b.slice(l) /= (2 * l + 1);
              Kb.zeros(Nrad, Nrad, nblock);
              if (kfrac  != 0.0) Kb += kfrac  * basis.exchange(ang_b);
              if (kshort != 0.0) Kb += kshort * basis.rs_exchange(ang_b);
              for (size_t l = 0; l < nblock; ++l)
                Exx += 0.5 * arma::trace(Kb.slice(l) * Pbl.slice(l));
            }
          }

          const double Etot = Ekin + Enuc + Econf + Ecoul + Exc + Exx;

          if (opts.verbosity > 0) {
            printf("Ekin %.10f  Enuc %.10f  Ecoul %.10f  Exc %.10f  Exx %.10f",
                    Ekin, Enuc, Ecoul, Exc, Exx);
            if (have_conf) printf("  Econf %.10f", Econf);
            printf("  Etot %.10f\n", Etot);
            fflush(stdout);
          }

          auto build_fock_block = [&](size_t l, const arma::cube & XC_cube,
                                       bool add_xc, const arma::cube & K_cube,
                                       bool add_k) -> helfem::Matrix {
            helfem::Matrix Fl = T + Vnuc + J;
            if (have_conf) Fl += Vconf;
            if (l > 0) Fl += l * (l + 1) * Tl;
            if (add_xc)
              Fl += helfem::to_eigen(arma::mat(XC_cube.slice(l)));
            if (add_k)
              Fl += helfem::to_eigen(arma::mat(K_cube.slice(l)));
            return Sinvh.transpose() * Fl * Sinvh;
          };

          if (restricted) {
            for (size_t l = 0; l < nblock; ++l)
              fock[l] = build_fock_block(l, XCa, have_xc, Ka, have_exx);
          } else {
            for (size_t l = 0; l < nblock; ++l) {
              fock[l]          = build_fock_block(l, XCa, have_xc, Ka, have_exx);
              fock[nblock + l] = build_fock_block(l, XCb, have_xc && opts.nelb > 0,
                                                   Kb, have_exx && opts.nelb > 0);
            }
          }
          return std::make_pair(Etot, fock);
        };

        OpenOrbitalOptimizer::FockMatrix<OOO_Real> CoreH(nblock * nparttype);
        for (size_t t = 0; t < nparttype; ++t) {
          for (size_t l = 0; l < nblock; ++l) {
            helfem::Matrix Hl = T + Vnuc;
            if (have_conf) Hl += Vconf;
            if (l > 0) Hl += l * (l + 1) * Tl;
            CoreH[t * nblock + l] = Sinvh.transpose() * Hl * Sinvh;
          }
        }

        OpenOrbitalOptimizer::SCFSolver<OOO_Real, OOO_Real> scfsolver(
            number_of_blocks_per_particle_type, maximum_occupation,
            number_of_particles, fock_builder, block_descriptions);
        scfsolver.verbosity(opts.verbosity);

        // Frozen per-l occupation: hand OOO a per-block particle count
        // vector so Aufbau is bypassed. Same pattern as atomic_ooo /
        // diatomic_ooo --readocc; here the caller supplies the per-l
        // occupation directly rather than reading occs.dat.
        const bool freeze_a = static_cast<int>(opts.fixed_per_l_a.n_elem) == lmax + 1;
        const bool freeze_b = (!restricted) && static_cast<int>(opts.fixed_per_l_b.n_elem) == lmax + 1;
        if (freeze_a || freeze_b) {
          Eigen::Matrix<OOO_Real, Eigen::Dynamic, 1> fixed =
              Eigen::Matrix<OOO_Real, Eigen::Dynamic, 1>::Zero(nblock * nparttype);
          if (freeze_a)
            for (int l = 0; l <= lmax; ++l)
              fixed(l) = static_cast<OOO_Real>(opts.fixed_per_l_a(l));
          if (freeze_b)
            for (int l = 0; l <= lmax; ++l)
              fixed(nblock + l) = static_cast<OOO_Real>(opts.fixed_per_l_b(l));
          scfsolver.fixed_number_of_particles_per_block(fixed);
        }

        scfsolver.initialize_with_fock(CoreH);
        scfsolver.run();

        // Extract results. Convert OOO's per-block orbital matrices
        // (in the Sinvh-orthonormal basis) back to AO coefficients
        // (Nbf, Nbf) per l, matching the arma::cube layout the
        // bespoke sadatom SCFSolver used to hand back.
        const auto orbitals    = scfsolver.get_orbitals();
        const auto occupations = scfsolver.get_orbital_occupations();

        AtomicSCFResult result;
        result.basis = basis;

        auto extract_channel = [&](size_t t, arma::cube & orbs_out, arma::ivec & occs_out) {
          orbs_out.zeros(Nrad, Nrad, nblock);
          occs_out.zeros(nblock);
          for (size_t l = 0; l < nblock; ++l) {
            const helfem::Matrix C_ao = Sinvh * orbitals[t * nblock + l];
            orbs_out.slice(l) = helfem::to_arma(C_ao);
            // Round OOO's Tbase (double) per-orbital occupations up to
            // the nearest integer total per l. Aufbau on integer
            // electron counts yields integer occupations for the
            // occupied orbitals; sum them.
            int total = 0;
            for (Eigen::Index i = 0; i < occupations[t * nblock + l].size(); ++i)
              total += static_cast<int>(std::round(occupations[t * nblock + l](i)));
            occs_out(l) = total;
          }
        };
        extract_channel(0, result.orbs_a, result.occs_a);
        if (!restricted)
          extract_channel(1, result.orbs_b, result.occs_b);

        return result;
      }

    }
  }
}
