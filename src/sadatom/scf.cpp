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
#include "../general/checkpoint.h"

#include "openorbitaloptimizer/scfsolver.hpp"

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

        // Divide each slice of a cube by a scalar (helfem::Cube has no
        // whole-object arithmetic). Uses true element-wise division to
        // match arma's `cube / scalar` bit-for-bit (arma and Eigen both
        // compute val / k, not val * (1/k)).
        auto divided_cube = [](const helfem::Cube & C, double f) {
          helfem::Cube out(C.size());
          for (size_t l = 0; l < C.size(); ++l) out[l] = C[l] / f;
          return out;
        };

        auto accumulate_density = [&](helfem::Matrix & Prad, helfem::Cube & Pl_cube,
                                       size_t l, const helfem::Matrix & orb,
                                       const helfem::Vector & occ, double & Ekin_out) {
          if (occ.cwiseAbs().maxCoeff() == 0.0) return;
          const helfem::Matrix C = Sinvh * orb;
          const helfem::Matrix P_l = C * occ.asDiagonal() * C.transpose();
          Prad += P_l;
          Pl_cube[l] = P_l;
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
          helfem::Cube XCa, XCb;
          helfem::Cube Pal(nblock, helfem::Matrix::Zero(Nrad, Nrad));
          helfem::Cube Pbl;

          if (restricted) {
            for (size_t l = 0; l < nblock; ++l)
              accumulate_density(Prad, Pal, l, orbitals[l], occupations[l], Ekin);
            if (have_xc) {
              grid.eval_Fxc(opts.x_func, opts.x_pars, opts.c_func, opts.c_pars,
                             divided_cube(Pal, angfac), XCa, Exc, nelnum, opts.dftthr);
              for (size_t l = 0; l < XCa.size(); ++l) XCa[l] /= angfac;
            }
          } else {
            Pbl.assign(nblock, helfem::Matrix::Zero(Nrad, Nrad));
            helfem::Matrix Prad_a = helfem::Matrix::Zero(Nrad, Nrad);
            helfem::Matrix Prad_b = helfem::Matrix::Zero(Nrad, Nrad);
            for (size_t l = 0; l < nblock; ++l) {
              accumulate_density(Prad_a, Pal, l, orbitals[l], occupations[l], Ekin);
              accumulate_density(Prad_b, Pbl, l, orbitals[nblock + l], occupations[nblock + l], Ekin);
            }
            Prad = Prad_a + Prad_b;
            if (have_xc) {
              grid.eval_Fxc(opts.x_func, opts.x_pars, opts.c_func, opts.c_pars,
                             divided_cube(Pal, angfac), divided_cube(Pbl, angfac), XCa, XCb,
                             Exc, nelnum, opts.nelb > 0, opts.dftthr);
              for (size_t l = 0; l < XCa.size(); ++l) XCa[l] /= angfac;
              if (opts.nelb > 0)
                for (size_t l = 0; l < XCb.size(); ++l) XCb[l] /= angfac;
            }
          }

          const double Enuc = (Prad * Vnuc).trace();
          const double Econf = have_conf ? (Prad * Vconf).trace() : 0.0;
          const helfem::Matrix J = basis.coulomb(Prad / angfac);
          const double Ecoul = 0.5 * (Prad * J).trace();

          helfem::Cube Ka, Kb;
          double Exx = 0.0;
          if (have_exx) {
            helfem::Cube ang_a = Pal;
            for (size_t l = 0; l < nblock; ++l)
              ang_a[l] /= restricted ? 2.0 * (2 * l + 1) : (2 * l + 1);
            Ka.assign(nblock, helfem::Matrix::Zero(Nrad, Nrad));
            if (kfrac  != 0.0) {
              const helfem::Cube Kx = basis.exchange(ang_a);
              for (size_t l = 0; l < nblock; ++l) Ka[l] += kfrac * Kx[l];
            }
            if (kshort != 0.0) {
              const helfem::Cube Kx = basis.rs_exchange(ang_a);
              for (size_t l = 0; l < nblock; ++l) Ka[l] += kshort * Kx[l];
            }
            for (size_t l = 0; l < nblock; ++l)
              Exx += 0.5 * (Ka[l] * Pal[l]).trace();
            if (!restricted) {
              helfem::Cube ang_b = Pbl;
              for (size_t l = 0; l < nblock; ++l)
                ang_b[l] /= (2 * l + 1);
              Kb.assign(nblock, helfem::Matrix::Zero(Nrad, Nrad));
              if (kfrac  != 0.0) {
                const helfem::Cube Kx = basis.exchange(ang_b);
                for (size_t l = 0; l < nblock; ++l) Kb[l] += kfrac * Kx[l];
              }
              if (kshort != 0.0) {
                const helfem::Cube Kx = basis.rs_exchange(ang_b);
                for (size_t l = 0; l < nblock; ++l) Kb[l] += kshort * Kx[l];
              }
              for (size_t l = 0; l < nblock; ++l)
                Exx += 0.5 * (Kb[l] * Pbl[l]).trace();
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

          auto build_fock_block = [&](size_t l, const helfem::Cube & XC_cube,
                                       bool add_xc, const helfem::Cube & K_cube,
                                       bool add_k) -> helfem::Matrix {
            helfem::Matrix Fl = T + Vnuc + J;
            if (have_conf) Fl += Vconf;
            if (l > 0) Fl += l * (l + 1) * Tl;
            if (add_xc)
              Fl += XC_cube[l];
            if (add_k)
              Fl += K_cube[l];
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

        // Initial-guess electron-nuclear potential. iguess 0 uses the
        // bare nuclear attraction (core-Hamiltonian guess); 1/2/3 use a
        // GSZ / SAP / Thomas-Fermi screened-nucleus model potential,
        // which typically converges materially faster. Only the guess
        // Fock matrix is affected -- the SCF Fock build above always
        // uses the true Vnuc.
        helfem::Matrix Vguess = Vnuc;
        if (opts.iguess != 0) {
          modelpotential::ModelPotential * model = nullptr;
          switch (opts.iguess) {
          case 1: model = new modelpotential::GSZAtom(opts.Z); break;
          case 2: model = new modelpotential::SAPAtom(opts.Z); break;
          case 3: model = new modelpotential::TFAtom(opts.Z);  break;
          default: throw std::logic_error("Unsupported iguess value (expected 0..3).\n");
          }
          Vguess = basis.model_potential(model);
          delete model;
        }

        OpenOrbitalOptimizer::FockMatrix<OOO_Real> CoreH(nblock * nparttype);
        for (size_t t = 0; t < nparttype; ++t) {
          for (size_t l = 0; l < nblock; ++l) {
            helfem::Matrix Hl = T + Vguess;
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
        const bool freeze_a = static_cast<int>(opts.fixed_per_l_a.size()) == lmax + 1;
        const bool freeze_b = (!restricted) && static_cast<int>(opts.fixed_per_l_b.size()) == lmax + 1;
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

        // --load path: read old basis + per-l density cube(s), project
        // per-l density into the current basis via cross-basis radial
        // overlap, then feed OOO's initialize_with_orbitals with the
        // eigen-decomposition of each projected block.
        if (opts.load_file.size()) {
          Checkpoint loadchk(opts.load_file, /*writemode=*/false);
          // Rebuild the old sadatom basis from the stored parameters.
          // We only need Nrad_old and the FE matrices needed for the
          // per-l density projection, which is fully determined by
          // rebuilding the TwoDBasis object.
          int old_Z = 0, old_lmax = 0, old_primbas = 0, old_nnodes = 0, old_Nquad = 0;
          loadchk.read("sadatom_Z",       old_Z);
          loadchk.read("sadatom_lmax",    old_lmax);
          loadchk.read("sadatom_primbas", old_primbas);
          loadchk.read("sadatom_nnodes",  old_nnodes);
          loadchk.read("sadatom_Nquad",   old_Nquad);
          helfem::Vector old_bval;
          loadchk.read("sadatom_bval", old_bval);

          // Reconstruct the old polynomial basis so the reassembled
          // FE basis matches the checkpoint's Nbf exactly.
          std::shared_ptr<const polynomial_basis::PolynomialBasis> old_poly(
              polynomial_basis::get_basis(old_primbas, old_nnodes));
          sadatom::basis::TwoDBasis oldbasis(old_Z, modelpotential::POINT_NUCLEUS,
                                              0.0, old_poly, opts.zeroder,
                                              old_Nquad, old_bval, old_lmax);
          const helfem::Matrix S12  = basis.overlap(oldbasis);
          const helfem::Matrix Sinvh_full = basis.Sinvh();
          const helfem::Matrix Pproj = Sinvh_full * Sinvh_full.transpose() * S12;
          const helfem::Matrix S_new = basis.overlap();

          // Read per-l density slices back into a cube. Each slice is
          // stored on disk as sadatom_Pal_l (helfem::Matrix, Nrad_old^2).
          const int old_nblock_read = old_lmax + 1;
          helfem::Cube Pal_old;
          if (loadchk.exist("sadatom_Pal_0")) {
            helfem::Matrix slice0;
            loadchk.read("sadatom_Pal_0", slice0);
            Pal_old.assign(old_nblock_read, helfem::Matrix::Zero(slice0.rows(), slice0.cols()));
            Pal_old[0] = slice0;
            for (int l = 1; l < old_nblock_read; ++l) {
              const std::string key = std::string("sadatom_Pal_") + std::to_string(l);
              helfem::Matrix sl;
              if (loadchk.exist(key)) {
                loadchk.read(key, sl);
                Pal_old[l] = sl;
              }
            }
          }
          helfem::Cube Pbl_old;
          if (!restricted && loadchk.exist("sadatom_Pbl_0")) {
            helfem::Matrix slice0;
            loadchk.read("sadatom_Pbl_0", slice0);
            Pbl_old.assign(old_nblock_read, helfem::Matrix::Zero(slice0.rows(), slice0.cols()));
            Pbl_old[0] = slice0;
            for (int l = 1; l < old_nblock_read; ++l) {
              const std::string key = std::string("sadatom_Pbl_") + std::to_string(l);
              helfem::Matrix sl;
              if (loadchk.exist(key)) {
                loadchk.read(key, sl);
                Pbl_old[l] = sl;
              }
            }
          }

          OpenOrbitalOptimizer::Orbitals<OOO_Real>            loaded_orbs(nblock * nparttype);
          OpenOrbitalOptimizer::OrbitalOccupations<OOO_Real>  loaded_occs(nblock * nparttype);

          auto fill_l = [&](size_t base, size_t l, const helfem::Cube & Pcube,
                             double per_l_electrons, double max_occ) {
            helfem::Matrix Pl_new;
            if (static_cast<int>(l) <= old_lmax && Pcube.size() > l) {
              Pl_new = Pproj * Pcube[l] * Pproj.transpose();
              const double trace_now = (Pl_new * S_new).trace();
              if (trace_now > 0 && per_l_electrons > 0)
                Pl_new *= per_l_electrons / trace_now;
            } else {
              Pl_new = helfem::Matrix::Zero(Nrad, Nrad);
            }
            const helfem::Matrix Porth = Sinvh_full.transpose() * Pl_new * Sinvh_full;
            Eigen::SelfAdjointEigenSolver<helfem::Matrix> es(Porth);
            if (es.info() != Eigen::Success)
              throw std::logic_error("--load: eigendecomposition of projected l block density failed");
            const helfem::Vector occ_eigs = es.eigenvalues();     // ascending
            const helfem::Matrix vec_eigs = es.eigenvectors();
            const Eigen::Index n = vec_eigs.cols();
            helfem::Matrix V(vec_eigs.rows(), n);
            helfem::Vector w(n);
            for (Eigen::Index i = 0; i < n; ++i) {
              V.col(i) = vec_eigs.col(n - 1 - i);
              w(i)     = std::min(std::max(occ_eigs(n - 1 - i), 0.0), max_occ);
            }
            loaded_orbs[base + l] = V;
            loaded_occs[base + l] = w;
          };

          // Per-l electron counts to renormalise into. Read from
          // checkpoint if present, else fall back to trace-preserving
          // (no rescaling). Checkpoint stores integers as N x 1 matrices.
          Eigen::VectorXi per_l_a, per_l_b;
          if (loadchk.exist("sadatom_occs_a")) {
            Eigen::MatrixXi tmp;
            loadchk.read("sadatom_occs_a", tmp);
            per_l_a = tmp.col(0);
          }
          if (loadchk.exist("sadatom_occs_b")) {
            Eigen::MatrixXi tmp;
            loadchk.read("sadatom_occs_b", tmp);
            per_l_b = tmp.col(0);
          }

          for (size_t l = 0; l < nblock; ++l) {
            double per_l = (static_cast<int>(l) < per_l_a.size())
                             ? static_cast<double>(per_l_a(l)) : 0.0;
            double max_occ = restricted ? 2.0 * (2 * l + 1) : (2.0 * l + 1);
            fill_l(0, l, Pal_old, per_l, max_occ);
          }
          if (!restricted) {
            for (size_t l = 0; l < nblock; ++l) {
              double per_l = (static_cast<int>(l) < per_l_b.size())
                               ? static_cast<double>(per_l_b(l)) : 0.0;
              double max_occ = 2.0 * l + 1;
              fill_l(nblock, l, Pbl_old, per_l, max_occ);
            }
          }
          scfsolver.initialize_with_orbitals(loaded_orbs, loaded_occs);
        } else {
          scfsolver.initialize_with_fock(CoreH);
        }
        scfsolver.run();

        // Extract results. Convert OOO's per-block orbital matrices
        // (in the Sinvh-orthonormal basis) back to AO coefficients
        // (Nbf, Nbf) per l, matching the arma::cube layout the
        // bespoke sadatom SCFSolver used to hand back.
        const auto orbitals    = scfsolver.get_orbitals();
        const auto occupations = scfsolver.get_orbital_occupations();

        AtomicSCFResult result;
        result.basis = basis;

        auto extract_channel = [&](size_t t, helfem::Cube & orbs_out, Eigen::VectorXi & occs_out) {
          orbs_out.assign(nblock, helfem::Matrix::Zero(Nrad, Nrad));
          occs_out = Eigen::VectorXi::Zero(nblock);
          for (size_t l = 0; l < nblock; ++l) {
            const helfem::Matrix C_ao = Sinvh * orbitals[t * nblock + l];
            orbs_out[l] = C_ao;
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

        // Rebuild the converged per-l radial density cube(s) from the
        // final orbitals + integer per-l occupations (Aufbau filling,
        // consistent with the converged ground state and with the
        // checkpoint written below). Used both for --save and for the
        // gensap effective-potential / SAP-table output in main.cpp.
        auto build_cube = [&](const helfem::Cube & orbs_ao, const Eigen::VectorXi & occs_per_l,
                               helfem::Cube & Pcube_out) {
          Pcube_out.assign(nblock, helfem::Matrix::Zero(Nrad, Nrad));
          for (size_t l = 0; l < nblock; ++l) {
            if (occs_per_l(l) <= 0) continue;
            // For an integer per-l total N in a manifold of capacity
            // 2*(2l+1) restricted or (2l+1) unrestricted, split
            // evenly across the N/max_orb lowest orbitals.
            const helfem::Matrix & orb_l = orbs_ao[l];
            const int norb = orb_l.cols();
            const double per_orb = (restricted ? 2.0 * (2 * l + 1) : 2.0 * l + 1);
            // AO density with occupations set: pick occs from OOO
            // internal (they are what's converged). For simplicity
            // fall back to a diagonal-per-orbital occupation of
            // occs_per_l(l) / norb capped at per_orb.
            helfem::Vector occ_vec = helfem::Vector::Zero(norb);
            double remaining = static_cast<double>(occs_per_l(l));
            for (int i = 0; i < norb && remaining > 0; ++i) {
              const double take = std::min<double>(per_orb, remaining);
              occ_vec(i) = take;
              remaining -= take;
            }
            Pcube_out[l] = orb_l * occ_vec.asDiagonal() * orb_l.transpose();
          }
        };

        build_cube(result.orbs_a, result.occs_a, result.Pl_a);
        helfem::Matrix Prad_tot = helfem::Matrix::Zero(Nrad, Nrad);
        for (size_t l = 0; l < nblock; ++l)
          Prad_tot += result.Pl_a[l];
        if (!restricted) {
          build_cube(result.orbs_b, result.occs_b, result.Pl_b);
          for (size_t l = 0; l < nblock; ++l)
            Prad_tot += result.Pl_b[l];
        }
        result.Prad = Prad_tot;

        // --save path: write basis-defining params + per-l AO density
        // cube(s) + per-l electron counts. Rebuilding a matching basis
        // needs (Z, lmax, bval); the density cube is used by --load.
        if (opts.save_file.size()) {
          Checkpoint savechk(opts.save_file, /*writemode=*/true);
          savechk.write("sadatom_Z",       opts.Z);
          savechk.write("sadatom_lmax",    opts.lmax);
          savechk.write("sadatom_primbas", opts.poly->get_id());
          savechk.write("sadatom_nnodes",  opts.poly->get_nnodes());
          savechk.write("sadatom_Nquad",   opts.Nquad);
          savechk.write("sadatom_bval", opts.bval);

          for (size_t l = 0; l < nblock; ++l)
            savechk.write(std::string("sadatom_Pal_") + std::to_string(l),
                          result.Pl_a[l]);
          {
            // Checkpoint stores integers as N x 1 matrices.
            Eigen::MatrixXi oa(result.occs_a.size(), 1);
            for (Eigen::Index i = 0; i < result.occs_a.size(); ++i) oa(i, 0) = result.occs_a(i);
            savechk.write("sadatom_occs_a", oa);
          }
          if (!restricted) {
            for (size_t l = 0; l < nblock; ++l)
              savechk.write(std::string("sadatom_Pbl_") + std::to_string(l),
                            result.Pl_b[l]);
            Eigen::MatrixXi ob(result.occs_b.size(), 1);
            for (Eigen::Index i = 0; i < result.occs_b.size(); ++i) ob(i, 0) = result.occs_b(i);
            savechk.write("sadatom_occs_b", ob);
          }
          printf("Saved results to %s\n", opts.save_file.c_str());
        }

        return result;
      }

    }
  }
}
