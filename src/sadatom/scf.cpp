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
#include "../general/scf_driver_common.h"

#include "openorbitaloptimizer/scfsolver.hpp"

#include <Eigen/Eigenvalues>

namespace helfem {
  namespace sadatom {
    namespace scf {

      /// Occupation below which a saved orbital is treated as
      /// unoccupied and dropped from the checkpoint.
      ///
      /// Zero but for exact zeros: the checkpoint keeps everything the
      /// solver produced, and any decision about what is small enough to
      /// throw away belongs to whoever consumes the record. Only
      /// orbitals the SCF itself reports as exactly empty are skipped,
      /// and those carry no charge by definition.
      ///
      /// This used to be 1e-6, which cost the tabulated atoms their
      /// neutrality: the discarded occupation is precisely the effective
      /// charge the SAP potential is left with at large r, so a
      /// millionth of an electron dropped here became a spurious
      /// -1e-6/r tail on a neutral atom. tools/gen_atomdb.py now sets
      /// its own cutoff and refuses to emit a database that is not
      /// neutral, so the truncation is visible and checked where it is
      /// actually made.
      static const double occ_save_threshold = 0.0;

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

        // Shared implementation of the Fock build. `log_line`, when
        // non-null, collects the energy breakdown instead of it going
        // straight to stdout: a batched build evaluates its entries
        // concurrently, and interleaved printf from several threads
        // would be unreadable and out of order. Everything else here is
        // either a local or a const capture, so the body is safe to run
        // from several threads at once.
        auto build_fock = [&](const OpenOrbitalOptimizer::DensityMatrix<OOO_Real, OOO_Real> & dm,
                              std::string * log_line,
                              helfem::scf_driver::FockTimer::Components * tc) {
          const auto & orbitals    = dm.first;
          const auto & occupations = dm.second;
          // Component timings go into the caller's own Components, never
          // into shared state: a batched build runs these concurrently.
          Timer tbuild, tcomp;

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
            if (tc) tc->density += tcomp.get();
            if (have_xc) {
              tcomp.set();
              grid.eval_Fxc(opts.x_func, opts.x_pars, opts.c_func, opts.c_pars,
                             divided_cube(Pal, angfac), XCa, Exc, nelnum, opts.dftthr);
              for (size_t l = 0; l < XCa.size(); ++l) XCa[l] /= angfac;
              if (tc) tc->xc += tcomp.get();
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
            if (tc) tc->density += tcomp.get();
            if (have_xc) {
              tcomp.set();
              grid.eval_Fxc(opts.x_func, opts.x_pars, opts.c_func, opts.c_pars,
                             divided_cube(Pal, angfac), divided_cube(Pbl, angfac), XCa, XCb,
                             Exc, nelnum, opts.nelb > 0, opts.dftthr);
              for (size_t l = 0; l < XCa.size(); ++l) XCa[l] /= angfac;
              if (opts.nelb > 0)
                for (size_t l = 0; l < XCb.size(); ++l) XCb[l] /= angfac;
              if (tc) tc->xc += tcomp.get();
            }
          }

          const double Enuc = (Prad * Vnuc).trace();
          const double Econf = have_conf ? (Prad * Vconf).trace() : 0.0;
          tcomp.set();
          const helfem::Matrix J = basis.coulomb(Prad / angfac);
          const double Ecoul = 0.5 * (Prad * J).trace();
          if (tc) tc->coulomb += tcomp.get();

          helfem::Cube Ka, Kb;
          double Exx = 0.0;
          tcomp.set();
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

          if (tc) tc->exchange += tcomp.get();

          const double Etot = Ekin + Enuc + Econf + Ecoul + Exc + Exx;

          if (opts.verbosity > 0) {
            std::ostringstream line;
            line << std::fixed << std::setprecision(10)
                 << "Ekin " << Ekin << "  Enuc " << Enuc << "  Ecoul " << Ecoul
                 << "  Exc " << Exc << "  Exx " << Exx;
            if (have_conf) line << "  Econf " << Econf;
            line << "  Etot " << Etot << "\n";
            if (log_line) {
              *log_line = line.str();
            } else {
              fputs(line.str().c_str(), stdout);
              fflush(stdout);
            }
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
          if (tc) tc->total = tbuild.get();
          return std::make_pair(Etot, fock);
        };

        // Per-component wall clock for the Fock build, plus the time
        // spent outside it in the SCF solver.
        helfem::scf_driver::FockTimer ftimer;

        OpenOrbitalOptimizer::FockBuilder<OOO_Real, OOO_Real> fock_builder =
            [&](const OpenOrbitalOptimizer::DensityMatrix<OOO_Real, OOO_Real> & dm) {
          ftimer.enter();
          helfem::scf_driver::FockTimer::Components tc;
          auto ret = build_fock(dm, nullptr, &tc);
          ftimer.add_build(tc);
          if (opts.verbosity >= 5) ftimer.print_build(have_xc, have_exx);
          ftimer.leave();
          return ret;
        };

        // Batched Fock build. The solver hands us every density it wants
        // evaluated at once -- the ODA polytope's axis vertices, which
        // share a set of orbitals and differ only in their occupations --
        // and they are independent, so evaluate them concurrently.
        //
        // Going parallel across the batch rather than inside each build
        // is deliberate. The per-build parallelism is over radial
        // elements and over output angular momenta, and it scales badly:
        // measured on atom-in-jellium, six threads bought 1.15x. The
        // batch entries are perfectly independent, so this is the level
        // that actually has parallelism in it. Nested OpenMP is off by
        // default, so the inner regions collapse to serial and no
        // oversubscription results.
        OpenOrbitalOptimizer::BatchedFockBuilder<OOO_Real, OOO_Real> batched_fock_builder =
            [&](const std::vector<OpenOrbitalOptimizer::DensityMatrix<OOO_Real, OOO_Real>> & dms) {
          ftimer.enter();
          const size_t n = dms.size();
          std::vector<OpenOrbitalOptimizer::FockBuilderReturn<OOO_Real, OOO_Real>> out(n);
          std::vector<std::string> lines(n);
          // One Components per entry, exactly as for the log lines: the
          // entries below run concurrently, so nothing shared may be
          // written from inside the loop.
          std::vector<helfem::scf_driver::FockTimer::Components> comps(n);

          // Fill any lazily-built caches while still serial: the builds
          // below only read them, but filling concurrently would race.
          if (have_xc)
            grid.prime_quadrature_cache(opts.x_func, opts.c_func);

          if (n > 1) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
            for (long i = 0; i < (long) n; ++i)
              out[i] = build_fock(dms[i], &lines[i], &comps[i]);
          } else {
            for (size_t i = 0; i < n; ++i)
              out[i] = build_fock(dms[i], &lines[i], &comps[i]);
          }
          // Serial again: safe to fold the per-entry timings in.
          for (const auto & c : comps) ftimer.add_build(c);

          // Report in batch order, so the log reads the same whether or
          // not the entries were evaluated concurrently.
          for (const auto & line : lines)
            if (line.size()) fputs(line.c_str(), stdout);
          if (opts.verbosity >= 5) ftimer.print_build(have_xc, have_exx);
          fflush(stdout);
          ftimer.leave();
          return out;
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
          case 4: model = new modelpotential::SAPFEAtom(opts.Z);  break;
          default: throw std::logic_error("Unsupported iguess value (expected 0..4).\n");
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
        scfsolver.set_batched_fock_builder(batched_fock_builder);
        scfsolver.set("verbosity", opts.verbosity);
        scfsolver.set("maximum_iterations", opts.maxiter);
        scfsolver.set("convergence_threshold", opts.convthr);

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
              polynomial_basis::make_basis(old_primbas, old_nnodes));
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
            // A density is contravariant: P~ = Sinvh^-1 P Sinvh^-T, and
            // Sinvh^T S Sinvh = I gives Sinvh^-1 = Sinvh^T S. Without
            // the S factors this is the Fock rule, which returns
            // S^-1 P S^-1 -- see the same fix in
            // scf_driver::fill_block_from_density.
            const helfem::Matrix SC    = S_new * Sinvh_full;
            const helfem::Matrix Porth = SC.transpose() * Pl_new * SC;
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
        scfsolver.set("methods", opts.scf_methods);
        scfsolver.print_citation();
        scfsolver.run();
        if (opts.verbosity >= 5) ftimer.print_summary(have_xc, have_exx);

        // Extract results. Convert OOO's per-block orbital matrices
        // (in the Sinvh-orthonormal basis) back to AO coefficients
        // (Nbf, Nbf) per l, matching the arma::cube layout the
        // bespoke sadatom SCFSolver used to hand back.
        const auto orbitals    = scfsolver.get_orbitals();
        const auto occupations = scfsolver.get_orbital_occupations();

        AtomicSCFResult result;
        result.basis = basis;

        auto extract_channel = [&](size_t t, helfem::Cube & orbs_out, Eigen::VectorXi & occs_out,
                                   std::vector<helfem::Vector> & occs_orb_out) {
          orbs_out.assign(nblock, helfem::Matrix::Zero(Nrad, Nrad));
          occs_out = Eigen::VectorXi::Zero(nblock);
          occs_orb_out.assign(nblock, helfem::Vector());
          for (size_t l = 0; l < nblock; ++l) {
            const helfem::Matrix C_ao = Sinvh * orbitals[t * nblock + l];
            orbs_out[l] = C_ao;
            // Keep OOO's per-orbital occupations verbatim: they are what
            // was converged, and they are fractional whenever OOO
            // optimizes the occupations.
            occs_orb_out[l] = occupations[t * nblock + l];
            // Rounded per-l total, for the checkpoint and the consumers
            // that want an integer electron count per channel.
            occs_out(l) = static_cast<int>(std::round(occs_orb_out[l].sum()));
          }
        };
        extract_channel(0, result.orbs_a, result.occs_a, result.occs_orb_a);
        if (!restricted)
          extract_channel(1, result.orbs_b, result.occs_b, result.occs_orb_b);

        // Rebuild the converged per-l radial density cube(s) from the
        // final orbitals + integer per-l occupations (Aufbau filling,
        // consistent with the converged ground state and with the
        // checkpoint written below). Used both for --save and for the
        // gensap effective-potential / SAP-table output in main.cpp.
        auto build_cube = [&](const helfem::Cube & orbs_ao,
                               const std::vector<helfem::Vector> & occs_per_orb,
                               helfem::Cube & Pcube_out) {
          Pcube_out.assign(nblock, helfem::Matrix::Zero(Nrad, Nrad));
          for (size_t l = 0; l < nblock; ++l) {
            // Use the converged per-orbital occupations directly. They
            // already encode the Aufbau filling in the integer case, and
            // unlike a re-derived filling they stay correct when OOO has
            // optimized fractional occupations.
            const helfem::Vector & occ_vec = occs_per_orb[l];
            if (occ_vec.size() == 0 || occ_vec.cwiseAbs().maxCoeff() == 0.0) continue;
            const helfem::Matrix & orb_l = orbs_ao[l];
            Pcube_out[l] = orb_l * occ_vec.asDiagonal() * orb_l.transpose();
          }
        };

        build_cube(result.orbs_a, result.occs_orb_a, result.Pl_a);
        helfem::Matrix Prad_tot = helfem::Matrix::Zero(Nrad, Nrad);
        for (size_t l = 0; l < nblock; ++l)
          Prad_tot += result.Pl_a[l];
        if (!restricted) {
          build_cube(result.orbs_b, result.occs_orb_b, result.Pl_b);
          for (size_t l = 0; l < nblock; ++l)
            Prad_tot += result.Pl_b[l];
        }
        result.Prad = Prad_tot;

        // --save path: write basis-defining params + per-l AO density
        // cube(s) + per-l electron counts. Rebuilding a matching basis
        // needs (Z, lmax, bval); the density cube is used by --load.
        //
        // The WAVE FUNCTIONS are written alongside the densities: the
        // occupied AO orbital coefficients per l, plus their exact
        // (possibly fractional) occupations. A density is a lossy
        // record -- it cannot be taken apart into orbitals again -- so
        // the orbitals are what a downstream consumer needs to rebuild
        // the state, project it into another basis, or reconstruct the
        // effective potential without tabulating and interpolating it.
        if (opts.save_file.size()) {
          Checkpoint savechk(opts.save_file, /*writemode=*/true);
          savechk.write("sadatom_Z",       opts.Z);
          savechk.write("sadatom_lmax",    opts.lmax);
          savechk.write("sadatom_primbas", opts.poly->id());
          savechk.write("sadatom_nnodes",  opts.poly->nnodes());
          savechk.write("sadatom_Nquad",   opts.Nquad);
          savechk.write("sadatom_bval", opts.bval);

          for (size_t l = 0; l < nblock; ++l)
            savechk.write(std::string("sadatom_Pal_") + std::to_string(l),
                          result.Pl_a[l]);
          // Occupied orbitals + their exact occupations, per l. Columns
          // whose occupation is numerically zero are dropped: they are
          // the unoccupied remainder of the block and carry no
          // information about the state.
          auto write_orbitals = [&](const helfem::Cube & orbs,
                                     const std::vector<helfem::Vector> & occs,
                                     const char * ctag) {
            for (size_t l = 0; l < nblock; ++l) {
              const helfem::Vector & occ_l = occs[l];
              std::vector<Eigen::Index> keep;
              for (Eigen::Index i = 0; i < occ_l.size(); ++i)
                if (std::abs(occ_l(i)) > occ_save_threshold) keep.push_back(i);

              helfem::Matrix C(orbs[l].rows(), (Eigen::Index) keep.size());
              helfem::Matrix o((Eigen::Index) keep.size(), 1);
              for (size_t k = 0; k < keep.size(); ++k) {
                C.col((Eigen::Index) k) = orbs[l].col(keep[k]);
                o((Eigen::Index) k, 0)  = occ_l(keep[k]);
              }
              savechk.write(std::string("sadatom_C") + ctag + "l_" + std::to_string(l), C);
              savechk.write(std::string("sadatom_occ") + ctag + "l_" + std::to_string(l), o);
            }
          };
          write_orbitals(result.orbs_a, result.occs_orb_a, "a");
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
            write_orbitals(result.orbs_b, result.occs_orb_b, "b");
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
