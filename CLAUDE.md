# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is HelFEM

Helsinki Finite Element Method suite for fully numerical Hartree-Fock and density-functional theory (DFT) calculations on atoms and diatomic molecules. Implements finite element basis sets with Lagrange (LIP) and Hermite (HIP) interpolating polynomials, supporting LDA/GGA/meta-GGA functionals via libxc.

## Building

Requires an out-of-source build. The quickest path:

```bash
./compile.sh        # builds into objdir/, installs to install/bin/
```

Manual build:

```bash
cmake -B objdir -DCMAKE_INSTALL_PREFIX=install
cmake --build objdir -j9
cmake --install objdir
```

All dependencies are located with `find_package` / `pkg-config` and consumed as
imported targets — there is **no** machine-specific config file to maintain. If a
dependency lives in a non-standard prefix, add it to `CMAKE_PREFIX_PATH`:

```bash
cmake -B objdir -DCMAKE_PREFIX_PATH="/opt/libxc;$(brew --prefix)"
```

**Key CMake options:**
- `HELFEM_BINARIES=OFF` — build only `libhelfem` (skips the HDF5 and libxc requirements)
- `HELFEM_EIGEN_BLAS` (default `ON`) — route Eigen's dense products (GEMM/GEMV/SYRK) through an external BLAS via `EIGEN_USE_BLAS`, instead of Eigen's own kernels. Vendor BLAS is typically faster for large dense products, and it makes Eigen reproduce that BLAS bit-for-bit. Requires a **32-bit-integer (LP64) BLAS** (stock Eigen's BLAS backend is int32-only and HelFEM does **not** patch Eigen). When a BLAS is linked, LAPACKE is also probed and, if found, Eigen's dense decompositions (including the SCF eigensolves) are routed to LAPACK via `EIGEN_USE_LAPACKE`. **Graceful fallback:** if no LP64 BLAS is found, the configure prints a notice and uses Eigen's header-only kernels — BLAS is never a hard dependency (and missing LAPACKE just leaves the decompositions on Eigen's own code). Set `OFF` to force the header-only kernels (the machine-independent-reproducibility choice). There is no ILP64 knob: Eigen indexes with a 64-bit type unconditionally.
- `HELFEM_FETCH_LIBXC` (default `ON`) — libxc is looked for first as a CMake package (`LibxcConfig.cmake`, which a CMake-built libxc installs), then through pkg-config (`libxc.pc`); if neither is found, libxc `HELFEM_LIBXC_FETCH_TAG` (default `7.1.2`) is cloned from gitlab.com/libxc/libxc and built static alongside HelFEM, then installed with it (see *Fetched dependencies* below). `OFF` makes a missing libxc a configure error instead. Building it needs CMake ≥ 3.21. On a compute node without network access, set `FETCHCONTENT_SOURCE_DIR_LIBXC` to a local checkout.
- `HELFEM_WITH_RECI` (default `OFF`) — build `helfem-reci`, the CASCI half of CASSCF (`src/general/cas_ci.{h,cpp}`), on reci's determinant/CSF engine. reci is looked for as an installed package (`find_package(reci COMPONENTS core)`), else fetched from `git@github.com:philipmnel/reci.git` at a pinned commit (SSH: the repository is private) and built core-only (`RECI_CORE_ONLY`: no Fortran, TREXIO, diaglib or molpro). Its core brings Kokkos and C++20, both confined to `cas_ci.cpp`: nothing else in HelFEM is compiled as C++20. A program using it must hold a `helfem::cas::KokkosScope` in `main()`. reci's CI is real, which suffices for any active space: orbitals with m != 0 are complex, but their integrals are real numbers, with (tu|vw) = (vw|tu) = (ut|wv) though not (tu|vw) = (ut|vw). The tensor is therefore passed through unsymmetrised -- symmetrising it would be the real-ORBITAL assumption, which does not hold.
- `HELFEM_USE_OPENMP` (default `ON`) — consumed as the `OpenMP::OpenMP_CXX` imported target. On **macOS**, AppleClang ships no OpenMP runtime; `brew install libomp` first (or configure with `-DHELFEM_USE_OPENMP=OFF`). The old unprefixed `USE_OPENMP` is still honoured for a standalone build, with a deprecation warning; it is ignored when HelFEM is consumed by another project, where an unprefixed name would collide.

**Dependencies:** Eigen ≥ 3.4 (auto-fetched if absent), BLAS/LAPACK (optional; used when found and `HELFEM_EIGEN_BLAS=ON`, the default),
wignernj (auto-fetched), OpenOrbitalOptimizer (auto-fetched); binaries additionally
need HDF5 (C++ interface) and libxc (fetched if absent, see `HELFEM_FETCH_LIBXC`). The Legendre special functions are C++ (no
Fortran compiler required).

**Fetched dependencies** are built static with position-independent code and
installed into a HelFEM-private subtree, `<prefix>/lib/helfem-deps`, which
`helfemConfig.cmake` searches explicitly (`NO_DEFAULT_PATH`) and nothing else
does. Both halves matter. Every fetched target is referenced by the exported
HelFEM targets, so leaving one out of the install (as OpenOrbitalOptimizer and
OpenTrustRegion once were) makes `find_package(helfem)` fail for every
consumer. Installing one where `find_package` looks by default (as Eigen,
wignernj and libxc once were) makes the next configure against the same
prefix find that copy instead of fetching, freezing it at whatever snapshot
was installed first. The same freeze can come from the user package registry
(`~/.cmake/packages`), which Eigen's own `export(PACKAGE)` writes a build-tree
path into; the fetch disables that. All of this lives in the
`helfem_fetch_privately` macro in the top-level `CMakeLists.txt`, which also
keeps the subprojects' options (`BUILD_SHARED_LIBS`, libxc's `ENABLE_FORTRAN`,
Eigen's `BUILD_TESTING`, ...) out of an including project's cache.

## Running tests

After build, test binaries are in `objdir/src/` (or `install/bin/`):

```bash
./install/bin/legendre_test    # Legendre function tests
./install/bin/gaunt_test       # Gaunt coefficient tests
./install/bin/sphtest          # Spherical harmonic tests
./install/bin/atomic_itest     # Atomic integration test
./install/bin/atomdb_test      # Tabulated atomic wave functions
```

## Architecture

The codebase has three layers:

### 1. `libhelfem/` — Core FEM library

Low-level library with no HDF5/libxc dependency. Compiled as `helfem` static/shared library.

- **Polynomial bases** (`PolynomialBasis`, `LIPBasis`, `HIPBasis`, `GeneralHIPBasis`, `LegendreBasis`): Implement different polynomial basis families for finite elements. `LIPBasis_eval.cpp` and `HIPBasis_eval.cpp` are large auto-generated evaluation tables.
- **`FiniteElementBasis`**: Assembles element-level bases into a global FEM basis on a radial grid.
- **`RadialBasis`**: Wraps `FiniteElementBasis` for quantum-chemistry use (boundary conditions, overlap/kinetic matrix assembly).
- **Grid** (`grid.cpp`): Generates radial grids (linear, quadratic, polynomial, exponential, geometric).
- **Quadrature** (`lobatto.cpp`, `quadrature.cpp`, `chebyshev.cpp`): Gauss-Lobatto and Chebyshev quadrature rules.
- **Nucleus models** (`PointNucleus`, `GaussianNucleus`, `SphericalNucleus`, `HollowNucleus`, `RegularizedNucleus`): Nuclear charge distributions for the electron-nuclear potential.

### 2. `src/general/` + application subdirs → `helfem-common` library

Higher-level library linking against `helfem`. Contains SCF infrastructure, DFT wrappers, and basis sets for each geometry:

- **`general/`**: SCF helpers (`scf_helpers.cpp`), DIIS (`diis.cpp`), L-BFGS (`lbfgs.cpp`), Gaunt coefficients (`gaunt.cpp`), DFT functional interface (`dftfuncs.cpp`), checkpoint I/O (`checkpoint.cpp`), superposition of atomic potentials (`sap.cpp`), model potentials (`model_potential.cpp`), tabulated atomic wave functions (`atomdb.cpp` + the generated `atomdb_data.cpp`).

  `sap.cpp` and `atomdb.cpp` are two representations of the same SAP potential. `sap.cpp` tabulates the effective charge on a fixed radial grid and interpolates between knots; `atomdb.cpp` stores the *orbitals* of the same spherically averaged atoms and derives the density, the Hartree screening and the LDA exchange screening at whatever r is asked for, with no interpolation. Storing the orbitals rather than the potential keeps the tabulated object at the lowest polynomial degree in the problem — the density is twice the degree of the orbitals and the potential higher still, so a finite-element representation of the potential would need a finer grid than the one the orbitals were solved on. `tools/gen_atomdb.py` builds `atomdb_data.cpp` from `gensap` checkpoints; `objdir/src/atomdb_dump | tools/gen_sap_table.py src/general/sap.cpp` then rebuilds `sap.cpp` from the evaluator, so the interpolated table is a tabulation of exactly what `SAPFEAtom` returns and the two differ only by the interpolation (~2e-4 in Zeff).
- **`atomic/`**: 2D angular + radial basis (`TwoDBasis.cpp`) for spherical atoms; DFT integration grid.
- **`sadatom/`**: Spherically-averaged atom solver (`solver.cpp`) — faster than `atomic/` for symmetric ground states; used to generate SAP initial guesses (`gensap`).
- **`diatomic/`**: Prolate spheroidal coordinate basis (`basis.cpp`) and 2D quadrature (`twodquadrature.cpp`) for diatomic molecules.
- **`legendre/`**: Fortran 90 library for associated Legendre functions; wrapped via `Legendre_Wrapper.f90` / `Legendre_Wrapper.h`.

### 3. Executables

| Binary | Source | Purpose |
|--------|--------|---------|
| `atomic` | `src/atomic/main.cpp` | Full HF/DFT for atoms |
| `diatomic` | `src/diatomic/main.cpp` | Full HF/DFT for diatomics |
| `gensap` | `src/sadatom/main.cpp` | Generate SAP initial guess |
| `harmonic` | `src/harmonic/main.cpp` | Harmonic oscillator test |
| `softcoulomb` | `src/harmonic/softcoulomb.cpp` | Soft Coulomb test |
| `diatomic_cbasis` | `src/diatomic/corebasis.cpp` | Core basis analysis |
| `diatomic_1e` | `src/diatomic/1e.cpp` | One-electron diatomic |
| `diatomic_cpl` | `src/diatomic/completeness.cpp` | Completeness profile |
| `diatomic_dline` | `src/diatomic/density_line.cpp` | Density along bond axis |
| `diatomic_dgrid` | `src/diatomic/density_grid.cpp` | Density on 2D grid |
| `aij` | `src/sadatom/aij.cpp` | Atom in jellium |

## Second-order convergence

The atom-in-jellium problem optimizes the occupations as well as the
orbitals, and that is what makes it converge badly at first order. When
two orbitals are degenerate, moving density between them costs nothing to
first order in the orbital energies, so the gradient is flat along exactly
the direction that matters while the real cost -- a dense coupling through
the Coulomb and XC kernels -- is entirely second order. The spin-restricted
Fe atom is the standard case: 4s and 3d come out at the same orbital
energy and sit in different `l` blocks, so no orbital rotation connects
them at all and their whole interaction is `<4s 4s|W|3d 3d>`. ODA alone
does not converge it in 2000 iterations; `--secondorder=1
--fractionaloccupations=1` reaches a gradient of 1e-9 in a couple of seconds.

`--secondorder=1` runs the first-order solver for `--preiter` iterations
and then hands over to a trust-region optimizer built on OpenTrustRegion
(`src/general/trustregion_scf.{h,cpp}`, over the wrapper in
`otr_solver.{h,cpp}`). Each solve optimizes within one occupation
*pattern*, meaning which shells are fractionally occupied. Between solves
the optimizer changes the pattern itself; see "The second-order phase can
start from the guess" below.

**Fractional occupations are opt-in.** `--fractionaloccupations` (default off)
decides what the occupations are.

- **Off: integer occupations.** Every block holds a fixed whole number of
  electrons, filled in orbital order, and only the orbitals are parameters.
  An orbital can still be partly filled when it stands for several
  electrons: a spherically averaged shell in `gensap`/`aij`, or a mirrored
  +-m pair at `--symmetry=3`. A Fe beta 3d^1 is one electron in a
  five-fold orbital. A handover with non-integer block counts is rounded
  first. A converged pass that violates aufbau is re-occupied in energy
  order, Roothaan-style, and solved again. When that oscillates, the
  optimizer returns to the lowest converged pattern and warns that it is
  not aufbau. Spin-restricted Fe does exactly that, between 4s^2 3d^6 and
  4s^0 3d^8, and ends at -1258.8862694608, the pure-state solution 0.032
  Eh above the ensemble one.
- **On: fractional occupations.** The block fillings are parameters as
  well. The optimizer stops at shell closings and releases KKT violations
  by moving charge, as described below. That is what an atom with
  degenerate shells in different blocks needs, and it is a different
  problem from an integer calculation, not a better solver for the same
  one.

`--stabilityanalysis=1` runs OpenTrustRegion's stability analysis whenever the
second-order phase converges. The analysis covers the orbital rotations
within the symmetry blocks, and the occupation coordinates when there are
any. OpenTrustRegion reports only an unstable point, which it then follows
downhill, so the driver prints a line when the final point passes.
OpenTrustRegion also runs the check by itself on any solve that starts at
a stationary point. That is why unrestricted H2 at R=5 leaves the
spin-symmetric saddle that first order converges to (-0.8593120879) and
reaches the broken-symmetry minimum (-0.9998401893) even without the
option.

Two habits worth keeping:

- **`--sotest=1e-4` before trusting a result on a new system.** It checks
  the analytic gradient and Hessian against finite differences of the
  energy, including mixed `d1 . H d2` forms that a single-direction check
  cannot see. Nothing downstream distinguishes a wrong Hessian from a hard
  problem. The kernel is the functional's own for an LDA, a GGA and a
  tau-meta-GGA alike; only a *laplacian*-dependent meta-GGA still lacks
  its kernel channel.

  `--sotest` is nevertheless a pass/fail test only for an LDA on the
  `atomic` and `diatomic` grids, and a measurement above it. That is a
  limit of the *reference*, not of the kernel. Those workers evaluate tau
  from genuine theta/phi derivatives, which carry `1/(r sin theta)` and
  reach ~1e15 near the axis; the four-point energy polarization has its
  own truncation there, and it lands at 1e-5 (PBE) to 3e-3 (TPSS) however
  exact the Hessian is. `gensap`/`aij` gate on the laplacian instead,
  because a spherically averaged radial grid has no such term.

  To test the kernel itself, use `HELFEM_CHECK_DFOCK` below, which
  compares the response Fock matrix directly and reaches the same floor
  for LDA, PBE and TPSS alike.

  Two knobs help when a `d . H d` row disagrees:
  `HELFEM_CHECK_DFIELDS=1` compares the perturbed density, density
  gradient and tau against central differences of `update_density`,
  isolating the field evaluation from the kernel; `HELFEM_CHECK_DFOCK=1`
  compares the whole response Fock matrix against a central difference of
  the ground-state one. Both print to stderr.
- **Read the conditioning report at `--verbosity=1`.** It prints the
  spread of the uncoupled Hessian diagonal and how much of it is
  near-singular. Exact zeros there are not ill-conditioning, they are a
  bug: they mean the parameter set contains rotations that do not change
  the density.

`--soredfac` defaults to 3e-1, a hundred times looser than
OpenTrustRegion's own default, and that is deliberate. A step whose
microiterations missed their residual-reduction target is discarded
however good it is -- probed directly, one discarded step lowered the
energy by 5.3e-4 and landed on the minimum -- so tightening this does not
sharpen the answer, it makes the solver throw away good steps until it
stalls. Loosening it is not a trade either: on the cases that already
worked it is *cheaper* (Fe LDA 65 Hessian-vector products to 48, Fe PBE 81
to 55) at identical energies. The same over-tight target was also what
made `--sosolver=tcg` appear broken.

**The second-order phase can start from the guess.** `--preiter=0` hands
the guess orbitals straight to the optimizer, and the optimizer finds the
occupation pattern itself. With integer occupations it does so by the
aufbau re-occupation above. With `--fractionaloccupations=1` it moves between
patterns with three mechanisms.

- **Kinks stop a step.** E(n) has a kink wherever a fractional orbital
  fills or empties. An occupation step is cut at the first kink it
  reaches. If the cut point is lower in energy, it is adopted, the
  parametrization is rebuilt for the new pattern, and the solve restarts
  there. A step that crossed the kink would follow a quadratic model of
  the wrong pattern.
- **KKT violations release a block.** By Janak's theorem dE/dn_i = eps_i,
  so at the minimum of  min E(n) s.t. 0<=n_i<=w, sum n_i = N  a full
  orbital lies below the Fermi level and an empty one above it. A point
  that violates this is stationary only for the *restricted* problem with
  that block pinned. That is why its gradient can be tiny while its energy
  is still wrong. When the RMS gradient is below max(conv_tol, 1e-4), the
  optimizer checks these conditions on canonical orbitals and moves a
  nudge of charge across the violation. This includes all-integer
  patterns, where the highest full orbital of one block lies above the
  lowest empty orbital of another.
- **A recurring pattern ends the restarts.** The restarts stop when a
  pattern recurs without having lowered the energy. A cap of 100 restarts
  is only a safety net.

Measured from the core guess at `--preiter=0 --fractionaloccupations=1`, compared
with the code before these mechanisms:

| case | before | now |
|---|---|---|
| `aij` Fe | 6.1 Eh high | exact, 3 restarts |
| `aij` Cm | 100 Eh high | exact, 29 restarts, 4.7 s |
| `gensap` Cr, Nb, Fe-TPSS | 5-7 Eh high | exact, 3-5 restarts |

Spin-restricted PBE Fe2 at `--symmetry=3` runs `--scfmethods=DIIS
--preiter=0 --fractionaloccupations=1` straight from the SAP guess. It reaches the same energy as the
DIIS+ODA+CG handover (-2526.8717316453) in 2 restarts and about 130
Hessian-vector products. First order alone does not converge that system
in 128 iterations.

`--preiter` still defaults to 100. A good handover saves restarts, and a
restart costs a full rebuild of the parametrization. Before reporting a
second-order result on a new system, check the "Aufbau check from the final
Fock matrix" line that `diatomic` prints at `--verbosity=1`. It
diagonalizes the final Fock matrix independently of the optimizer and
lists the electrons per block.

Known weakness: the second-order phase is sensitive to where the
first-order phase hands over. Fe with TPSS, Cr and Nb all stall at
`--preiter=40` and all converge in a second at 20, 30, 50 or 60. This is
not cured, only made cheap -- those cases now stop within a second or two
a few parts in 10^7 from the minimum, rather than grinding for 900 seconds
far away from it. A work ceiling, counted per macroiteration so it fires
on work-without-progress rather than on an absolute budget, is what stops
them, and it says what happened. If a system matters and stalls, move
`--preiter`.

`gensap` carries the same optimizer over the same parametrization, since
the spherically averaged atom is the same problem with the jellium
switched off. The options are spelled the same (`--secondorder`,
`--preiter`, `--sotest`, ...), and the two drivers can be compared
directly: `aij` with `--njellium=0 --rs=0` IS the bare atom, and on Fe
both reach `-1258.9186876173` through separate energy expressions, Fock
builders and SCF drivers.

That case is worth knowing about, because in `gensap` first order does
not merely converge slowly -- it stops at `-1258.8157`, **0.103 Eh above
the minimum**, without printing a convergence line. The second-order
phase repairs it outright.

Two things are specific to `gensap`:

- **Exact exchange goes through the Hessian too.** `gensap` supports HF
  and hybrids, which `aij` does not. Exchange is linear in the density,
  so its response is exact; `--sotest` confirms it (`d1.H.d1` agrees with
  finite differences to 1.9e-10 at HF), and on Fe the first- and
  second-order answers agree to 2e-10 Eh.
- **Frozen occupations and `--secondorder` are refused together.** The
  occupations are among the variables the optimizer optimizes, so
  `--fixed-per-l` style pinning has no meaning there, and the driver
  throws rather than silently ignoring one of the two.

The handover sensitivity is more visible here than in `aij`, because
`gensap` defaults to the SAP initial guess (`--iguess=2`) and lands
somewhere different. On Fe with that guess the energy is right from
`--preiter` 30 upwards, but the RMS gradient stalls between 1e-8 and 4e-7
and the convergence line is not printed; with the core guess
(`--iguess=0`) the same run reaches 4.9e-10. The energy is not what is
uncertain -- it repeats to all printed digits -- only the flag.

### `atomic` and `diatomic`

The same optimizer drives these two as well. What differs is the basis
bookkeeping: they orthonormalize each symmetry block separately over its
own subset of the basis functions, so `trscf::Optimizer` takes one
`Sinvh` per block rather than one shared radial basis. Each driver splits
its Fock build into a density-driven core plus the OpenOrbitalOptimizer
wrapper, so the second-order phase shares the energy expression rather
than restating it.

**The XC response is LDA-shaped in both.** Only the density-density
kernel block is used, which is exact for an LDA and an approximation
beyond it. That costs iterations, not correctness -- every step is
validated against the true energy by the trust-region ratio test -- and
`--sotest` reports the deviation instead of failing. Measured: PBE on N
converges to the same energy as first order at an RMS gradient of 4.9e-9.
`DFTGridWorkerBase::set_response_potential` already takes the gradient
and tau channels, so that is the seam when they arrive.

**Occupations are integer in `atomic`**, so the free occupation
coordinates vanish of their own accord and it reduces to a pure
orbital-rotation Newton step. Nothing special-cases that. `diatomic` is
where they can be fractional -- `--symmetry=3` splits pi+ from pi- and
spreads the occupations over both -- and the same parametrization picks
them up. Note the arithmetic: it takes TWO fractionally occupied blocks
before there is any occupation freedom at all, since the sum-zero
subspace of one free block has dimension zero. Spin-restricted O2 at
`--symmetry=3` has exactly one (pi*, 2 electrons of a maximum 4) and
reports `0 occupation transfers among 1 fractionally occupied blocks`.

**Mirrored blocks are the trap.** A +m / -m block stands for two
degenerate spatial orbitals and its maximum occupation counts both, so
its density is shared equally between them. That halving must live in the
scatter, which is the one point BOTH density builds go through -- the
SCF's and the trust-region optimizer's own. Put it in the SCF's alone and
the optimizer double-counts every mirrored block: Ne at `--symmetry=3`
hands over 5.5 Eh wrong and never converges.

That class of mistake is invisible to `--sotest`, which differentiates
one energy expression against finite differences of *itself* and so
agrees perfectly with a wrong one. It needs a case with occupied mirrored
blocks, which is what `diatomic-Ne-dummy-lda-absm-secondorder` is for.
When adding a second-order path to a new driver, check the handover
energy against the first-order one before trusting any derivative check.

Occupation coupling is where the exact preconditioner earns itself, and it
takes two occupation coordinates to see it -- with one the block is 1x1
and there is nothing to capture. Curium (5f, 6d and 7s fractionally
occupied at once) is the demonstration: the exact block reaches an RMS
gradient of 9.7e-10 where the floored diagonal strands the solve 3.8e-3
above the minimum. Almost every open-shell atom has only one such
coordinate, so most systems will never show the difference.

## Current-corrected meta-GGAs

HelFEM's basis functions are complex, chi = f exp(i m phi), and its
density matrices real. An open shell that occupies +m and -m unequally
-- a p_{+1} electron, a pi_{+1} one at `--symmetry=1` -- therefore
carries a paramagnetic current
j_p = sum_uv P_uv Im(chi_u^* grad chi_v), and the plain kinetic energy
density counts it: tau of a p_{+1} orbital exceeds that of a p_0 one by
|j_p|^2 / 2 rho. A tau-meta-GGA then sees the M_L components of one term
as different densities. `--current` (in `atomic` and `diatomic`) evaluates
every tau-dependent functional at the gauge-invariant
tau~ = tau - |j_p|^2 / (2 rho) instead, per spin. It is opt-in and a no-op
for functionals without tau; `gensap`/`aij` have no such option because a
spherically averaged density carries no current.

The chain rule is generated, not written: libxckernel's `helfemwriter`
(0.6.0 and later) emits `xck_helfem_{vxc,fxc}_cmgga_tau_*` into
`src/general/xckernel_fxc.h`, derived for a complex basis with a real
symmetric P. That convention is the one that matters -- a real
perturbation of a complex basis MOVES the current, so the response needs
the perturbed current too, and a kernel derived for the opposite
convention (real basis, current from Im P) would silently drop it.
The base worker forms tau~ for libxc, turns the ground-state vrho into
u = vrho + vtau |j|^2 / 2 rho^2 and adds the current channel
y = -vtau j / rho (`build_vcurrent`, after `build_vgrad`); each worker
supplies j_p and assembles y against the pair's current density.

- **3D workers** (atomic, diatomic): three components, from the complex
  basis values, so they hold whatever the density matrix couples.
- **Pure-m worker**: the current is purely azimuthal,
  j_phi = sum_m m rho_m / h_phi. Its y term is the one piece of the XC
  matrix that is ODD in m, so the +m / -m block mirroring in `eval_Fxc`
  negates it for the mirrored block rather than copying it. At
  `--symmetry=3` the shared +m / -m density makes j_phi vanish exactly.
- **Empty spin channels.** Below the density threshold a channel's
  current is dropped from tau~, the potential and the kernel alike, since
  j / rho is 0/0 there (the beta channel of a hydrogen atom); doing it in
  all three keeps the Fock matrix the derivative of the energy.

`current_mgga_test` is the check, and each part of it can fail:
- A density with no current is bit-identical with and without `--current`.
- One occupied pi_{+1} orbital has tau~ = tau_W, where TPSS correlation
  vanishes: -5.4e-3 Eh without the current, -2.6e-12 with it, on both
  grids.
- The pure-m and 3D workers agree to 1e-14 in Exc and the whole XC
  matrix for a current-carrying density -- two independent
  implementations of j and of the sign convention between j and y.
- The pure-m XC matrix matches finite differences of the energy to a few
  1e-9, and the response of both grids matches a central difference of
  the XC matrix to 3e-8. Dropping the perturbed current from the kernel
  puts the latter at 0.17, so the check does see it.

End to end, through the `diatomic` driver: OH X 2Pi with TPSS and the
beta pi electron pinned in m = +1 (`--readocc`, `--symmetry=1`) gives the
same first-iteration energy on the pure-m and 3D grids to every printed
digit with `--current`, and `--secondorder` takes it from an RMS gradient
of 1.8e-6 to 2.1e-9 in two macroiterations -- the response is exact
enough for quadratic convergence.

The derivative checks need a density with NO spin channel made of a
single orbital. One orbital has tau = tau_W (z = 1), which is where
libxc clamps sigma <= 8 rho tau, so the energy has a kink there that a
central difference straddles: the plain TPSS Fock matrix "fails" such a
check by 3-20% with the current off just as much as on. The same z = 1
limit is a real hazard in production. TPSS correlation of one electron
has vtau proportional to rho/tau_W, which diverges wherever grad rho = 0;
the H 2p_0 density has such points, and an SCF on it diverges (kinetic
energy 330 Eh after one step) with or without `--current`. With
`--current`, 2p_{+1} reaches z = 1 too and breaks in exactly the same way.
That is the functional, not the correction: tau~ makes the two
components behave alike, and here they are alike in failing.

Measured on B 2p^1 (TPSS, unrestricted, `--symmetry=2 --lmax=1 --nelem=5
--nnodes=15`, configuration pinned with `--readocc`):
M_L = 0 is -24.6672086289 either way (a real orbital carries no current),
and M_L = 1 goes from -24.6581195074 to -24.6613424157, i.e. the spurious
M_L splitting falls from 9.1 to 5.9 mEh. It does not vanish, and should
not be expected to: the p_0 and p_{+1} densities really are different
shapes, which a semilocal functional can still tell apart.
