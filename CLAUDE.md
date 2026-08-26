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
- `USE_OPENMP` (default `ON`) — consumed as the `OpenMP::OpenMP_CXX` imported target. On **macOS**, AppleClang ships no OpenMP runtime; `brew install libomp` first (or configure with `-DUSE_OPENMP=OFF`).

**Dependencies:** Eigen ≥ 3.4 (auto-fetched if absent), BLAS/LAPACK (optional; used when found and `HELFEM_EIGEN_BLAS=ON`, the default),
wignernj (auto-fetched), OpenOrbitalOptimizer (auto-fetched); binaries additionally
need HDF5 (C++ interface) and libxc. The Legendre special functions are C++ (no
Fortran compiler required).

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

## Second-order convergence in `aij` and `gensap`

The atom-in-jellium problem optimizes the occupations as well as the
orbitals, and that is what makes it converge badly at first order. When
two orbitals are degenerate, moving density between them costs nothing to
first order in the orbital energies, so the gradient is flat along exactly
the direction that matters while the real cost -- a dense coupling through
the Coulomb and XC kernels -- is entirely second order. The spin-restricted
Fe atom is the standard case: 4s and 3d come out at the same orbital
energy and sit in different `l` blocks, so no orbital rotation connects
them at all and their whole interaction is `<4s 4s|W|3d 3d>`. ODA alone
does not converge it in 2000 iterations; `--secondorder=1` reaches a
gradient of 1e-9 in a couple of seconds.

`--secondorder=1` runs the first-order solver for `--preiter` iterations
and then hands over to a trust-region optimizer built on OpenTrustRegion
(`src/general/trustregion_scf.{h,cpp}`, over the wrapper in
`otr_solver.{h,cpp}`). The first-order phase only has to find the
occupation *pattern* -- which shells are fractionally occupied -- and get
close enough for a quadratic model to mean something; the second-order
phase optimizes within that pattern and is not the thing that should be
deciding to open a closed shell.

Two habits worth keeping:

- **`--sotest=1e-4` before trusting a result on a new system.** It checks
  the analytic gradient and Hessian against finite differences of the
  energy, including mixed `d1 . H d2` forms that a single-direction check
  cannot see. Nothing downstream distinguishes a wrong Hessian from a hard
  problem. For a GGA or meta-GGA the Hessian is *deliberately* approximate
  -- the kernel keeps only its density-density block -- so `--sotest`
  measures the deviation (0.14% for PBE, 0.08% for TPSS) instead of
  failing.
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

**Hand over from a well-converged first-order solution.** `--preiter`
defaults to 100 and lower is not merely slower, it is silently wrong: the
second-order phase optimizes *within* an occupation pattern and cannot
discover one. From the core guess (`--preiter=0`) Fe converges in a second
to an RMS gradient of 5e-9 and an energy 6.1 Eh too high; Cm lands 124 Eh
too high. Even at `--preiter=20` Fe converges to 2.2e-9 at the *pure-state*
solution, 0.032 Eh above the ensemble one, because 4s came over exactly
full. A block frozen on the wrong side of the Fermi level violates the KKT
conditions of  min E(n) s.t. 0<=n_i<=w, sum n_i = N  -- by Janak's theorem
dE/dn_i = eps_i, so a full orbital must lie below eps_F and an empty one
above it. Such a point is stationary for the *restricted* problem with that
block pinned, which is why the gradient there can be tiny and the energy
still wrong. The optimizer detects this and releases the block rather than
merely warning, which repairs the answer outright: Fe at `--preiter=20`
goes from 0.032 Eh high to exact, and Cm at 20 from 0.029 Eh high to exact.
It does not rescue a start from the bare core guess, where the orbitals
themselves are wrong rather than the occupations.

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

Occupation coupling is where the exact preconditioner earns itself, and it
takes two occupation coordinates to see it -- with one the block is 1x1
and there is nothing to capture. Curium (5f, 6d and 7s fractionally
occupied at once) is the demonstration: the exact block reaches an RMS
gradient of 9.7e-10 where the floored diagonal strands the solve 3.8e-3
above the minimum. Almost every open-shell atom has only one such
coordinate, so most systems will never show the difference.
