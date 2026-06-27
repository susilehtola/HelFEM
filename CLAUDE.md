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
mkdir objdir && cd objdir
cmake .. -DUSE_OPENMP=ON -DCMAKE_INSTALL_PREFIX=../install -DCMAKE_BUILD_TYPE=Release
make -j9 install
```

**Key CMake options:**
- `HELFEM_FIND_DEPS=ON` — auto-discover Armadillo/HDF5/libxc via `find_package`
- `HELFEM_BINARIES=OFF` — build only `libhelfem` (skips HDF5 and libxc requirements)
- `HELFEM_CMAKE_SYSTEM=OFF` — ignore `CMake.system` (useful for multiple build dirs)
- `HELFEM_BLAS_PROVIDED_EXTERNALLY=ON` — don't link BLAS/LAPACK into libhelfem; parent project (e.g. an embedding application with its own MKL/OpenBLAS) must provide them. Sets `-DARMA_DONT_USE_WRAPPER`. The parent must match integer width — see ARMA_64BIT_WORD / ARMA_BLAS_LONG.

**System configuration:** `CMake.system` (symlink to e.g. `CMake.fedora`) sets compiler flags, library paths, and Armadillo/BLAS configuration for the local machine. This is where 64-bit BLAS indices, flexiblas, and non-standard install paths are configured.

**Dependencies:**
- Core (`libhelfem`): Armadillo ≥ v9
- Binaries (`src/`): HDF5 (C++ interface), libxc
- Fortran compiler: for Legendre special functions (`src/legendre/`)

## Running tests

After build, test binaries are in `objdir/src/` (or `install/bin/`):

```bash
./install/bin/legendre_test    # Legendre function tests
./install/bin/gaunt_test       # Gaunt coefficient tests
./install/bin/sphtest          # Spherical harmonic tests
./install/bin/atomic_itest     # Atomic integration test
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

- **`general/`**: SCF helpers (`scf_helpers.cpp`), DIIS (`diis.cpp`), L-BFGS (`lbfgs.cpp`), Gaunt coefficients (`gaunt.cpp`), DFT functional interface (`dftfuncs.cpp`), checkpoint I/O (`checkpoint.cpp`), superposition of atomic potentials (`sap.cpp`), model potentials (`model_potential.cpp`).
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
