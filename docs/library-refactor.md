# Library restructuring plan: a reusable FEM core

Status: proposal for discussion. Prototype in `docs/autoconv_prototype.cpp`.

**Phase 3 is done** (branch `libhelfemqc-split`): the quantum-chemistry layer
now lives in a separate `libhelfemqc` target (`helfem::helfemqc`, in-tree alias
`helfem::qc`) that links PUBLIC to `helfem`. Final classification, with two
deviations from the original table below:

- **Moved to libhelfemqc** — the concrete nucleus models (`PointNucleus`,
  `GaussianNucleus`, `HollowNucleus`, `SphericalNucleus`, `RegularizedNucleus`,
  header + `.cpp`), the FE Coulomb/exchange assembler `CoulombExchangeFE.h`, and
  the atom-centred orbital bases `NAORadialBasis.h` / `GTO.h` / `STO.h`.
- **Kept in libhelfem (general)** — the abstract base
  `modelpotential::ModelPotentialT<T>` (`ModelPotential.{h,cpp}`), whose
  `virtual T V(T r) const = 0` interface `RadialBasis` consumes by pointer, and
  the trivial `RadialPotentialT` (`RadialPotential.*`, general `r^n` math). Also
  kept: `erfc_expn` (`erfc_expn.{h,cpp}`), a general screened-Coulomb Green's
  function that `quadrature::erfc_integral` depends on — it did **not** move;
  instead its namespace `helfem::atomic::erfc_expn` was renamed to
  `helfem::erfc_expn` (dropping the misleading `atomic`).

`helfem-common` now links `helfemqc` PUBLIC; the installed CMake package chains
`helfem::helfemqc` into `helfem::helfem`'s interface. Pure move — energies are
bit-identical to master.

## Goal

Turn `libhelfem` into a **proper, reusable, domain-agnostic finite-element
library**, and keep the atomic / diatomic quantum chemistry as *applications*
on top of it. Concretely, a two-library stack:

```
libhelfem     FEM discretization + matrix-element / 2e-Green's-function
              evaluators.  Templated on the scalar type, auto-converging to
              eps(T).  Knows nothing about electrons.
   |
libhelfemqc   Radial quantum chemistry: potentials (nucleus models, model
              potentials), Coulomb/exchange kernels + angular coupling,
              GTO/STO/NAO.  Supplies "f(x)" and its breakpoints to libhelfem.
   |
HelFEM        Applications: atomic/, diatomic/, sadatom/, SCF, DFT.
```

## Where we start from (better than a blank slate)

The codebase already leans this way:

- **The unifying primitive already exists.** `FiniteElementBasis::matrix_element(
  lhder, rhder, xq, wq, f)` is exactly `<d^k B | f(x) | d^l B>`: left/right
  derivative orders and an arbitrary weight `f` (`std::function<T(T)>`, null =
  unit). There is a matching `vector_element` and generic-callable overloads.
- **`RadialBasis` already keeps physics out.** Its `kinetic()` *excludes* the
  centrifugal term — "caller adds `l(l+1)*kinetic_l()`". Angular momentum is
  already the caller's business.
- **The r^2 measure tames the obvious singularity.** The point-nucleus element
  is `-<R| r |R>` (the `1/r` is cancelled by the spherical Jacobian) — a
  *polynomial* weight, the *easy* case, not a hard one.

So this is ~60% a consolidation, not a rewrite.

## Component classification

| Component | Today | Target | Note |
|---|---|---|---|
| polynomial bases (LIP/HIP/HIP2/HIP3/Legendre), quadrature, grid | lib1dfem | **libhelfem** | fold lib1dfem in — no separate customer for "bases" vs "assembly" |
| `FiniteElementBasis` (+ `matrix_element`) | libhelfem | **libhelfem** | the general assembly + the unifying evaluator |
| `RadialBasis` | libhelfem | **libhelfem** | radial/spherical FEM is domain-neutral; already physics-free once `f` is injected |
| radial 2e primitives (`twoe_integral`, `yukawa_integral` per multipole L) | libhelfem | **libhelfem** | the `r<^L/r>^(L+1)` Green's function is math, not chemistry |
| concrete nucleus models (`PointNucleus`, `GaussianNucleus`, ...) | libhelfem | **libhelfemqc** | these are just `f(r)` — the electron-nuclear charge distributions |
| `ModelPotential` (abstract base) / `RadialPotential` | libhelfem | **libhelfem** | the `virtual V(r)` interface `RadialBasis` consumes + the general `r^n` potential stay general |
| `CoulombExchangeFE` | libhelfem | **libhelfemqc** | kernel choice + angular coupling + Fock assembly is chemistry |
| `erfc_expn` | libhelfem | **libhelfem** | general screened-Coulomb Green's function `quadrature` depends on; kept general, namespace `atomic::` dropped |
| `NAORadialBasis`, `GTO`, `STO` | libhelfem | **libhelfemqc** | atomic orbitals |
| gaunt, SCF, DFT, checkpoint, atomic/diatomic bases | helfem-common | **HelFEM app** | unchanged |

## The auto-converging matrix-element evaluator

Replace the caller-supplied `(xq, wq)` with an evaluator that refines its own
quadrature until the element is stable to `eps(T)`. The prototype
(`docs/autoconv_prototype.cpp`, run below) establishes the strategy empirically:

```
  A poly x^2    double  n=4  err=0         | _Float128 n=4  err=0
  B exp(-x)     double  n=8  err=0         | _Float128 n=13 err=0
  C kink |x-.5| double  naive n=400 err=1.3e-6 STALL | split@kink n=3 err=0
```

Design that follows from it:

1. **Polynomial / unit `f`: no iteration.** Compute the exact Gauss-Lobatto
   order from the integrand degree (`deg(B_k)+deg(B_l)+deg(f)`), evaluate once.
   Covers overlap, kinetic, point-nucleus, polynomial confinement — the bulk.
2. **Smooth non-polynomial `f`: order-refine.** Double the order until
   `|I_n - I_{n-1}| <= eps(T)*|I_n|`. Converges geometrically and, crucially,
   the required order grows with `T` on its own (8 at double, 13 at quad) — the
   element reaches *its type's* machine precision with no tuning. Covers finite
   nuclei, model potentials, erfc/Yukawa weights.
3. **Non-smooth `f` (kinks, integrable singularities): subdivide, don't
   over-quadrature.** Across a cusp, order-refinement *stalls* — at ~1.3e-6 for
   the 2e Green's function, and quad buys nothing. The evaluator must take the
   breakpoint locations and split there; each smooth sub-panel then converges by
   rule (2). **The QC layer supplies `f` together with its breakpoints** (e.g.
   `r=r'` for the Coulomb kernel, the nuclear boundary for a finite nucleus).

Proposed general interface (schematic):

```cpp
// libhelfem
Mat<T> matrix_element(int lhder, int rhder,
                      const std::function<T(T)>& f,           // null = unit
                      const std::vector<T>& breakpoints = {}, // non-smooth points of f
                      int poly_degree_of_f = -1);             // >=0 => exact order, no refine
```

Chemistry side becomes declarative: `overlap = matrix_element(0,0, r^2)`,
`nuclear = -matrix_element(0,0, r)`, `finite_nuclear =
matrix_element(0,0, r^2*V(r), {R_nuc})`, etc.

## Two-electron integrals

`twoe_integral(L, iel)` (the `r<^L/r>^(L+1)` radial Green's function per
multipole) stays a **libhelfem** primitive — it's the 2-coordinate analogue of
the matrix element, and its cusp on the diagonal is exactly a breakpoint case.
What is chemistry and moves to **libhelfemqc**: which multipoles couple, the
Gaunt angular factors, Coulomb-vs-exchange pairing, the erfc/Yukawa kernel
choice, and the Fock build.

## Phasing (one PR each)

1. **Make `matrix_element` the single 1e assembly path.** Rewrite
   `RadialBasis`'s `overlap/kinetic/nuclear/potential/...` as thin `(k,l,f)`
   callers; delete the duplicates. Bit-exactness gate against master.
2. **Auto-convergence.** Add the degree-aware / order-refining / breakpoint-aware
   evaluator; make the QC callers pass breakpoints. Gate: existing energies
   bit-identical at double (exact-order path), and a known-hard element
   (finite-nucleus, 2e cusp) converges to eps(T) at double **and** gains digits
   at quad.
3. **Split `libhelfemqc`. (DONE)** Move the concrete nucleus models,
   `CoulombExchangeFE`, `NAO`/`GTO`/`STO` up into a new `libhelfemqc` target;
   `libhelfem` keeps FEM + evaluators plus the domain-neutral pieces the
   evaluators depend on (the `ModelPotential` abstract base, `RadialPotential`,
   and `erfc_expn`). `helfem-common` links `libhelfemqc`. Downstream
   `find_package` consumer test passes.
4. **Fold `lib1dfem` into `libhelfem`.** Collapse the base-primitive library into
   the FEM library; one general target.

## Open questions / risks

- **Breakpoint plumbing.** Every non-smooth `f` must carry its breakpoints. Are
  they always known analytically (nucleus radius, `r=r'`)? Yes for the current
  set; a general library should still allow a fallback adaptive bisection when
  they are not.
- **Naming.** `libhelfem` (general) + `libhelfemqc` (chemistry) as agreed. The
  `helfem::` C++ namespace is shared; the QC pieces likely want a
  `helfem::qc::` sub-namespace to mirror the target split.
- **Performance.** The exact-order short-circuit must cover the SCF hot path so
  auto-convergence never adds cost where the integrand is polynomial.
- **`erfc_expn` namespace.** ~~Still lives in `atomic::`; rename as part of the
  QC move.~~ Done in Phase 3: renamed `helfem::atomic::erfc_expn` ->
  `helfem::erfc_expn`. The function itself stayed in `libhelfem` (general
  screened-Coulomb Green's function), it was not moved to the QC layer.
