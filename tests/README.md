# HelFEM integration tests

Regression tests for the `atomic`, `diatomic`, `gensap` (spherically-averaged
atom) and `aij` (atom-in-jellium) drivers, covering the four axes that
matter:

1. spin-restricted and spin-unrestricted
2. quick s-only systems (H, He, H2) and harder ones with p/pi orbitals
   (Ne, Ar, Kr, BH, CH, NH, OH, FH, N2, CO)
3. with and without orbital symmetry
4. HF, LDA, GGA, mGGA (both tau- and laplacian-dependent) and hybrids

`cases.json` is data; `run_tests.py` is independent of what it contains. Add
or remove systems by editing the JSON.

## Running

    tests/run_tests.py --generate refs/mine.json      # record what this build does
    tests/run_tests.py --check    refs/mine.json      # compare against a recorded set

    tests/run_tests.py --check refs/mine.json --tag quick      # subset by tag
    tests/run_tests.py --check refs/mine.json --skip-slow
    tests/run_tests.py --check refs/mine.json -j 6             # cases run concurrently

`--bindir` defaults to `objdir/src`.

## What is compared, and why

The drivers print a full decomposition:

    kinetic ... nuclear ... Coulomb ... XC ... Exx ... total ... (nel err ...)

Every component is recorded, not just the total, because the component that
moved localises the fault: XC drifting while kinetic holds still points at the
DFT grid; kinetic drifting points at the basis; `nel err` drifting points at
the quadrature.

Converged **occupations** are recorded separately and reported as a distinct
kind of failure. A system with near-degenerate solutions can converge to a
different *state* rather than to a wrong *number* (task #45), and those two
failures need different responses — one is physics, the other is a bug.

## Determinism: read this before believing a failure

Converged energies are a deterministic function of `OMP_NUM_THREADS`. They are
bit-reproducible at a fixed thread count and can differ *between* thread
counts, because the OpenMP reduction over the XC energy sums in a different
order (~1e-16) and a near-degenerate system amplifies that into a different
SCF solution. Measured on atomic O/PBE: 2 threads gives -74.7827460224,
6 threads gives -74.7707877743 — a 0.012 Eh gap, entirely reproducible on
each side.

Consequences:

- every case runs at `OMP_NUM_THREADS=1`, recorded in the reference metadata;
- `--jobs` parallelises across *cases*, never within one, so it cannot change
  a result;
- before calling a difference a regression, confirm the thread count matched.

The same applies to the BLAS. This machine uses FlexiBLAS, a runtime
dispatcher whose backend can change without anything in the repo changing, so
the reference metadata records it too.

## Basis sets: everything is at the CBS limit

**Every case runs at a converged basis.** HelFEM exists to produce
basis-set-limit numbers, so a suite pinned at some arbitrary small basis would
exercise a regime the code is not for, would miss faults living in the
auto-convergence machinery and the high-L integrals, and would leave every
reference value physically meaningless — not comparable to literature or to
another code.

For the **atomic** cases this factorises. The angular direction is not a
convergence question at all: the orbitals are L^2 eigenfunctions, so
`lmax = max occupied l` is *exact*, not truncated. Verified — Ne at
lmax = 1, 2, 3 agrees to 3e-10, and the Be unoccupied-p invariant asserts the
same fact. So lmax is 0 for H/He/Be/Na/Mg/K/Ca, 1 for N/O/Ne/P/Ar, 2 for
Cr/Zn/Kr, and only the radial grid needs converging: `nelem=5, nnodes=15`
reproduces the nelem = 10/15/20 limit exactly (He -2.8616799956,
Ne -128.5470981094). Field cases keep a polarisable lmax=2, since a field
mixes l.

Minimal lmax is also *cheaper*, because it stops carrying angular channels
that contribute nothing: Ne/LDA went from ~150 s at lmax=2 to 12 s at lmax=1.

For the **diatomic** cases the grids come from `diatomic_cbasis`, which
reports the converged element and partial-wave counts directly. Note `--lmax`
there takes a per-|m| list (`--lmax=11,7` means 11 partial waves for m=0, 7
for m=1, and two entries means mmax=1).

## Cross-version references

The interesting use is checking today's code against a known-good build:

    git worktree add /tmp/wt-old <commit>
    cmake -B /tmp/wt-old/bd -S /tmp/wt-old && cmake --build /tmp/wt-old/bd -j
    tests/run_tests.py --generate refs/old.json --bindir /tmp/wt-old/bd/src

    tests/run_tests.py --check refs/old.json          # against the current build

**The pre-Eigen anchor is `141d982`** (2026-06-29), the parent of
`073aa42 "v2: Eigen migration Phase 1"`.

Two caveats that make a strict comparison against it impossible, and which
should temper how its numbers are read:

- **It predates OpenOrbitalOptimizer entirely.** There is no OOO in its
  CMakeLists; it used the bespoke in-tree SCF drivers. So the comparison spans
  two rewrites at once — the linear algebra *and* the SCF solver.
- **The convergence controls differ.** The old CLI had `--convthr` / `--maxit`
  / `--diis*`; the current one has `--scfmethods`. The same convergence
  threshold cannot be requested from both.

So a cross-era comparison establishes *"both converge to the same stationary
point"*, not *"both produce the same number to 1e-10"*. Use a loose tolerance
(1e-6) across eras and lean on the occupation check to catch genuine
state changes. Within one era, 1e-8 or tighter is appropriate.

Every physics flag needed by `cases.json` (`Z, M, lmax, mmax, nelem, nnodes,
method, symmetry, restricted, nela, nelb`) exists unchanged in both eras, so
the case list itself needs no adaptation.

## Configuration-dependent tests: the three components of O 3P

Oxygen is 1s2 2s2 2p4, so at M=3 (nela=5, nelb=3) exactly one p orbital is
doubly occupied, and *which* one selects the M_L component of the 3P term.
`--readocc` pins the choice via an `occs.dat` of `(nocca, noccb, m)` rows
(requires `--symmetry=1`):

| doubly occupied | m=0   | m=+1  | m=-1  | M_L | HF energy       |
|-----------------|-------|-------|-------|-----|-----------------|
| p0              | `3 3` | `1 0` | `1 0` |  0  | -74.8175462069  |
| p+              | `3 2` | `1 1` | `1 0` | -1  | -74.8146025595  |
| p-              | `3 2` | `1 0` | `1 1` | +1  | -74.8146025595  |

**p+ and p- must be exactly degenerate** — the two are related by reflection,
so this is a symmetry identity with a known answer rather than a recorded
number. Measured agreement is to all ten printed digits, at HF and at PBE.
The p0 component sits 2.94 mEh away, which is the physical M_L splitting.

This is also the mechanism behind the O/PBE irreproducibility chased earlier:
unpinned, the SCF selects among these components depending on arbitrarily
small perturbations (thread count, for instance), which is why unpinned
open-shell oxygen makes a poor reference and a pinned one makes a good test.

## The +-|m| symmetry level (--symmetry=3)

`--symmetry=3` solves one block per |m| rather than one per m, with the
two-fold degeneracy folded into the block's maximum occupation. That puts
the constraint on the *occupations* as well as on the Fock matrix, which is
what the retired `--maverage` failed to do: it averaged F over +-m but left
Aufbau free to drop an odd pi electron wholly into one of two exactly
degenerate blocks, so the density broke the symmetry the Fock was forced to
have, and the SCF stalled with the DIIS error pinned at 1.8e-5.

Two kinds of test, because there are two things to get wrong.

**It must not change a closed-shell answer.** Ne (2p^6) through the diatomic
code fills m = -1, 0, +1, so the |m| blocks carry a real degeneracy -- but
the shell is closed and nothing breaks the symmetry, so grouping the blocks
must land on the same solution as solving each m separately. HF makes this
the test of the exchange +-m mirror (only armed at `--symmetry=3`); LDA
tests the pure-m XC grid's mirror instead. Measured 3e-10 apart at HF and
identical to all printed digits at LDA, hence the 1e-9 tolerance. H2 covers
the degenerate case where only m=0 exists, so the machinery must be an exact
no-op.

**It must converge where the old flag could not.** CH (2-Pi) has one pi
electron and no way to place it symmetrically unless the occupations are
constrained; at `--symmetry=3` it comes out as 0.5 in each of +-1. The
recorded -38.1696015354 is therefore *above* the symmetry-broken CH
solutions (-38.2817165629 at `--symmetry=1`) and is not comparable to them:
it is a different, cylindrically averaged state. The point of the test is
that it converges at all.

## Running the suite

`ctest` registers one test per case, so a failure names the case and the
timings show where the run went:

    ctest -j6              # quick tier: 49 cases + the unit binaries
    ctest -L unit          # the self-checking binaries alone, ~2 s
    ctest -R Ne-dummy      # one family
    ctest -N               # list everything, including what is disabled

Cases outside the quick tier are registered but DISABLED, so they appear
in `ctest -N` and are skipped by a plain run. Enable them with

    cmake -DHELFEM_SLOW_TESTS=ON <builddir> && ctest -j6

The case list is read from `cases.json` at configure time, so a new case
registers itself -- but you have to re-run cmake after editing the file.

Invariants compare cases against each other and therefore cannot live in
a per-case test; `integration.invariants` covers them for the quick tier.
The harness can still be driven directly, which is what CI does:

    python3 run_tests.py --check refs/ci.json --tag ci --bindir <builddir>/src -j4

## Cross-code invariants

Every element in this suite is closed-shell, single-s-electron or half-filled,
so its density is spherically symmetric. The spherical average that `gensap`
solves is therefore *exact* for these systems, not an approximation -- the two
codes are solving the same problem and must agree. Measured: Ne
-128.2334805607 and Cr -1042.2182811427 identical between `gensap` and
`atomic` to every printed digit, N to 2e-10 (convergence-limited, hence the
1e-8 tolerance).

Likewise `aij` with `--njellium=0` is the bare atom, and must reproduce
`gensap` exactly: -240.3560448251 from both for Al.

These are the strongest tests here. Every other case validates the code
against its own recorded past; these validate two independently written
solvers against each other, with no reference values involved at all. Cr is
the striking one -- a 24-electron transition metal, the full (l,m)-resolved
treatment against the spherically-averaged one, agreeing exactly.

## Invariants

`cases.json` declares sets of cases that must agree with *each other*,
independent of any reference:

- a closed-shell system gives the same energy restricted or polarized (one per
  closed-shell element in the set);
- the O 3P p+ and p- components are exactly degenerate;
- E(+Ez) = E(-Ez), and E(+Bz) = E(-Bz) with m-symmetric occupations;
- an angular momentum present in the basis but unoccupied does not change the
  energy: Be at `--lmax=0` equals pinned Be at `--lmax=2`.

These are the most valuable tests here, and for a reason worth stating.
Every *reference* case records what the code currently does; if the reference
was generated from broken code, the test enshrines the breakage. The
invariants are symmetry identities with answers known independently of any
build, so they can detect a fault that predates the first reference. They also
stay valid when reference values legitimately change.

## Pinning configurations

Systems with near-degenerate solutions need their configuration pinned, or the
SCF picks among the candidates according to arbitrarily small perturbations
and the recorded value is not reproducible. Be is the worked example: its
2s/2p near-degeneracy let an unpinned run settle on a p-contaminated solution
1.4e-6 away from the clean one.

Pin at the right granularity. `--symmetry=1` counts particles per m, and Be's
m=0 block holds 1s, 2s *and* 2p0 — so a per-m count cannot separate 2s from
2p0 and the degeneracy survives the pin. `--symmetry=2` counts per (l,m) and
does separate them. At `--lmax=0` there is no p in the basis at all, so
`--symmetry=1` is sufficient there.

Fields need pinning for the same reason: a field breaks the degeneracy that
Aufbau would otherwise resolve arbitrarily.

## What is deliberately absent

**Laplacian-dependent meta-GGAs** (BR89 and relatives). They are numerically
unstable and diverge in any reasonable calculation, so they cannot serve as
stable references — a run that fails to converge is not a regression signal.
`increment_mgga_lapl` therefore has no test coverage here; that is a known
gap, not an oversight.

## Continuous integration

Two workflows, split by runtime:

- `.github/workflows/ci.yml` — every push and pull request. Builds, then runs
  the `ci` tag: 26 cases that finish in minutes, chosen to still cover both
  spin modes, HF/LDA/GGA/mGGA/hybrid, a pinned configuration, a field and a
  diatomic. Six of the thirteen invariants are fully covered by this tier.
- `.github/workflows/weekly.yml` — Sundays, plus manual dispatch. The whole
  suite, including the heavy elements (Cr, Zn, Kr), the range-separated
  hybrids and the diatomics beyond H2.

Both run `--smoke`, not `--check`, and that is deliberate. A recorded energy
depends on the CPU and the BLAS, so a reference set generated on one machine
cannot be compared bit-for-bit on a GitHub runner; checking it there would
either fail spuriously or need a tolerance so loose it catches nothing. The
invariants have no such problem — they are symmetry identities that hold on
any machine to convergence precision — so CI enforces those, and additionally
fails on any crash or timeout.

Both workflows also upload the energies they produced as an artifact. That is
how a platform-matched reference set gets created: take the artifact from a
green run and promote it, rather than transcribing numbers by hand.
