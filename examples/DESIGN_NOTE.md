# Per-multipole radial J/K accessor — design note

## Problem

An external consumer (e.g. a BSD-licensed atomic-SCF library wanting to use
HelFEM as a "numerical atomic orbital" backend) needs per-multipole radial
Coulomb and exchange matrices:

```
radial_J(k, la, lb, D)_ij  =  Σ_mn  D_mn  R^k(la_i la_j ; lb_m lb_n)
radial_K(k, la, lb, D)_ij  =  Σ_mn  D_mn  R^k(la_i lb_m ; la_j lb_n)
```

where `R^k` is the Slater radial integral (no Condon–Shortley `a^k` / `b^k`
angular factor — the consumer applies that). `la, lb` are angular-momentum
labels; `D` is a per-block density matrix in radial indices.

Inside HelFEM the per-multipole-`L` radial contraction already exists — it
sits between the Gaunt-coefficient angular sums in
`src/atomic/TwoDBasis.cpp::coulomb()` (lines 879–931) and `::exchange()`
(lines 1009–1098). The consumer just needs it surfaced without going
through the Gaunt sums.

## Recommendation: small public accessor on TwoDBasis

Add two thin public methods to `helfem::atomic::basis::TwoDBasis`:

```cpp
/// Per-multipole radial Coulomb: given the (L,M)-aggregated density
/// Paux_LM (Nrad x Nrad), return the (L,M) radial Coulomb matrix
/// (Nrad x Nrad) — i.e. the inner contraction of TwoDBasis::coulomb()
/// without the outer Gaunt sums.
arma::mat radial_coulomb(int L, const arma::mat & Paux_LM) const;

/// Per-multipole radial exchange between angular blocks (la, lb):
/// given a single (la, lb) density block D (Nrad x Nrad), return the
/// (la, lb) contribution to the radial K matrix at multipole L — i.e.
/// the inner contraction of TwoDBasis::exchange() without the outer
/// Gaunt sums.
arma::mat radial_exchange(int L, int la, int lb, const arma::mat & D) const;
```

Both methods refactor existing inner loops, so the implementation is a
straightforward extract — no new numerical machinery. Approximate cost: ~80
lines of code, no header churn beyond the two prototypes.

The consumer would then assemble its Fock matrix as

```cpp
for (int L = 0; L <= Lmax; ++L) {
  arma::mat Paux_L = aggregate_density(L, /*all (la,lb) blocks*/);
  arma::mat J_L = basis.radial_coulomb(L, Paux_L);
  // ... distribute J_L back to per-(la,lb) blocks with consumer's a^k weights
  for (auto const & [la, lb] : block_pairs) {
    arma::mat K_Llalb = basis.radial_exchange(L, la, lb, D[la][lb]);
    // ... multiply by consumer's b^k(la, lb, L) angular factor
  }
}
```

### Why this over the alternatives

**(a) Public accessor (recommended).** Smallest possible API extension; no
copy-paste of the radial-element-contraction code into the consumer.
TwoDBasis stays the single source of truth for FE radial 2e integrals.
Cost: ~80 LOC refactor inside TwoDBasis.cpp + 2 lines in TwoDBasis.h.

**(b) Compile-along.** The consumer pulls
`src/atomic/{TwoDBasis,basis}.cpp` plus the needed
`src/general/{gaunt,model_potential,scf_helpers}.cpp` into its own build.
Avoids modifying HelFEM. But: drags in HDF5 and libxc transitively (because
`src/general` deps), licensing situation is messier, and the consumer has
to track every internal API change. **Only recommended as a stop-gap.**

**(c) Expose a new `RadialIntegralEngine` class.** Cleanly separate the
radial-only assembly from `TwoDBasis` (which carries angular state too).
This is the right long-term shape, especially once libhelfem grows the
`RadialBasis` abstract base + `FEMRadialBasis`/`NAORadialBasis` subclasses
in the planned refactor (see the Phase 2 plan). For now (a) is a
better-cost/value drop-in.

## Validation hook

The simplest sanity check is built into
`examples/atomic_radial_export.cpp`: configure `TwoDBasis` with a
single-`l` shell (`lval = {l}, mval = {0}`), and `coulomb(P)` / `exchange(P)`
reduce to the monopole (k=0) radial J/K. The example runs He HF this way
and matches the reference HF energy −2.86167999561 Eh to ~3 ×10⁻¹².

Once `radial_coulomb`/`radial_exchange` land, the existing TwoDBasis
`coulomb()`/`exchange()` should be re-implemented in terms of them with no
numerical change — that gives a regression test for free.

## ILP64 / LP64 — sharp edge for any consumer

Independent of which accessor lands, document prominently that consumers
must build with `-DARMA_64BIT_WORD -DARMA_BLAS_LONG` and link the same
64-bit BLAS (e.g. `libflexiblas64`) as the installed libhelfem. The
consumer's original "free(): invalid next size" report was an LP64/ILP64
mismatch inside LAPACK, not a HelFEM bug.

This belongs in a top-level `INSTALL.md` paragraph, not just the example.
