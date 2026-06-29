#!/usr/bin/env python3
"""Exhaustive radial DF round-trip: all (i, j, m, n) quadruples for a
small He basis, with intra-element vs inter-element error split.

Uses lmax=0 (s-shell only) so the angular-summed basis.coulomb yields
the bare R^0 directly (Gaunt(0,0,0,0,0,0)^2 * 4*pi/(2*0+1) = 1). For
every (m, n), one J call gives R^0(*, *, m, n) as a Nrad x Nrad matrix.

Splits errors into:
  - intra-element: (i, j) and (m, n) both in the SAME element
  - inter-element: (i, j) and (m, n) in DIFFERENT elements
  - mixed:         (i, j) spans element boundary (or both pairs touch
                   different element sets via shared boundary func)
"""
import sys
import numpy as np

from helfem import AtomicBasis


def element_set(basis, idx):
    """Return frozenset of element indices that contain radial index idx.
    Boundary-shared functions belong to multiple elements."""
    Nel = basis.Nel()
    elems = set()
    for iel in range(Nel):
        first, last = basis.radial_element_range(iel)
        if first <= idx <= last:
            elems.add(iel)
    return frozenset(elems)


def run(label, primbas, nnodes, nelem, Rmax=10.0):
    print(f"\n=== Exhaustive radial DF round-trip ({label}) ===")
    basis = AtomicBasis(Z=2, lmax=0, mmax=0,
                        primbas=primbas, nnodes=nnodes, nelem=nelem, Rmax=Rmax)
    Nrad = basis.Nrad()
    Nel = basis.Nel()
    print(f"  Nrad = {Nrad}, Nel = {Nel}")
    for iel in range(Nel):
        first, last = basis.radial_element_range(iel)
        print(f"  element {iel}: indices [{first}..{last}] (size {last-first+1})")

    # Per-radial-index element membership
    elem_sets = [element_set(basis, i) for i in range(Nrad)]
    print(f"  element sets per radial idx:")
    for i in range(Nrad):
        marker = " <- BOUNDARY (shared)" if len(elem_sets[i]) > 1 else ""
        print(f"    idx {i}: elements {sorted(elem_sets[i])}{marker}")

    # --- Reference R^0(*, *, m, n) via the J helper at lmax=0 ---
    # For lmax=0 s-shell, basis.coulomb(P) = R^0(P) directly.
    print("\n=== Computing reference R^0 tensor (Nrad^4 entries) ===")
    R0_ref = np.zeros((Nrad, Nrad, Nrad, Nrad))
    for m in range(Nrad):
        for n in range(m, Nrad):
            P = np.zeros((Nrad, Nrad))
            if m == n:
                P[m, m] = 1.0
            else:
                P[m, n] = 0.5
                P[n, m] = 0.5
            J = basis.coulomb(P)   # = R^0(*, *, m, n) for lmax=0 s-shell
            R0_ref[:, :, m, n] = J
            if m != n:
                R0_ref[:, :, n, m] = J
    print(f"  R0_ref filled (max |R0| = {np.max(np.abs(R0_ref)):.4e})")

    # --- DF reconstruction ---
    print("\n=== Computing radial DF factors ===")
    B = basis.radial_df_factors(tol=1e-12)
    print(f"  Got {len(B)} multipoles, naux_0 = {B[0].shape[0]}")

    # Σ_Q B[0][Q, i, j] * B[0][Q, m, n] for all (i, j, m, n)
    # einsum: 'Qij,Qmn->ijmn'
    print("=== Reconstructing R^0 via einsum and computing errors ===")
    R0_recon = np.einsum('Qij,Qmn->ijmn', B[0], B[0])
    diff = R0_recon - R0_ref
    abs_diff = np.abs(diff)
    print(f"  Max abs error overall: {abs_diff.max():.4e}")

    # --- Split errors by intra vs inter element ---
    print("\n=== Error breakdown by pair-element relationship ===")
    intra_max = 0.0
    inter_max = 0.0
    intra_count = 0
    inter_count = 0
    intra_worst = None
    inter_worst = None
    for i in range(Nrad):
        for j in range(Nrad):
            ij_elems = elem_sets[i] & elem_sets[j]
            if not ij_elems:
                continue   # cross-element (i,j) pair has zero density
            for m in range(Nrad):
                for n in range(Nrad):
                    mn_elems = elem_sets[m] & elem_sets[n]
                    if not mn_elems:
                        continue
                    err = abs_diff[i, j, m, n]
                    # Both pairs share an element?
                    if ij_elems & mn_elems:
                        intra_count += 1
                        if err > intra_max:
                            intra_max = err
                            intra_worst = (i, j, m, n,
                                           R0_ref[i, j, m, n],
                                           R0_recon[i, j, m, n])
                    else:
                        inter_count += 1
                        if err > inter_max:
                            inter_max = err
                            inter_worst = (i, j, m, n,
                                           R0_ref[i, j, m, n],
                                           R0_recon[i, j, m, n])

    print(f"  Intra-element (pairs share an element): {intra_count} quadruples")
    print(f"    max err = {intra_max:.4e}")
    if intra_worst:
        i, j, m, n, ref, recon = intra_worst
        print(f"    worst: R^0({i},{j},{m},{n})  ref={ref:+.6e}  recon={recon:+.6e}")
    print(f"  Inter-element (no shared element):     {inter_count} quadruples")
    print(f"    max err = {inter_max:.4e}")
    if inter_worst:
        i, j, m, n, ref, recon = inter_worst
        print(f"    worst: R^0({i},{j},{m},{n})  ref={ref:+.6e}  recon={recon:+.6e}")

    print(f"\nSummary: intra max err {intra_max:.2e}, inter max err {inter_max:.2e}")

    if inter_max > 1e-9 or intra_max > 1e-9:
        print("FAIL")
        return 1
    print("PASS")
    return 0


def main():
    # LIP basis with shared boundary (noverlap=1)
    rc = run("LIP, nelem=2, nnodes=6", primbas=4, nnodes=6, nelem=2)
    if rc: return rc
    # HIP basis with 2 shared boundary functions (noverlap=2)
    rc = run("HIP, nelem=2, nnodes=6", primbas=5, nnodes=6, nelem=2)
    if rc: return rc
    # Larger basis stress test
    rc = run("LIP, nelem=4, nnodes=8 (Nrad~28)", primbas=4, nnodes=8, nelem=4, Rmax=15.0)
    if rc: return rc
    print("\nALL EXHAUSTIVE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
