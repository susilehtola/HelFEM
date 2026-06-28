#!/usr/bin/env python3
"""Smoke test for the HelFEM Python bindings.

Builds an atomic basis for He (Z=2, l=0), computes the matrix elements,
and verifies the lowest H 1s eigenvalue (bare T+V on the single-electron
H Hamiltonian) matches the analytic -0.5 Eh.
"""
import numpy as np
import sys

import helfem

def main():
    # Z=1 H atom, l=0 only. Modest FE basis -- 15-LIP, 20 elements.
    basis = helfem.AtomicBasis(
        Z=1, lmax=0, mmax=0,
        primbas=4, nnodes=15, nelem=20, Rmax=40.0,
    )
    print(f"AtomicBasis(Z=1, l=m=0): Nbf={basis.Nbf()} (Nrad={basis.Nrad()}, Nang={basis.Nang()})")
    print(f"  lvals = {basis.lvals()}")
    print(f"  mvals = {basis.mvals()}")

    S = basis.overlap()
    T = basis.kinetic()
    V = basis.nuclear()
    H = T + V
    print(f"  S shape = {S.shape}, dtype = {S.dtype}")

    # Generalised symmetric eigenproblem H c = e S c via canonical
    # orthogonalisation.
    sval, svec = np.linalg.eigh(S)
    Sinvh = svec @ np.diag(1.0 / np.sqrt(sval)) @ svec.T
    Hp = Sinvh.T @ H @ Sinvh
    eval_h, _ = np.linalg.eigh(Hp)
    print(f"  H 1s eigenvalue = {eval_h[0]:.10f}  (analytic -0.5)")
    assert abs(eval_h[0] + 0.5) < 1e-9, "H 1s eigenvalue off"

    # Now a He calc: J / K with a density.
    print()
    he = helfem.AtomicBasis(
        Z=2, lmax=0, mmax=0,
        primbas=4, nnodes=15, nelem=20, Rmax=40.0,
    )
    print(f"AtomicBasis(Z=2, l=m=0): Nbf={he.Nbf()}")
    # Toy density: 2 * |c_1s><c_1s| using the bare H+ 1s as an initial guess.
    S2 = he.overlap(); T2 = he.kinetic(); V2 = he.nuclear()
    Hb = T2 + V2
    sv2, vc2 = np.linalg.eigh(S2)
    Sinvh2 = vc2 @ np.diag(1.0 / np.sqrt(sv2)) @ vc2.T
    _, evec = np.linalg.eigh(Sinvh2.T @ Hb @ Sinvh2)
    Cmo = Sinvh2 @ evec
    P = 2.0 * np.outer(Cmo[:, 0], Cmo[:, 0])

    J, K = he.get_jk(P)
    print(f"  ||J||_F = {np.linalg.norm(J):.6f}")
    print(f"  ||K||_F = {np.linalg.norm(K):.6f}")
    # Hartree energy: 1/2 * trace(P @ J)
    print(f"  E_J = {0.5*np.trace(P @ J):.6f}")
    # Exchange (HF): -1/2 * trace(P @ K)  (note PySCF sign convention: F = h + J - K)
    print(f"  E_K = {-0.5*np.trace(P @ K):.6f}")

    print()
    print("SMOKE TEST PASSED")
    return 0

if __name__ == "__main__":
    sys.exit(main())
