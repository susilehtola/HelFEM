"""build_active_eri must REFUSE complex orbitals, not silently symmetrise them.

HelFEM's atomic basis uses complex spherical harmonics, so an orbital with
m != 0 is complex, and for it (tu|vw) != (ut|vw). This package stores a real
8-fold tensor because that is what PySCF takes, and so cannot represent such a
pair -- see REAL ORBITALS ONLY in helfem/pyscf_driver.py. Before the guard it
symmetrised the pair density and carried on, which is exact only for m = 0 and
was invisible to every other test here, all of which run at lmax = 0.

Three checks, each of which the unguarded code gets wrong in a different way:

  1. a pure m = +1 orbital is refused,
  2. a pure m = 0 orbital, in the SAME lmax = 1 basis, is accepted -- so the
     guard tests the orbitals, not merely whether the basis has m != 0 in it,
  3. and the accepted numbers are unchanged: the tensor equals the old
     symmetrised construction to roundoff. Not to the bit -- the two call
     coulomb() on different arguments, and linearity holds only in exact
     arithmetic -- so the bar is 1e-14 relative.
"""

import numpy as np

from helfem import AtomicBasis
from helfem.pyscf_driver import ComplexOrbitalError, build_active_eri


def pure_channel_orbitals(basis, m, count):
    """`count` orbitals living entirely in one angular channel with that m.

    Solved within the channel, so they are genuine eigenfunctions of that
    channel's one-electron Hamiltonian rather than arbitrary vectors.
    """
    nrad, nang = basis.Nrad(), basis.Nang()
    assert basis.Nbf() == nang * nrad, "expected contiguous per-channel blocks"
    ch = [i for i, mv in enumerate(basis.mvals()) if mv == m][0]
    sl = slice(ch * nrad, (ch + 1) * nrad)
    H, S = basis.hcore()[sl, sl], basis.overlap()[sl, sl]
    # generalised eigenproblem via S^{-1/2}
    w, V = np.linalg.eigh(S)
    X = V @ np.diag(w ** -0.5) @ V.T
    _, U = np.linalg.eigh(X @ H @ X)
    C = np.zeros((basis.Nbf(), count))
    C[sl, :] = X @ U[:, :count]
    return C


def old_symmetrised(basis, C):
    """The pre-guard construction, kept here only as the reference."""
    n = C.shape[1]
    eri = np.empty((n, n, n, n))
    for t in range(n):
        for u in range(t, n):
            P = np.outer(C[:, t], C[:, u])
            blk = 0.5 * (C.T @ basis.coulomb(P + P.T) @ C)
            eri[t, u] = blk
            eri[u, t] = blk
    return eri


def main():
    b = AtomicBasis(Z=4, lmax=1, mmax=1, primbas=4, nnodes=6, nelem=3,
                    Rmax=20.0, igrid=4, zexp=2.0)
    print(f"Be lmax=1: Nbf={b.Nbf()}  channels m={list(b.mvals())}")
    nfail = 0

    # 1. a complex orbital pair must be refused
    Cm0 = pure_channel_orbitals(b, 0, 2)
    Cp1 = pure_channel_orbitals(b, +1, 1)
    try:
        build_active_eri(b, np.hstack([Cm0, Cp1]))
        print("  m=+1 orbital in the active space     FAIL (was accepted)")
        nfail += 1
    except ComplexOrbitalError as e:
        print(f"  m=+1 orbital in the active space     ok (refused: {str(e)[:60]}...)")

    # 2. real orbitals in the same basis must be accepted
    try:
        eri = build_active_eri(b, Cm0)
        print("  m=0 orbitals, same lmax=1 basis      ok (accepted)")
    except ComplexOrbitalError:
        print("  m=0 orbitals, same lmax=1 basis      FAIL (refused)")
        nfail += 1
        eri = None

    # 3. ...and with the numbers unchanged
    if eri is not None:
        ref = old_symmetrised(b, Cm0)
        rel = np.max(np.abs(eri - ref)) / np.max(np.abs(ref))
        ok = rel < 1e-14
        print(f"  same as the old construction         {'ok' if ok else 'FAIL'}"
              f" (relative diff {rel:.1e})")
        nfail += 0 if ok else 1

    print("REAL ORBITAL GUARD TEST " + ("FAILED" if nfail else "PASSED"))
    raise SystemExit(1 if nfail else 0)


if __name__ == "__main__":
    main()
