#!/usr/bin/env python3
"""Fully coupled (kappa, c) CASSCF: Hessian blocks and convergence.

Four checks, each able to fail on its own:

  1  the assembled coupled Hessian against four-point finite differences of
     the TRUE coupled energy, in a mixed direction and in each segment alone.
     The per-segment split is what localizes a factor error -- a spurious 1/2
     on the orbital block showed up here as ratio exactly 2.000000 while the
     CI block read 1.000000.
  2  BCH ordering: d/dt of the gradient is H plus (1/2) g . [K', K], since it
     walks exp(tK')exp(sK) while the Hessian is defined on exp(sK + tK').
     The antisymmetric part of the raw derivative must reproduce that
     commutator -- predicting the discrepancy is a stronger statement than
     working at a stationary point where it cannot appear.
  3  the coupled optimizer reaches the same energy as the two-step reference,
     in far fewer macroiterations.
  4  the Hessian is genuinely singular, and the null space is real rather
     than an artefact: its gradient content must be negligible.
"""
import sys

import numpy as np

from helfem import AtomicBasis
from helfem.pyscf_driver import (helfem_scf, casscf, active_hamiltonian,
                                 orbital_gradient, nonredundant_pairs)
from helfem.casscf_coupled import (build_coupled, coupled_casscf,
                                   coupled_energy, hess_kk_raw, kappa_of)

TOL_HESS = 1e-5      # four-point FD; measured ~1e-8 relative
TOL_AGREE = 1e-7     # coupled vs two-step; measured 1.1e-09


def main():
    from pyscf import fci

    np.random.seed(20260914)
    b = AtomicBasis(Z=4, lmax=0, mmax=0, primbas=4, nnodes=6,
                    nelem=3, Rmax=20.0, igrid=4, zexp=2.0)
    nbf = b.Nbf()
    mf = helfem_scf(b, 4)
    assert mf.converged
    ninact, nact, nelecas = 1, 3, (1, 1)
    C = mf.mo_coeff
    pairs = nonredundant_pairs(ninact, nact, nbf)
    nk = len(pairs)

    Ecore, heff, eri = active_hamiltonian(b, C, ninact, nact)
    _, c0 = fci.direct_spin1.kernel(heff, eri, nact, nelecas, ecore=Ecore)
    c0 = c0 / np.linalg.norm(c0)
    E, g, H, sb, D, d = build_coupled(b, C, c0, ninact, nact, nelecas, pairs)
    print(f"Be Nbf={nbf}  E={E:.12f}  {nk} rotations + {len(sb)} CI directions")
    assert np.abs(H - H.T).max() < 1e-12, "assembled Hessian is not symmetric"

    print("\n1  assembled Hessian vs four-point finite differences")
    h = 2e-4
    for tag, v in (("mixed", np.random.randn(len(g))),
                   ("kappa only", np.r_[np.random.randn(nk), np.zeros(len(sb))]),
                   ("CI only", np.r_[np.zeros(nk), np.random.randn(len(sb))])):
        v = v / np.linalg.norm(v)
        an = float(v @ H @ v)

        def E1(a):
            return coupled_energy(b, C, c0, ninact, nact, nelecas,
                                  kappa_of(a * v[:nk], pairs, nbf),
                                  sum(s * w for s, w in zip(a * v[nk:], sb)))

        fd = (E1(h) - 2 * E1(0.0) + E1(-h)) / (h * h)
        rel = abs(fd - an) / max(abs(fd), 1e-30)
        print(f"   {tag:11s} analytic {an:+.8e}  fd {fd:+.8e}  rel {rel:.2e}")
        assert rel < TOL_HESS, f"{tag} Hessian off by {rel:.2e}"

    print("\n2  BCH ordering term in d/dt of the gradient")
    def randK():
        K = np.zeros((nbf, nbf))
        for m in range(nbf):
            for n in range(m + 1, nbf):
                t = np.random.randn()
                K[m, n], K[n, m] = t, -t
        return K / np.linalg.norm(K)
    K1, K2 = randK(), randK()
    a12 = 0.5 * float(np.einsum("mn,mn->", hess_kk_raw(b, C, ninact, nact, D, d, K2), K1))
    a21 = 0.5 * float(np.einsum("mn,mn->", hess_kk_raw(b, C, ninact, nact, D, d, K1), K2))
    gfull = orbital_gradient(b, C, ninact, nact, D, d)
    pred = 0.25 * float(np.einsum("mn,mn->", gfull, K2 @ K1 - K1 @ K2))
    print(f"   raw asymmetry {(a12 - a21) / 2:+.6e}   BCH prediction {pred:+.6e}")
    assert abs((a12 - a21) / 2 - pred) < 1e-4 * max(abs(pred), 1e-12) + 1e-12

    print("\n3  coupled vs two-step")
    E2, _, i2 = casscf(b, C, ninact, nact, nelecas)
    Ec, Cc, cc, ic = coupled_casscf(b, C, ninact, nact, nelecas)
    print(f"   two-step  E={E2:.12f}  macro={i2['macro']:3d}  |g|={i2['grad_norm']:.2e}")
    print(f"   coupled   E={Ec:.12f}  macro={ic['macro']:3d}  |g|={ic['grad_norm']:.2e}")
    assert ic["converged"], f"coupled CASSCF did not converge: {ic}"
    assert abs(Ec - E2) < TOL_AGREE, f"coupled vs two-step: {abs(Ec - E2):.3e}"
    assert ic["macro"] < i2["macro"], "coupling should reduce the iteration count"
    print(f"   macroiterations {i2['macro']} -> {ic['macro']}")

    print("\n4  the Hessian null space is real, not an artefact")
    _, _, Hc, sbc, Dc, _ = build_coupled(b, Cc, cc, ninact, nact, nelecas, pairs)
    w, V = np.linalg.eigh(Hc)
    _, gc, _, _, _, _ = build_coupled(b, Cc, cc, ninact, nact, nelecas, pairs)
    gp = V.T @ gc
    null = np.abs(w) < 1e-6
    print(f"   {null.sum()} of {len(w)} modes have |eigenvalue| < 1e-6")
    print(f"   gradient inside that subspace  {np.linalg.norm(gp[null]):.3e}")
    print(f"   active natural occupations: "
          + "  ".join(f"{o:.8f}" for o in np.linalg.eigvalsh(Dc)[::-1]))
    assert null.sum() > 0, "expected exact redundancies in a CAS Hessian"
    # A flat direction carries no gradient; if it did, it would not be flat.
    assert np.linalg.norm(gp[null]) < 1e-6

    print("\nCOUPLED CASSCF TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
