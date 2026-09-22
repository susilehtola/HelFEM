#!/usr/bin/env python3
"""Validate the CASSCF orbital gradient against the true CAS energy.

The gradient at fixed CI coefficients is g = 2(F - F^T) with the generalized
Fock matrix of helfem.pyscf_driver.generalized_fock. It is checked here by
central-differencing `frozen_ci_energy`, which rebuilds the CAS energy from
`active_hamiltonian` and so shares no algebra with the gradient formula.

That separation is the point. CLAUDE.md records the standing trap: a derivative
check that differentiates one expression against finite differences of *itself*
agrees perfectly with a consistently wrong one, so it validates the derivative
and not the physics.

Four checks:
  1  every rotation class matches finite differences, including the redundant
     ones -- at frozen CI an active-active rotation does change the energy
  2  a unit-norm direction through the whole space, Richardson-extrapolated.
     (Normalizing matters: the central-difference truncation error scales as
     ||K||^3 h^2, so an unnormalized random direction fails at 6e-2 for
     entirely uninteresting reasons.)
  3  at a converged CASCI the frozen-CI gradient equals the CI-relaxed one,
     because the CI is variational
  4  the full active space reproduces the FCI energy in that space
"""
import sys

import numpy as np
from scipy.linalg import expm

from helfem import AtomicBasis
from helfem.pyscf_driver import (helfem_scf, active_hamiltonian, casci,
                                 build_active_eri, frozen_ci_energy,
                                 orbital_gradient)

TOL_PAIR = 1e-6         # measured worst 1.1e-8, FD-noise limited
TOL_RICH = 1e-8         # measured 1.5e-10
TOL_RELAX = 1e-8        # measured 2.5e-11
STEP = 1e-4


def main():
    from pyscf import fci

    np.random.seed(20260914)
    b = AtomicBasis(Z=4, lmax=0, mmax=0, primbas=4, nnodes=6,
                    nelem=3, Rmax=20.0, igrid=4, zexp=2.0)
    nbf = b.Nbf()
    mf = helfem_scf(b, 4)                     # Be, 1s^2 2s^2
    assert mf.converged, "reference RHF did not converge"
    C = mf.mo_coeff
    print(f"Be  Nbf={nbf}  RHF={mf.e_tot:.12f}")

    ninact, nact, nelecas = 1, 3, (1, 1)
    E0, heff, eri = active_hamiltonian(b, C, ninact, nact)
    E_cas, civec = fci.direct_spin1.kernel(heff, eri, nact, nelecas, ecore=E0)
    D, d = fci.direct_spin1.make_rdm12(civec, nact, nelecas)
    print(f"CAS(2,{nact}) ninact={ninact}: {E_cas:.12f}")

    # the frozen-CI energy must reproduce the CASCI energy at the reference
    E_frozen = frozen_ci_energy(b, C, ninact, nact, D, d)
    assert abs(E_frozen - E_cas) < 1e-12, f"frozen-CI energy off by {E_frozen-E_cas:.3e}"

    g = orbital_gradient(b, C, ninact, nact, D, d)
    assert np.abs(g + g.T).max() < 1e-14, "gradient is not antisymmetric"
    print(f"|g|max = {np.abs(g).max():.3e}")

    nocc = ninact + nact
    nvirt_probe = min(nocc + 3, nbf)
    classes = {
        "inactive-active":  [(i, nocc - nact + t)
                             for i in range(ninact) for t in range(nact)],
        "inactive-virtual": [(i, v) for i in range(ninact)
                             for v in range(nocc, nvirt_probe)],
        "active-virtual":   [(ninact + t, v) for t in range(nact)
                             for v in range(nocc, nvirt_probe)],
        "active-active":    [(ninact + t, ninact + u)
                             for t in range(nact) for u in range(t + 1, nact)],
    }
    print("\n1  finite differences of the true frozen-CI energy")
    for name, pairs in classes.items():
        worst = 0.0
        for (m, n) in pairs:
            K = np.zeros((nbf, nbf))
            K[m, n], K[n, m] = 1.0, -1.0
            fd = (frozen_ci_energy(b, C @ expm(+STEP * K), ninact, nact, D, d)
                  - frozen_ci_energy(b, C @ expm(-STEP * K), ninact, nact, D, d)) / (2 * STEP)
            worst = max(worst, abs(fd - g[m, n]))
        print(f"   {name:18s} worst |fd - analytic| = {worst:.3e}")
        assert worst < TOL_PAIR, f"{name} gradient off by {worst:.3e}"

    print("\n2  unit-norm direction through the whole space, Richardson")
    K = np.zeros((nbf, nbf))
    for m in range(nbf):
        for n in range(m + 1, nbf):
            v = np.random.randn()
            K[m, n], K[n, m] = v, -v
    K /= np.linalg.norm(K)
    analytic = 0.5 * np.einsum("mn,mn->", g, K)

    def fd(h):
        return (frozen_ci_energy(b, C @ expm(+h * K), ninact, nact, D, d)
                - frozen_ci_energy(b, C @ expm(-h * K), ninact, nact, D, d)) / (2 * h)

    f1, f2 = fd(2 * STEP), fd(STEP)
    rich = f2 + (f2 - f1) / 3.0
    print(f"   analytic   {analytic:+.12e}")
    print(f"   h={2*STEP:.0e}    {f1:+.12e}  err {abs(f1-analytic):.3e}")
    print(f"   h={STEP:.0e}    {f2:+.12e}  err {abs(f2-analytic):.3e}")
    print(f"   Richardson {rich:+.12e}  err {abs(rich-analytic):.3e}")
    assert abs(f1 - analytic) > abs(f2 - analytic), "error should fall with h"
    assert abs(rich - analytic) < TOL_RICH, f"direction off by {abs(rich-analytic):.3e}"

    print("\n3  CI relaxation must not move the gradient at a converged CASCI")
    def relaxed(kappa):
        e0, hh, ee = active_hamiltonian(b, C @ expm(kappa), ninact, nact)
        return fci.direct_spin1.kernel(hh, ee, nact, nelecas, ecore=e0)[0]
    m, n = 0, ninact
    K1 = np.zeros((nbf, nbf))
    K1[m, n], K1[n, m] = 1.0, -1.0
    fd_rel = (relaxed(+STEP * K1) - relaxed(-STEP * K1)) / (2 * STEP)
    print(f"   pair ({m},{n}): relaxed fd {fd_rel:+.10e}  frozen analytic "
          f"{g[m, n]:+.10e}  diff {abs(fd_rel-g[m, n]):.3e}")
    assert abs(fd_rel - g[m, n]) < TOL_RELAX

    print("\n4  full active space reproduces FCI in that space")
    E_full = casci(b, C, 0, nbf, 4)[0]
    E_fci = fci.direct_spin1.kernel(C.T @ b.hcore() @ C,
                                    build_active_eri(b, C),
                                    nbf, (2, 2), ecore=0.0)[0]
    print(f"   CAS(4,{nbf}) {E_full:.12f}   FCI {E_fci:.12f}   "
          f"diff {abs(E_full-E_fci):.3e}")
    assert abs(E_full - E_fci) < 1e-10

    print("\nCAS GRADIENT TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
