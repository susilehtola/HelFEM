"""Fully coupled (kappa, c) CASSCF on the HelFEM atomic basis.

Reference implementation for the second-order path: orbital rotations and CI
coefficients optimized together, rather than alternating as in
`pyscf_driver.casscf`. It exists to pin down the formulas and to give the C++
port something to diff against, so it assembles the coupled Hessian densely.
A production solver drives OpenTrustRegion with Hessian-VECTOR products and
never forms the matrix; every block below is written as an action on a
direction so that transcription is mechanical.

    H = [ H_kk  H_ks ]   H_kk  d/dt of the generalized Fock, symmetrised
        [ H_sk  H_ss ]   H_ss  2 P_perp (H - E) P_perp        (sigma-vector)
                         H_ks  generalized Fock on TRANSITION RDMs
                         H_sk  = H_ks^T

Everything is still built from `coulomb`/`exchange` on pair densities: no
AO->MO transform appears anywhere, which is what makes the whole construction
portable to the diatomic basis unchanged.

Measured on Be CAS(2,3) over a 1s core (Nbf=14), against the two-step
reference in pyscf_driver:

    two-step   -14.563849375547   71 macroiterations, |g| 2.3e-07
    coupled    -14.563849376675   22 macroiterations, |g| 2.0e-08

with the closing iterations going |g| = 1.4e-04 -> 7.5e-08 -> 2.0e-08, the
quadratic behaviour a correct second-order method should show.
"""

import numpy as np

from .pyscf_driver import (active_hamiltonian, generalized_fock,
                           inactive_fock, nonredundant_pairs, rotation_matrix)

__all__ = ["ci_hamiltonian", "coupled_energy", "fock_response_kappa",
           "hess_ss", "hess_ks", "hess_kk_raw", "build_coupled",
           "coupled_casscf"]


def ci_hamiltonian(heff, eri, nact, nelecas):
    """A closure applying the active-space H to a CI vector."""
    from pyscf import fci

    h2e = fci.direct_spin1.absorb_h1e(heff, eri, nact, nelecas, 0.5)
    return lambda c: fci.direct_spin1.contract_2e(h2e, c, nact, nelecas)


def kappa_of(x, pairs, norb):
    K = np.zeros((norb, norb))
    for (m, n), v in zip(pairs, x):
        K[m, n], K[n, m] = v, -v
    return K


def ci_basis(c0):
    """Orthonormal basis of the space orthogonal to |0>.

    Deterministic (an SVD null space, not a QR of random columns): a basis
    that changes between macroiterations makes a run irreproducible for no
    gain.
    """
    from scipy.linalg import null_space

    N = null_space(np.asarray(c0).reshape(1, -1))
    return [N[:, i].reshape(np.asarray(c0).shape) for i in range(N.shape[1])]


def coupled_energy(basis, C0, c0, ninact, nact, nelecas, kap, s):
    """E(kappa, s) with |Psi(s)> = (|0> + s)/|| |0> + s ||, rebuilt from
    scratch so finite differences of it are an independent check."""
    from scipy.linalg import expm

    C = np.asarray(C0) @ expm(kap)
    Ecore, heff, eri = active_hamiltonian(basis, C, ninact, nact)
    c = np.asarray(c0) + s
    c = c / np.linalg.norm(c)
    return Ecore + float(np.dot(c.ravel(),
                                ci_hamiltonian(heff, eri, nact, nelecas)(c).ravel()))


def fock_response_kappa(basis, C, ninact, nact, D, d, dkap, hcore=None):
    """d/dt generalized_fock(C exp(t dkap)) at t = 0, RDMs frozen.

    Every piece of the generalized Fock is a coulomb/exchange call on a
    density built from C, so its derivative is the same calls on the
    one-index-transformed densities (dC = C dkap) plus contraction terms.
    """
    h = basis.hcore() if hcore is None else np.asarray(hcore)
    nocc = ninact + nact
    C = np.asarray(C)
    Cx = C @ dkap
    Ci, Ca = C[:, :ninact], C[:, ninact:nocc]
    Cxi, Cxa = Cx[:, :ninact], Cx[:, ninact:nocc]

    PI, dPI = 2.0 * (Ci @ Ci.T), 2.0 * (Cxi @ Ci.T + Ci @ Cxi.T)
    Vi = h + basis.coulomb(PI) - 0.5 * basis.exchange(PI)
    dVi = basis.coulomb(dPI) - 0.5 * basis.exchange(dPI)
    FI = C.T @ Vi @ C
    dFI = Cx.T @ Vi @ C + C.T @ Vi @ Cx + C.T @ dVi @ C

    PA = Ca @ D @ Ca.T
    dPA = Cxa @ D @ Ca.T + Ca @ D @ Cxa.T
    Va = basis.coulomb(PA) - 0.5 * basis.exchange(PA)
    dVa = basis.coulomb(dPA) - 0.5 * basis.exchange(dPA)
    dFA = Cx.T @ Va @ C + C.T @ Va @ Cx + C.T @ dVa @ C

    dF = np.zeros((C.shape[1], nocc))
    dF[:, :ninact] = 2.0 * (dFI[:, :ninact] + dFA[:, :ninact])
    dF[:, ninact:nocc] = dFI[:, ninact:nocc] @ D
    for t in range(nact):
        for u in range(nact):
            Ptu = Ca @ d[t, u] @ Ca.T
            dPtu = Cxa @ d[t, u] @ Ca.T + Ca @ d[t, u] @ Cxa.T
            J = basis.coulomb(0.5 * (Ptu + Ptu.T))
            dJ = basis.coulomb(0.5 * (dPtu + dPtu.T))
            dF[:, ninact + t] += (Cx.T @ (J @ Ca[:, u])
                                  + C.T @ (dJ @ Ca[:, u])
                                  + C.T @ (J @ Cxa[:, u]))
    return dF


def hess_kk_raw(basis, C, ninact, nact, D, d, dkap):
    """d/dt of the orbital gradient along dkap, square over all orbitals.

    This is H_kk plus (1/2) g . [dkap, .]: differentiating the gradient walks
    a PRODUCT of exponentials, exp(t K')exp(s K), while the Hessian is defined
    on exp(sK + tK'), and BCH separates them by that commutator. The two agree
    only where the gradient vanishes, so callers wanting the Hessian must
    symmetrise -- `build_coupled` does. Verified: the antisymmetric part of
    this matches (1/2) g . [K', K] to five significant figures.
    """
    C = np.asarray(C)
    norb, nocc = C.shape[1], ninact + nact
    dFf = np.zeros((norb, norb))
    dFf[:, :nocc] = fock_response_kappa(basis, C, ninact, nact, D, d, dkap)
    return 2.0 * (dFf - dFf.T)


def hess_ss(basis, C, c0, ninact, nact, nelecas, ds):
    """2 P_perp (H - E) P_perp ds -- one sigma-vector."""
    Ecore, heff, eri = active_hamiltonian(basis, C, ninact, nact)
    Hop = ci_hamiltonian(heff, eri, nact, nelecas)
    Eact = float(np.dot(np.asarray(c0).ravel(), Hop(c0).ravel()))
    r = Hop(ds) - Eact * ds
    r = r - float(np.dot(np.asarray(c0).ravel(), r.ravel())) * c0
    return 2.0 * r


def hess_ks(basis, C, c0, ninact, nact, nelecas, ds, pairs):
    """Orbital gradient of 2<ds|H(kappa)|0>, via the transition RDMs."""
    from pyscf import fci

    Dt, dt = fci.direct_spin1.trans_rdm12(ds, c0, nact, nelecas)
    Dt = 0.5 * (Dt + Dt.T)
    dt = 0.5 * (dt + dt.transpose(1, 0, 3, 2))
    F = generalized_fock(basis, C, ninact, nact, Dt, dt, core_occ=0.0)
    norb, nocc = np.asarray(C).shape[1], ninact + nact
    Ff = np.zeros((norb, norb))
    Ff[:, :nocc] = F
    g = 2.0 * (Ff - Ff.T)
    return 2.0 * np.array([g[m, n] for (m, n) in pairs])


def build_coupled(basis, C, c0, ninact, nact, nelecas, pairs):
    """(E, gradient, dense Hessian, ci_basis, D, d) at the current reference."""
    from pyscf import fci
    from .pyscf_driver import orbital_gradient

    C = np.asarray(C)
    norb = C.shape[1]
    Ecore, heff, eri = active_hamiltonian(basis, C, ninact, nact)
    Hop = ci_hamiltonian(heff, eri, nact, nelecas)
    Eact = float(np.dot(np.asarray(c0).ravel(), Hop(c0).ravel()))
    D, d = fci.direct_spin1.make_rdm12(c0, nact, nelecas)

    sb = ci_basis(c0)
    nk, ns = len(pairs), len(sb)

    g = orbital_gradient(basis, C, ninact, nact, D, d)
    g_k = np.array([g[m, n] for (m, n) in pairs])
    r = Hop(c0) - Eact * c0
    r = r - float(np.dot(np.asarray(c0).ravel(), r.ravel())) * c0
    g_s = np.array([2.0 * float(np.dot(v.ravel(), r.ravel())) for v in sb])

    H = np.zeros((nk + ns, nk + ns))
    for j in range(nk):
        Kj = kappa_of(np.eye(nk)[j], pairs, norb)
        col = hess_kk_raw(basis, C, ninact, nact, D, d, Kj)
        # No 1/2. The scalar form contracts the FULL antisymmetric matrix and
        # so double-counts each pair; extracting pair entries directly does
        # not. Halving this block is invisible until the optimizer stalls.
        H[:nk, j] = [col[a, b] for (a, b) in pairs]
    H[:nk, :nk] = 0.5 * (H[:nk, :nk] + H[:nk, :nk].T)   # BCH ordering term
    for j, v in enumerate(sb):
        hv = hess_ss(basis, C, c0, ninact, nact, nelecas, v)
        H[nk:, nk + j] = [float(np.dot(w.ravel(), hv.ravel())) for w in sb]
        H[:nk, nk + j] = hess_ks(basis, C, c0, ninact, nact, nelecas, v, pairs)
    H[nk:, :nk] = H[:nk, nk:].T
    H[nk:, nk:] = 0.5 * (H[nk:, nk:] + H[nk:, nk:].T)
    return Ecore + Eact, np.concatenate([g_k, g_s]), H, sb, D, d


def coupled_casscf(basis, C, ninact, nact, nelecas, tol=1e-7, maxmacro=100,
                   trust=0.3, null_tol=1e-6, curv_floor=1e-3, verbose=False):
    """Fully coupled (kappa, c) CASSCF. Returns (E, C, civec, info)."""
    from pyscf import fci

    C = np.array(C, dtype=float, copy=True)
    norb = C.shape[1]
    pairs = nonredundant_pairs(ninact, nact, norb)
    nk = len(pairs)
    Ecore, heff, eri = active_hamiltonian(basis, C, ninact, nact)
    _, c0 = fci.direct_spin1.kernel(heff, eri, nact, nelecas, ecore=Ecore)
    c0 = c0 / np.linalg.norm(c0)

    gn = np.inf
    for macro in range(maxmacro):
        E, g, H, sb, D, d = build_coupled(basis, C, c0, ninact, nact,
                                          nelecas, pairs)
        gn = np.linalg.norm(g)
        if verbose:
            print(f"   macro {macro:3d}  E = {E:.12f}  |g| = {gn:.3e}")
        if gn < tol:
            return E, C, c0, {"converged": True, "macro": macro,
                              "grad_norm": gn}

        # Newton in the COMPLEMENT of the Hessian null space.
        #
        # A CAS Hessian is genuinely singular, and not only through the
        # active-active rotations already excluded from `pairs`: an active
        # orbital with zero natural occupation makes every rotation touching
        # it exactly flat. Measured at the Be CAS(2,3) solution, 10 of 51
        # modes have |eigenvalue| < 1e-6 and carry a gradient of 1.7e-9.
        #
        # A Levenberg shift does not handle that. It leaves those modes with
        # |g/w| up to 8e-2 from dividing ~0 by ~0; the trust radius then clips
        # the whole step and throttles the informative directions, and the
        # optimizer stalls at |g| ~ 1e-2 having looked merely slow. Dropping
        # the modes costs nothing, since the energy does not depend on them.
        w, V = np.linalg.eigh(H)
        keep = np.abs(w) > null_tol
        inv = np.zeros_like(w)
        inv[keep] = 1.0 / np.maximum(w[keep], curv_floor)
        step = -(V @ (inv * (V.T @ g)))
        nrm = np.linalg.norm(step)
        if nrm > trust:
            step *= trust / nrm

        for _ in range(25):
            Ct = C @ rotation_matrix(step[:nk], pairs, norb)
            ct = c0 + sum(s * v for s, v in zip(step[nk:], sb))
            ct = ct / np.linalg.norm(ct)
            Ec2, h2, e2 = active_hamiltonian(basis, Ct, ninact, nact)
            Et = Ec2 + float(np.dot(
                ct.ravel(), ci_hamiltonian(h2, e2, nact, nelecas)(ct).ravel()))
            if Et < E - 1e-14:
                C, c0 = Ct, ct
                trust = min(trust * 1.4, 1.0)
                break
            step *= 0.4
            trust *= 0.4
        else:
            return E, C, c0, {"converged": bool(gn < 1e-5), "macro": macro,
                              "grad_norm": gn, "note": "line search exhausted"}
    return E, C, c0, {"converged": False, "macro": maxmacro, "grad_norm": gn}
