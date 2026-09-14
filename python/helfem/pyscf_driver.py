"""PySCF driver for HelFEM's atomic finite-element basis.

HelFEM never materializes the four-index AO integral tensor: it exposes
`coulomb(P)` and `exchange(P)`, which are linear in the density and carry all
the element-locality and Gaunt-selection optimizations. Everything here is
built from those two calls.

The one identity the whole module rests on is that `coulomb` contracts the
*second* pair:

    coulomb(P)_{mu nu} = sum_{la si} (mu nu|la si) P_{la si}

so feeding it a pair density P = C_t C_u^T + transpose and contracting the free
pair against C gives a row of MO integrals,

    (tu|vw) = C_v^T . coulomb(C_t C_u^T + h.c.) . C_w / 2

at a cost of one `coulomb` call per pair. With C = I that extracts the full AO
tensor (Nbf(Nbf+1)/2 calls -- tests and small bases only); with C restricted to
an active space it is exactly the integral list a CASSCF needs, which is why no
AO->MO transform is required anywhere.

Verified against HelFEM's own contractions and against PySCF on He
(Nbf=14, lmax=0, nelem=3, nnodes=6, Rmax=20):

    J, K rebuilt from the extracted tensor vs coulomb()/exchange()  2e-16, 5e-16
    8-fold permutational symmetry of the extracted tensor           2e-16
    PySCF RHF on these integrals vs HelFEM's own `atomic` binary    all 10 digits
                                      (-2.859592542390 vs -2.8595925424)
    build_active_eri vs PySCF ao2mo on the same tensor              2e-16

Sign conventions: the bindings return POSITIVE K (PySCF style), i.e. the Fock
matrix is h + J - K for a spin density and h + J - K/2 for a total density.
HelFEM's internal `exchange()` has the HF minus already applied; the bindings
flip it back so PySCF can subtract.
"""

import numpy as np

__all__ = [
    "build_active_eri",
    "install_full_eri",
    "inactive_fock",
    "active_hamiltonian",
    "helfem_scf",
    "casci",
    "generalized_fock",
    "orbital_gradient",
    "frozen_ci_energy",
    "nonredundant_pairs",
    "rotation_matrix",
    "casscf",
]


def build_active_eri(basis, C):
    """Chemist-notation (tu|vw) over the columns of `C`.

    `C` is (Nbf, n) in the non-orthonormal FEM basis; the result is
    (n, n, n, n). Costs n(n+1)/2 `coulomb` calls and no AO->MO transform.

    Pass C = np.eye(Nbf) to extract the full AO tensor.
    """
    C = np.asarray(C, dtype=float)
    if C.ndim != 2:
        raise ValueError("build_active_eri: C must be 2-D (Nbf, n)")
    n = C.shape[1]
    eri = np.empty((n, n, n, n))
    for t in range(n):
        for u in range(t, n):
            # P + P^T is what coulomb() expects (a symmetric density). For
            # t == u that is 2 C_t C_t^T, so the 0.5 below is right in both
            # cases and needs no special case.
            P = np.outer(C[:, t], C[:, u])
            J = basis.coulomb(P + P.T)
            blk = 0.5 * (C.T @ J @ C)
            eri[t, u] = blk
            eri[u, t] = blk
    return eri


def install_full_eri(basis, mf):
    """Give a PySCF SCF object the full HelFEM AO tensor in 8-fold packed form.

    Nbf(Nbf+1)/2 `coulomb` calls and Nbf^4/8 storage, so this is for tests and
    small bases. Prefer `helfem_scf(..., full_eri=False)`, which routes PySCF's
    get_jk straight at HelFEM and never forms the tensor.
    """
    from pyscf import ao2mo

    nbf = basis.Nbf()
    eri = build_active_eri(basis, np.eye(nbf))
    mf._eri = ao2mo.restore(8, eri, nbf)
    return mf


def inactive_fock(basis, C, ninact, hcore=None):
    """(F_inactive, E_inactive) for the lowest `ninact` orbitals doubly occupied.

    F = h + J[P] - K[P]/2 and E = Tr[P h] + Tr[P (J - K/2)]/2 with
    P = 2 sum_i C_i C_i^T, i.e. the closed-shell RHF expressions. F restricted
    to the active space is the CASSCF h_eff; E is the core energy the CI adds
    to its eigenvalue.
    """
    h = basis.hcore() if hcore is None else np.asarray(hcore, dtype=float)
    Ci = np.asarray(C, dtype=float)[:, :ninact]
    P = 2.0 * (Ci @ Ci.T)
    J = basis.coulomb(P)
    K = basis.exchange(P)
    veff = J - 0.5 * K
    F = h + veff
    E = np.einsum("ij,ij->", P, h) + 0.5 * np.einsum("ij,ij->", P, veff)
    return F, E


def active_hamiltonian(basis, C, ninact, nact, hcore=None):
    """The CAS Hamiltonian: (E_inactive, h_eff, eri).

    `h_eff` is (nact, nact), `eri` is (nact,)*4 in chemist notation. Together
    with the active-space RDMs these reproduce the total energy as

        E = E_inactive + sum_pq h_eff_pq D_pq + sum_pqrs eri_pqrs Gamma_pqrs / 2

    which is the identity to assert when wiring up any CI solver.
    """
    F, E_inact = inactive_fock(basis, C, ninact, hcore)
    Ca = np.asarray(C, dtype=float)[:, ninact:ninact + nact]
    return E_inact, Ca.T @ F @ Ca, build_active_eri(basis, Ca)


def helfem_scf(basis, nelectron, full_eri=False, conv_tol=1e-11,
               max_cycle=200, verbose=0):
    """Converge a PySCF RHF on the HelFEM basis; returns the `mf` object.

    `full_eri=False` (the default) patches PySCF's get_jk to call HelFEM
    directly, which is what the bindings were built for and scales to any
    basis. `full_eri=True` installs the explicit tensor instead, which is
    what lets PySCF's own ao2mo/FCI paths run unmodified -- useful as an
    independent reference, prohibitive beyond a few tens of functions.

    There is no nuclear repulsion for a single atom, so energy_nuc is zeroed.
    """
    from pyscf import gto, scf

    S = basis.overlap()
    h = basis.hcore()

    mol = gto.M(verbose=verbose)
    mol.nelectron = int(nelectron)
    mol.incore_anyway = True

    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args, **kwargs: h
    mf.get_ovlp = lambda *args, **kwargs: S
    mf.energy_nuc = lambda *args, **kwargs: 0.0

    if full_eri:
        install_full_eri(basis, mf)
    else:
        def get_jk(mol=None, dm=None, hermi=1, with_j=True, with_k=True,
                   omega=None):
            dm = np.asarray(dm, dtype=float)
            if dm.ndim != 2:
                raise NotImplementedError(
                    "helfem_scf's get_jk handles a single 2-D density only; "
                    "pass full_eri=True for multi-density (UHF) paths.")
            return basis.get_jk(dm)
        mf.get_jk = get_jk

    mf.conv_tol = conv_tol
    mf.max_cycle = max_cycle
    mf.kernel()
    return mf


def casci(basis, C, ninact, nact, nelecas, ecore_shift=0.0, nroots=1):
    """CASCI at fixed orbitals via PySCF's FCI solver.

    Returns (energy, civec, h_eff, eri, E_inactive). `nelecas` is an int or
    an (nalpha, nbeta) tuple. This is the reference CI used to validate the
    active-space export before a production solver is wired in.
    """
    from pyscf import fci

    E_inact, h_eff, eri = active_hamiltonian(basis, C, ninact, nact)
    if isinstance(nelecas, (int, np.integer)):
        nb = int(nelecas) // 2
        na = int(nelecas) - nb
        nelecas = (na, nb)
    e, civec = fci.direct_spin1.kernel(
        h_eff, eri, nact, nelecas,
        ecore=E_inact + ecore_shift, nroots=nroots)
    return e, civec, h_eff, eri, E_inact


def generalized_fock(basis, C, ninact, nact, D, d, hcore=None):
    """CASSCF generalized Fock matrix F[m][p]: m over all orbitals, p over
    inactive+active.

        F[m][i] = 2 ( F^I[m][i] + F^A[m][i] )               i inactive
        F[m][t] = sum_u D_tu F^I[m][u] + Q[m][t]            t active
        Q[m][t] = sum_uvw d_tuvw (mu|vw)

    with F^I = h + J[P_I] - K[P_I]/2 for P_I = 2 sum_i C_i C_i^T, and
    F^A = J[P_A] - K[P_A]/2 for P_A = sum_tu D_tu C_t C_u^T.

    Every term is a `coulomb`/`exchange` call on a pair density; Q costs one
    `coulomb` per active (t, u) pair, contracting the 2-RDM slice
    sum_vw d_tuvw C_v C_w^T. No AO->MO transform.
    """
    h = basis.hcore() if hcore is None else np.asarray(hcore, dtype=float)
    C = np.asarray(C, dtype=float)
    D = np.asarray(D, dtype=float)
    d = np.asarray(d, dtype=float)
    nocc = ninact + nact
    Ci = C[:, :ninact]
    Ca = C[:, ninact:nocc]

    PI = 2.0 * (Ci @ Ci.T)
    FI = C.T @ (h + basis.coulomb(PI) - 0.5 * basis.exchange(PI)) @ C

    PA = Ca @ D @ Ca.T
    FA = C.T @ (basis.coulomb(PA) - 0.5 * basis.exchange(PA)) @ C

    F = np.zeros((C.shape[1], nocc))
    F[:, :ninact] = 2.0 * (FI[:, :ninact] + FA[:, :ninact])
    # D is symmetric for a real CI vector, so D and D^T are interchangeable.
    F[:, ninact:nocc] = FI[:, ninact:nocc] @ D

    for t in range(nact):
        for u in range(nact):
            Ptu = Ca @ d[t, u] @ Ca.T
            # (mn|vw) is symmetric in (v, w), so symmetrising the contraction
            # argument is exact and keeps coulomb() on the symmetric densities
            # it expects.
            J = basis.coulomb(0.5 * (Ptu + Ptu.T))
            F[:, ninact + t] += C.T @ (J @ Ca[:, u])
    return F


def orbital_gradient(basis, C, ninact, nact, D, d, hcore=None):
    """g[m][n] = 2 (F_mn - F_nm), the CAS energy gradient with respect to an
    orbital rotation at FIXED CI coefficients, square over all orbitals.

    At a converged CASCI the CI is variational, so this also equals the
    CI-relaxed gradient; verified to 2.5e-11 on Be CAS(2,3).
    """
    norb = np.asarray(C).shape[1]
    nocc = ninact + nact
    Ffull = np.zeros((norb, norb))
    Ffull[:, :nocc] = generalized_fock(basis, C, ninact, nact, D, d, hcore)
    return 2.0 * (Ffull - Ffull.T)


def frozen_ci_energy(basis, C, ninact, nact, D, d):
    """The CAS energy at these orbitals with the RDMs held fixed.

    Rebuilt from `active_hamiltonian`, so it shares no algebra with
    `generalized_fock` -- which is what makes central differences of it a real
    check on the gradient rather than a re-expansion of it. (CLAUDE.md's
    standing warning: a derivative test that differentiates one expression
    against finite differences of itself agrees perfectly with a consistently
    wrong one.)
    """
    E0, heff, eri = active_hamiltonian(basis, C, ninact, nact)
    return (E0 + np.einsum("pq,pq->", heff, np.asarray(D))
            + 0.5 * np.einsum("pqrs,pqrs->", eri, np.asarray(d)))


def nonredundant_pairs(ninact, nact, norb):
    """Orbital-rotation pairs a CAS energy actually depends on:
    inactive->active, inactive->virtual, active->virtual.

    Excluded by construction: inactive-inactive and virtual-virtual, which
    leave the energy exactly invariant, and **active-active**, which is
    redundant for a CAS because the CI absorbs it. Dropping active-active is
    not an optimization -- keeping it makes the orbital Hessian singular by
    construction, and HelFEM's own conditioning diagnostic treats exact zeros
    in the Hessian diagonal as a bug (it is, for an SCF; it is expected here).
    """
    nocc = ninact + nact
    pairs = [(i, p) for i in range(ninact) for p in range(ninact, norb)]
    pairs += [(t, v) for t in range(ninact, nocc) for v in range(nocc, norb)]
    return pairs


def rotation_matrix(x, pairs, norb):
    """exp(kappa) for the antisymmetric kappa built from amplitudes `x`."""
    from scipy.linalg import expm

    K = np.zeros((norb, norb))
    for (m, n), v in zip(pairs, x):
        K[m, n], K[n, m] = v, -v
    return expm(K)


def _casscf_precond(basis, C, ninact, nact, D, pairs, floor=0.05):
    """2 |n_p - n_q| (F^I_qq - F^I_pp), floored -- the CAS analogue of the RHF
    (eps_a - eps_i) denominator. The floor stops a near-degenerate pair from
    producing an enormous step."""
    nocc = ninact + nact
    FI_ao, _ = inactive_fock(basis, C, ninact)
    FI = np.diag(C.T @ FI_ao @ C)
    occ = np.zeros(C.shape[1])
    occ[:ninact] = 2.0
    occ[ninact:nocc] = np.clip(np.diag(np.asarray(D)), 0.0, 2.0)
    h = np.array([2.0 * abs(occ[m] - occ[n]) * (FI[n] - FI[m])
                  for (m, n) in pairs])
    return np.maximum(np.abs(h), floor)


def casscf(basis, C, ninact, nact, nelecas, tol=1e-7, maxmacro=300,
           trust=0.2, verbose=False):
    """Two-step CASSCF: CI solved at fixed orbitals, orbitals stepped on the
    resulting RDMs, repeat. Returns (E, C, info).

    Each macroiteration works at x = 0 relative to the current orbitals and
    folds the accepted step back into C. That re-referencing is not a detail:
    dE/dx equals the generalized-Fock gradient ONLY at x = 0, because away
    from it the Frechet derivative of the matrix exponential enters. An
    optimizer handed the gradient from a rotated point under a frozen-reference
    parametrization gets an inconsistent objective/gradient pair and thrashes
    -- measured, on Be CAS(2,3): it crawled to -14.56062 while the true
    minimum is -14.56385.

    The step is preconditioned steepest descent with a backtracking line search
    on the true energy, so the energy decreases monotonically. That converges
    reliably but only linearly; the quadratic convergence is the business of
    the second-order path, not of this reference implementation.
    """
    from pyscf import fci

    C = np.array(C, dtype=float, copy=True)
    norb = C.shape[1]
    pairs = nonredundant_pairs(ninact, nact, norb)

    def solve(Cx):
        E0, heff, eri = active_hamiltonian(basis, Cx, ninact, nact)
        e, civec = fci.direct_spin1.kernel(heff, eri, nact, nelecas, ecore=E0)
        Dx, dx = fci.direct_spin1.make_rdm12(civec, nact, nelecas)
        return e, Dx, dx

    E, D, d = solve(C)
    nci = 1
    gnorm = np.inf
    for macro in range(maxmacro):
        g = orbital_gradient(basis, C, ninact, nact, D, d)
        gvec = np.array([g[m, n] for (m, n) in pairs])
        gnorm = np.linalg.norm(gvec)
        if verbose:
            print(f"   macro {macro:3d}  E = {E:.12f}  |g| = {gnorm:.3e}")
        if gnorm < tol:
            return E, C, {"converged": True, "macro": macro,
                          "grad_norm": gnorm, "ci_solves": nci}

        step = -gvec / _casscf_precond(basis, C, ninact, nact, D, pairs)
        step *= min(1.0, trust / max(np.abs(step).max(), 1e-30))

        for _ in range(20):
            Ct = C @ rotation_matrix(step, pairs, norb)
            Et, Dt, dt = solve(Ct)
            nci += 1
            if Et < E - 1e-14:
                C, E, D, d = Ct, Et, Dt, dt
                trust = min(trust * 1.3, 1.0)
                break
            step *= 0.4
            trust *= 0.4
        else:
            # No downhill step remains along the preconditioned direction.
            # At a small gradient that IS convergence, to line-search
            # resolution; at a large one it is a genuine failure.
            return E, C, {"converged": bool(gnorm < 1e-5), "macro": macro,
                          "grad_norm": gnorm, "ci_solves": nci,
                          "note": "line search exhausted"}
    return E, C, {"converged": False, "macro": maxmacro,
                  "grad_norm": gnorm, "ci_solves": nci}
