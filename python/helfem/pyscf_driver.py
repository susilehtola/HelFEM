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
