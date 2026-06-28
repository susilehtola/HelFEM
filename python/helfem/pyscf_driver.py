"""PySCF driver shim for HelFEM atomic integrals.

Build a pyscf.scf object whose one- and two-electron integrals come
from a helfem.AtomicBasis. The pyscf object is otherwise normal --
all of pyscf's HF / DFT / CASSCF / CC / MP2 / TDDFT machinery flows
on top, treating HelFEM as just another integral backend.

Usage:

    from helfem.pyscf_driver import helfem_scf
    mf, basis = helfem_scf(Z=2, lmax=0)  # He at l=0
    e_hf = mf.kernel()                    # full RHF SCF in the FE basis
    # ...then pass mf to pyscf.mcscf.CASSCF, pyscf.cc.CCSD, etc.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg

import pyscf
from pyscf import gto, scf, ao2mo

import helfem


_ELEMENT_SYMBOLS = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C",
    7: "N", 8: "O", 9: "F", 10: "Ne",
}


def helfem_scf(
    Z: int,
    *,
    lmax: int = 0,
    mmax: int = None,
    charge: int = 0,
    spin: int = None,
    verbose: int = 0,
    primbas: int = 4,
    nnodes: int = 15,
    nelem: int = 20,
    Rmax: float = 40.0,
    igrid: int = 4,
    zexp: float = 2.0,
    finitenuc: int = 0,
    Rrms: float = 0.0,
):
    """Build a pyscf RHF/ROHF object backed by HelFEM atomic integrals.

    Parameters
    ----------
    Z : int
        Nuclear charge.
    lmax, mmax : int
        Maximum angular momenta of the HelFEM atomic basis. mmax
        defaults to lmax.
    charge : int
        Net charge (default 0).
    spin : int or None
        2*S (number of unpaired electrons). Default: minimum (parity
        of nelectron).
    verbose : int
        PySCF verbosity (passed to the wrapping Mole).
    primbas, nnodes, nelem, Rmax, igrid, zexp, finitenuc, Rrms : ...
        HelFEM atomic-basis construction parameters; see
        helfem.AtomicBasis for meaning.

    Returns
    -------
    (mf, basis)
        `mf` is a pyscf.scf.RHF (or ROHF, if `spin > 0`) object whose
        get_ovlp / get_hcore / get_jk hooks call into the HelFEM
        atomic basis. `basis` is the underlying `helfem.AtomicBasis`.
    """
    if mmax is None:
        mmax = lmax
    nelectron = Z - charge
    if spin is None:
        spin = nelectron % 2

    # Build the HelFEM basis. This dominates the per-call cost (~1s
    # for a typical atomic FE basis); reuse `basis` across SCF calls.
    basis = helfem.AtomicBasis(
        Z=Z, lmax=lmax, mmax=mmax,
        primbas=primbas, nnodes=nnodes, nelem=nelem, Rmax=Rmax,
        igrid=igrid, zexp=zexp, finitenuc=finitenuc, Rrms=Rrms,
    )
    Nbf = basis.Nbf()

    # Build a stub pyscf Mole. The sto-3g basis here is just for
    # PySCF's bookkeeping (atomic symbol, charge, spin, electron count);
    # every integral method on the SCF object gets overridden below.
    elem = _ELEMENT_SYMBOLS.get(Z, "Ne")
    mol = gto.M(
        atom=f"{elem} 0 0 0",
        basis="sto-3g",
        spin=spin,
        charge=charge,
        verbose=verbose,
    )

    # Tell PySCF the AO dimension is Nbf, not the sto-3g count.
    mol.nao_nr = lambda *a, **kw: Nbf
    # Atomic problem: no inter-nuclear repulsion.
    mol.energy_nuc = lambda *a, **kw: 0.0

    # Pick the SCF flavour.
    mf = scf.RHF(mol) if spin == 0 else scf.ROHF(mol)

    # Pre-compute the static matrices once (overlap, core Hamiltonian).
    S = basis.overlap()
    H = basis.hcore()

    # Override the matrix builders. PySCF's signatures pass `mol` first.
    mf.get_ovlp = lambda *a, **kw: S
    mf.get_hcore = lambda *a, **kw: H

    def _get_jk(_mol_arg, dm, *args, **kwargs):
        # PySCF may pass dm as a 2D (N, N) (closed shell or total) or
        # a 3D (2, N, N) (spin-resolved). Handle both.
        dm = np.asarray(dm)
        if dm.ndim == 2:
            return basis.get_jk(dm)
        elif dm.ndim == 3 and dm.shape[0] == 2:
            dm_a, dm_b = dm[0], dm[1]
            Ja, Ka = basis.get_jk(dm_a)
            Jb, Kb = basis.get_jk(dm_b)
            # Coulomb sees the TOTAL density; exchange is spin-resolved.
            J_total = Ja + Jb
            vj = np.stack([J_total, J_total], axis=0)
            vk = np.stack([Ka, Kb], axis=0)
            return vj, vk
        else:
            raise NotImplementedError(
                f"helfem_scf.get_jk: unexpected dm shape {dm.shape}")
    mf.get_jk = _get_jk

    # Initial guess: diagonalise the core Hamiltonian in the FE basis.
    # PySCF's default 'minao' guess goes through pyscf.scf.atom_hf which
    # tries to build min-AO matrix elements; that path doesn't know
    # about HelFEM, so we override with an explicit core guess.
    def _init_core(*a, **kw):
        eval_, evec = scipy.linalg.eigh(H, S)
        n_alpha = (nelectron + spin) // 2
        n_beta = nelectron - n_alpha
        if spin == 0:
            return 2.0 * evec[:, :n_alpha] @ evec[:, :n_alpha].T
        else:
            Pa = evec[:, :n_alpha] @ evec[:, :n_alpha].T
            Pb = evec[:, :n_beta]  @ evec[:, :n_beta].T
            return np.stack([Pa, Pb], axis=0)
    mf.get_init_guess = _init_core

    return mf, basis


def build_active_eri(basis, mo_active):
    """Build the active-space 4-index ERI tensor (chemists' notation) via
    repeated J calls into a HelFEM basis. Returns the UNPACKED tensor of
    shape (N, N, N, N) where N = mo_active.shape[1].

    Cost: N*(N+1)/2 J builds (one per unique active orbital pair) plus
    N**4 traces. For a typical active space N ~ 5-20 on top of an
    Nbf ~ 300-3000 atomic FE basis, the J builds dominate at
    ~ms each. Negligible compared to the CI / orbital-opt loop.

    Math: with the symmetrised pair density
        P_ij_sym = (c_i c_j^T + c_j c_i^T) / 2   for i != j
                 = c_i c_i^T                       for i == j
    we have J(P_ij_sym)_{ab} = (ab | ij)   (since the AO Coulomb kernel
    is symmetric under (alpha, beta) swap, the asymmetric c_i c_j^T
    and c_j c_i^T pieces average to the symmetric product), and so
        (mn | ij) = c_m^T . J(P_ij_sym) . c_n.
    """
    N = mo_active.shape[1]
    eri = np.zeros((N, N, N, N))
    for i in range(N):
        for j in range(i, N):
            ci = mo_active[:, i]
            cj = mo_active[:, j]
            if i == j:
                P = np.outer(ci, ci)
            else:
                P = 0.5 * (np.outer(ci, cj) + np.outer(cj, ci))
            Jij = basis.coulomb(P)
            # (mn | ij) = c_m^T . J . c_n.
            # JC = J . mo_active  -> (Nbf, N)   then mo_active.T @ JC -> (N, N).
            mtJn = mo_active.T @ (Jij @ mo_active)        # shape (N, N)
            eri[:, :, i, j] = mtJn
            if i != j:
                eri[:, :, j, i] = mtJn
    # Enforce the (mn) <-> (ij) pair-swap symmetry that's required by
    # PySCF's 8-fold-packed (s8) format. The block we computed
    # satisfies (mn | ij) symmetry in m<->n and i<->j, but symmetrising
    # over pair-swap protects against small numerical asymmetries.
    eri = 0.5 * (eri + eri.transpose(2, 3, 0, 1))
    return eri


def install_eri_builder(mc, basis):
    """Patch a pyscf.mcscf.CAS* object (CASCI or CASSCF) so it builds the
    active-space ERI by repeated J calls into `basis` instead of from
    `mol.intor("int2e")`.

    PySCF's `CASCI.get_h2eff(mo_coeff)` is what the SCF kernel calls;
    it returns the active-space ERI in 4-fold-packed form. We compute
    the unpacked tensor via build_active_eri then pack with
    pyscf.ao2mo.restore. Also patch `ao2mo` for callers that go
    through that route (e.g. CASSCF orbital optimisation).

    Returns `mc` (the same object) for chaining.
    """
    def _build(mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = mc.mo_coeff
        if mo_coeff.shape[1] > mc.ncas:
            mo_active = mo_coeff[:, mc.ncore:mc.ncore + mc.ncas]
        else:
            mo_active = mo_coeff
        eri_full = build_active_eri(basis, mo_active)
        return ao2mo.restore('s4', eri_full, mc.ncas)

    mc.get_h2eff = _build
    mc.ao2mo     = _build
    return mc


def helfem_casci(mf, basis, ncas, nelecas):
    """Build a pyscf.mcscf.CASCI driven by HelFEM integrals on top of an
    `mf` from helfem_scf. Wraps `install_eri_builder` so the user gets a
    ready-to-call CASCI object.
    """
    from pyscf import mcscf
    mc = mcscf.CASCI(mf, ncas, nelecas)
    return install_eri_builder(mc, basis)


def helfem_casscf(mf, basis, ncas, nelecas):
    """Same as helfem_casci but with orbital optimisation (CASSCF)."""
    from pyscf import mcscf
    mc = mcscf.CASSCF(mf, ncas, nelecas)
    return install_eri_builder(mc, basis)


# -- NAO basis (projection of an FE AtomicBasis onto its physical
# atomic orbitals) -----------------------------------------------------

class NAOAtomicBasis:
    """Thin Python wrapper that presents a NAO subspace of a HelFEM
    AtomicBasis as if it were itself an AtomicBasis. Built from an
    (Nbf_FE x Nbf_NAO) coefficient matrix C; every matrix element
    forwards to the wrapped FE basis via the C-transform.

    NAOAtomicBasis is interchangeable with AtomicBasis in helfem_scf
    and helfem_casci (it implements the same overlap / kinetic /
    nuclear / hcore / get_jk methods).

    For He at a high-resolution FE basis the lowest HF virtual
    orbitals are continuum-like (see commit message of PR #91); the
    NAO projection picks just the lowest few per-shell MOs, giving a
    compact, physical-ish basis on which CASCI / CASSCF behave.
    """
    def __init__(self, atomic_basis, C):
        self._fe = atomic_basis
        self._C  = np.asarray(C, dtype=float)
        if self._C.shape[0] != atomic_basis.Nbf():
            raise ValueError(
                f"C.shape[0]={self._C.shape[0]} != Nbf_FE={atomic_basis.Nbf()}")

    def Nbf(self):  return self._C.shape[1]
    def Nrad(self): return self._C.shape[1]   # flat (no separate angular)
    def Nang(self): return 1
    def lvals(self):  return []
    def mvals(self):  return []

    def overlap(self):
        return self._C.T @ self._fe.overlap() @ self._C
    def kinetic(self):
        return self._C.T @ self._fe.kinetic() @ self._C
    def nuclear(self):
        return self._C.T @ self._fe.nuclear() @ self._C
    def hcore(self):
        return self._C.T @ self._fe.hcore() @ self._C

    def get_jk(self, dm):
        dm = np.asarray(dm)
        if dm.ndim == 2:
            P_fe = self._C @ dm @ self._C.T
            J, K = self._fe.get_jk(P_fe)
            return self._C.T @ J @ self._C, self._C.T @ K @ self._C
        elif dm.ndim == 3 and dm.shape[0] == 2:
            dm_a, dm_b = dm[0], dm[1]
            P_fe_a = self._C @ dm_a @ self._C.T
            P_fe_b = self._C @ dm_b @ self._C.T
            Ja, Ka = self._fe.get_jk(P_fe_a)
            Jb, Kb = self._fe.get_jk(P_fe_b)
            J_total = Ja + Jb
            vj = np.stack([self._C.T @ J_total @ self._C] * 2, axis=0)
            vk = np.stack([self._C.T @ Ka @ self._C,
                           self._C.T @ Kb @ self._C], axis=0)
            return vj, vk
        else:
            raise NotImplementedError(f"unexpected dm shape {dm.shape}")

    def coulomb(self, P_nao):
        """Single-channel J (used by build_active_eri / helfem_casci)."""
        P_fe = self._C @ P_nao @ self._C.T
        J_fe = self._fe.coulomb(P_fe)
        return self._C.T @ J_fe @ self._C


def extract_per_shell_naos(mf, atomic_basis, keep_per_shell):
    """Slice mf.mo_coeff into per-angular-shell blocks and keep the
    lowest `keep_per_shell[iang]` HF orbital from each shell. Returns
    a (Nbf_FE, sum(keep_per_shell)) coefficient matrix.

    Exploits the fact that for a spherically-symmetric atom the HF
    Fock matrix is block-diagonal in (l, m), so each HF MO lives on
    exactly one angular shell to numerical precision.
    """
    lvals = atomic_basis.lvals()
    mvals = atomic_basis.mvals()
    Nrad  = atomic_basis.Nrad()
    Nang  = atomic_basis.Nang()
    if len(keep_per_shell) != Nang:
        raise ValueError(
            f"keep_per_shell has {len(keep_per_shell)} entries, expected {Nang}")
    mo_coeff  = mf.mo_coeff
    mo_energy = mf.mo_energy
    if mo_coeff is None:
        raise ValueError("mf.kernel() must have been called -- mo_coeff is None")

    # For each MO, find its dominant angular shell.
    shell_mass = np.empty((Nang, mo_coeff.shape[1]))
    for iang in range(Nang):
        block = mo_coeff[iang*Nrad:(iang+1)*Nrad, :]
        shell_mass[iang] = np.linalg.norm(block, axis=0)
    dominant_shell = np.argmax(shell_mass, axis=0)
    # Sanity check: every MO is dominantly on one shell (>=99%).
    max_mass = np.max(shell_mass, axis=0)
    weak = np.where(max_mass < 0.99)[0]
    if weak.size:
        # Numerically blurred -- e.g., near-degenerate shells -- but
        # the per-shell SCF for an atom shouldn't normally produce
        # blurred MOs. Warn loudly if so.
        print(f"[helfem extract_per_shell_naos] warning: {weak.size} MOs "
              f"are not dominantly on a single shell (max mass < 0.99); "
              f"NAO basis quality may suffer.")

    C_cols = []
    for iang in range(Nang):
        keep = keep_per_shell[iang]
        if keep <= 0:
            continue
        # MOs dominantly on this shell, sorted by orbital energy.
        on_shell = np.where(dominant_shell == iang)[0]
        on_shell_sorted = on_shell[np.argsort(mo_energy[on_shell])]
        for k in on_shell_sorted[:keep]:
            C_cols.append(mo_coeff[:, k])
    if not C_cols:
        raise ValueError("extract_per_shell_naos: kept zero NAOs")
    return np.column_stack(C_cols)


def helfem_nao_scf(mf_fe, atomic_basis, keep_per_shell, **scf_kwargs):
    """Convenience: build an NAOAtomicBasis from the lowest per-shell
    HF MOs of `mf_fe`, then wire a fresh pyscf SCF object on top.

    Returns (mf_nao, basis_nao). `mf_nao.kernel()` is trivial since the
    NAOs are already HF eigenstates of the full FE problem (within the
    NAO subspace); use the resulting mf as the parent of CASCI/CASSCF.
    """
    if mf_fe.mo_coeff is None:
        mf_fe.kernel()
    C = extract_per_shell_naos(mf_fe, atomic_basis, keep_per_shell)
    basis_nao = NAOAtomicBasis(atomic_basis, C)
    return _build_scf_for_basis(basis_nao, mf_fe.mol.nelectron,
                                 mf_fe.mol.spin, mf_fe.mol.charge,
                                 mf_fe.mol.verbose), basis_nao


def _build_scf_for_basis(basis, nelectron, spin, charge, verbose):
    """Internal: build a pyscf SCF object on top of an arbitrary
    basis-shaped object (AtomicBasis or NAOAtomicBasis). Factored out
    of helfem_scf so it can be reused by helfem_nao_scf."""
    Nbf = basis.Nbf()
    elem = _ELEMENT_SYMBOLS.get(nelectron - charge + charge, "Ne")  # crude
    mol = gto.M(
        atom=f"{elem} 0 0 0",
        basis="sto-3g",
        spin=spin, charge=charge, verbose=verbose,
    )
    mol.nao_nr     = lambda *a, **kw: Nbf
    mol.energy_nuc = lambda *a, **kw: 0.0

    mf = scf.RHF(mol) if spin == 0 else scf.ROHF(mol)
    S = basis.overlap()
    H = basis.hcore()
    mf.get_ovlp  = lambda *a, **kw: S
    mf.get_hcore = lambda *a, **kw: H
    def _get_jk(_mol_arg, dm, *args, **kwargs):
        dm = np.asarray(dm)
        if dm.ndim == 2:
            return basis.get_jk(dm)
        elif dm.ndim == 3 and dm.shape[0] == 2:
            return basis.get_jk(dm)
        else:
            raise NotImplementedError(f"_get_jk: unexpected dm shape {dm.shape}")
    mf.get_jk = _get_jk

    def _init_core(*a, **kw):
        eval_, evec = scipy.linalg.eigh(H, S)
        n_alpha = (nelectron + spin) // 2
        n_beta  = nelectron - n_alpha
        if spin == 0:
            return 2.0 * evec[:, :n_alpha] @ evec[:, :n_alpha].T
        else:
            Pa = evec[:, :n_alpha] @ evec[:, :n_alpha].T
            Pb = evec[:, :n_beta]  @ evec[:, :n_beta].T
            return np.stack([Pa, Pb], axis=0)
    mf.get_init_guess = _init_core
    return mf
