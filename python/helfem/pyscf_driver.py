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
from pyscf import gto, scf

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
