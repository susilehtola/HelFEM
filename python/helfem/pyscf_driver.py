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

    Cost: N**2 J builds (one per ordered active orbital pair) plus
    N**4 traces. For a typical active space N ~ 5-20 on top of an
    Nbf ~ 300-3000 atomic FE basis, the J builds dominate at
    ~ms each. Negligible compared to the CI / orbital-opt loop.

    Math: with the ASYMMETRIC pair-density indicator P = c_i c_j^T,
        J(P)_{ab} = sum_{c,d} P_{cd} (ab|cd) = (ab | ij)
    and the active-space entry
        (mn | ij) = c_m^T . J(P) . c_n.

    Why asymmetric P (not the half-sum of c_i c_j^T and c_j c_i^T): for
    complex Y_lm AOs (HelFEM's atomic basis), (mn|ij) and (mn|ji) have
    DIFFERENT m-conservation patterns -- one can be nonzero while the
    other is zero. Averaging mixes them and contaminates the tensor
    with spurious m-non-conserving entries (this was a real bug: He
    lmax=1 CCSD gave a 5 mEh-wrong correlation energy through this
    contamination; see install_full_eri's deprecation note below).
    For REAL AO bases (Y_l0 only, or real spherical harmonics) the
    two halves coincide and either form gives the same answer.
    """
    N = mo_active.shape[1]
    eri = np.zeros((N, N, N, N))
    for i in range(N):
        for j in range(N):
            ci = mo_active[:, i]
            cj = mo_active[:, j]
            # Asymmetric pair-density indicator: J(c_i c_j^T) extracts
            # exactly (.|ij), with no (.|ji) contamination.
            P = np.outer(ci, cj)
            Jij = basis.coulomb(P)
            # (mn | ij) = c_m^T . J . c_n.
            # JC = J . mo_active  -> (Nbf, N)   then mo_active.T @ JC -> (N, N).
            eri[:, :, i, j] = mo_active.T @ (Jij @ mo_active)
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
    `mf` from helfem_scf. Auto-installs the full AO ERI on mf if not
    already present, then constructs a standard pyscf CASCI -- no
    custom mc.ao2mo override needed, all of pyscf's downstream
    machinery (FCI solver, etc.) flows on top.
    """
    from pyscf import mcscf
    if getattr(mf, "_eri", None) is None:
        install_full_eri(mf, basis)
    return mcscf.CASCI(mf, ncas, nelecas)


def helfem_casscf(mf, basis, ncas, nelecas):
    """Same as helfem_casci but with orbital optimisation (CASSCF).

    Requires mf._eri to have been pre-populated with the AO ERI tensor
    (use install_full_eri below). CASSCF uses PySCF's _ERIS internals
    which need the AO ERI; the install_eri_builder shim works for
    CASCI but not CASSCF (CASSCF needs vhf_c, ppaa, papa, ... beyond
    what install_eri_builder provides).
    """
    from pyscf import mcscf
    if getattr(mf, "_eri", None) is None:
        install_full_eri(mf, basis)
    return mcscf.CASSCF(mf, ncas, nelecas)


# -- State symmetry classification ------------------------------------
#
# HelFEM atomic AOs have definite (l, m), so the one-electron L_z and
# the orbital part of L^2 are diagonal in the AO basis. For a CI
# state's 1RDM in the AO basis:
#   <L_z>           = sum_i D_ii * m_i              (EXACT)
#   <L^2_orbital>   = sum_i D_ii * l_i * (l_i + 1)  (ONE-BODY part)
#
# <M_L> is exact because L_z commutes with H for atoms (spherical
# Coulomb), so all determinants in a CI ground state share its M_L.
# Different CI roots may have different M_L; this gives a one-line
# symmetry classification of states without setting up PySCF's full
# mol.symm_orb plumbing.
#
# <L^2_orbital> is the orbital ONE-BODY part of the total L^2. It's a
# coarse "L-character" diagnostic, NOT the full <L^2> of an atomic
# eigenstate (the full L^2 includes orbital cross-coupling from L_+/L_-
# that this expression misses).
#
# To target a specific (M_L, 2S+1) sector in CASCI: set mc.spin = 2S,
# run with nroots = K large enough to capture the target state, and
# pick by M_L using state_ML / classify_states. This is the "post-hoc
# symmetry filter" workflow; for a "pre-emptive irrep targeting" via
# wfnsym, full mol.symm_orb support is the natural follow-on.

def _Lz_diagonal_AO(atomic_basis):
    """Internal: diagonal of L_z in the raw AtomicBasis (not NAO).
    Each AO is on one angular shell with definite m_l."""
    Nrad = atomic_basis.Nrad()
    Lz = np.zeros(atomic_basis.Nbf())
    for iang, m in enumerate(atomic_basis.mvals()):
        Lz[iang * Nrad : (iang + 1) * Nrad] = m
    return Lz


def _Lorb2_diagonal_AO(atomic_basis):
    """Internal: diagonal of orbital L^2 in the raw AtomicBasis."""
    Nrad = atomic_basis.Nrad()
    L2 = np.zeros(atomic_basis.Nbf())
    for iang, l in enumerate(atomic_basis.lvals()):
        L2[iang * Nrad : (iang + 1) * Nrad] = l * (l + 1)
    return L2


def Lz_matrix(basis):
    """L_z one-electron operator matrix in `basis`. For an AtomicBasis,
    diagonal with value m_l for each AO. For an NAOAtomicBasis, the
    matrix is C^T diag(Lz_AO) C in the NAO basis."""
    if isinstance(basis, NAOAtomicBasis):
        Lz_ao = _Lz_diagonal_AO(basis._fe)
        return (basis._C.T * Lz_ao) @ basis._C
    return np.diag(_Lz_diagonal_AO(basis))


def Lorb2_matrix(basis):
    """Orbital L^2 one-electron operator in `basis`. See top-of-section
    note: this is the ONE-BODY l*(l+1) trace, not the full many-body
    <L^2>."""
    if isinstance(basis, NAOAtomicBasis):
        L2_ao = _Lorb2_diagonal_AO(basis._fe)
        return (basis._C.T * L2_ao) @ basis._C
    return np.diag(_Lorb2_diagonal_AO(basis))


def _get_state_dm_ao(mc, state_idx):
    """Internal: get the AO-basis 1RDM for the requested CI state.
    Passes the CI vector explicitly (PySCF's make_rdm1 takes ci as a
    kwarg, so swapping mc.ci isn't enough -- the saved-then-restore
    pattern returns mc.ci[0] for multi-root)."""
    if isinstance(mc.ci, (list, tuple)):
        ci = mc.ci[state_idx]
    else:
        if state_idx != 0:
            raise IndexError(
                f"requested state {state_idx} but mc has a single CI vector")
        ci = mc.ci
    # PySCF CASBase.make_rdm1 accepts ci as a kwarg.
    return mc.make_rdm1(ci=ci)


def state_ML(mc, basis, state_idx=0):
    """Compute <M_L> for CI state `state_idx` of a helfem-driven
    CASCI/CASSCF. EXACT (L_z conserved for atoms)."""
    dm_ao = _get_state_dm_ao(mc, state_idx)
    return float(np.trace(dm_ao @ Lz_matrix(basis)))


def state_S2(mc, state_idx=0):
    """Compute <S^2> for CI state `state_idx`. Uses PySCF's
    fcisolver.spin_square. Returns (S^2, 2S+1) as PySCF does."""
    if isinstance(mc.ci, (list, tuple)):
        ci = mc.ci[state_idx]
    else:
        if state_idx != 0:
            raise IndexError(
                f"requested state {state_idx} but mc has a single CI vector")
        ci = mc.ci
    return mc.fcisolver.spin_square(ci, mc.ncas, mc.nelecas)


def state_Lorb2(mc, basis, state_idx=0):
    """Compute <L^2_orbital> (one-body L^2 trace) for CI state
    `state_idx`. Coarse L-character diagnostic; see top-of-section
    note for the relation to the full many-body <L^2>."""
    dm_ao = _get_state_dm_ao(mc, state_idx)
    return float(np.trace(dm_ao @ Lorb2_matrix(basis)))


def classify_states(mc, basis):
    """Return per-state (energy, M_L, L^2_orbital) for all CI roots of
    a multi-root CASCI/CASSCF. Returns a list of dicts sorted by state
    index (= ascending energy in PySCF convention). For single-root
    mc, returns a single-entry list."""
    if isinstance(mc.ci, (list, tuple)):
        n_states = len(mc.ci)
        energies = mc.e_tot if hasattr(mc, "e_tot") and \
            isinstance(mc.e_tot, (list, tuple, np.ndarray)) else None
    else:
        n_states = 1
        energies = [mc.e_tot] if hasattr(mc, "e_tot") else None
    out = []
    for k in range(n_states):
        e = energies[k] if energies is not None else None
        ml = state_ML(mc, basis, state_idx=k)
        l2 = state_Lorb2(mc, basis, state_idx=k)
        s2, mult = state_S2(mc, state_idx=k)
        out.append({
            "state": k,
            "energy": e,
            "M_L": ml,
            "L^2_orbital": l2,
            "S^2": s2,
            "2S+1": mult,
        })
    return out


# -- Post-HF convenience wrappers (MP2, CCSD, FCI) -------------------
#
# install_full_eri makes any of PySCF's post-HF methods Just Work on
# top of a helfem-driven mf. These wrappers provide a uniform one-line
# API alongside helfem_casci / helfem_casscf, and auto-install the AO
# ERI tensor on mf if not already there.

def helfem_mp2(mf, basis):
    """MP2 on top of HelFEM SCF. Auto-installs the AO ERI on mf."""
    from pyscf import mp
    if getattr(mf, "_eri", None) is None:
        install_full_eri(mf, basis)
    return mp.MP2(mf)


def helfem_ccsd(mf, basis):
    """CCSD on top of HelFEM SCF. Auto-installs the AO ERI on mf."""
    from pyscf import cc
    if getattr(mf, "_eri", None) is None:
        install_full_eri(mf, basis)
    return cc.CCSD(mf)


def helfem_fci(mf, basis):
    """Full CI on top of HelFEM SCF. Returns a pyscf CASCI driver
    configured for the full basis (ncas = Nbf, nelecas from mf.mol).

    For small NAO bases (Nbf <= ~16) FCI is tractable and gives the
    exact basis-set energy. For larger bases use CASCI on a chosen
    active space instead.
    """
    nelectron = int(mf.mol.nelectron)
    return helfem_casci(mf, basis, ncas=basis.Nbf(), nelecas=nelectron)


def natural_orbitals(mc):
    """Extract natural orbitals from a converged CASCI / CASSCF result.

    Returns (occupations, no_coeff_ao):
      occupations  : (Nbf,) ndarray, natural-orbital occupation numbers
                     sorted descending (so the most-occupied NO is at
                     column 0).
      no_coeff_ao  : (Nbf, Nbf) ndarray, AO-basis coefficients of the
                     natural orbitals (i.e. NO_alpha = sum_mu
                     no_coeff_ao[mu, alpha] * AO_mu).

    The NOs diagonalise the AO-basis 1RDM in the metric S:
        S . D_AO . S . no_coeff = S . no_coeff . diag(occ)
    """
    dm_ao = mc.make_rdm1()                          # AO-basis 1RDM
    S = mc._scf.get_ovlp()
    # Solve generalised eigenvalue D_ao c = n S^-1 c, equivalently
    # use S^(1/2) basis to make it standard symmetric.
    sval, svec = scipy.linalg.eigh(S)
    Sinvh = svec @ np.diag(1.0 / np.sqrt(sval)) @ svec.T
    Sh    = svec @ np.diag(np.sqrt(sval))       @ svec.T
    # Transform D_AO -> S^(1/2) D_AO S^(1/2), which is the 1RDM in the
    # symmetric-orthonormal basis. Diagonalise standard symmetric.
    D_sym = Sh @ dm_ao @ Sh
    occ, U = scipy.linalg.eigh(D_sym)
    # Back-transform NOs to AO basis: AO -> sym-orth via S^(1/2)^-1 = Sinvh.
    no_coeff_ao = Sinvh @ U
    # Sort descending by occupation.
    order = np.argsort(occ)[::-1]
    return occ[order], no_coeff_ao[:, order]


def helfem_no_truncated_basis(mc, fe_basis, n_keep, occ_threshold=1e-6):
    """Build a NAOAtomicBasis from the top-N natural orbitals of an mc.

    n_keep  : number of NOs to keep (-1 = keep all with occ > threshold).
    occ_threshold : NOs with occupation below this are dropped even if
                    n_keep > current_count.

    `fe_basis` must be the underlying AtomicBasis (or NAOAtomicBasis)
    that `mc._scf` was built on. The returned NAOAtomicBasis wraps
    `fe_basis` with the NO coefficients composed appropriately.
    """
    occ, no_coeff = natural_orbitals(mc)
    # If mc was built on an NAOAtomicBasis, fe_basis is that wrapper;
    # the no_coeff is in the NAO basis. Compose with the NAO C matrix
    # to get coeffs in the underlying FE basis.
    if isinstance(fe_basis, NAOAtomicBasis):
        no_coeff_in_fe = fe_basis._C @ no_coeff
        underlying = fe_basis._fe
    else:
        no_coeff_in_fe = no_coeff
        underlying = fe_basis
    # Filter by n_keep and threshold.
    keep_mask = occ > occ_threshold
    if n_keep > 0:
        # Keep at most n_keep, ordered by occupation (already sorted).
        keep_mask = np.zeros_like(occ, dtype=bool)
        keep_mask[:n_keep] = True
        # ALSO apply threshold
        keep_mask &= (occ > occ_threshold)
    kept = no_coeff_in_fe[:, keep_mask]
    if kept.shape[1] == 0:
        raise ValueError(
            f"helfem_no_truncated_basis: no NOs survive (n_keep={n_keep}, "
            f"occ_threshold={occ_threshold}, max_occ={occ.max():.3e})")
    return NAOAtomicBasis(underlying, kept), occ[keep_mask]


def install_full_eri(mf, basis):
    """Pre-compute the full Nbf^4 AO ERI tensor for `basis` and store it
    on mf._eri (8-fold packed form). Once installed, PySCF's
    standard ao2mo / CASCI / CASSCF / MP2 / CCSD etc. machinery works
    against the HelFEM basis through normal code paths -- no per-method
    overrides needed.

    !!! DEPRECATED for complex-Y_lm AOs (mmax > 0). !!!

    HelFEM's atomic basis labels each AO by a definite (l, m)
    quantum-mechanical magnetic-quantum-number, i.e. the AOs are
    complex spherical harmonics Y_l,m(Omega) * u(r). For lmax = 0 only
    the (l=0, m=0) shell exists and Y_00 is real-valued, so the
    AO ERI tensor has full 8-fold permutational symmetry and the
    s8-packed mf._eri + PySCF post-HF stack works correctly.

    For any shell with m != 0 (lmax > 0 / mmax > 0):
      - The AO ERI is REAL (m-conservation) but only 2-fold symmetric
        ((mn|ij) = (ij|mn)); the other PySCF-assumed symmetries
        ((mn|ij) = (nm|ij), (mn|ij) = (mn|ji)) FAIL because complex
        AOs conjugate the bra differently.
      - PySCF's CCSD / CASSCF / MP2 internals derive equations from
        the 8-fold-symmetric ERI; running them on a 2-fold-symmetric
        tensor gives WRONG correlation energies. (Empirically, He
        lmax=1 mmax=1 install_full_eri -> CCSD gave a 5.6 mEh wrong
        correlation; the correct route via radial_df_factors gave the
        right answer.)
      - Even with build_active_eri's symmetrisation bug fixed (this PR:
        asymmetric P, m-conservation respected), s8 packing throws
        away non-symmetric entries, and s1 packing into PySCF's CCSD
        still doesn't reach the right answer because PySCF assumes
        full 8-fold symmetry internally.

    The supported general path is:
        basis.radial_df_factors(tol) -> Bk
    consumed by libatomscf's cderi_from_radial_df + PySCF DF post-HF,
    which builds the AO ERI in the REAL spherical-harmonic basis where
    full 8-fold symmetry holds. See radial_df_factors() docstring on
    the C++ AtomicBasis side; the angular Gaunt assembly and PySCF DF
    wiring live in libatomscf, not here.

    This function now raises NotImplementedError when basis has any
    shell with m != 0. For real-Y_lm AOs (m == 0 only) it still works
    and is useful as a small-case cross-check.
    """
    mvals = list(basis.mvals())
    complex_shells = [(i, basis.lvals()[i], m) for i, m in enumerate(mvals) if m != 0]
    if complex_shells:
        raise NotImplementedError(
            "install_full_eri is complex-AO-unsafe: HelFEM uses complex Y_lm,\n"
            "and PySCF's 8-fold-symmetric post-HF machinery doesn't produce\n"
            f"correct correlation energies on a 2-fold-symmetric ERI.\n"
            f"  basis has {len(complex_shells)} shell(s) with m != 0:\n"
            f"    {complex_shells[:3]}{' ...' if len(complex_shells) > 3 else ''}\n"
            "\n"
            "Use the radial_df_factors() + libatomscf cderi path for post-HF\n"
            "calculations at lmax > 0 (or mmax > 0):\n"
            "\n"
            "    B = basis.radial_df_factors(tol=1e-12)\n"
            "    cderi = libatomscf.cderi_from_radial_df(B, basis.lvals(),\n"
            "                                            basis.mvals(),\n"
            "                                            basis.Nrad())\n"
            "    # feed cderi to PySCF DF-CCSD / DF-CASSCF / etc.\n"
            "\n"
            "install_full_eri remains correct (and useful as a cross-check)\n"
            "for the lmax == 0, mmax == 0 case (s-shell-only, real AOs)."
        )
    Nbf = basis.Nbf()
    # build_active_eri with identity coefficients returns the AO ERI.
    eri_unpacked = build_active_eri(basis, np.eye(Nbf))
    # Store in 8-fold-packed form (valid for real AOs, lmax=0 only).
    mf._eri = ao2mo.restore('s8', eri_unpacked, Nbf)
    return mf


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

    def radial_df_factors(self, tol=1e-10):
        """Density-fitted (Cholesky) BARE radial Slater integrals.

        Forwards to the underlying AtomicBasis: the NAO projection is an
        angular-side transformation (per-shell C matrix), so the RADIAL
        DF factors are inherited unchanged from the underlying FE basis.
        libatomscf consumes (lvals/mvals from the underlying basis) +
        (per-NAO angular coefficients) at angular assembly.

        Note: the radial structure (Nrad, the FE element layout) is
        invariant under the NAO projection; the per-NAO selection only
        affects which AOs are kept downstream.
        """
        return self._fe.radial_df_factors(tol)


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
