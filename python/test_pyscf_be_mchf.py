#!/usr/bin/env python3
"""Be 1s^2(2s^2 + 2p^2) multi-configuration HF via PySCF + HelFEM.

The original Froese-Fischer MC-HF calculation: Be ground state with
the 1s shell frozen as core, the 2s + 2p valence as a 4-orbital
active space holding the 2 valence electrons. CASSCF reoptimises
both the active-orbital shapes and the CI coefficients.

Reference: experimental Be ground state energy = -14.6674 Eh.
HF/basis-limit Be = -14.5730 Eh, so correlation ~ 94 mEh.
The 1s^2(2s^2 + 2p^2) MC-HF captures the dominant near-degeneracy
correlation (~50 mEh), the rest comes from dynamical correlation
needing higher excitations / larger active spaces.
"""
import sys
import numpy as np

from helfem.pyscf_driver import (
    helfem_scf, helfem_nao_scf, helfem_casci, helfem_casscf,
    install_full_eri,
)

REF_BE_HF       = -14.573023
REF_BE_MCHF_2s2p = -14.6172    # approximate Froese-Fischer MC-HF result
REF_BE_EXACT    = -14.66736

def main():
    print("=== HF for Be (Z=4, lmax=1, mmax=1) -- compact FE basis ===")
    mf_fe, basis_fe = helfem_scf(
        Z=4, lmax=1, mmax=1,
        primbas=4, nnodes=8, nelem=5, Rmax=10.0,
        verbose=0,
    )
    print(f"  FE Nbf = {basis_fe.Nbf()}, shells = {list(zip(basis_fe.lvals(), basis_fe.mvals()))}")
    e_hf_fe = mf_fe.kernel()
    print(f"  RHF (full FE) = {e_hf_fe:.10f} Eh   (ref ~{REF_BE_HF})")
    print()

    print("=== Extract NAOs: 2 s + 1 each (p_-1, p_0, p_+1) = 5 NAOs ===")
    # Angular shell order from angular_basis(lmax=1, mmax=1):
    # likely (0,0), (1,-1), (1,0), (1,+1).  Keep 2 from s, 1 from each p.
    keep = []
    for (l, m) in zip(basis_fe.lvals(), basis_fe.mvals()):
        if l == 0:
            keep.append(2)
        elif l == 1:
            keep.append(1)
        else:
            keep.append(0)
    print(f"  keep_per_shell = {keep}")
    mf_nao, basis_nao = helfem_nao_scf(mf_fe, basis_fe, keep_per_shell=keep)
    e_hf_nao = mf_nao.kernel()
    print(f"  NAO Nbf = {basis_nao.Nbf()}")
    print(f"  RHF (NAO) = {e_hf_nao:.10f} Eh   (matches FE: delta = {e_hf_nao - e_hf_fe:+.3e})")
    print(f"  NAO MO energies: {mf_nao.mo_energy}")
    print()

    print("=== Install full AO ERI in NAO basis ===")
    install_full_eri(mf_nao, basis_nao)
    print(f"  mf._eri.shape = {mf_nao._eri.shape}")
    print()

    # For Be: 4 electrons, ncore = (4 - 2) / 2 = 1 (auto from CASCI).
    # active = 2s + 2p_-1 + 2p_0 + 2p_+1 = 4 orbitals.
    # 2 active electrons -> CASCI(2, 4) is exactly Froese-Fischer's
    # 1s^2(2s^2 + 2p^2) (the 2p^2 part covers all three p orientations
    # by symmetry).
    print("=== CASCI(2, 4)  -- 1s^2 frozen, 2s+2p active ===")
    mc_ci = helfem_casci(mf_nao, basis_nao, ncas=4, nelecas=2)
    mc_ci.verbose = 0
    e_ci = mc_ci.kernel()[0]
    print(f"  ncore = {mc_ci.ncore} (PySCF auto-set)")
    print(f"  CASCI = {e_ci:.10f}   corr = {e_ci - e_hf_nao:+.6f} Eh")
    print()

    print("=== CASSCF(2, 4) -- the Froese-Fischer MC-HF calc ===")
    mc_scf = helfem_casscf(mf_nao, basis_nao, ncas=4, nelecas=2)
    mc_scf.verbose = 0
    e_scf = mc_scf.kernel()[0]
    print(f"  CASSCF = {e_scf:.10f}   corr = {e_scf - e_hf_nao:+.6f} Eh")
    print()

    print("--- Summary (Be ground state) ---")
    print(f"  HF (NAO)              = {e_hf_nao:.6f}")
    print(f"  CASCI(2, 4)           = {e_ci:.6f}")
    print(f"  CASSCF(2, 4) (MC-HF)  = {e_scf:.6f}")
    print(f"  Froese-Fischer ref    = {REF_BE_MCHF_2s2p:.6f}")
    print(f"  Exact (FCI extrap.)   = {REF_BE_EXACT:.6f}")
    print(f"  HF/basis-limit ref    = {REF_BE_HF:.6f}")
    print()
    print(f"  MC-HF correlation captured: {abs(e_scf - e_hf_nao)*1000:.2f} mEh")
    print(f"  Total Be corr ~94 mEh; MC-HF captures the near-degeneracy piece (~50 mEh).")
    print(f"  Remaining ~40 mEh = dynamical correlation, needs higher excitations.")
    assert e_scf <= e_ci + 1e-10, "CASSCF should be at or below CASCI"
    print("\nPASS")
    return 0

if __name__ == "__main__":
    sys.exit(main())
