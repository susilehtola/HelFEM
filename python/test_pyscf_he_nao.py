#!/usr/bin/env python3
"""He CASCI on an NAO-projected basis -- the recommended workflow.

Pipeline:
  1. High-resolution HelFEM atomic SCF.
  2. Extract per-shell NAOs (lowest few MOs per angular shell).
  3. Build NAOAtomicBasis on the small NAO subspace.
  4. CASCI on the NAO basis -- physical orbitals, real correlation.

For He at lmax=1, keep_per_shell=[5, 5] gives 10 NAOs (5 s + 5 p_z).
CAS(2, 10) in the NAO basis captures the angular correlation that
the full FE basis's continuum-dominated virtuals fail to express
in a CAS(2, 5).
"""
import sys
import numpy as np

from helfem.pyscf_driver import (
    helfem_scf, helfem_nao_scf, helfem_casci, NAOAtomicBasis,
)

def main():
    # Use a compact FE basis (nelem=5, nnodes=8, Rmax=10) so the HF
    # virtuals are physical (2s, 2p_z, ...) rather than continuum-like
    # artifacts of a high-resolution basis. The NAO extraction is mostly
    # a no-op here (every MO is kept after per-shell slicing); see the
    # commit message and PR description for the recommended NO-iteration
    # or sadatom-NAO recipe when starting from a high-resolution basis.
    print("=== Step 1: HF for He at lmax=1 (compact FE basis) ===")
    mf_fe, basis_fe = helfem_scf(
        Z=2, lmax=1, mmax=0,
        primbas=4, nnodes=8, nelem=5, Rmax=10.0,
        verbose=0,
    )
    print(f"  FE Nbf = {basis_fe.Nbf()}, shells = {list(zip(basis_fe.lvals(), basis_fe.mvals()))}")
    e_hf_fe = mf_fe.kernel()
    print(f"  RHF (full FE) = {e_hf_fe:.10f} Eh")
    print()

    print("=== Step 2: extract per-shell NAOs (5 per shell) ===")
    mf_nao, basis_nao = helfem_nao_scf(
        mf_fe, basis_fe,
        keep_per_shell=[5, 5],   # 5 s + 5 p_z = 10 NAOs total
    )
    print(f"  NAO Nbf = {basis_nao.Nbf()}")
    e_hf_nao = mf_nao.kernel()
    print(f"  RHF (NAO subspace) = {e_hf_nao:.10f} Eh")
    print(f"  Delta vs full FE: {e_hf_nao - e_hf_fe:+.3e} Eh")
    print(f"  NAO MO energies: {mf_nao.mo_energy[:6]}")
    print()

    print("=== Step 3: CASCI(2, 5) on NAO basis ===")
    mc = helfem_casci(mf_nao, basis_nao, ncas=5, nelecas=2)
    mc.verbose = 0
    e_cas_5 = mc.kernel()[0]
    corr_5 = e_cas_5 - e_hf_nao
    print(f"  CAS(2, 5) = {e_cas_5:.10f} Eh   corr = {corr_5:+.6f}")
    print()

    print("=== Step 4: CASCI(2, 10) -- full NAO subspace ===")
    mc10 = helfem_casci(mf_nao, basis_nao, ncas=10, nelecas=2)
    mc10.verbose = 0
    e_cas_10 = mc10.kernel()[0]
    corr_10 = e_cas_10 - e_hf_nao
    print(f"  CAS(2, 10) = {e_cas_10:.10f} Eh   corr = {corr_10:+.6f}")
    print()

    print("--- Summary ---")
    print(f"  HF (full FE)     = {e_hf_fe:.6f}")
    print(f"  HF (NAO subset)  = {e_hf_nao:.6f}")
    print(f"  CAS(2, 5)        = {e_cas_5:.6f}  corr {corr_5:+.4e}")
    print(f"  CAS(2, 10) = full = {e_cas_10:.6f}  corr {corr_10:+.4e}")
    print(f"  He FCI / basis-limit ref: -2.903724")

    # NAO-subspace HF should equal full-FE HF (NAOs span the occupied).
    assert abs(e_hf_nao - e_hf_fe) < 1e-9, "NAO HF differs from FE HF"
    # CAS should be variationally below HF in the NAO basis.
    assert e_cas_5 <= e_hf_nao + 1e-10
    assert e_cas_10 <= e_cas_5 + 1e-10
    print("\nPASS")
    return 0

if __name__ == "__main__":
    sys.exit(main())
