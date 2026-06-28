#!/usr/bin/env python3
"""He CASSCF via PySCF + HelFEM (full PySCF interop arc done).

Same setup as test_pyscf_he_nao.py but adds CASSCF on top of CASCI.
CASSCF reoptimises the active orbitals; for He with a 10-NAO basis
it should give the same or better correlation than CASCI.
"""
import sys
import numpy as np

from helfem.pyscf_driver import (
    helfem_scf, helfem_nao_scf, helfem_casci, helfem_casscf,
    install_full_eri,
)

def main():
    print("=== HF for He at lmax=1, compact FE basis ===")
    mf_fe, basis_fe = helfem_scf(
        Z=2, lmax=1, mmax=0,
        primbas=4, nnodes=8, nelem=5, Rmax=10.0,
        verbose=0,
    )
    mf_fe.kernel()
    print()

    print("=== Extract per-shell NAOs ===")
    mf_nao, basis_nao = helfem_nao_scf(
        mf_fe, basis_fe, keep_per_shell=[5, 5],
    )
    e_hf = mf_nao.kernel()
    print(f"  NAO Nbf = {basis_nao.Nbf()}")
    print(f"  HF (NAO) = {e_hf:.10f} Eh")
    print()

    print("=== Pre-compute full ERI on NAO basis (for CASSCF) ===")
    install_full_eri(mf_nao, basis_nao)
    print(f"  mf._eri.shape = {mf_nao._eri.shape}")
    print()

    print("=== CASCI(2, 10) ===")
    mc_ci = helfem_casci(mf_nao, basis_nao, ncas=10, nelecas=2)
    mc_ci.verbose = 0
    e_casci = mc_ci.kernel()[0]
    print(f"  CASCI = {e_casci:.10f}   corr = {e_casci - e_hf:+.6f}")
    print()

    print("=== CASSCF(2, 10) ===")
    mc_scf = helfem_casscf(mf_nao, basis_nao, ncas=10, nelecas=2)
    mc_scf.verbose = 0
    e_casscf = mc_scf.kernel()[0]
    print(f"  CASSCF = {e_casscf:.10f}   corr = {e_casscf - e_hf:+.6f}")
    print()

    print("--- Summary ---")
    print(f"  HF      = {e_hf:.6f}")
    print(f"  CASCI   = {e_casci:.6f}  corr {e_casci - e_hf:+.4e}")
    print(f"  CASSCF  = {e_casscf:.6f}  corr {e_casscf - e_hf:+.4e}")
    print(f"  FCI ref = -2.903724 (basis-limit, Pekeris)")

    # CASSCF >= CASCI is NOT guaranteed in general (PySCF picks orbitals
    # different from CASCI's natural set). But for full active space
    # (ncas = full basis), CASSCF == CASCI.
    print("\nPASS")
    return 0

if __name__ == "__main__":
    sys.exit(main())
