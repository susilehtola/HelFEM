#!/usr/bin/env python3
"""He natural-orbital truncation demo: run CASCI in a generous NAO basis,
extract NOs by occupation, see what fraction of orbitals carry the
wave function. Then re-run CASCI in the truncated NO basis to verify
the truncation preserves the correlation.
"""
import sys
import numpy as np

from helfem.pyscf_driver import (
    helfem_scf, helfem_nao_scf, helfem_casci, helfem_casscf,
    install_full_eri, natural_orbitals, helfem_no_truncated_basis,
    _build_scf_for_basis,
)

def main():
    print("=== HF for He at lmax=1, compact basis ===")
    mf_fe, basis_fe = helfem_scf(
        Z=2, lmax=1, mmax=0,
        primbas=4, nnodes=8, nelem=5, Rmax=10.0,
        verbose=0,
    )
    mf_fe.kernel()
    print()

    print("=== Big NAO basis: keep 5 per shell (10 NAOs total) ===")
    mf_nao, basis_nao = helfem_nao_scf(
        mf_fe, basis_fe, keep_per_shell=[5, 5],
    )
    e_hf = mf_nao.kernel()
    print(f"  Nbf = {basis_nao.Nbf()}, HF = {e_hf:.10f}")
    print()

    print("=== CASCI(2, 10) -- full active ===")
    mc_big = helfem_casci(mf_nao, basis_nao, ncas=10, nelecas=2)
    mc_big.verbose = 0
    e_big = mc_big.kernel()[0]
    print(f"  CAS(2, 10) = {e_big:.10f}   corr = {e_big - e_hf:+.6f} Eh")
    print()

    print("=== Natural orbital occupations ===")
    occ, no_coeff = natural_orbitals(mc_big)
    for i, n in enumerate(occ):
        print(f"  NO {i}: occ = {n:.6e}")
    print()

    # Find K such that cumulative occupation captures > 99.99% (out of 2 e).
    cum = np.cumsum(occ)
    total = occ.sum()
    for K in range(1, len(occ) + 1):
        if cum[K-1] / total > 0.999999:
            break
    print(f"=== {K} NOs capture > 99.9999% of the wave function ===")
    print(f"  Cumulative occupation at K=1..{len(occ)}: "
          f"{[f'{cum[i]/total:.6f}' for i in range(len(occ))]}")
    print()

    print(f"=== Re-run CASCI in K={K}-NO truncated basis ===")
    basis_no, kept_occ = helfem_no_truncated_basis(mc_big, basis_nao, n_keep=K)
    print(f"  Truncated basis Nbf = {basis_no.Nbf()}")
    mf_no = _build_scf_for_basis(basis_no, nelectron=2, spin=0, charge=0, verbose=0)
    install_full_eri(mf_no, basis_no)
    e_hf_no = mf_no.kernel()
    print(f"  HF (NO basis) = {e_hf_no:.10f}  (delta from NAO HF = {e_hf_no - e_hf:+.3e})")
    mc_trunc = helfem_casci(mf_no, basis_no, ncas=K, nelecas=2)
    mc_trunc.verbose = 0
    e_trunc = mc_trunc.kernel()[0]
    print(f"  CAS({nelec_to_str(2)}, {K}) on NO basis = {e_trunc:.10f}")
    print(f"  corr (NO truncated) = {e_trunc - e_hf_no:+.6f}")
    print(f"  vs full-basis CAS(2, 10) corr = {e_big - e_hf:+.6f}")
    print(f"  loss from truncation: {abs(e_trunc - e_big):.3e} Eh")

    print()
    print("--- Summary ---")
    print(f"  HF (NAO 10) = {e_hf:.6f}")
    print(f"  CAS(2, 10)  = {e_big:.6f}  corr {e_big - e_hf:+.4e}")
    print(f"  CAS(2, {K}) on NOs = {e_trunc:.6f}  corr {e_trunc - e_hf_no:+.4e}")
    print(f"  NO truncation loss: {abs(e_trunc - e_big):.3e} Eh")
    # NO truncation is lossy because the NOs aren't HF eigenstates --
    # HF in the NO basis converges to a slightly different (variationally
    # higher) reference. The CASCI values are very close though.
    assert abs(e_trunc - e_big) < 1e-4, "NO truncation should preserve correlation to ~1e-4"
    print("\nPASS")
    return 0

def nelec_to_str(n):  return str(n)

if __name__ == "__main__":
    sys.exit(main())
