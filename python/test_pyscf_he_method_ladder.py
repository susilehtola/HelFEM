#!/usr/bin/env python3
"""He on a small NAO basis, running through the full PySCF post-HF
method ladder via helfem: HF -> MP2 -> CCSD -> FCI.

Demonstrates the install_full_eri pattern works for all of PySCF's
correlated methods (not just CASCI/CASSCF), and exercises the
classify_states helper with the new S^2 / 2S+1 entries.
"""
import sys
from helfem.pyscf_driver import (
    helfem_scf, helfem_nao_scf, helfem_casci, helfem_mp2, helfem_ccsd,
    helfem_fci, install_full_eri, classify_states,
)

def main():
    print("=== He at lmax=1 compact basis, NAO subset ===")
    mf_fe, basis_fe = helfem_scf(
        Z=2, lmax=1, mmax=0,
        primbas=4, nnodes=8, nelem=5, Rmax=10.0,
        verbose=0,
    )
    mf_fe.kernel()
    mf, basis = helfem_nao_scf(mf_fe, basis_fe, keep_per_shell=[5, 5])
    e_hf = mf.kernel()
    print(f"  Nbf(NAO) = {basis.Nbf()},  HF = {e_hf:.10f} Eh")

    print("\n=== MP2 ===")
    mp2 = helfem_mp2(mf, basis)
    e_corr_mp2, _ = mp2.kernel()
    print(f"  MP2 correlation = {e_corr_mp2:+.6f} Eh")
    print(f"  HF + MP2        = {e_hf + e_corr_mp2:.10f} Eh")

    print("\n=== CCSD ===")
    ccsd = helfem_ccsd(mf, basis)
    ccsd.verbose = 0
    e_corr_ccsd, _, _ = ccsd.kernel()
    print(f"  CCSD correlation = {e_corr_ccsd:+.6f} Eh")
    print(f"  HF + CCSD        = {e_hf + e_corr_ccsd:.10f} Eh")

    print("\n=== FCI (full basis) ===")
    fci = helfem_fci(mf, basis)
    fci.verbose = 0
    e_fci = fci.kernel()[0]
    print(f"  FCI energy      = {e_fci:.10f} Eh")
    print(f"  FCI correlation = {e_fci - e_hf:+.6f} Eh")

    # State classification (S^2 included).
    print("\n=== State classification (FCI, 3 lowest roots) ===")
    fci3 = helfem_fci(mf, basis)
    fci3.verbose = 0
    fci3.fcisolver.nroots = 3
    fci3.kernel()
    print(f"  {'state':>5} {'energy':>14} {'M_L':>6} {'L^2_orb':>9} {'S^2':>6} {'2S+1':>5}")
    for s in classify_states(fci3, basis):
        print(f"  {s['state']:>5} {s['energy']:>14.6f} {s['M_L']:>6.2f} "
              f"{s['L^2_orbital']:>9.4f} {s['S^2']:>6.3f} {s['2S+1']:>5.2f}")

    print("\n--- Summary ---")
    print(f"  HF        = {e_hf:.6f}")
    print(f"  HF + MP2  = {e_hf + e_corr_mp2:.6f}  corr {e_corr_mp2:+.4e}")
    print(f"  HF + CCSD = {e_hf + e_corr_ccsd:.6f}  corr {e_corr_ccsd:+.4e}")
    print(f"  FCI       = {e_fci:.6f}  corr {e_fci - e_hf:+.4e}")
    print(f"  Expected: MP2 < CCSD <= FCI (in magnitude of correlation)")

    # Sanity: For 2-electron systems CCSD == FCI (CCSD is exact for 2e).
    assert abs((e_hf + e_corr_ccsd) - e_fci) < 1e-7, \
        "CCSD should equal FCI for a 2-electron system"
    print("\nPASS (CCSD == FCI for 2-electron He, as expected)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
