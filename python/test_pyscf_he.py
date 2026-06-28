#!/usr/bin/env python3
"""End-to-end test: He RHF via the HelFEM PySCF driver.

Reference: He HF/cc-pV5Z = -2.86167999561 Eh (this is the converged
basis-limit HF energy; should be reproducible to ~12 digits via the
HelFEM atomic FE basis).
"""
import sys
import numpy as np

from helfem.pyscf_driver import helfem_scf

REF_HE_HF = -2.861679995612  # Hartree-Fock basis limit for He

def main():
    mf, basis = helfem_scf(
        Z=2, lmax=0, mmax=0,
        primbas=4, nnodes=15, nelem=20, Rmax=40.0,
        verbose=4,  # show PySCF SCF iterations
    )
    print(f"\nHelFEM AtomicBasis: Nbf={basis.Nbf()}")
    print(f"PySCF: {type(mf).__name__} on {type(mf.mol).__name__}")
    print()

    e_hf = mf.kernel()
    print(f"\n--- Result ---")
    print(f"He HF energy      = {e_hf:.12f} Eh")
    print(f"Reference (basis limit) = {REF_HE_HF:.12f} Eh")
    print(f"Error             = {abs(e_hf - REF_HE_HF):.3e}")

    if abs(e_hf - REF_HE_HF) < 1e-9:
        print("\nPASS")
        return 0
    else:
        print("\nFAIL")
        return 1

if __name__ == "__main__":
    sys.exit(main())
