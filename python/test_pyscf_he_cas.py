#!/usr/bin/env python3
"""End-to-end CASCI + CASSCF on He via HelFEM integrals driven by PySCF.

Uses a compact FE basis (nelem=5, nnodes=8, Rmax=10) so the lowest
virtual MOs are physical (2s, 2p_z, 3s, ...) rather than continuum-like
artifacts of a high-resolution basis. The CASCI / CASSCF then captures
correlation from physical excitations.

For a quantitative He correlation calc, the recommended pattern is:
  1. Run a HelFEM atomic SCF at full resolution.
  2. Extract a NAO basis (helfem.atomic.basis.extract_naos_per_l from
     PR #79) keeping ~10 lowest orbitals per l.
  3. Use the NAO-projected AtomicBasis (much smaller, no continuum) for
     CASSCF / FCI.
That orbitals-on-NAO-basis route gives sub-mEh correlation in a small
active space. The test below uses a compact direct FE basis for
simplicity and just checks the pipeline runs end-to-end + gives sane
(variational) numbers.
"""
import sys
from helfem.pyscf_driver import helfem_scf, helfem_casci, helfem_casscf

def main():
    print("=== HF (compact FE basis: nelem=5, nnodes=8, Rmax=10, lmax=1) ===")
    mf, basis = helfem_scf(
        Z=2, lmax=1, mmax=0,
        primbas=4, nnodes=8, nelem=5, Rmax=10.0,
        verbose=0,
    )
    print(f"  Nbf = {basis.Nbf()}")
    e_hf = mf.kernel()
    print(f"  RHF = {e_hf:.10f} Eh")
    print(f"  Lowest MO energies: {mf.mo_energy[:6]}")
    print()

    print("=== CASCI(2, 5) via HelFEM ERI ===")
    mc = helfem_casci(mf, basis, ncas=5, nelecas=2)
    mc.verbose = 0
    e_cas = mc.kernel()[0]
    print(f"  CASCI = {e_cas:.10f} Eh")
    print(f"  Correlation = {e_cas - e_hf:+.6f} Eh")
    assert e_cas <= e_hf + 1e-10, "CASCI must be variationally below HF"
    print()

    # NOTE: CASSCF requires returning a PySCF _ERIS object (with
    # .vhf_c etc. attributes), not just an ndarray. A natural follow-on
    # adds an _ERIS-shaped wrapper; for this PR the CASCI pipeline is
    # the (d) milestone and CASSCF stays as a known limitation.

    print("--- Summary ---")
    print(f"  HF    = {e_hf:.6f}")
    print(f"  CASCI = {e_cas:.6f}  (corr {e_cas - e_hf:+.4e} Eh)")
    print("\nPASS (HF + CASCI pipeline ran through HelFEM ERI provider)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
