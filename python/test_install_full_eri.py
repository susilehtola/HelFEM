#!/usr/bin/env python3
"""Regression test for install_full_eri / build_active_eri.

Three checks:
 (1) Bug-fix verification: build_active_eri (asymmetric P) produces an
     AO ERI tensor that satisfies m-conservation for complex Y_lm AOs.
     The old code symmetrised c_i c_j^T + c_j c_i^T which mixed
     (mn|ij) with (mn|ji) -- different m-conservation patterns --
     contaminating the tensor with m-non-conserving entries.

 (2) Real-AO case (lmax = 0): install_full_eri still works and gives a
     self-consistent CCSD result for the He s-shell-only basis.

 (3) Complex-AO case (lmax > 0): install_full_eri raises a clear
     NotImplementedError pointing to the radial_df_factors path.

The 5.6 mEh wrong-correlation bug at He lmax=1 mmax=1 (reported by
the external consumer) is what motivated the symmetrisation fix +
the explicit complex-AO guard. The supported general path for
post-HF with HelFEM's complex AOs is radial_df_factors + libatomscf's
cderi (which builds the AO ERI in the real spherical-harmonic basis
where full 8-fold symmetry holds and PySCF post-HF works).
"""
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import helfem
from helfem.pyscf_driver import (
    helfem_scf, build_active_eri, install_full_eri,
)


def check(cond, msg):
    if not cond:
        print(f"  FAIL: {msg}")
        sys.exit(1)
    print(f"  ok: {msg}")


def test_m_conservation_complex_AOs():
    """Build the He lmax=1 mmax=1 AO ERI via build_active_eri (after
    the asymmetric-P fix) and confirm m-conservation holds for every
    nonzero entry."""
    print("\n=== Test 1: m-conservation for complex Y_lm AOs ===")
    P = dict(Z=2, lmax=1, mmax=1, nnodes=6, nelem=2, Rmax=40)
    b = helfem.AtomicBasis(**P)
    Nbf, Nrad = b.Nbf(), b.Nrad()
    mvals = list(b.mvals())
    print(f"  Nbf={Nbf}, shells={list(zip(b.lvals(), mvals))}")

    eri = build_active_eri(b, np.eye(Nbf))

    # m_l of AO index p = iang*Nrad + irad
    m_a = np.array([mvals[p // Nrad] for p in range(Nbf)])
    # m-conservation: m_a - m_b + m_c - m_d == 0 for any nonzero entry
    nonzero = np.abs(eri) > 1e-10
    ML = m_a[:, None, None, None] - m_a[None, :, None, None] + \
         m_a[None, None, :, None] - m_a[None, None, None, :]
    violations = nonzero & (ML != 0)
    print(f"  nonzero entries: {nonzero.sum()}")
    print(f"  m-conservation violations: {violations.sum()}")
    check(violations.sum() == 0,
          "all nonzero (mn|ij) satisfy m-conservation m_m - m_n + m_i - m_j = 0")


def test_install_full_eri_lmax0():
    """For lmax = 0 (real-AO basis), install_full_eri works correctly
    and produces a self-consistent CCSD result."""
    print("\n=== Test 2: install_full_eri at lmax = 0 (real-AO case) ===")
    P = dict(Z=2, lmax=0, mmax=0, nnodes=6, nelem=2, Rmax=40)
    b = helfem.AtomicBasis(**P)
    print(f"  Nbf={b.Nbf()}, shells={list(zip(b.lvals(), b.mvals()))}")
    mf, _ = helfem_scf(**P)
    mf.mol.incore_anyway = True
    mf.kernel()
    install_full_eri(mf, b)
    check(mf._eri is not None, "install_full_eri sets mf._eri")

    from pyscf import cc
    ccs = cc.CCSD(mf); ccs.verbose = 0; ccs.kernel()
    print(f"  E_HF = {mf.e_tot:.8f}, E_corr(CCSD) = {ccs.e_corr:.6f}")
    # For He lmax=0 with this compact basis: ~-0.0227 (basis-dependent;
    # basis-limit partial-wave value is ~-0.0126).
    check(-0.035 < ccs.e_corr < -0.005,
          "CCSD correlation in expected range for compact s-only He basis")


def test_install_full_eri_complex_raises():
    """For lmax > 0 (complex-Y_lm AOs), install_full_eri raises a clear
    NotImplementedError pointing at radial_df_factors."""
    print("\n=== Test 3: install_full_eri raises for complex AOs ===")
    P = dict(Z=2, lmax=1, mmax=1, nnodes=6, nelem=2, Rmax=40)
    b = helfem.AtomicBasis(**P)
    mf, _ = helfem_scf(**P); mf.kernel()

    raised = False
    msg = ""
    try:
        install_full_eri(mf, b)
    except NotImplementedError as e:
        raised = True
        msg = str(e)
    check(raised, "install_full_eri raises NotImplementedError for lmax=1 mmax=1")
    check("radial_df_factors" in msg,
          "error message points at radial_df_factors as the supported route")
    check("complex" in msg.lower() or "m != 0" in msg,
          "error message explains the complex-AO root cause")


def main():
    test_m_conservation_complex_AOs()
    test_install_full_eri_lmax0()
    test_install_full_eri_complex_raises()
    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
