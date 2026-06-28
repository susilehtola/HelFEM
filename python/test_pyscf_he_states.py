#!/usr/bin/env python3
"""He: multi-root CASCI + state classification via <M_L> and <L^2_orb>.

Demonstrates: run CASCI with nroots=K, classify each CI root by its
exact <M_L> and coarse <L^2_orbital> diagnostic. For He CAS(2, 10)
at lmax=1 we expect:
  - Ground state    : 1s^2     M_L=0  L^2_orb ~ 0    (1S)
  - First excited   : ~ 1s 2p_z, M_L=0  L^2_orb ~ 2   (3P / 1P M_L=0)
  - Pi excited (m=+/-1): M_L=+/-1, L^2_orb ~ 2
  etc.

Note: lmax=1, mmax=0 only has m=0 angular shells, so all states have
M_L=0. To see different M_L values, use mmax >= 1 (which adds p_+/-1
shells).
"""
import sys
import numpy as np

from helfem.pyscf_driver import (
    helfem_scf, helfem_nao_scf, helfem_casci, install_full_eri,
    classify_states,
)

def run(lmax, mmax, label):
    print(f"\n=== He at lmax={lmax}, mmax={mmax} ({label}) ===")
    mf_fe, basis_fe = helfem_scf(
        Z=2, lmax=lmax, mmax=mmax,
        primbas=4, nnodes=8, nelem=5, Rmax=10.0,
        verbose=0,
    )
    mf_fe.kernel()

    # Keep_per_shell: pick 3 NAOs per (l,m) shell.
    keep = [3] * basis_fe.Nang()
    mf_nao, basis_nao = helfem_nao_scf(mf_fe, basis_fe, keep_per_shell=keep)
    e_hf = mf_nao.kernel()
    print(f"  NAO Nbf = {basis_nao.Nbf()},  HF = {e_hf:.6f} Eh")
    print(f"  shells (l,m) = {list(zip(basis_fe.lvals(), basis_fe.mvals()))}")

    # Multi-root CASCI: 5 lowest states.
    mc = helfem_casci(mf_nao, basis_nao, ncas=basis_nao.Nbf(), nelecas=2)
    mc.verbose = 0
    mc.fcisolver.nroots = 10
    mc.kernel()

    print(f"\n  State classification (5 roots):")
    print(f"  {'state':>5} {'energy (Eh)':>14} {'M_L':>6}  {'L^2_orb':>10}  {'character':>20}")
    print(f"  {'-'*5} {'-'*14} {'-'*6}  {'-'*10}  {'-'*20}")
    for s in classify_states(mc, basis_nao):
        # Character guess
        ml = s["M_L"]
        l2 = s["L^2_orbital"]
        if abs(l2) < 0.5:
            char = "S-like (l~0)"
        elif abs(l2 - 2.0) < 0.5:
            char = "P-like (l~1)"
        elif abs(l2 - 1.0) < 0.5:
            char = "mixed S+P"
        else:
            char = f"l^2_avg~{l2:.1f}"
        print(f"  {s['state']:>5} {s['energy']:>14.6f} {ml:>6.2f}  {l2:>10.4f}  {char:>20}")


def main():
    run(lmax=1, mmax=0, label="m=0 only -- all states M_L=0")
    run(lmax=1, mmax=1, label="m=-1..+1 -- p_x, p_y in basis")
    print("\nPASS")
    return 0

if __name__ == "__main__":
    sys.exit(main())
