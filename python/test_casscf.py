#!/usr/bin/env python3
"""Two-step CASSCF against PySCF driven with the SAME HelFEM integrals.

This is the tightest check available in this project: two independent orbital
optimizers, identical input integrals, must land on the same minimum. It tests
the active-space export, the RDM contract, the generalized Fock and the
optimizer together, and unlike a derivative check it cannot be fooled by a
consistently wrong energy expression -- PySCF is not sharing any of the
algebra, only the integrals.

Measured on Be CAS(2,3) over a 1s core (Nbf=14): the two agree to 1.6e-10.
"""
import sys

import numpy as np

from helfem import AtomicBasis
from helfem.pyscf_driver import helfem_scf, casci, casscf

TOL_AGREE = 1e-8        # measured 1.6e-10
TOL_GRAD = 1e-5         # preconditioned steepest descent; measured 2.3e-7


def main():
    from pyscf import mcscf

    b = AtomicBasis(Z=4, lmax=0, mmax=0, primbas=4, nnodes=6,
                    nelem=3, Rmax=20.0, igrid=4, zexp=2.0)
    nbf = b.Nbf()
    # full_eri: PySCF's mcscf goes through ao2mo, so it needs the tensor.
    mf = helfem_scf(b, 4, full_eri=True)
    assert mf.converged, "reference RHF did not converge"
    print(f"Be  Nbf={nbf}  RHF = {mf.e_tot:.12f}")

    ninact, nact, nelecas = 1, 3, (1, 1)
    E_casci = casci(b, mf.mo_coeff, ninact, nact, nelecas)[0]
    print(f"CASCI at RHF orbitals      = {E_casci:.12f}")

    E_ours, C_opt, info = casscf(b, mf.mo_coeff, ninact, nact, nelecas)
    print(f"HelFEM two-step CASSCF     = {E_ours:.12f}")
    print(f"   macro={info['macro']}  |g|={info['grad_norm']:.3e}  "
          f"CI solves={info['ci_solves']}  converged={info['converged']}")
    assert info["converged"], f"CASSCF did not converge: {info}"
    assert info["grad_norm"] < TOL_GRAD
    assert E_ours < E_casci + 1e-12, "CASSCF must not be above CASCI"

    mc = mcscf.CASSCF(mf, nact, sum(nelecas))
    mc.conv_tol, mc.conv_tol_grad = 1e-12, 1e-8
    mc.max_cycle_macro, mc.verbose = 200, 0
    mc.kernel()
    print(f"PySCF mcscf.CASSCF         = {mc.e_tot:.12f}  (ncore={mc.ncore})")
    assert mc.ncore == ninact, f"PySCF chose a different core: {mc.ncore}"

    diff = abs(E_ours - mc.e_tot)
    print(f"\n|HelFEM - PySCF| = {diff:.3e}")
    assert diff < TOL_AGREE, f"optimizers disagree by {diff:.3e}"

    # The orbitals themselves must NOT be expected to match. Beyond the usual
    # active-active redundancy, this solution has an active natural orbital
    # with zero occupation: all its 1- and 2-RDM elements vanish, so its
    # column of the generalized Fock is exactly zero and every rotation
    # involving it is EXACTLY redundant. The two codes stop at different
    # points along that flat direction -- measured, the occupied spaces
    # differ in one direction by 3.9 degrees while the energies agree to
    # 1.6e-10.
    #
    # Worth knowing for the second-order path: this is a genuine null space of
    # the orbital Hessian that survives after active-active pairs are already
    # excluded, so a solver that assumes non-singularity will need to project
    # it out (and HelFEM's conditioning report, which treats exact zeros in
    # the Hessian diagonal as a bug, will flag it).
    _, Dfin, _ = _final_rdms(b, C_opt, ninact, nact, nelecas)
    occ = np.linalg.eigvalsh(Dfin)[::-1]
    print("active natural occupations: "
          + "  ".join(f"{o:.8f}" for o in occ))
    assert abs(occ.sum() - sum(nelecas)) < 1e-9

    # What IS determined: the inactive (core) density, an observable.
    S = b.overlap()
    P_ours = 2.0 * C_opt[:, :ninact] @ C_opt[:, :ninact].T
    P_ref = 2.0 * mc.mo_coeff[:, :ninact] @ mc.mo_coeff[:, :ninact].T
    dcore = np.abs(P_ours - P_ref).max()
    print(f"core density |P_ours - P_pyscf|max = {dcore:.3e}")
    assert abs(np.trace(P_ours @ S) - 2 * ninact) < 1e-9
    assert dcore < 1e-4, f"core densities differ by {dcore:.3e}"

    print("\nCASSCF TESTS PASSED")
    return 0


def _final_rdms(basis, C, ninact, nact, nelecas):
    from pyscf import fci
    from helfem.pyscf_driver import active_hamiltonian
    E0, heff, eri = active_hamiltonian(basis, C, ninact, nact)
    e, civec = fci.direct_spin1.kernel(heff, eri, nact, nelecas, ecore=E0)
    D, d = fci.direct_spin1.make_rdm12(civec, nact, nelecas)
    return e, D, d


if __name__ == "__main__":
    sys.exit(main())
