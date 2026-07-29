#!/usr/bin/env python3
"""Regenerate PySCF's pyscf/dft/sap_data.py from the tabulated wave functions.

This is the Python twin of gen_sap_table.py: the same table of effective
charges, in the layout pyscf.dft.sap expects -- row 0 the radial grid,
rows 1..118 the effective charge of each element, linearly interpolated
between the tabulated radii.

Reads the "Z r Zeff ..." rows written by HelFEM's atomdb_dump, so the
PySCF table, HelFEM's sap.cpp and HelFEM's on-the-fly SAPFEAtom all
describe the same object.

The previous table was the exchange-only LDA potential of *unrestricted
Hartree-Fock* atoms. Fractionally occupied HF suffers from ghost
interaction error, so these are self-consistent spin-restricted LDA
exchange atoms instead, with the occupations left wherever the fractional
occupation optimizer put them.

Usage: objdir/src/atomdb_dump | gen_sap_pyscf.py path/to/sap_data.py
"""
import numpy as np
import sys

outfile = sys.argv[1]

Z, r, zeff = [], [], []
for line in sys.stdin:
    f = line.split()
    if len(f) < 3:
        continue
    Z.append(int(f[0]))
    r.append(float(f[1]))
    zeff.append(float(f[2]))
Z = np.array(Z)
nelem = Z.max()
nrad = (Z == 1).sum()
if Z.size != nelem * nrad:
    raise SystemExit("dump is ragged: %d rows for %d elements" % (Z.size, nelem))

rgrid = np.array(r[:nrad])
table = [rgrid] + [np.array(zeff[i * nrad:(i + 1) * nrad]) for i in range(nelem)]
for i in range(nelem):
    if not np.array_equal(np.array(r[i * nrad:(i + 1) * nrad]), rgrid):
        raise SystemExit("element %d is on a different radial grid" % (i + 1))
print("%d elements on %d radial points, cutoff %.14e" % (nelem, nrad, rgrid[-1]))

HEADER = '''# Copyright (c) 2020, Susi Lehtola
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
# * Neither the name of the <organization> nor the
# names of its contributors may be used to endorse or promote products
# derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
#
# GENERATED FILE -- DO NOT EDIT BY HAND. Regenerate with HelFEM's
# tools/gen_sap_pyscf.py.
#
# Routines for the implementation of the superposition of atomic
# potentials guess for electronic structure calculations, see
#
# S. Lehtola, "Assessment of Initial Guesses for Self-Consistent Field
# Calculations. Superposition of Atomic Potentials: Simple yet
# Efficient", J. Chem. Theory Comput. 15, 1593 (2019).
# DOI: 10.1021/acs.jctc.8b01089
#
# This function evaluates the effective charge of a neutral atom,
# Z(r) = Z - r [v_Hartree(r) + v_x(r)], from spherically symmetric
# spin-restricted exchange-only LDA calculations. The occupations are
# those found by the fractional occupation optimizer, which for the d
# and f blocks are generally not integers.
#
# S. Lehtola, L. Visscher, E. Engel, Efficient implementation of the
# superposition of atomic potentials initial guess for electronic
# structure calculations in Gaussian basis sets, J. Chem. Phys. 152,
# 144105 (2020). DOI: 10.1063/5.0004046
#
# The potentials have been calculated for the ground-states of
# spherically symmetric atoms at the non-relativistic level of theory
# as described in
#
# S. Lehtola, "Fully numerical calculations on atoms with fractional
# occupations and range-separated exchange functionals", Phys. Rev. A
# 101, 012516 (2020). DOI: 10.1103/PhysRevA.101.012516
#
# using accurate finite-element calculations as described in
#
# S. Lehtola, "Fully numerical Hartree-Fock and density functional
# calculations. I. Atoms", Int. J. Quantum Chem. e25945 (2019).
# DOI: 10.1002/qua.25945

import numpy
sap_Zeff = numpy.asarray([
'''

with open(outfile, "w") as f:
    f.write(HEADER)
    f.write(",\n".join(
        "[" + ", ".join(" %.14e" % v for v in row) + "]" for row in table))
    f.write("])\n")
print("wrote %s" % outfile)
