#!/usr/bin/env python3
"""Compile the 118 gensap checkpoints into src/general/atomdb_data.cpp.

The database ships the *wave function*: per (Z, l) a set of radial
orbitals expanded in a shared finite-element basis, together with their
(generally fractional) occupations. Everything else -- the density, the
Hartree potential, the SAP effective charge -- is derived from these at
evaluation time, so the table never has to represent an object of higher
polynomial degree than the orbitals themselves.

Usage: gen_atomdb.py <checkpoint dir> <output.cpp> [occupation cutoff]

The cutoff decides which orbitals make it into the database. It has to
live here rather than in the SCF, because it is a property of the shipped
table and not of the calculation: the checkpoints keep everything the
solver produced.

It defaults to 1e-14, one part in ~1e13 of the smallest electron count in
the database, which is where a double stops resolving the total. Every
orbital whose occupation is above the noise floor is kept, so the atoms
come out neutral to 4e-12 electrons in total -- against 1.4e-13 for the
untruncated set, i.e. the cutoff is not what limits the neutrality.

Dropping to a cutoff of 0 keeps 8967 orbitals rather than 3412 and takes
the generated file from 13 MB to 33 MB, all of it orbitals occupied at
1e-15 and below. Raising it to the 1e-6 the SCF used to save at is what
gave the shipped table its spurious -2.6e-6/r tail: the discarded
occupation is exactly the effective charge left at infinity.
"""
import h5py
import numpy as np
import sys

SRC = sys.argv[1]
OUT = sys.argv[2]
CUTOFF = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-14
MAXZ = 118

meta = None
bval = None
norb = np.zeros((MAXZ, 4), dtype=int)
dropped = np.zeros(MAXZ)
occ_all = []
coef_all = []

for Z in range(1, MAXZ + 1):
    with h5py.File("%s/wf_Z%d.chk" % (SRC, Z), "r") as f:
        m = (int(f["sadatom_lmax"][()]), int(f["sadatom_nnodes"][()]),
             int(f["sadatom_primbas"][()]), int(f["sadatom_Nquad"][()]))
        b = np.array(f["sadatom_bval"]).ravel()
        if meta is None:
            meta, bval = m, b
        assert m == meta and np.array_equal(b, bval), "record %d has a different basis" % Z
        for l in range(meta[0] + 1):
            C = np.array(f["sadatom_Cal_%d" % l])       # (norb, Nbf), one row per orbital
            o = np.array(f["sadatom_occal_%d" % l]).ravel()
            assert C.shape[0] == o.size
            keep = o > CUTOFF
            dropped[Z - 1] += o[~keep].sum()
            C, o = C[keep], o[keep]
            norb[Z - 1, l] = C.shape[0]
            occ_all.append(o)
            coef_all.append(C.ravel())

occ_all = np.concatenate(occ_all)
coef_all = np.concatenate(coef_all)

# Whatever occupation is discarded here is charge the tabulated atom no
# longer carries, and the SAP effective charge tends to exactly that at
# large r: a database that drops 1e-6 electrons ships a -1e-6/r tail on a
# nominally neutral atom. Fail rather than let that happen quietly again.
if dropped.max() > 1e-10:
    worst = int(dropped.argmax()) + 1
    raise SystemExit(
        "cutoff %.1e discards %.2e electrons from Z=%d -- the tabulated atom "
        "would not be neutral. Lower the cutoff, or check that the SCF is "
        "still conserving particle number." % (CUTOFF, dropped.max(), worst))
lmax, nnodes, primbas, nquad = meta
Nbf = int(coef_all.size // occ_all.size)
nelem = bval.size - 1

# Offsets: orbital index at which each (Z, l) block starts.
offset = np.zeros((MAXZ, 4), dtype=int)
run = 0
for Z in range(MAXZ):
    for l in range(lmax + 1):
        offset[Z, l] = run
        run += norb[Z, l]
assert run == occ_all.size


def numbers(vals, per_line, indent):
    out = []
    for i in range(0, len(vals), per_line):
        chunk = ", ".join("%.17e" % v for v in vals[i:i + per_line])
        out.append(indent + chunk + ("," if i + per_line < len(vals) else ""))
    return "\n".join(out)


def ints(rows, indent):
    return "\n".join(
        indent + "{" + ", ".join("%d" % v for v in row) + "}" +
        ("," if i + 1 < len(rows) else "") for i, row in enumerate(rows))


with open(OUT, "w") as f:
    f.write('''/*
 *                This source code is part of
 *
 *                          HelFEM
 *                             -
 * Finite element methods for electronic structure calculations on small systems
 *
 * Written by Susi Lehtola, 2018-
 * Copyright (c) 2018- Susi Lehtola
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * See the LICENSE file at the root of this source distribution
 * for the full license text.
 */

/* GENERATED FILE -- DO NOT EDIT BY HAND.

   Tabulated spherically averaged atomic wave functions for H..Og,
   from spin-restricted LDA exchange-only calculations run with

     gensap --Z=$Z --method=lda_x --M=0 --lmax=%d --nelem=%d --nnodes=%d

   Occupations were never pinned: they are whatever the fractional
   occupation optimizer converged to, which for the d and f blocks is
   generally not an integer. Orbitals occupied at or below %.0e are not
   stored, which leaves each atom neutral to better than 1e-12 electrons
   -- the SAP effective charge of these atoms decays to zero rather than
   to the charge a truncation forgot.

   The radial basis is shared by every record: %d-node LIPs on the %d
   elements delimited by `bval`, giving %d radial functions. Storing the
   orbitals rather than the potential keeps the tabulated object at the
   lowest polynomial degree in the problem -- the density is degree 2x
   the orbitals and the potential higher still, and would need a finer
   grid than the one the orbitals were solved on.
*/

#include "atomdb.h"

namespace helfem {
  namespace atomdb {
    namespace data {

''' % (lmax, nelem, nnodes, nnodes, CUTOFF, nelem, Nbf))

    f.write("      const int max_Z = %d;\n" % MAXZ)
    f.write("      const int lmax = %d;\n" % lmax)
    f.write("      const int Nbf = %d;\n" % Nbf)
    f.write("      const int nelem = %d;\n" % nelem)
    f.write("      const int nnodes = %d;\n" % nnodes)
    f.write("      const int primbas = %d;\n" % primbas)
    f.write("      const int nquad = %d;\n" % nquad)
    f.write("      const int norbital = %d;\n\n" % occ_all.size)

    f.write("      /* Element boundaries of the shared radial grid. */\n")
    f.write("      const double bval[%d] = {\n" % bval.size)
    f.write(numbers(bval, 3, "        ") + "};\n\n")

    f.write("      /* Number of stored orbitals, [Z-1][l]. */\n")
    f.write("      const int norb[%d][%d] = {\n" % (MAXZ, lmax + 1))
    f.write(ints(norb, "        ") + "};\n\n")

    f.write("      /* Index of the first stored orbital of (Z, l), [Z-1][l]. */\n")
    f.write("      const int offset[%d][%d] = {\n" % (MAXZ, lmax + 1))
    f.write(ints(offset, "        ") + "};\n\n")

    f.write("      /* Orbital occupations, indexed by the offsets above. */\n")
    f.write("      const double occupations[%d] = {\n" % occ_all.size)
    f.write(numbers(occ_all, 3, "        ") + "};\n\n")

    f.write("      /* Orbital expansion coefficients, orbital-major: the %d\n"
            "         coefficients of orbital `iorb` start at iorb*Nbf. */\n" % Nbf)
    f.write("      const double coefficients[%d] = {\n" % coef_all.size)
    f.write(numbers(coef_all, 3, "        ") + "};\n\n")

    f.write("    }\n  }\n}\n")

print("wrote %s: %d orbitals, %d coefficients" % (OUT, occ_all.size, coef_all.size))
print("occupation cutoff %.1e; discarded %.2e electrons in total, at most %.2e "
      "from one atom" % (CUTOFF, dropped.sum(), dropped.max()))
