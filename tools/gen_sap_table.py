#!/usr/bin/env python3
"""Regenerate the tabulated SAP effective charge in src/general/sap.cpp.

Reads the "Z r Zeff" rows written by the atomdb_dump binary and splices a
fresh table into sap.cpp, leaving the interpolation code around it alone.

Going through atomdb_dump rather than through gensap's result_<El>.dat
files means the table is a tabulation of exactly what SAPFEAtom
evaluates, so the interpolated and on-the-fly potentials differ only by
the interpolation itself.

Usage: objdir/src/atomdb_dump | gen_sap_table.py src/general/sap.cpp
"""
import numpy as np
import re
import sys

sapfile = sys.argv[1]

Z, r, zeff = [], [], []
for line in sys.stdin:
    f = line.split()
    if len(f) != 3:
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


def fmt(row, last):
    """One row of the 2D array in the layout the existing file uses:
    %.14e, three per line, each row opened with its own { and closed with
    },  -- the final row closing the outer brace too."""
    vals = ["%.14e" % v for v in row]
    out, line = [], "      {"
    for i, v in enumerate(vals):
        tail = ("}}" if last else "},") if i + 1 == len(vals) else ","
        if i and i % 3 == 0:
            out.append(line.rstrip())
            line = "       "
        line += v + tail + " "
    out.append(line.rstrip())
    return "\n".join(out)


body = "\n".join(fmt(row, i + 1 == len(table)) for i, row in enumerate(table))

src = open(sapfile).read()
src = re.sub(r"#define SAP_NRAD \d+", "#define SAP_NRAD %d" % nrad, src)
src = re.sub(r"#define SAP_NELEM \d+", "#define SAP_NELEM %d" % (nelem + 1), src)
src = re.sub(r"(double sap_cutoff_radius\(\) \{ return )[^;]+(; \})",
             r"\g<1>%.14e\g<2>" % rgrid[-1], src)
head, rest = src.split("static const double Zeff[SAP_NELEM][SAP_NRAD] = {\n", 1)
_, tail = rest.split("}};\n", 1)
open(sapfile, "w").write(
    head + "static const double Zeff[SAP_NELEM][SAP_NRAD] = {\n" + body + ";\n" + tail)
print("rewrote %s" % sapfile)
