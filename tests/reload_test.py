#!/usr/bin/env python3
"""Exact-reload test for diatomic --save / --load.

A converged run is saved and loaded back into the same basis. The loaded
density is then the converged one, so the restart guess must reproduce it
and the first SCF energy must already be the converged energy.

The converged energy alone cannot catch a bad restart: DIIS recovers from a
degraded guess and converges to the same energy anyway, merely in more
iterations. That is how --load at --symmetry=3 went on seeding every
mirrored +m/-m block with half its electrons -- a full re-SCF for every
"restart" -- without any test failing.

N2 at --symmetry=3 has its occupied pi shell in a mirrored block; at
--symmetry=1 the +m and -m blocks are separate. Both, restricted and
unrestricted.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

ARGS = ["--Z1=7", "--Z2=7", "--Rbond=2.07", "--nelem=3", "--lmax=7,5",
        "--method=HF", "--M=1", "--verbosity=1"]


def run(cmd, cwd):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stdout.write(p.stdout + p.stderr)
        raise SystemExit(f"FAILED: {' '.join(cmd)} exited with {p.returncode}")
    return p.stdout


def energies(out):
    its = [float(x) for x in re.findall(r"^Iteration \d+: .*? energy (\S+)", out, re.M)]
    conv = re.findall(r"^Converged to energy (\S+)!", out, re.M)
    if not its or not conv:
        sys.stdout.write(out)
        raise SystemExit("FAILED: no SCF iterations or no convergence line")
    return its, float(conv[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bindir", required=True)
    args = ap.parse_args()
    exe = os.path.join(os.path.abspath(args.bindir), "diatomic")
    nfail = 0
    for symmetry, restricted in ((3, 1), (3, 0), (1, 1)):
        with tempfile.TemporaryDirectory() as d:
            common = [exe] + ARGS + [f"--symmetry={symmetry}", f"--restricted={restricted}"]
            _, E = energies(run(common + ["--save=a.chk"], d))
            out = run(common + ["--load=a.chk"], d)
            its, E2 = energies(out)
            seeded = "WARNING: the" not in out and "Restart guess reproduces" in out
            ok = seeded and abs(its[0] - E) < 1e-7 and abs(E2 - E) < 1e-9
            nfail += not ok
            print(f"symmetry={symmetry} restricted={restricted}: saved {E:.10f}, first iteration after "
                  f"load {its[0]:.10f}, {len(its)} iterations, guess "
                  f"{'reproduces' if seeded else 'does NOT reproduce'} the density: "
                  f"{'ok' if ok else 'FAIL'}")
    print("RELOAD TEST " + ("FAILED" if nfail else "PASSED"))
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
