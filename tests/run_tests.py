#!/usr/bin/env python3
"""Integration test driver for HelFEM.

Two modes:

    run_tests.py --generate refs/current.json     # run everything, record results
    run_tests.py --check    refs/current.json     # run everything, compare

The point of separating them is the cross-version workflow: generate a
reference set with a known-good build, then check a newer build against it.

    git checkout <old>; cmake --build ...; tests/run_tests.py --generate refs/pre-eigen.json
    git checkout master; cmake --build ...; tests/run_tests.py --check    refs/pre-eigen.json

What is compared
----------------
Not just the total energy. The drivers print a full decomposition

    kinetic ... nuclear ... Coulomb ... XC ... Exx ... total ... (nel err ...)

and every component is recorded, so a failure says *which* part moved: a
drifting XC with a stable kinetic points at the DFT grid, a drifting kinetic
points at the basis, and a drifting `nel err` points at the quadrature.

Converged occupations are recorded too. Systems with near-degenerate
solutions can converge to a *different state* rather than to a wrong number
(see task #45), and an occupation change is a qualitatively different failure
from a numerical drift -- the runner reports it as such instead of just
printing a large delta.

Determinism
-----------
Every case runs at OMP_NUM_THREADS=1. Converged energies are a deterministic
function of the thread count, so it is part of the reference. The recorded
metadata also captures the git commit and the BLAS backend, because both can
move a result without any source change.
"""

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import tempfile
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# "kinetic -12.34 nuclear -1.2 ... total -14.5 (nel err 1.0e-12)".
# Field names vary between codes and are conditional on the run (Enucr,
# Eefield, Emfield, Econf appear only when nonzero), so scrape pairs rather
# than matching a fixed layout.
PAIR_RE = re.compile(r"([A-Za-z_]+)\s+(-?\d+\.\d+)")
NELERR_RE = re.compile(r"nel err\s+(-?[\d.]+e[-+]\d+)")
OCC_RE = re.compile(r"^([ab]:sym\d+|sym\d+) occupations:\s*(.*)$")

# Builds from before the Eigen migration print a labelled block instead of the
# one-line decomposition. Supporting both dialects is what makes a cross-era
# reference comparison possible at all; the field names are normalised to the
# modern short forms so the two are directly comparable.
LEGACY_RE = re.compile(r"^([A-Za-z][A-Za-z \-]*?)\s+energy:\s*(-?\d+\.\d+)\s*$")
# gensap and aij print no decomposition at all, only a converged total.
CONVERGED_RE = re.compile(r"Converged to energy\s+(-?\d+\.\d+)")

# OpenOrbitalOptimizer prints this on success, so it is a universal marker for
# every current driver. A run killed by a timeout still prints per-iteration
# energies, and a late SCF iterate is close enough to a converged one to slip
# through any sane tolerance -- so a truncated run can silently become a
# reference that then "passes" forever while asserting nothing. Require the
# marker. Pre-Eigen builds predate OOO and never print it, so the requirement
# is skipped for the legacy output format.

LEGACY_FIELDS = {
    "Kinetic": "kinetic",
    "Nuclear attraction": "nuclear",
    "Nuclear repulsion": "Enucr",
    "Coulomb": "Coulomb",
    "Exact exchange": "Exx",
    "Exchange-correlation": "XC",
    "Total": "total",
}


def load_cases(path):
    with open(path) as f:
        spec = json.load(f)
    return spec


def parse_output(text):
    """Extract the energy decomposition and the converged occupations."""
    energy_line = None
    for line in text.splitlines():
        if "total" in line and "kinetic" in line:
            energy_line = line  # keep the last one
    result = {}
    if energy_line is not None:
        for name, value in PAIR_RE.findall(energy_line):
            result[name] = float(value)
        m = NELERR_RE.search(energy_line)
        if m:
            result["nel_err"] = float(m.group(1))
    else:
        # Pre-Eigen dialect: a labelled block near the end of the run.
        for line in text.splitlines():
            m = LEGACY_RE.match(line.strip())
            if m:
                label = " ".join(m.group(1).split())
                if label in LEGACY_FIELDS:
                    result[LEGACY_FIELDS[label]] = float(m.group(2))
        if result:
            result["_format"] = "legacy"
        else:
            # gensap / aij: total only.
            m = None
            for line in text.splitlines():
                mm = CONVERGED_RE.search(line)
                if mm:
                    m = mm
            if m:
                result["total"] = float(m.group(1))
                result["_format"] = "total-only"

    # Occupations are printed once per SCF iteration; keep the final block.
    occs, block = {}, {}
    for line in text.splitlines():
        m = OCC_RE.match(line.strip())
        if m:
            key, val = m.group(1), " ".join(m.group(2).split())
            if key in block:      # a new block started
                occs, block = block, {}
            block[key] = val
    if block:
        occs = block
    if occs:
        result["_occupations"] = occs

    iters = text.count("Iteration ")
    if iters:
        result["_iterations"] = iters
    result["_converged"] = bool(CONVERGED_RE.search(text))
    return result


def run_case(case, defaults, bindir, timeout_override=None):
    # Absolute: each case runs in its own temp cwd, so a relative binary path
    # would be resolved against that directory rather than against ours.
    binary = os.path.abspath(os.path.join(bindir, case["code"]))
    if not os.path.exists(binary):
        return {"_error": "binary not found: %s" % binary}

    env = dict(os.environ)
    env.update(defaults.get("env", {}))
    env.update(case.get("env", {}))

    # Each case gets its own working directory. The drivers write checkpoint
    # files (helfem.chk and friends) into the cwd under fixed names, so cases
    # sharing a directory clobber each other's files -- which shows up as
    # sporadic, method-dependent failures only when --jobs > 1.
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="helfem-test-") as workdir:
        # Cases that pin a configuration need an input file (occs.dat) next to
        # the run. Writing it into the per-case directory keeps concurrent
        # cases with different configurations from reading each other's.
        for fname, content in case.get("input_files", {}).items():
            with open(os.path.join(workdir, fname), "w") as fh:
                fh.write(content)
        try:
            proc = subprocess.run(
                [binary] + case["args"],
                capture_output=True, text=True, env=env, cwd=workdir,
                timeout=(timeout_override or case.get("timeout_s", defaults.get("timeout_s", 900))),
            )
        except subprocess.TimeoutExpired:
            return {"_error": "timeout"}
    elapsed = time.time() - t0

    out = proc.stdout + proc.stderr
    parsed = parse_output(out)
    if "total" not in parsed:
        tail = "\n".join(out.strip().splitlines()[-5:])
        return {"_error": "no energy in output (exit %d)\n%s" % (proc.returncode, tail)}
    # A one-electron system has no two-electron interaction, so there is no SCF
    # to iterate and OOO never prints its convergence marker -- the single
    # diagonalisation IS the answer. Requiring the marker there rejects an
    # exactly-correct result. Recognise it by the absence of any iteration
    # rather than by inspecting the arguments.
    trivially_exact = parsed.get("_iterations", 0) == 0
    if not parsed.get("_converged") and not trivially_exact \
       and parsed.get("_format") != "legacy":
        return {"_error": "did not converge (no 'Converged to energy' in output; "
                          "ran %.0fs, %d iterations) -- refusing to record an "
                          "unconverged energy as a reference"
                          % (elapsed, parsed.get("_iterations", 0))}
    parsed["_seconds"] = round(elapsed, 1)
    if proc.returncode != 0:
        parsed["_exit"] = proc.returncode
    return parsed


def metadata(bindir):
    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                  cwd=HERE, timeout=20).stdout.strip()
        except Exception:
            return "?"
    return {
        "git_commit": sh("git rev-parse HEAD"),
        "git_describe": sh("git log --oneline -1"),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "bindir": os.path.abspath(bindir),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "(per-case default)"),
        "flexiblas": sh("flexiblas print current 2>/dev/null | head -1") or "n/a",
    }


def select(cases, args):
    out = []
    for c in cases:
        if args.filter and not re.search(args.filter, c["name"]):
            continue
        if args.tag and args.tag not in c.get("tags", []):
            continue
        if args.skip_slow and "slow" in c.get("tags", []):
            continue
        # Cases parked pending an upstream fix: they fail loudly rather than
        # silently (the convergence guard refuses to record a value), but there
        # is no point running them until the fix lands.
        if "held-out" in c.get("tags", []) and not args.include_held_out:
            continue
        out.append(c)
    return out


def compare(name, ref, got, tol, ctol=None):
    """Return a list of human-readable problems."""
    problems = []
    if "_error" in got:
        return ["ERROR: %s" % got["_error"]]
    if "_error" in ref:
        return ["reference itself is an error entry; regenerate it"]

    ref_occ = ref.get("_occupations")
    got_occ = got.get("_occupations")
    if ref_occ and got_occ and ref_occ != got_occ:
        problems.append("OCCUPATIONS CHANGED (converged to a different state, "
                        "not merely a different number)")
        for k in sorted(set(ref_occ) | set(got_occ)):
            if ref_occ.get(k) != got_occ.get(k):
                problems.append("    %-12s ref %-16s got %s"
                                % (k, ref_occ.get(k, "-"), got_occ.get(k, "-")))

    # The total is the physically meaningful invariant and gets the tight
    # tolerance. Individual components get a looser one because they are far
    # more sensitive: near a stationary point the energy is quadratic in the
    # density error while each component is linear, so two builds that stop at
    # slightly different points inside the same convergence basin show
    # component drift of O(delta) and total drift of O(delta^2). Measured
    # across the Eigen+OOO rewrite: components moved up to 1.1e-4 while every
    # total held to better than 1e-6. Component drift is worth seeing; on its
    # own it is not a correctness failure.
    comp_tol = ctol if ctol is not None else tol * 100.0
    for field in ("kinetic", "nuclear", "Coulomb", "XC", "Exx", "Enucr"):
        if field not in ref or field not in got:
            continue
        delta = got[field] - ref[field]
        if abs(delta) > comp_tol:
            problems.append("    %-8s ref %18.10f  got %18.10f  delta %+.3e"
                            % (field, ref[field], got[field], delta))

    if "total" in ref and "total" in got:
        delta = got["total"] - ref["total"]
        if abs(delta) > tol:
            problems.append("    %-8s ref %18.10f  got %18.10f  delta %+.3e  <-- TOTAL"
                            % ("total", ref["total"], got["total"], delta))
        elif problems:
            # Components moved but the total held: show it, since that is the
            # difference between "converged elsewhere" and "computed wrongly".
            problems.append("    (total agrees: %+.3e within %.1e -- component "
                            "drift only)" % (delta, tol))
    return problems


def check_invariants(spec, results):
    """Cases that must agree with each other, independent of any reference."""
    failures = []
    for inv in spec.get("invariants", []):
        names = inv["cases"]
        field = inv.get("field", "total")
        tol = inv.get("tolerance", 1e-7)
        vals = {}
        for n in names:
            r = results.get(n)
            if r and field in r:
                vals[n] = r[field]
        if len(vals) < 2:
            continue  # not enough cases were run to compare
        lo, hi = min(vals.values()), max(vals.values())
        # A relative bound is the right one whenever the quantities compared
        # are converged to a relative accuracy rather than an absolute one --
        # e.g. the atomic<->diatomic cross-checks, where the basis is converged
        # by cbasis to a relative threshold. A flat absolute bound would pass
        # He (-2.9 Eh) and fail N (-54 Eh) for no physical reason, and would be
        # hopeless by Kr (-2750 Eh).
        rel = inv.get("relative_tolerance")
        if rel is not None:
            scale = max(abs(lo), abs(hi), 1.0)
            tol = rel * scale
        if hi - lo > tol:
            failures.append((inv["name"], inv.get("_why", ""), vals, hi - lo, tol))
    return failures


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--generate", metavar="FILE", help="run cases and write references")
    mode.add_argument("--check", metavar="FILE", help="run cases and compare to references")
    mode.add_argument("--smoke", action="store_true",
                      help="run cases and check invariants, without any reference "
                           "file. Fails on a crash, a timeout, or an invariant "
                           "violation. This is the portable check: invariants are "
                           "symmetry identities that hold on any machine, whereas "
                           "recorded energies depend on CPU and BLAS.")
    p.add_argument("--cases", default=os.path.join(HERE, "cases.json"))
    p.add_argument("--bindir", default=os.path.join(HERE, os.pardir, "objdir", "src"))
    p.add_argument("--filter", help="regex on case name")
    p.add_argument("--tag", help="only cases carrying this tag")
    p.add_argument("--skip-slow", action="store_true")
    p.add_argument("--include-held-out", action="store_true",
                   help="also run cases parked pending an upstream fix")
    p.add_argument("--timeout", type=float,
                   help="per-case timeout in seconds, overriding the case and "
                        "default values; needed for slow historical builds")
    p.add_argument("--jobs", "-j", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                   help="cases to run concurrently; each case is single-threaded, "
                        "so this does not affect any result")
    p.add_argument("--tolerance", type=float,
                   help="tolerance on the TOTAL energy (the correctness criterion)")
    p.add_argument("--component-tolerance", type=float,
                   help="tolerance on individual energy components; defaults to "
                        "100x --tolerance, since components drift faster than the "
                        "total between builds that converge to slightly different points")
    args = p.parse_args()

    spec = load_cases(args.cases)
    defaults = spec.get("defaults", {})
    cases = select(spec["cases"], args)
    if not cases:
        sys.exit("no cases selected")

    ref_data = {}
    if args.smoke:
        args.generate = None
    if args.check:
        with open(args.check) as f:
            ref_data = json.load(f)

    print("Running %d case(s) from %s" % (len(cases), os.path.basename(args.cases)))
    print("Binaries: %s\n" % os.path.abspath(args.bindir))

    # Each case is pinned to one thread, so cases parallelise cleanly across
    # cores without perturbing any result. Output is collected and printed in
    # case order so a run is diffable regardless of completion order.
    results, failed, errored = {}, [], []
    if args.jobs > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(run_case, c, defaults, args.bindir, args.timeout): c["name"]
                       for c in cases}
            done = 0
            for fut in concurrent.futures.as_completed(futures):
                done += 1
                sys.stderr.write("\r  [%d/%d] finished  " % (done, len(cases)))
                sys.stderr.flush()
            sys.stderr.write("\r" + " " * 40 + "\r")
            results = {futures[f]: f.result() for f in futures}

    for case in cases:
        name = case["name"]
        sys.stdout.write("  %-32s " % name)
        sys.stdout.flush()
        got = results.get(name) if args.jobs > 1 else run_case(case, defaults, args.bindir, args.timeout)
        results[name] = got

        if "_error" in got:
            print("ERROR  (%s)" % got["_error"].splitlines()[0])
            errored.append(name)
            continue

        if args.generate or args.smoke:
            print("%18.10f  (%.0fs)" % (got["total"], got.get("_seconds", 0)))
            continue

        ref = ref_data.get("cases", {}).get(name)
        if ref is None:
            print("%18.10f  NO REFERENCE" % got["total"])
            continue
        tol = args.tolerance or case.get("tolerance", defaults.get("tolerance", 1e-8))
        problems = compare(name, ref, got, tol, args.component_tolerance)
        if problems:
            print("FAIL")
            for line in problems:
                print("      " + line)
            failed.append(name)
        else:
            print("ok   %18.10f" % got["total"])

    inv_failures = check_invariants(spec, results)
    if inv_failures:
        print("\nInvariant violations:")
        for iname, why, vals, spread, tol in inv_failures:
            print("  %s  (spread %.3e > tol %.1e)" % (iname, spread, tol))
            if why:
                print("    %s" % why)
            for n, v in sorted(vals.items()):
                print("      %-32s %18.10f" % (n, v))

    if args.smoke:
        ok = not errored and not inv_failures
        print("\n%d ran, %d errored, %d invariant violation(s)"
              % (len(results) - len(errored), len(errored), len(inv_failures)))
        return 0 if ok else 1

    if args.generate:
        payload = {"metadata": metadata(args.bindir), "cases": results}
        os.makedirs(os.path.dirname(os.path.abspath(args.generate)), exist_ok=True)
        with open(args.generate, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        print("\nWrote %d reference(s) to %s" % (len(results), args.generate))
        if errored:
            print("WARNING: %d case(s) errored and were recorded as errors: %s"
                  % (len(errored), ", ".join(errored)))
        return 1 if errored else 0

    print("\n%d passed, %d failed, %d errored, %d invariant violation(s)"
          % (len(results) - len(failed) - len(errored), len(failed),
             len(errored), len(inv_failures)))
    if ref_data.get("metadata"):
        m = ref_data["metadata"]
        print("Reference from %s (%s)" % (m.get("git_describe", "?"), m.get("date", "?")))
    return 1 if (failed or errored or inv_failures) else 0


if __name__ == "__main__":
    sys.exit(main())
