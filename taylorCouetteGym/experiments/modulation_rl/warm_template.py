#!/usr/bin/env python3
"""Warm the graded Sc=1075 RL case ONCE at the fixed mean speed (constant
300 rpm) for 60 s (~2.3 residence times; tau = V/Q ~ 26 s, see the inlet BC
comment in cases/side_outlet_grad_case/0.orig/U) and cache the final state as
0.warmed/ in a template clone of side_outlet_grad_case.

This is the continuous-manufacturing IC for the D-only modulation study:
episodes start from the statistically steady constant-300 operating state
instead of the pristine startup transient, so wallFlux decorrelates from the
episode clock and each block's flux is attributable to its own duty choice.
The base case in cases/ is NOT touched -- its pristine-IC benchmark contract
(taylor_couette_modulation.py docstring: "do not cache a 0.warmed/ in the
case") stays intact because Helpers.reset_case prefers 0.warmed over 0.orig
wherever it exists. The warmed template lives under results/warmed_grad300/
and trainers fan workers out from it exactly like
experiments/parallelized_catalysis_rl (warm-once-and-fan-out).

The warmup doubles as the constant-300 steady baseline: the rlMetrics lines
(t, Mz_kin, conv, cupC, wallFlux) are saved to warmup_metrics.csv and the
steady levels printed -- these set the wallFlux obs normalizer for the warmed
env and the constant-300 comparison point.

USAGE
    cd experiments/modulation_rl
    nohup python3 -u warm_template.py > results/warmed_grad300.log 2>&1 &
"""
import csv
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from taylor_couette_mixing.envs.helpers import Helpers  # noqa: E402

RPM = 2.0 * np.pi / 60.0
WARM_RPM = 300.0        # the D-only design's fixed mean w_b
WARM_DURATION = 60.0    # ~2.3 tau; conv reaches steady in ~40 s on this case

CASE_SRC = os.path.join(ROOT, "taylor_couette_mixing", "cases", "side_outlet_grad_case")
OUT_DIR = os.path.join(HERE, "results", "warmed_grad300")
CASE = os.path.join(OUT_DIR, "side_outlet_grad_case")


def clone_pristine():
    """Fresh clone whose 0/ is the PRISTINE IC. Prefer the shipped 0.orig/ (the
    Helpers-convention snapshot of the true IC) over 0/, which may carry stale
    omega entries from earlier runs; strip caches so the warmup provably starts
    from rest."""
    if os.path.isdir(CASE):
        shutil.rmtree(CASE)
    os.makedirs(OUT_DIR, exist_ok=True)
    shutil.copytree(CASE_SRC, CASE)

    orig = os.path.join(CASE, "0.orig")
    zero = os.path.join(CASE, "0")
    if os.path.isdir(orig):
        if os.path.isdir(zero):
            shutil.rmtree(zero)
        os.rename(orig, zero)
    for name in os.listdir(CASE):
        p = os.path.join(CASE, name)
        if not os.path.isdir(p):
            continue
        if name in ("0.warmed", "postProcessing", "dynamicCode"):
            shutil.rmtree(p)
            continue
        try:
            t = float(name)
        except ValueError:
            continue
        if t != 0.0:
            shutil.rmtree(p)


def main():
    print(f"[warm] cloning {CASE_SRC} -> {CASE}", flush=True)
    clone_pristine()

    h = Helpers(case_path=CASE)
    h.reset_case(mode="hard")   # snapshots pristine 0/ -> 0.orig, endTime=0

    print(f"[warm] pimpleFoam: constant {WARM_RPM:.0f} rpm x {WARM_DURATION:.0f} s "
          f"(first start compiles the coded inlet BC)", flush=True)
    metrics = h.do_simulation(WARM_RPM * RPM, WARM_DURATION)

    latest = h._get_latest_time()
    if latest == "0":
        raise RuntimeError("warmup did not advance simulation time")
    shutil.copytree(os.path.join(CASE, latest), os.path.join(CASE, "0.warmed"))

    keys = list(metrics[0].keys())
    csv_path = os.path.join(OUT_DIR, "warmup_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for m in metrics:
            w.writerow([f"{m.get(k, float('nan')):.8g}" for k in keys])

    t = np.array([m["t"] for m in metrics])
    conv = np.array([m["conv"] for m in metrics])
    wf = np.array([m["wallFlux"] for m in metrics])
    last = t >= t[-1] - 10.0
    prev = (t >= t[-1] - 20.0) & (t < t[-1] - 10.0)
    print(f"[warm] steady conv     = {np.median(conv[last]):.4f} "
          f"(drift vs previous 10 s: {abs(np.median(conv[last]) - np.median(conv[prev])):.4f})")
    print(f"[warm] steady wallFlux = {np.median(wf[last]):.4g} "
          f"(pristine-env normalizer wallflux_max = 1.32e-8)")

    # Lean template: reset_case now prefers the fresh 0.warmed/ -> 0/ becomes the
    # warmed state, stray time dirs + postProcessing go away, endTime back to 0.
    # Workers copytree this dir (inheriting compiled dynamicCode/) and start
    # instantly from the warmed IC.
    h.reset_case(mode="hard")
    print(f"[warm] done: template at {CASE} (0.warmed cached), "
          f"metrics at {csv_path}", flush=True)


if __name__ == "__main__":
    main()
