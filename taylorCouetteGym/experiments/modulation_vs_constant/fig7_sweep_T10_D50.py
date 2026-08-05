#!/usr/bin/env python3
"""Fig. 7 sweep variant: period T = 10 s, duty D = 50%.

Fills in the duty axis at T=10 s (now D = 15/20/30/50/100%). Per Eq. 8
(peak = w_b / D): bursts of 5 s at 2*w_b -- peaks {200, 400, 600, 800, 1000}
rpm, idle 5 s, 6 periods per 60 s episode, realized mean omega = w_b exactly.
The tamest sweep of the campaign (all peaks well-resolved, cheap dt).

Context: D=30% produced the first pulsed-beats-constant point (w_b=100, X
0.2690 vs 0.2591 at 11% less power) with an interior duty optimum on the grid
{15,20,30,100}%. D=50% (quasi-steady transfer ~ D^0.3 = 0.81 of constant)
brackets that optimum from above: does the low-w_b advantage grow, peak, or
hand back to constant?

Constant baselines symlinked from results/; only the 5 pulsed episodes run.
Results -> results_T10_D50/.

USAGE
    nohup python3 -u fig7_sweep_T10_D50.py > fig7_T10_D50.log 2>&1 &
    python3 fig7_sweep_T10_D50.py --analyze-only
Then: python3 compare_periods.py   # 6-family comparison figures + table
"""
import os
import sys

import fig7_sweep as F

F.PERIOD = 10.0
F.MODES = {"pulsed": 0.5, "constant": 1.0}
F.RESULTS_DIR = os.path.join(F.HERE, "results_T10_D50")

RESULTS_T25 = os.path.join(F.HERE, "results")


def link_constant_baselines():
    os.makedirs(F.RESULTS_DIR, exist_ok=True)
    for wb in F.WBS:
        src = os.path.join(RESULTS_T25, f"constant_wb{wb}")
        dst = os.path.join(F.RESULTS_DIR, f"constant_wb{wb}")
        if os.path.isdir(src) and not os.path.exists(dst):
            os.symlink(src, dst)


if __name__ == "__main__":
    link_constant_baselines()
    if "--modes" not in sys.argv and "--analyze-only" not in sys.argv:
        sys.argv += ["--modes", "pulsed"]
    F.main()
