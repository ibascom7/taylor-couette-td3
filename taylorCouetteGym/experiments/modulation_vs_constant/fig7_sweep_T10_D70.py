#!/usr/bin/env python3
"""Fig. 7 sweep variant: period T = 10 s, duty D = 70%.

Brackets the interior duty optimum (known to lie in (60%, 100%) at low-mid
w_b: D=60% beats constant at w_b<=300 and D=100% IS constant). Per Eq. 8:
bursts of 7 s at 1.43*w_b -- peaks {143, 286, 429, 571, 714} rpm, idle 3 s
(~5 swirl-decay times), 6 periods per 60 s episode, mean omega = w_b exactly.
Runs alongside the D=80% sweep (fig7_sweep_T10_D80.py).

Constant baselines symlinked from results/; only the 5 pulsed episodes run.
Results -> results_T10_D70/.

USAGE
    nohup python3 -u fig7_sweep_T10_D70.py > fig7_T10_D70.log 2>&1 &
    python3 fig7_sweep_T10_D70.py --analyze-only
Then: python3 compare_periods.py
"""
import os
import sys

import fig7_sweep as F

F.PERIOD = 10.0
F.MODES = {"pulsed": 0.7, "constant": 1.0}
F.RESULTS_DIR = os.path.join(F.HERE, "results_T10_D70")

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
