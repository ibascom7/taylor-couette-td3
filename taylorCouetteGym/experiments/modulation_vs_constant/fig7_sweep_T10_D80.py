#!/usr/bin/env python3
"""Fig. 7 sweep variant: period T = 10 s, duty D = 80%.

The right flank of the duty axis (optimum known to lie in (60%, 100%) at
low-mid w_b). Per Eq. 8: bursts of 8 s at 1.25*w_b -- peaks {125, 250, 375,
500, 625} rpm, idle 2 s (~3 swirl-decay times), 6 periods per 60 s episode,
mean omega = w_b exactly. Runs alongside the D=70% sweep
(fig7_sweep_T10_D70.py).

Constant baselines symlinked from results/; only the 5 pulsed episodes run.
Results -> results_T10_D80/.

USAGE
    nohup python3 -u fig7_sweep_T10_D80.py > fig7_T10_D80.log 2>&1 &
    python3 fig7_sweep_T10_D80.py --analyze-only
Then: python3 compare_periods.py
"""
import os
import sys

import fig7_sweep as F

F.PERIOD = 10.0
F.MODES = {"pulsed": 0.8, "constant": 1.0}
F.RESULTS_DIR = os.path.join(F.HERE, "results_T10_D80")

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
