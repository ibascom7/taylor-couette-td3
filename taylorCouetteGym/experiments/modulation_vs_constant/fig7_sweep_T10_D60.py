#!/usr/bin/env python3
"""Fig. 7 sweep variant: period T = 10 s, duty D = 60%.

Brackets the interior duty optimum from above (duty axis now
D = 15/20/30/50/60/100% at T=10 s). Per Eq. 8 (peak = w_b / D): bursts of 6 s
at 1.67*w_b -- peaks {167, 333, 500, 667, 833} rpm, idle 4 s, 6 periods per
60 s episode, realized mean omega = w_b exactly. All peaks tame and cheap.

Context: D=50% beats constant at w_b<=300 (film-renewal overshoot: each
spin-up transiently out-transfers its own speed's steady state, and the
just-renewed film keeps a fat idle tail). D=60% trades renewal frequency
unchanged (same T) for taller on-fraction and shorter idle (4 s ~ 6 swirl
decay times): if the renewal bonus dominates, D=60% ~ D=50%; if the concave
quasi-steady term dominates (D^0.3 = 0.86 of constant), it slides toward
constant. Either way it locates the duty optimum's right flank.

Constant baselines symlinked from results/; only the 5 pulsed episodes run.
Results -> results_T10_D60/.

USAGE
    nohup python3 -u fig7_sweep_T10_D60.py > fig7_T10_D60.log 2>&1 &
    python3 fig7_sweep_T10_D60.py --analyze-only
Then: python3 compare_periods.py   # 7-family comparison figures + table
"""
import os
import sys

import fig7_sweep as F

F.PERIOD = 10.0
F.MODES = {"pulsed": 0.6, "constant": 1.0}
F.RESULTS_DIR = os.path.join(F.HERE, "results_T10_D60")

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
