#!/usr/bin/env python3
"""Fig. 7 sweep variant: T = 10 s, D = 80%, WARMED continuous-manufacturing IC.

The T = 10 s counterpart of fig7_sweep_warmed_T2p5_D80.py (same warmed
constant-300 template, same duty): bursts of 8 s at 1.25*w_b, idle 2 s, from
the steady operating state. Together the two warmed sweeps bracket the pinned
duty-env period (T = 5 s) from both sides for the duty_v1 benchmark table.

Constant baselines are SYMLINKED from results_warmed_T2p5_D80/ -- the warmed
constants generated there; the pristine constants in results/ are NOT
comparable. Run the T2p5 variant first. Only the 5 pulsed episodes run here.
Results -> results_warmed_T10_D80/.

USAGE
    nohup python3 -u fig7_sweep_warmed_T10_D80.py > fig7_warmed_T10_D80.log 2>&1 &
    python3 fig7_sweep_warmed_T10_D80.py --analyze-only
"""
import os
import sys

import fig7_sweep as F

F.PERIOD = 10.0
F.MODES = {"pulsed": 0.8, "constant": 1.0}
F.RESULTS_DIR = os.path.join(F.HERE, "results_warmed_T10_D80")
F.WARMUP_RPM = 300.0

RESULTS_WARMED_T2P5 = os.path.join(F.HERE, "results_warmed_T2p5_D80")


def link_constant_baselines():
    os.makedirs(F.RESULTS_DIR, exist_ok=True)
    missing = []
    for wb in F.WBS:
        src = os.path.join(RESULTS_WARMED_T2P5, f"constant_wb{wb}")
        dst = os.path.join(F.RESULTS_DIR, f"constant_wb{wb}")
        if os.path.isdir(src):
            if not os.path.exists(dst):
                os.symlink(src, dst)
        else:
            missing.append(wb)
    if missing:
        print(f"WARNING: warmed constant baselines missing for wb={missing} "
              f"(run fig7_sweep_warmed_T2p5_D80.py first)", flush=True)


if __name__ == "__main__":
    link_constant_baselines()
    if "--modes" not in sys.argv and "--analyze-only" not in sys.argv:
        sys.argv += ["--modes", "pulsed"]
    F.main()
