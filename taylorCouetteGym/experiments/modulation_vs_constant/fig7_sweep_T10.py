#!/usr/bin/env python3
"""Fig. 7 sweep variant: period T = 10 s (was 25 s), everything else identical.

Motivation (from the T=25 timeline analysis): the fluid swirl dies ~0.6 s after
each burst, so at T=25 the reactor is a dead pipe for ~18 of every 25 s and
pulsed loses to constant everywhere. T=10 keeps D=20% (bursts of 2 s at 5*w_b,
idle 8 s) but (a) shrinks the dead time per cycle, (b) gives each ~26 s-residence
parcel ~2.6 burst doses instead of ~1, and (c) 60 s = exactly 6 periods, so the
realized mean omega is exactly w_b (the T=25 run's burst-first 2.4 periods gave
1.25*w_b). Prediction: pulsed moves toward (maybe past?) constant.

The CONSTANT baselines are period-independent, so they are NOT re-run: the five
constant_wb* runs are symlinked from results/ into results_T10/ before analysis,
and the plots/table include them. Only the 5 pulsed episodes run (5 cores,
~1.5 h; slowest is wb=500 with 2500 rpm bursts).

USAGE
    nohup python3 -u fig7_sweep_T10.py --modes pulsed > fig7_T10.log 2>&1 &
    python3 fig7_sweep_T10.py --analyze-only          # replot results_T10/
Then compare the three families (constant, pulsed T=25, pulsed T=10):
    python3 compare_periods.py
"""
import os
import sys

import fig7_sweep as F

F.PERIOD = 10.0
F.RESULTS_DIR = os.path.join(F.HERE, "results_T10")

RESULTS_T25 = os.path.join(F.HERE, "results")


def link_constant_baselines():
    """Symlink the (period-independent) constant runs from results/ so the
    T=10 analysis includes them. rmtree-safe: only re-run modes are wiped, and
    shutil.rmtree refuses symlinks anyway."""
    os.makedirs(F.RESULTS_DIR, exist_ok=True)
    for wb in F.WBS:
        src = os.path.join(RESULTS_T25, f"constant_wb{wb}")
        dst = os.path.join(F.RESULTS_DIR, f"constant_wb{wb}")
        if os.path.isdir(src) and not os.path.exists(dst):
            os.symlink(src, dst)


if __name__ == "__main__":
    link_constant_baselines()
    if "--modes" not in sys.argv and "--analyze-only" not in sys.argv:
        sys.argv += ["--modes", "pulsed"]   # constants are linked, never re-run
    F.main()
