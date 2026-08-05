#!/usr/bin/env python3
"""Fig. 7 sweep variant: period T = 10 s, duty D = 15% (was 20%).

Motivated by the paper's Fig. 6 (conversion INCREASES as duty decreases at
fixed mean speed, max at their grid edge D=20%). Per Eq. 8 (peak = w_b / D)
the bursts hit 6.67*w_b: 1.5 s at {667, 1333, 2000, 2667, 3333} rpm followed
by 8.5 s idle, 6 periods per 60 s episode, realized mean omega = w_b exactly.

CAUTION: the w_b=400 (2667 rpm) and w_b=500 (3333 rpm) peaks exceed the 2500
rpm ceiling ever tested on this mesh:
  - stability: untested; the engine's divergence detection reports and excludes
    a blown-up episode, and the template's initial deltaT is lowered to 0.002 s
    so the impulsive t=0 start doesn't Courant-spike (adjustTimeStep takes over
    immediately after);
  - film resolution: at 3333 rpm the concentration film (~20-29 um, Re^-0.7
    scaling) is marginal against the 25 um wall cell -> w_b=500 conversion may
    read low even if stable.

Quasi-steady concavity says D=15% loses transfer vs D=20% (0.15*6.67^0.7 = 0.57
vs 0.62 of constant), while the paper's Fig. 6 trend says lower D wins -- this
run tests whose physics our resolved-film wedge follows.

Constant baselines are duty/period-independent -> symlinked from results/, only
the 5 pulsed episodes run (5 cores). Results -> results_T10_D15/.

USAGE
    nohup python3 -u fig7_sweep_T10_D15.py > fig7_T10_D15.log 2>&1 &
    python3 fig7_sweep_T10_D15.py --analyze-only
Then: python3 compare_periods.py   # 4-family comparison figures + table
"""
import os
import sys

import fig7_sweep as F

F.PERIOD = 10.0
F.MODES = {"pulsed": 0.15, "constant": 1.0}
F.RESULTS_DIR = os.path.join(F.HERE, "results_T10_D15")

RESULTS_T25 = os.path.join(F.HERE, "results")

_prepare_template = F.prepare_template


def prepare_template_small_dt():
    """Original template prep + smaller initial deltaT: the 3333 rpm impulsive
    start would see Co ~ O(80) on the stock 0.01 s first step."""
    tdir = _prepare_template()
    F.foam_set(tdir, "deltaT", "0.002", "system/controlDict")
    return tdir


F.prepare_template = prepare_template_small_dt


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
