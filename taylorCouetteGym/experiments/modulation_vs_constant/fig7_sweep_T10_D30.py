#!/usr/bin/env python3
"""Fig. 7 sweep variant: period T = 10 s, duty D = 30%.

Completes the duty axis at T=10 s (with D=15%, 20%, and constant=100% already
measured). Per Eq. 8 (peak = w_b / D): bursts of 3 s at 3.33*w_b -- peaks
{333, 667, 1000, 1333, 1667} rpm, all well below the 2500 rpm ceiling and
well-resolved by the 25 um wall cell. Idle is 7 s (~10 swirl-decay times vs
~13 at D=20%). 6 periods per 60 s episode, realized mean omega = w_b exactly.

Quasi-steady prediction: duty-averaged transfer ~ D^0.3 of constant -> 0.70 at
D=30% vs 0.62 at D=20% -> D=30% should land BETWEEN D=20% and constant. If it
does, conversion is monotone in D on this reactor at T=10 s (no interior pulsed
duty optimum for TD3 within Lopez's waveform family).

Constant baselines symlinked from results/; only the 5 pulsed episodes run.
Results -> results_T10_D30/.

USAGE
    nohup python3 -u fig7_sweep_T10_D30.py > fig7_T10_D30.log 2>&1 &
    python3 fig7_sweep_T10_D30.py --analyze-only
Then: python3 compare_periods.py   # 5-family comparison figures + table
"""
import os
import sys

import fig7_sweep as F

F.PERIOD = 10.0
F.MODES = {"pulsed": 0.3, "constant": 1.0}
F.RESULTS_DIR = os.path.join(F.HERE, "results_T10_D30")

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
