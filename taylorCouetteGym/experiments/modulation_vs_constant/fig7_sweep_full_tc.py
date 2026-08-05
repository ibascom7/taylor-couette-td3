#!/usr/bin/env python3
"""Reference job on the FULL-HEIGHT reactor: Gamma = 30 (Lopez-Guajardo geometry).

Same style as the td3_prep sweep -- pulsed (D=80%, T=10 s) AND constant at
w_b = {100..500} rpm -- but on cases/full_tc_cat_case: the graded Sc=1075 mesh
(25 um wall cells, D = 1e-8) extended from H = 38.1 mm to the paper's full
H = 190.5 mm (16,650 cells, 5x). Same feed (100 mL/min) -> tau = H/u ~ 130 s,
so tau/T ~ 13: the paper's amortization regime, where each parcel averages
~13 pulse periods and idles cannot starve the outlet. This is the reference
for how the conversion-vs-power picture SHOULD look when the reactor is long.

Episodes are 300 s (~2.3 tau) so the last-period window is quasi-steady.
COST: ~230 CPU-s per sim-second (smoke-calibrated) -> ~19-21 h per episode,
10 episodes in parallel -> ~1 day wall clock. ~11 GB of 1 s time folders.

USAGE
    nohup python3 -u fig7_sweep_full_tc.py > fig7_full_tc.log 2>&1 &
    python3 fig7_sweep_full_tc.py --analyze-only
"""
import os

import fig7_sweep as F

F.CASE_NAME = "full_tc_cat_case"
F.EPISODE = 300.0
F.PERIOD = 10.0
F.MODES = {"pulsed": 0.8, "constant": 1.0}
F.RESULTS_DIR = os.path.join(F.HERE, "results_full_tc")

if __name__ == "__main__":
    F.main()
