#!/usr/bin/env python3
"""Fig. 7 sweep variant: T = 2.5 s, D = 80%, WARMED continuous-manufacturing IC.

The static champion waveform (fig7_sweep_td3_prep: R- = 0.270 at w_b = 300)
re-benchmarked from the WARMED constant-300 steady state instead of the
pristine startup transient -- the IC the duty-cycle RL runs
(experiments/modulation_rl/parallel_train_duty.py, tags duty_v1_s*) train
from, so these numbers are the direct benchmark row for that campaign.

The template spins at constant 300 rpm for 60 s (~2.3 tau) ONCE and the final
state is promoted to 0/ (fig7_sweep.WARMUP_RPM); every episode then starts
warmed at t = 0. BOTH modes run here -- the warmed constant baselines CANNOT
be symlinked from results/ (those started pristine, not comparable) and this
sweep is where they are generated; fig7_sweep_warmed_T10_D80.py then symlinks
them. 10 episodes -> results_warmed_T2p5_D80/.

USAGE (run BEFORE the T10 variant; it consumes this sweep's constants)
    nohup python3 -u fig7_sweep_warmed_T2p5_D80.py > fig7_warmed_T2p5_D80.log 2>&1 &
    python3 fig7_sweep_warmed_T2p5_D80.py --analyze-only
"""
import os

import fig7_sweep as F

F.PERIOD = 2.5
F.MODES = {"pulsed": 0.8, "constant": 1.0}
F.RESULTS_DIR = os.path.join(F.HERE, "results_warmed_T2p5_D80")
F.WARMUP_RPM = 300.0

if __name__ == "__main__":
    F.main()
