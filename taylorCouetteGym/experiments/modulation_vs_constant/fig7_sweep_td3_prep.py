#!/usr/bin/env python3
"""TD3-prep sweep: D = 80%, T = 10 s pulsed AND constant, 50 s episodes.

50 s is the planned RL training episode length. Both modes are run fresh at
50 s (10 episodes, 10 cores): pulsed bursts of 8 s at 1.25*w_b (5 full periods,
mean omega = w_b exactly) and the constant baselines. X = mean conversion over
the last full period [40, 50] s.

NB determinism: the solver is deterministic, so these episodes are bit-identical
to the first 50 s of the existing 60 s runs (results/ and results_T10_D80/) --
the fresh runs double as a pipeline-determinism check against the predicted
values, and give clean endTime=50 case trees mirroring the RL episodes.

This dataset is self-contained: its fig7/table plots (written by the engine
into results_td3_prep/) cover ONLY these 50 s runs and are NOT added to
compare_periods.py (the duty-axis families were 60 s -- different windows).

USAGE
    nohup python3 -u fig7_sweep_td3_prep.py > fig7_td3_prep.log 2>&1 &
    python3 fig7_sweep_td3_prep.py --analyze-only
"""
import os

import fig7_sweep as F

F.EPISODE = 50.0
F.PERIOD = 10.0
F.MODES = {"pulsed": 0.8, "constant": 1.0}
F.RESULTS_DIR = os.path.join(F.HERE, "results_td3_prep")

if __name__ == "__main__":
    F.main()
