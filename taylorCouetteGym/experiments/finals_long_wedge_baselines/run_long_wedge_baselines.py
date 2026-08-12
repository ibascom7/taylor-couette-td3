#!/usr/bin/env python3
"""FINALS baseline grid, LONG WEDGE cell (Gamma = 30, full Lopez-Guajardo H).

Same nine regime-placed waveforms as the short-wedge grid (see
finals_short_wedge_baselines/run_short_wedge_baselines.py -- the single source
of truth for the run table and reward convention), on full_tc_cat_case: the
case behind experiments/modulation_vs_constant/results_full_tc (16,650 cells).
TIME BASE = THIS CELL'S RESIDENCE TIME tau ~ 130 s: pulse period T = tau,
episode = 5*tau = 650 s, X window = the last full period [520, 650] s -- the
same tau-block convention as the finals TD3 runs and the short-wedge grid.

COST (from the fig7 full-TC calibration, ~230 CPU-s per sim-second at mean
300 rpm, dt Courant-scales with omega): per 650 s episode roughly
mean-300 ~42-48 h, mean-750 ~105-120 h, mean-1500 ~210-240 h, 1 core each.
Submit as a slurm array (one episode per task) and RESUBMIT WITH --resume
after a walltime kill: episodes continue from their latest saved time folder
(the mwvf tasks will need 1-2 resubmits).

USAGE
    python3 run_long_wedge_baselines.py --smoke              # ~25 min pipeline test
    python3 run_long_wedge_baselines.py --index 0            # one episode (array task)
    python3 run_long_wedge_baselines.py --index 2 --resume   # continue after timeout
    python3 run_long_wedge_baselines.py --list
    python3 run_long_wedge_baselines.py --analyze-only
"""

import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_CORE = os.path.normpath(os.path.join(
    HERE, "..", "finals_short_wedge_baselines", "run_short_wedge_baselines.py"))
_spec = importlib.util.spec_from_file_location("finals_baselines_core", _CORE)
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)

B.CASE_NAME = "full_tc_cat_case"        # the results_full_tc reference case
B.TAU = 130.0                           # residence time V/Q [s] (5x the short cell)
B.PERIOD = B.TAU                        # pulse period = one tau = TD3 block
B.EPISODE = 5.0 * B.TAU                 # 5 residence times = 650 s
B.WRITE_INTERVAL = 10.0                 # 65 time folders/episode (~2.4 GB)
B.RESULTS_DIR = os.path.join(HERE, "results")
B.CELL_TAG = "Gamma=30 wedge (long)"

if __name__ == "__main__":
    B.main()
