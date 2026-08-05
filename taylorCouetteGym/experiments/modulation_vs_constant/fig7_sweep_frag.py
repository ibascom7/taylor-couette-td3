#!/usr/bin/env python3
"""Fragmentation probes: D = 80%, w_b = 300, 50 s episodes, period from argv.

Tests whether splitting the idle time into more, shorter idles (shorter period
at fixed duty) buys conversion -- the `s` axis of the planned block-constrained
TD3 action space. Benchmarks at 50 s (results_td3_prep, X over [40,50]):
constant-300 = 0.3402, D=80 T=10 = 0.3744.

    python3 fig7_sweep_frag.py 5      # 4 s at 375 rpm / 1 s idle  -> results_frag_T5/
    python3 fig7_sweep_frag.py 2.5    # 2 s at 375 rpm / 0.5 s idle -> results_frag_T2p5/

The 0.5 s idle of T=2.5 sits at the measured 0.6 s swirl-decay time -- it probes
whether renewal still completes when the idle barely outlives the vortices.
Single pulsed episode per invocation (w_b=300 only); ~45 min.
"""
import os
import sys

import fig7_sweep as F

period = float(sys.argv[1])
tag = str(period).replace(".", "p").rstrip("0").rstrip("p") if "." in str(period) else str(period)
F.EPISODE = 50.0
F.PERIOD = period
F.MODES = {"pulsed": 0.8, "constant": 1.0}
F.RESULTS_DIR = os.path.join(F.HERE, f"results_frag_T{tag}")

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--modes", "pulsed", "--wbs", "300"]
    F.main()
