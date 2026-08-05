#!/usr/bin/env python3
"""Spin sweep on the GRADED-mesh, Sc=1075 side-outlet case.

Same 25 um-wall-cell graded mesh as the Sc=107.5 run, but D lowered 10x to 1e-8
(Sc = nu/D = 1075) -- 10x closer to the real fluid (Sc~1e4). The point: at Sc=107.5
the wall transfer was so efficient the reactor SATURATED (Da~2-5, conversion flat &
high). Dropping to Sc=1075 pushes Da below ~1 (mass-transfer-LIMITED regime), so
conversion should RISE with speed -- while the 25 um cells still resolve the film
(~1.5-4 cells across it). Expected steady conversion ~0.36 (250 rpm) -> ~0.69 (1000 rpm).
NB the 1000 rpm film is ~39 um (~1.5 cells), so that point is marginally resolved and may
undershoot slightly; the 250-750 rise is well-resolved.

Sweeps {0,250,500,750,1000,2500} rpm x 150 s -> results_graded_sc1075/. ~4 h/episode at
1000 rpm (dt set by the flow, so same speed as the Sc=107.5 graded run). Stable (checked)
up to 1000 rpm.

The 2500 rpm point was added later (see USAGE): at 2500 rpm the film thins to ~20-35 um
(delta_c ~ Re^-0.7), i.e. right around the 25 um wall cell -> that point is only
MARGINALLY resolved (~1-1.5 cells across the film) and may undershoot / sit near the
one-cell floor, and the higher Courant on the fine cells risks divergence (the engine
detects it, reports the raw number, and excludes it from the plots). It runs ~2.5x
slower than 1000 rpm (smaller dt), so budget ~8-14 h wall-clock.

USAGE
    # full 6-speed sweep from scratch (expensive -- recomputes the 5 finished speeds):
    nohup python3 -u graded_mesh_sweep_sc1075.py > graded_sc1075.log 2>&1 &

    # INCREMENTAL: run ONLY the new 2500 rpm episode, reuse the 5 finished runs, and
    # re-plot every speed + rebuild the summary table (this is how 2500 was added):
    nohup python3 -u graded_mesh_sweep_sc1075.py --rpms 2500 > graded_sc1075_2500.log 2>&1 &

    python3 graded_mesh_sweep_sc1075.py --analyze-only    # replot all speeds on disk
"""
import os
import re

import steady_state_conversion_sweep as S

S.RESULTS_DIR = os.path.join(S.HERE, "results_graded_sc1075")

# 2500 rpm appended to the original {0,250,500,750,1000}; 400 added 2026-07-17 to
# settle the 60 s "constant 400 > 500" anomaly from the modulation_vs_constant
# campaign (the steady curve had no point between 250 and 500). analyze() spans
# whatever is in RPMS; use `--rpms <n>` to compute only new points.
S.RPMS = [0, 250, 400, 500, 750, 1000, 2500]

S.CASES = {
    "side_outlet_case_sc1075_graded": {
        "duration": 150.0,
        "label": "Graded mesh, Sc=1075 (25 um wall cell)",
        "short": "graded_sc1075",
        "table_col": "graded Sc=1075\n(25 um wall)",
        "conv_re": re.compile(
            r"SIDE_OUTLET_CONVERSION\s+t=(?P<t>[-+\d.eE]+)\s+cOut=(?P<cout>[-+\d.eE]+)"
            r"\s+conversion=(?P<conv>[-+\d.eE]+)"),
        "torque_re": re.compile(
            r"ROTATIONAL_POWER\s+t=(?P<t>[-+\d.eE]+)\s+Omega=(?P<omega>[-+\d.eE]+)"
            r"\s+Mz_wedge=(?P<mz>[-+\d.eE]+)"),
        "torque_dynamic": True,
    }
}
S.DEFAULT_WORKERS = len(S.CASES) * len(S.RPMS)   # 5

if __name__ == "__main__":
    S.main()
