#!/usr/bin/env python3
"""Spin sweep on the GRADED-mesh, Sc=107.5 side-outlet case.

Goal: a WALL-RESOLVED resolved-gradient case that is both stable and captures the
concentration film, by combining two moves:
  - raise D to 1e-7 (Sc = nu/D = 107.5) so the film is thick (~90-240 um across
    0-1000 rpm) and the scalar Peclet drops ~100x (stable -- no Sc=1e4 blow-ups); and
  - a RADIALLY GRADED mesh (side_outlet_case_sc107_graded): 37 radial cells graded
    smoothly (~1.09 expansion/cell) from ~543 um at the inner wall down to a 25 um
    cell at the OUTER (catalytic) wall -- which resolves the 88-240 um film with ~3-10
    cells. (25 um is sized to the Sc=107.5 film; a 3 um wall cell -- what you'd need for
    the real Sc=1e4 film -- makes the timestep collapse to ~4e-5 and the episode take
    ~75 h, because the fine cells sit in the near-outer-wall vortex/outlet flow.)

Sweeps {0, 250, 500, 750, 1000} rpm x 150 s and reports exactly like the other sweeps
(table + conversion-vs-omega / -time / -power, motor power). Outputs -> results_graded/.

RUNTIME: ~4 h per episode at 1000 rpm (measured), i.e. the 5-speed sweep is an overnight
run (wall-clock ~ the slowest single episode, run in parallel). Cheaper: use a coarser
wall cell (rebuild the mesh -- see the case blockMeshDict). NB Sc=107.5 is more diffusive
than the real reactant (Sc~1e4), so absolute numbers are not Lopez-quantitative; this is a
clean, stable, wall-RESOLVED demonstration that conversion responds to omega once the film
is actually captured.

USAGE
    python3 graded_mesh_sweep.py                 # 5 speeds x 150 s (~overnight)
    python3 graded_mesh_sweep.py --smoke
    python3 graded_mesh_sweep.py --analyze-only
    python3 graded_mesh_sweep.py --workers 5
"""
import os
import re

import steady_state_conversion_sweep as S

S.RESULTS_DIR = os.path.join(S.HERE, "results_graded")

S.CASES = {
    "side_outlet_case_sc107_graded": {
        "duration": 150.0,
        "label": "Graded mesh, Sc=107.5 (25 um wall cell)",
        "short": "graded_sc107",
        "table_col": "graded Sc=107.5\n(25 um wall)",
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
