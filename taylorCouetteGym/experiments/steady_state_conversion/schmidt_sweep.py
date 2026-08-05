#!/usr/bin/env python3
"""Schmidt-number sweep on the COARSE (15 x 90) side-outlet case.

Complementary to the mesh-refinement study. There, we held Sc = 1e4 fixed and refined
the mesh to resolve the ~20-45 um wall film (which diverges before it converges). Here
we do the opposite: hold the COARSE mesh fixed and LOWER the Schmidt number (raise the
diffusivity D) so the wall film THICKENS (delta_c ~ Sc^-1/3) until the coarse cells can
resolve it. Same conclusion, opposite knob -- and, because Pe = Re*Sc drops ~100-1000x,
the scalar is well-damped, so these runs are fast and stable (no fine-mesh blow-ups).

Cases (coarse 15x90, only D changes; nu, geometry, flow all identical to side_outlet_case):
    side_outlet_case_sc100   D = nu/100  = 1.08e-7   (film ~ resolvable on 423 um cells)
    side_outlet_case_sc1000  D = nu/1000 = 1.08e-8
The Sc = 1e4 point is READ from the existing coarse run in results/side_outlet_case
(no re-run needed) and shown as the third point on the Schmidt plot.

WHAT IT SHOWS: as Sc drops from 1e4 toward 100 on the SAME coarse mesh, the outlet
conversion should climb and REGAIN its omega-dependence -- demonstrating that the flat
"constant conversion" at Sc = 1e4 was a resolution artifact (the film was thinner than a
cell), NOT physics. NB the absolute numbers at Sc = 100/1000 are a DIFFERENT (more
diffusive) fluid, not Lopez -- this validates the methodology, it is not a quantitative
match to the real reactor.

Reuses the steady_state_conversion_sweep engine (identical table + conversion-vs-omega
/ -time / -power plots, motor power) PLUS a dedicated schmidt_sweep.png (conversion vs
Sc, one line per rpm, including the Sc=1e4 reference). Outputs -> results_schmidt/.

USAGE
    python3 schmidt_sweep.py                 # both Sc, 5 speeds, 150 s (fast: ~30-45 min)
    python3 schmidt_sweep.py --smoke
    python3 schmidt_sweep.py --analyze-only
"""
import csv
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import steady_state_conversion_sweep as S

S.RESULTS_DIR = os.path.join(S.HERE, "results_schmidt")
DURATION_S = 150.0
SCHMIDT = [100, 1000]
SC_REF = 10000          # the real-fluid point, read from the existing coarse run in results/


def _spec(sc):
    return {
        "duration": DURATION_S,
        "label": f"Sc = {sc}  (D = {S.NU / sc:.2e})",
        "short": f"sc{sc}",
        "table_col": f"Sc={sc}",
        "sc": sc,
        "conv_re": re.compile(
            r"SIDE_OUTLET_CONVERSION\s+t=(?P<t>[-+\d.eE]+)\s+cOut=(?P<cout>[-+\d.eE]+)"
            r"\s+conversion=(?P<conv>[-+\d.eE]+)"),
        "torque_re": re.compile(
            r"ROTATIONAL_POWER\s+t=(?P<t>[-+\d.eE]+)\s+Omega=(?P<omega>[-+\d.eE]+)"
            r"\s+Mz_wedge=(?P<mz>[-+\d.eE]+)"),
        "torque_dynamic": True,
    }


S.CASES = {f"side_outlet_case_sc{sc}": _spec(sc) for sc in SCHMIDT}
S.DEFAULT_WORKERS = len(S.CASES) * len(S.RPMS)   # 10, coarse -> fast


def _sc_ref_reference():
    """Steady conversion per rpm of the existing Sc=1e4 COARSE run (results/side_outlet_case),
    for the third point on the Schmidt plot. {} if that run isn't present."""
    ref = {}
    base = os.path.join(S.HERE, "results", "side_outlet_case")
    for rpm in S.RPMS:
        p = os.path.join(base, f"rpm_{rpm}_timeseries.csv")
        if not os.path.isfile(p):
            continue
        t, c = [], []
        with open(p) as f:
            rd = csv.reader(f); next(rd, None)
            for row in rd:
                if len(row) >= 2:
                    t.append(float(row[0])); c.append(float(row[1]))
        if c:
            t = np.array(t); c = np.array(c)
            phys = (c >= -0.02) & (c <= 1.02); c = c[phys]; t = t[phys]
            if len(c):
                ref[rpm] = float(np.median(c[t >= t[-1] - max(2.0, 0.2 * t[-1])]))
    return ref


def plot_schmidt(results):
    fig, ax = plt.subplots(figsize=(8.5, 6))
    cmap = plt.get_cmap("viridis")
    ref = _sc_ref_reference()
    for rpm in S.RPMS:
        pts = []
        for cn, spec in S.CASES.items():
            r = next((x for x in results if x["case"] == cn and x["rpm"] == rpm and x["ok"]), None)
            if r:
                pts.append((spec["sc"], r["conv_steady"]))
        if rpm in ref:
            pts.append((SC_REF, ref[rpm]))
        pts = sorted(pts)
        if len(pts) < 2:
            continue
        col = cmap(rpm / max(S.RPMS) if max(S.RPMS) else 0)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=col, lw=1.9, ms=7,
                label=f"{rpm} rpm")
    ax.set_xscale("log")
    ax.set_xlabel("Schmidt number  Sc = nu/D      (<- thicker film / resolvable      real fluid, Sc~1e4 ->)")
    ax.set_ylabel("steady-state outlet conversion")
    ax.set_title("Coarse 15x90 mesh: conversion vs Schmidt number\n"
                 "omega-response collapses as Sc->1e4 (wall film thins below the cell size)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="inner wall", loc="best")
    if not ref:
        ax.text(0.02, 0.02, "(Sc=1e4 reference not found in results/side_outlet_case)",
                transform=ax.transAxes, fontsize=8, color="crimson")
    fig.tight_layout()
    out = os.path.join(S.RESULTS_DIR, "schmidt_sweep.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


_orig_analyze = S.analyze


def analyze(results):
    _orig_analyze(results)
    try:
        plot_schmidt(results)
    except Exception as e:
        print(f"  (schmidt plot skipped: {e})")


S.analyze = analyze


if __name__ == "__main__":
    S.main()
