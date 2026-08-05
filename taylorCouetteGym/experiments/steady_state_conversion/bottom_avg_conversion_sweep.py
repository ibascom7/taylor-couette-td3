#!/usr/bin/env python3
"""Bottom-cells conversion measurement sweep (Wang/Yuhe request).

WHAT THIS ANSWERS
-----------------
Does CHANGING HOW cOut IS MEASURED change the (flat) conversion-vs-omega result?

After the steady_state_conversion presentation, Wang and Yuhe asked for a different
outlet-concentration measurement: instead of the flux-weighted (cup-mixing) average
over the side_outlet patch faces, take the AVERAGE CONCENTRATION OF THE CELLS AT THE
BOTTOM OF THE CONTAINER -- the 1-cell-tall axial band adjacent to the closed bottom
wall (the band the side outlet drains from). cOut becomes a volume-weighted cell
average there; conversion stays 1 - cOut/c0.

This sweeps side_outlet_case_bottomavg -- a byte-identical clone of side_outlet_case
(same mesh, physics, BCs) whose ONLY change is the conversion functionObject -- at the
same constant speeds {0, 250, 500, 750, 1000} rpm, 180 s each, episodes in parallel.
The functionObject also logs the OLD cup-mixing outlet value from the very same run,
so the two measurement styles are compared on identical flow fields (and the cup value
cross-checks against the original results/ sweep).

EXPECTED: conversion still ~flat in omega (the measurement style should not create an
omega dependence the resolved-gradient physics doesn't have) -- the values of cOut
shift, but the flatness verdict shouldn't.

USAGE
-----
    python3 bottom_avg_conversion_sweep.py                # full sweep (~35 min)
    python3 bottom_avg_conversion_sweep.py --smoke        # ~2 min pipeline test
    python3 bottom_avg_conversion_sweep.py --analyze-only # replot from results_bottomavg/
    python3 bottom_avg_conversion_sweep.py --rpms 0,1000  # subset re-run

Outputs go to results_bottomavg/ (the original results/ tree is untouched).
"""
import csv
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import steady_state_conversion_sweep as S

# Separate output tree so the coarse-vs-wallmodel results/ is untouched.
S.RESULTS_DIR = os.path.join(S.HERE, "results_bottomavg")

CASE = "side_outlet_case_bottomavg"

S.CASES = {
    CASE: {
        "duration": 180.0,   # matches the original side_outlet_case sweep
        "label": "Original, bottom-cells cOut",
        "short": "bottomavg",
        "table_col": "bottom cells\n(vol. avg)",
        # PRIMARY metric: the new bottom-cells volume average, from the
        # bottomCellsConversion coded functionObject.
        "conv_re": re.compile(
            r"BOTTOM_CELLS_CONVERSION\s+t=(?P<t>[-+\d.eE]+)\s+cOut=(?P<cout>[-+\d.eE]+)"
            r"\s+conversion=(?P<conv>[-+\d.eE]+)"),
        # torque FO is unchanged from side_outlet_case
        "torque_re": re.compile(
            r"ROTATIONAL_POWER\s+t=(?P<t>[-+\d.eE]+)\s+Omega=(?P<omega>[-+\d.eE]+)"
            r"\s+Mz_wedge=(?P<mz>[-+\d.eE]+)"),
        "torque_dynamic": True,
    },
}
S.DEFAULT_WORKERS = len(S.RPMS)   # 5 episodes, 1 core each

# The OLD measurement (cup-mixing at the side_outlet patch), logged by the same
# functionObject on the same run -- the direct measurement-style comparison.
CUP_RE = re.compile(
    r"BOTTOM_CELLS_CONVERSION\s+t=(?P<t>[-+\d.eE]+)\s+.*?convOutletCup=(?P<conv>[-+\d.eE]+)")


def _steady(t, c):
    """Steady value of a conversion series -- same robust recipe as the engine
    (physical mask + median over the trailing window)."""
    t, c = np.asarray(t, float), np.asarray(c, float)
    if not len(c):
        return float("nan")
    win = S.steady_window(t[-1])
    phys = (c >= -0.02) & (c <= 1.02)
    late = c[phys & (t >= t[-1] - win)]
    return float(np.median(late)) if len(late) else float("nan")


def _cup_series(rpm):
    """(t, conv) of the OLD cup-mixing measurement, re-parsed from this sweep's own
    log.pimpleFoam for the given rpm. Empty arrays if the log is missing."""
    log = os.path.join(S.RESULTS_DIR, CASE, f"rpm_{rpm}", "log.pimpleFoam")
    if not os.path.isfile(log):
        return np.array([]), np.array([])
    with open(log, errors="replace") as f:
        text = f.read()
    t, c = [], []
    for m in CUP_RE.finditer(text):
        t.append(float(m.group("t")))
        c.append(float(m.group("conv")))
    return np.array(t), np.array(c)


def _original_reference():
    """Steady conversion per rpm from the ORIGINAL side_outlet_case sweep (results/),
    i.e. the cup-mixing measurement in the presentation figure. {} if not on disk."""
    ref = {}
    base = os.path.join(S.HERE, "results", "side_outlet_case")
    for rpm in S.RPMS:
        p = os.path.join(base, f"rpm_{rpm}_timeseries.csv")
        if not os.path.isfile(p):
            continue
        t, c = [], []
        with open(p) as f:
            rd = csv.reader(f)
            next(rd, None)
            for row in rd:
                if len(row) >= 2:
                    t.append(float(row[0])); c.append(float(row[1]))
        if c:
            ref[rpm] = _steady(t, c)
    return ref


def plot_timeseries_compare(results):
    """Overwrites the engine's per-case time-series figure: the bottom-cells-average
    conversion only (the old cup-mixing values stay in conversion_vs_omega.png and
    measurement_comparison.csv)."""
    spec = S.CASES[CASE]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    cmap = plt.get_cmap("viridis")
    rs = sorted([r for r in results if r["case"] == CASE and r["ok"]],
                key=lambda x: x["rpm"])
    for r in rs:
        col = cmap(r["rpm"] / max(S.RPMS) if max(S.RPMS) else 0)
        ax.plot(r["t_conv"], r["conv"], color=col, lw=1.8, label=f"{r['rpm']} rpm")
    sv = [r["conv_steady"] for r in rs if abs(r["conv_steady"]) < 1.0]
    if sv:
        ax.set_ylim(0, max(max(sv) * 1.9, 0.03))
    ax.set_xlabel("time [s]")
    ax.set_ylabel("conversion  (1 - cOut / c0)")
    ax.set_title("Conversion approach to steady state (bottom-cells avg)")
    ax.grid(True, alpha=0.3)
    ax.legend(title="inner wall", loc="best")
    fig.tight_layout()
    out = os.path.join(S.RESULTS_DIR, f"conversion_vs_time__{spec['short']}.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_measurement_comparison(results):
    """Overwrites the engine's conversion_vs_omega.png with the head-to-head of the
    measurement styles: NEW bottom-cells average vs OLD cup-mixing (same run) vs the
    ORIGINAL results/ sweep (independent runs, the presentation figure)."""
    rs = sorted([r for r in results if r["case"] == CASE and r["ok"]],
                key=lambda x: x["rpm"])
    if not rs:
        return
    fig, ax = plt.subplots(figsize=(8, 5.5))
    rpms = [r["rpm"] for r in rs]

    ax.plot(rpms, [r["conv_steady"] for r in rs], "o-", color="#d1495b", lw=2, ms=8,
            label="NEW: bottom-cells volume avg")

    cup = {rpm: _steady(*_cup_series(rpm)) for rpm in rpms}
    if any(v == v for v in cup.values()):
        ax.plot(rpms, [cup[r] for r in rpms], "s--", color="#2e6f95", lw=1.8, ms=7,
                label="OLD: outlet cup-mixing (same run)")

    ref = _original_reference()
    common = [r for r in rpms if r in ref]
    if common:
        ax.plot(common, [ref[r] for r in common], "^:", color="#66a182", lw=1.5, ms=7,
                label="OLD: original results/ sweep")

    ax.set_xlabel("inner-wall speed [rpm]")
    ax.set_ylabel("steady-state conversion  (1 - cOut / c0)")
    ax.set_title("Steady-state conversion vs angular velocity\n"
                 "bottom-cells measurement vs cup-mixing measurement")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_xticks(S.RPMS)
    fig.tight_layout()
    out = os.path.join(S.RESULTS_DIR, "conversion_vs_omega.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out} (measurement comparison)")


def write_comparison_csv(results):
    ref = _original_reference()
    path = os.path.join(S.RESULTS_DIR, "measurement_comparison.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rpm", "conv_bottom_cells", "conv_cup_same_run", "conv_cup_original_sweep"])
        for r in sorted([r for r in results if r["case"] == CASE and r["ok"]],
                        key=lambda x: x["rpm"]):
            rpm = r["rpm"]
            cup = _steady(*_cup_series(rpm))
            w.writerow([rpm, f"{r['conv_steady']:.6g}", f"{cup:.6g}",
                        f"{ref.get(rpm, float('nan')):.6g}"])
    print(f"  wrote {path}")


# Hook onto the engine's analyze() (which S.main() calls). Our figures intentionally
# overwrite the engine's generic time-series / vs-omega plots with correctly-labelled,
# comparison-bearing versions.
_orig_analyze = S.analyze


def analyze(results):
    _orig_analyze(results)
    try:
        plot_timeseries_compare(results)
        plot_measurement_comparison(results)
        write_comparison_csv(results)
    except Exception as e:
        print(f"  (measurement-comparison outputs skipped: {e})")


S.analyze = analyze


if __name__ == "__main__":
    S.main()
