#!/usr/bin/env python3
"""Mesh-refinement (convergence) study of the resolved-gradient catalytic wall.

Runs the SAME resolved-gradient side-outlet physics as side_outlet_case on a sequence
of UNIFORM (ungraded) meshes that are refined in the wall-normal (radial) direction:

    case                 NR    dr = gap/NR    cells
    side_outlet_case_r16  16     397 um        1936
    side_outlet_case_r32  32     198 um        3872
    side_outlet_case_r64  64      99 um        7744
    side_outlet_case_r128 128     50 um       15488   (the ~1-2 day long pole)

Each is swept at {0, 250, 500, 750, 1000} rpm for 150 s (~5.8 residence times at
100 mL/min -> safely steady; check the steady_drift column). Only the radial spacing dr
changes (NZ = 120 fixed, simpleGrading 1 1 1 -> no grading), because the unresolved
scale is the Sc~1e4 concentration film at the OUTER wall (~17-45 um), whose gradient is
radial. The question: does the outlet conversion CLIMB / gain omega-dependence as dr
shrinks? If it keeps rising, the coarse result was under-resolved; if it plateaus, the
flatness is closer to physical. (No desktop-feasible uniform mesh fully resolves the
film -- even dr=50 um is still coarser than it -- so this is a refinement TREND, read
the slope, not an absolute converged value.)

Reporting is IDENTICAL to steady_state_conversion_sweep.py (it reuses that engine): a
summary table with one column per resolution, conversion-vs-omega / -vs-time / -vs-power
plots, motor power -- PLUS a dedicated `mesh_convergence.png` (steady conversion vs dr,
one line per rpm, with the Sherwood wall-model value as a dotted reference and the film
thickness shaded). Outputs go to results_fine/.

USAGE
    python3 fine_mesh_sweep.py                                   # all 4 meshes (~1.2 days; NR=128 dominates)
    python3 fine_mesh_sweep.py --cases side_outlet_case_r16,side_outlet_case_r32,side_outlet_case_r64
                                                                 # overnight 3-point study (drop NR=128)
    python3 fine_mesh_sweep.py --smoke                           # quick pipeline test
    python3 fine_mesh_sweep.py --analyze-only                    # replot (incl partial progress)
    python3 fine_mesh_sweep.py --workers 8                       # cap parallelism

Runtime (measured/estimated on a 16-core desktop at 150 s; the sweep is parallel so
wall-clock ~ the slowest single episode): NR=16 ~30 min, NR=32 ~2 h, NR=64 ~7 h,
NR=128 ~1.2 days. Cheaper meshes finish first; run --analyze-only any time to see the
partial convergence. Drop NR=128 (--cases ...r16,...r32,...r64) for an overnight study.
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
S.RESULTS_DIR = os.path.join(S.HERE, "results_fine")

GAP_MM = 6.35                 # annular gap
FILM_UM = (17.0, 45.0)        # Sc~1e4 concentration film thickness (from Sh = a Re^b Sc^c)
RADIAL_CELLS = (16, 32, 64, 128)
# Episode length. Feed 100 mL/min -> residence time tau ~ 26 s, so 150 s ~ 5.8 tau is
# comfortably steady (>130 s) while trimming ~17% off the expensive fine runs vs 180 s.
# The `steady_drift` column / conversion-vs-time plots confirm each run plateaued.
DURATION_S = 150.0


def _spec(nr):
    dr = GAP_MM * 1000.0 / nr  # radial cell size [um]
    return {
        "duration": DURATION_S,
        "label": f"NR={nr}  (dr={dr:.0f} um)",
        "short": f"nr{nr}",
        "table_col": f"dr={dr:.0f}um\n(NR{nr})",
        "nr": nr,
        "dr_um": dr,
        # same conversion / torque functionObjects as side_outlet_case
        "conv_re": re.compile(
            r"SIDE_OUTLET_CONVERSION\s+t=(?P<t>[-+\d.eE]+)\s+cOut=(?P<cout>[-+\d.eE]+)"
            r"\s+conversion=(?P<conv>[-+\d.eE]+)"),
        "torque_re": re.compile(
            r"ROTATIONAL_POWER\s+t=(?P<t>[-+\d.eE]+)\s+Omega=(?P<omega>[-+\d.eE]+)"
            r"\s+Mz_wedge=(?P<mz>[-+\d.eE]+)"),
        "torque_dynamic": True,
    }


S.CASES = {f"side_outlet_case_r{nr}": _spec(nr) for nr in RADIAL_CELLS}
S.DEFAULT_WORKERS = 16   # 20 episodes; cheap meshes free their cores early


def _wallmodel_reference():
    """Steady conversion of the Sherwood wall model per rpm (from the coarse-vs-wallmodel
    run in results/), used as a dotted reference on the convergence plot. {} if absent."""
    ref = {}
    base = os.path.join(S.HERE, "results", "side_outlet_cat_wallmodel")
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
            t = np.array(t); c = np.array(c)
            win = max(2.0, 0.2 * t[-1])
            ref[rpm] = float(np.mean(c[t >= t[-1] - win]))
    return ref


def plot_convergence(results):
    fig, ax = plt.subplots(figsize=(8.5, 6))
    cmap = plt.get_cmap("viridis")
    ref = _wallmodel_reference()
    have_ref = False
    for rpm in S.RPMS:
        pts = sorted([r for r in results if r["rpm"] == rpm and r["ok"]],
                     key=lambda r: S.CASES[r["case"]]["dr_um"])
        if not pts:
            continue
        col = cmap(rpm / max(S.RPMS) if max(S.RPMS) else 0)
        xs = [S.CASES[r["case"]]["dr_um"] for r in pts]
        ys = [r["conv_steady"] for r in pts]
        ax.plot(xs, ys, "o-", color=col, lw=1.9, ms=7, label=f"{rpm} rpm")
        if rpm in ref:
            ax.axhline(ref[rpm], color=col, ls=":", lw=1.2, alpha=0.8)
            have_ref = True
    ax.axvspan(FILM_UM[0], FILM_UM[1], color="grey", alpha=0.18)
    ax.set_xscale("log")
    ax.set_xlim(15, 460)
    ymax = ax.get_ylim()[1]
    ax.text(np.sqrt(FILM_UM[0] * FILM_UM[1]), ymax * 0.96,
            "Sc~1e4 film\n(need dr below\nthis to resolve)",
            ha="center", va="top", fontsize=8, color="dimgray")
    ax.set_xlabel("radial (wall-normal) cell size  dr  [um]      (finer  <--   -->  coarser)")
    ax.set_ylabel("steady-state outlet conversion")
    title = "Mesh convergence of the resolved-gradient wall"
    if have_ref:
        title += "\n(dotted = Sherwood wall-model value; a flat line = mesh-converged)"
    else:
        title += "\n(a flat line = mesh-converged; a rising line = still under-resolved)"
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="inner wall", loc="best")
    div = sorted({r["rpm"] for r in results if r.get("diverged")})
    if div:
        ax.text(0.015, 0.015,
                f"NR=128 (50 µm) diverged at {div} rpm — Courant overshoot on the tiny\n"
                f"cells blows up the Sc~1e4 scalar; those points excluded (see table).",
                transform=ax.transAxes, fontsize=7.5, color="crimson", va="bottom")
    fig.tight_layout()
    out = os.path.join(S.RESULTS_DIR, "mesh_convergence.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


# Hook the convergence plot onto the engine's analyze() (which S.main() calls).
_orig_analyze = S.analyze


def analyze(results):
    _orig_analyze(results)
    try:
        plot_convergence(results)
    except Exception as e:
        print(f"  (convergence plot skipped: {e})")


S.analyze = analyze


if __name__ == "__main__":
    S.main()
