#!/usr/bin/env python3
"""Steady-state conversion vs. angular velocity sweep (side-outlet catalytic reactor).

WHAT THIS ANSWERS
-----------------
Does the outlet conversion actually depend on the inner-wall angular velocity?

We run two independent constant-omega spin sweeps on the SAME mesh/geometry and
compare how conversion responds to rotation:

  1. side_outlet_case            -- Yuhe's ORIGINAL physics: the catalytic outer
                                    wall is a RESOLVED-GRADIENT sink (c = 0 at the
                                    wall, consumption = molecular diffusion of c to
                                    the wall). With the corrected Sc ~ 1e4 the wall
                                    concentration film is ~microns thick and the
                                    coarse mesh cannot resolve it -- so this is the
                                    "does the raw model even see rotation?" case.

  2. side_outlet_cat_wallmodel   -- OUR wall model: the wall consumption is a
                                    Sherwood mass-transfer sink k_c(Re)*C whose rate
                                    RESPONDS to rotation by construction.

Both cases share a byte-identical wedge mesh, the same transport constants
(nu = 1.075e-5, D = 1e-9 -> Sc ~ 1e4, Lopez-Guajardo), and the same feed rate
(100 mL/min, unified). They differ ONLY in the catalytic-wall treatment, so the
outer-wall BC is the single controlled variable and omega is swept within each case.
(The inlet c0 is nominally 50 vs 1, but the scalar is linear/passive so the normalised
conversion 1 - cup-mixing c / c0 is c0-independent.)

For each case we run 5 INDEPENDENT episodes at {0, 250, 500, 750, 1000} rpm, each
from a fresh (pre-filled) initial condition, holding omega constant for the whole
episode (pimpleFoam in the standard way). Episodes run in parallel, one CPU core
each. When all runs finish, the script writes a table (rpm -> steady conversion)
plus time-series and conversion-vs-omega plots into results/.

USAGE
-----
    python3 steady_state_conversion_sweep.py                # full sweep
    python3 steady_state_conversion_sweep.py --smoke        # ~2 min pipeline test
    python3 steady_state_conversion_sweep.py --workers 8    # cap parallelism
    python3 steady_state_conversion_sweep.py --analyze-only # replot from results/

Requires: OpenFOAM v2506 on PATH (pimpleFoam, foamDictionary) + python numpy,
matplotlib. Designed for a 16-core desktop; safe to start and walk away.
"""

import argparse
import csv
import math
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
CASES_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "taylor_couette_mixing", "cases"))
RESULTS_DIR = os.path.join(HERE, "results")

# The paper's electric-motor power model (Lopez-Guajardo Eqs 18-23). This is the
# SAME power methodology the wall-model / catalysis experiments use, and it is a
# PURE FUNCTION OF omega (independent of the CFD case) -- so applying it to both
# cases makes power computed identically for both, and needs no CFD data. Loaded
# in isolation (it only imports numpy) so we don't drag in the gym package.
_MP_PATH = os.path.normpath(os.path.join(HERE, "..", "..", "taylor_couette_mixing", "motor_power.py"))
_spec = importlib.util.spec_from_file_location("motor_power", _MP_PATH)
motor_power = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(motor_power)


def motor_power_const(omega):
    """Paper motor electrical power P_e [W] at a CONSTANT angular speed omega.
    Constant omega -> the inertia and di/dt terms vanish, so electrical_power is
    flat and we read its value directly. (Uses electrical_power, which needs only
    np.gradient, NOT average_power/energy which call np.trapezoid -- so this works
    under both numpy 1.x and 2.x.) Same for both cases at a given rpm."""
    t = np.array([0.0, 0.5, 1.0])
    w = np.full(3, float(omega))
    return float(np.mean(motor_power.electrical_power(t, w)))

RPMS = [0, 250, 500, 750, 1000]

# Geometry (for the Reynolds number reported in the table).
RI = 0.0254            # inner radius [m]
RO = 0.03175           # outer radius [m]
GAP = RO - RI          # characteristic length [m]
NU = 1.0752688172043011e-05   # kinematic viscosity [m2/s]
RHO = 930.0            # density [kg/m3]
WEDGE_FULL_SCALE = 72.0   # 360deg / 5deg wedge -> full-device power scaling

# Per-case settings. `duration` is the episode length in seconds of sim time.
CASES = {
    "side_outlet_case": {
        "duration": 180.0,
        "label": "Original (resolved-gradient wall)",
        "short": "original_resolved",
        "table_col": "original\n(resolved wall)",
        # conversion printed by the sideOutletConversion coded functionObject
        "conv_re": re.compile(
            r"SIDE_OUTLET_CONVERSION\s+t=(?P<t>[-+\d.eE]+)\s+cOut=(?P<cout>[-+\d.eE]+)"
            r"\s+conversion=(?P<conv>[-+\d.eE]+)"),
        # inner-wall torque from rotationalPower_innerWall FO. Mz_wedge is DYNAMIC
        # (includes rho); divide by rho to get the kinematic Mz_kin the wall-model
        # case reports, so both feed the identical viscous-power formula.
        "torque_re": re.compile(
            r"ROTATIONAL_POWER\s+t=(?P<t>[-+\d.eE]+)\s+Omega=(?P<omega>[-+\d.eE]+)"
            r"\s+Mz_wedge=(?P<mz>[-+\d.eE]+)"),
        "torque_dynamic": True,     # Mz_wedge = rho * Mz_kin  ->  Mz_kin = Mz_wedge / rho
    },
    "side_outlet_cat_wallmodel": {
        "duration": 180.0,   # matches side_outlet_case so both integrate over the same time
        "label": "Wall model (Sherwood sink)",
        "short": "wallmodel_sherwood",
        "table_col": "wall model\n(Sherwood)",
        # conv AND the KINEMATIC torque Mz_kin come from the same METRICS line
        "conv_re": re.compile(
            r"METRICS\s+t=(?P<t>[-+\d.eE]+)\s+Mz_kin=(?P<mz>[-+\d.eE]+)\s+conv=(?P<conv>[-+\d.eE]+)"),
        "torque_re": re.compile(
            r"METRICS\s+t=(?P<t>[-+\d.eE]+)\s+Mz_kin=(?P<mz>[-+\d.eE]+)"),
        "torque_dynamic": False,    # already kinematic
    },
}

DEFAULT_WORKERS = len(CASES) * len(RPMS)   # 10 -> all episodes at once (1 core each)


def rpm_to_omega(rpm):
    return rpm * 2.0 * math.pi / 60.0


def reynolds(rpm):
    return rpm_to_omega(rpm) * RI * GAP / NU


def steady_window(t_end):
    """Seconds at the end of an episode to average for the steady-state value.
    Keyed to the ACTUAL final time reached (robust to short/smoke runs and to an
    episode that diverged early)."""
    return max(2.0, 0.2 * t_end)


# --------------------------------------------------------------------------- #
# OpenFOAM helpers
# --------------------------------------------------------------------------- #
def foam_set(case, entry, value, dictfile):
    subprocess.run(
        ["foamDictionary", "-entry", entry, "-set", str(value), dictfile],
        cwd=case, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def clean_run_artifacts(case):
    """Remove numeric time dirs > 0 and postProcessing; keep 0/, constant/, system/,
    and dynamicCode/ (compiled coded sources)."""
    for name in os.listdir(case):
        p = os.path.join(case, name)
        if os.path.isdir(p):
            try:
                t = float(name)
            except ValueError:
                continue
            if t != 0.0:
                shutil.rmtree(p)
    pp = os.path.join(case, "postProcessing")
    if os.path.isdir(pp):
        shutil.rmtree(pp)


def prepare_template(case_name):
    """Clone the base case into results/<case>/_template, normalise its controlDict
    for a single independent episode, and run pimpleFoam for one deltaT so the coded
    sources COMPILE once. Clones of this template inherit dynamicCode/ and start
    instantly without recompiling."""
    base = os.path.join(CASES_DIR, case_name)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Base case not found: {base}")
    tdir = os.path.join(RESULTS_DIR, case_name, "_template")
    if os.path.isdir(tdir):
        shutil.rmtree(tdir)
    os.makedirs(os.path.dirname(tdir), exist_ok=True)
    shutil.copytree(base, tdir)

    # Strip any cached ICs / stale time dirs so only the pristine 0/ remains.
    for extra in ("0.orig", "0.warmed"):
        p = os.path.join(tdir, extra)
        if os.path.isdir(p):
            shutil.rmtree(p)
    clean_run_artifacts(tdir)

    # One clean independent episode starts from t=0.
    foam_set(tdir, "startFrom", "startTime", "system/controlDict")
    foam_set(tdir, "startTime", "0", "system/controlDict")
    foam_set(tdir, "writeControl", "adjustableRunTime", "system/controlDict")
    foam_set(tdir, "writeInterval", "10", "system/controlDict")
    foam_set(tdir, "purgeWrite", "0", "system/controlDict")
    foam_set(tdir, "runTimeModifiable", "no", "system/controlDict")

    # Compile coded sources with a throwaway one-step run (omega=0; the compiled
    # library is independent of the omega VALUE).
    foam_set(tdir, "boundaryField.inner_wall.omega", "0", "0/U")
    foam_set(tdir, "endTime", "0.05", "system/controlDict")
    env = dict(os.environ, OMP_NUM_THREADS="1")
    r = subprocess.run(["pimpleFoam"], cwd=tdir, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(
            f"[{case_name}] template compile run failed:\n{r.stderr[-2000:]}\n{r.stdout[-1500:]}")
    clean_run_artifacts(tdir)   # drop the tiny compile-run time dir; keep dynamicCode/
    return tdir


# --------------------------------------------------------------------------- #
# One episode
# --------------------------------------------------------------------------- #
def parse_log(case_name, text):
    """Parse pimpleFoam stdout -> (t_conv, conv, t_tq, mz_kin) arrays. mz_kin is the
    KINEMATIC inner-wall torque (Yuhe reports the dynamic Mz_wedge = rho*Mz_kin, so we
    divide by rho; the wall model reports Mz_kin directly), so both cases feed the
    identical viscous-power formula."""
    spec = CASES[case_name]
    t_conv, conv = [], []
    for m in spec["conv_re"].finditer(text):
        t_conv.append(float(m.group("t")))
        conv.append(float(m.group("conv")))
    t_tq, mz = [], []
    for m in spec["torque_re"].finditer(text):
        t_tq.append(float(m.group("t")))
        val = float(m.group("mz"))
        mz.append(val / RHO if spec["torque_dynamic"] else val)
    return np.array(t_conv), np.array(conv), np.array(t_tq), np.array(mz)


def metrics_from_log(case_name, text, rpm, duration):
    """Compute an episode result dict from a pimpleFoam log. Shared by the live run
    and by --analyze-only, so re-running --analyze-only recomputes power with the
    CURRENT methodology WITHOUT re-running CFD.

    Power reported two ways, both computed IDENTICALLY for the two cases:
      power_motor -- the paper's electric-motor model (motor_power.py, Eqs 18-23),
                     a pure function of omega -> identical for both cases at a given
                     rpm. This is the headline "motor power".
      power_visc  -- CFD viscous-drag power, full device = 72*rho*|Mz_kin|*|omega|,
                     from the inner-wall kinematic torque. A cross-check: the two
                     cases share the same momentum field, so this should match."""
    omega = rpm_to_omega(rpm)
    t_conv, conv, t_tq, mz_kin = parse_log(case_name, text)

    end_t = float(t_conv[-1]) if len(t_conv) else float(duration)
    win = steady_window(end_t)
    conv_final = conv_steady = drift = float("nan")
    converged = False
    diverged = False
    if len(conv):
        # Robust steady stats. Conversion is physically in [0,1]. Two fine-mesh failure
        # modes: (a) an occasional scalar-boundedness BLIP (one cOut>c0 / <0 sample) ->
        # reject via the physical mask + MEDIAN so a lone outlier can't skew the value;
        # (b) full numerical DIVERGENCE -- the advection-dominated Sc~1e4 scalar blows up
        # when the Courant overshoots on tiny cells (e.g. r128 at rotating speeds, Co->5)
        # -> almost no physical samples remain in the window, so mark the episode diverged
        # (ok=False) rather than report a garbage 1e40 median.
        phys = (conv >= -0.02) & (conv <= 1.02)
        tcp, ccp = t_conv[phys], conv[phys]
        late = ccp[tcp >= end_t - win] if len(ccp) else np.array([])
        if len(late) < 3:
            # numerical divergence. Report the raw (obviously huge) value so it still
            # shows in the table -- ok=False keeps it OUT of the plots so the axes stay
            # readable, but the number is preserved and clearly nonphysical.
            diverged = True
            raw_late = conv[t_conv >= end_t - win]
            conv_steady = float(np.mean(raw_late)) if len(raw_late) else float(conv[-1])
            conv_final = float(conv[-1])
        else:
            prev = ccp[(tcp >= end_t - 2 * win) & (tcp < end_t - win)]
            conv_final = float(ccp[-1])
            conv_steady = float(np.median(late))
            drift = float(abs(np.median(late) - np.median(prev))) if len(prev) else float("nan")
            converged = (not math.isnan(drift)) and (drift < 0.01 or drift < 0.05 * abs(conv_steady) + 1e-6)

    power_motor = motor_power_const(omega)   # same for both cases (function of omega)
    if len(mz_kin):
        p_visc = WEDGE_FULL_SCALE * RHO * np.abs(mz_kin) * abs(omega)
        power_visc = float(np.mean(p_visc[t_tq >= t_tq[-1] - win])) if len(t_tq) else float("nan")
    else:
        power_visc = float("nan")

    return dict(
        case=case_name, rpm=rpm, omega=omega, re=reynolds(rpm), duration=duration,
        ok=(len(conv) > 0 and not diverged), diverged=diverged,
        conv_final=conv_final, conv_steady=conv_steady,
        drift=drift, converged=converged, power_motor=power_motor, power_visc=power_visc,
        wall_s=float("nan"), t_conv=t_conv, conv=conv,
    )


def run_episode(case_name, template, rpm, duration):
    """Run one constant-omega episode. Returns a result dict."""
    omega = rpm_to_omega(rpm)
    workdir = os.path.join(RESULTS_DIR, case_name, f"rpm_{rpm}")
    if os.path.isdir(workdir):
        shutil.rmtree(workdir)
    shutil.copytree(template, workdir)          # inherits compiled dynamicCode/
    clean_run_artifacts(workdir)

    foam_set(workdir, "boundaryField.inner_wall.omega", repr(omega), "0/U")
    foam_set(workdir, "endTime", repr(float(duration)), "system/controlDict")

    env = dict(os.environ, OMP_NUM_THREADS="1")
    t0 = time.time()
    r = subprocess.run(["pimpleFoam"], cwd=workdir, capture_output=True, text=True, env=env)
    wall = time.time() - t0

    with open(os.path.join(workdir, "log.pimpleFoam"), "w") as f:
        f.write(r.stdout)
    with open(os.path.join(workdir, "log.err"), "w") as f:
        f.write(r.stderr)

    res = metrics_from_log(case_name, r.stdout, rpm, duration)
    res["ok"] = res["ok"] and (r.returncode == 0)
    res["wall_s"] = wall

    # Save the per-episode conversion time series.
    ts_csv = os.path.join(RESULTS_DIR, case_name, f"rpm_{rpm}_timeseries.csv")
    with open(ts_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "conversion"])
        for t, c in zip(res["t_conv"], res["conv"]):
            w.writerow([f"{t:.6g}", f"{c:.8g}"])

    status = "OK" if res["ok"] and len(res["conv"]) else ("FAILED" if r.returncode != 0 else "NO-DATA")
    print(f"  [{status:7s}] {case_name:26s} {rpm:5d} rpm  "
          f"conv_steady={res['conv_steady']:.4f}  drift={res['drift']:.2e}  "
          f"P_motor={res['power_motor']:.3g} W  P_visc={res['power_visc']:.3g} W  "
          f"({wall/60:.1f} min, reached t={(res['t_conv'][-1] if len(res['t_conv']) else 0):.0f}s)",
          flush=True)
    if r.returncode != 0:
        print(f"           stderr tail: {r.stderr.strip()[-300:]}", flush=True)
    return res


# --------------------------------------------------------------------------- #
# Analysis / plotting
# --------------------------------------------------------------------------- #
def load_results_from_logs():
    """Rebuild per-episode results by RE-PARSING each saved rpm_<n>/log.pimpleFoam
    (for --analyze-only). Recomputes conversion AND power with the CURRENT code, so a
    power-methodology change takes effect on an already-completed run without any CFD
    re-run. Falls back to the conversion CSV if a log is missing."""
    results = []
    for case_name, spec in CASES.items():
        for rpm in RPMS:
            log = os.path.join(RESULTS_DIR, case_name, f"rpm_{rpm}", "log.pimpleFoam")
            if os.path.isfile(log):
                with open(log, errors="replace") as f:
                    results.append(metrics_from_log(case_name, f.read(), rpm, spec["duration"]))
                continue
            # fallback: conversion-only from the saved CSV (no power/torque there)
            ts = os.path.join(RESULTS_DIR, case_name, f"rpm_{rpm}_timeseries.csv")
            if not os.path.isfile(ts):
                continue
            t, c = [], []
            with open(ts) as f:
                rd = csv.reader(f); next(rd, None)
                for row in rd:
                    if len(row) >= 2:
                        t.append(float(row[0])); c.append(float(row[1]))
            t = np.array(t); c = np.array(c)
            if len(c):
                win = steady_window(t[-1])
                conv_steady = float(np.mean(c[t >= t[-1] - win]))
                conv_final = float(c[-1])
            else:
                conv_steady = conv_final = float("nan")
            results.append(dict(
                case=case_name, rpm=rpm, omega=rpm_to_omega(rpm), re=reynolds(rpm),
                duration=spec["duration"], ok=len(c) > 0, diverged=False,
                conv_final=conv_final, conv_steady=conv_steady, drift=float("nan"),
                converged=True, power_motor=motor_power_const(rpm_to_omega(rpm)),
                power_visc=float("nan"), wall_s=float("nan"), t_conv=t, conv=c))
    return results


def write_summary_csv(results):
    path = os.path.join(RESULTS_DIR, "summary_table.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "label", "rpm", "omega_rad_s", "Re",
                    "conversion_steady", "conversion_final", "steady_drift", "converged",
                    "diverged", "power_motor_W", "power_visc_cfd_W", "wall_minutes"])
        for r in results:
            wall_min = r.get("wall_s", float("nan"))
            wall_min = wall_min / 60 if wall_min == wall_min else float("nan")  # nan-safe
            w.writerow([
                r["case"], CASES[r["case"]]["label"], r["rpm"],
                f"{r['omega']:.4f}", f"{r['re']:.1f}",
                f"{r['conv_steady']:.6g}", f"{r['conv_final']:.6g}",   # %g -> diverged shows as e.g. 3.3e+07
                f"{r['drift']:.3e}", int(bool(r["converged"])),
                int(bool(r.get("diverged"))),
                f"{r['power_motor']:.6g}", f"{r['power_visc']:.6g}",
                f"{wall_min:.2f}"])
    return path


def disp_conv(r):
    """Display string for a conversion cell: the value if valid, the (obviously huge)
    diverged value if the episode blew up, else 'failed'/'--'."""
    if r is None:
        return "--"
    if r["ok"]:
        return f"{r['conv_steady']:.4f}"
    if r.get("diverged"):
        return f"{r['conv_steady']:.1e} DIVERGED"
    return "failed"


def print_headline_table(results):
    print("\n" + "=" * 72)
    print("STEADY-STATE CONVERSION vs ANGULAR VELOCITY")
    print("=" * 72)
    for case_name, spec in CASES.items():
        rs = {r["rpm"]: r for r in results if r["case"] == case_name}
        print(f"\n{spec['label']}   [{case_name}]")
        print(f"  {'RPM':>5s} | {'omega[rad/s]':>12s} | {'Re':>7s} | {'steady conversion':>18s}")
        print("  " + "-" * 52)
        vals = []
        for rpm in RPMS:
            r = rs.get(rpm)
            if r and r["ok"]:
                vals.append(r["conv_steady"])
            cs = disp_conv(r) if r else "(failed)"
            print(f"  {rpm:>5d} | {rpm_to_omega(rpm):>12.3f} | {reynolds(rpm):>7.0f} | {cs:>18s}")
        if vals:
            rng = max(vals) - min(vals)
            print(f"  -> conversion range across omega: {rng:.4f} "
                  f"({'DEPENDS on omega' if rng > 0.01 else 'nearly FLAT in omega'})")
    print("=" * 72 + "\n")


def plot_timeseries(results):
    for case_name, spec in CASES.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        cmap = plt.get_cmap("viridis")
        rs = [r for r in results if r["case"] == case_name and r["ok"]]
        for r in sorted(rs, key=lambda x: x["rpm"]):
            frac = r["rpm"] / max(RPMS) if max(RPMS) else 0
            ax.plot(r["t_conv"], r["conv"], color=cmap(frac), lw=1.8,
                    label=f"{r['rpm']} rpm")
        # Bound the y-view to ~1.9x the steady level so transient numerical spikes (the
        # under-damped Sc~1e4 scalar can briefly overshoot on fine meshes) don't crush
        # every real curve into a thin strip. The full traces are still drawn; only the
        # view is limited.
        sv = [r["conv_steady"] for r in rs if r["ok"] and abs(r["conv_steady"]) < 1.0]
        if sv:
            ax.set_ylim(-0.01, max(max(sv) * 1.9, 0.03))
        ax.set_xlabel("time [s]")
        ax.set_ylabel("outlet conversion  (1 - cup c / c0)")
        ax.set_title(f"Conversion approach to steady state\n{spec['label']}")
        ax.grid(True, alpha=0.3)
        ax.legend(title="inner wall", loc="best")
        fig.tight_layout()
        out = os.path.join(RESULTS_DIR, f"conversion_vs_time__{spec['short']}.png")
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"  wrote {out}")


def plot_conversion_vs_omega(results):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    markers = {"side_outlet_case": ("o-", "#d1495b"),
               "side_outlet_cat_wallmodel": ("s-", "#2e6f95")}
    for case_name, spec in CASES.items():
        rs = sorted([r for r in results if r["case"] == case_name and r["ok"]],
                    key=lambda x: x["rpm"])
        if not rs:
            continue
        style, color = markers.get(case_name, ("o-", None))
        ax.plot([r["rpm"] for r in rs], [r["conv_steady"] for r in rs], style,
                color=color, lw=2, ms=8, label=spec["label"])
    ax.set_xlabel("inner-wall speed [rpm]")
    ax.set_ylabel("steady-state outlet conversion")
    ax.set_title("Steady-state conversion vs angular velocity")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_xticks(RPMS)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "conversion_vs_omega.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_conversion_vs_power(results):
    """Steady conversion vs the paper motor power (same power methodology + axis for
    both cases). Because motor power is a function of omega only, the two cases share
    the SAME x-values at each rpm -- so vertical gaps show which wall treatment gives
    more conversion at equal motor power."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    markers = {"side_outlet_case": ("o-", "#d1495b"),
               "side_outlet_cat_wallmodel": ("s-", "#2e6f95")}
    for case_name, spec in CASES.items():
        rs = sorted([r for r in results if r["case"] == case_name and r["ok"]],
                    key=lambda x: x["power_motor"])
        if not rs:
            continue
        style, color = markers.get(case_name, ("o-", None))
        xs = [r["power_motor"] for r in rs]
        ys = [r["conv_steady"] for r in rs]
        ax.plot(xs, ys, style, color=color, lw=2, ms=8, label=spec["label"])
        for r in rs:
            ax.annotate(f"{r['rpm']}", (r["power_motor"], r["conv_steady"]),
                        textcoords="offset points", xytext=(6, 5), fontsize=8, color=color)
    ax.set_xlabel("motor electrical power  [W]   (Lopez-Guajardo Eqs 18-23)")
    ax.set_ylabel("steady-state outlet conversion")
    ax.set_title("Steady-state conversion vs motor power\n(labels = rpm; same power model for both cases)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "conversion_vs_power.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_summary_table_image(results):
    """Render the headline table (rpm -> steady conversion for each case) as a PNG."""
    fig, ax = plt.subplots(figsize=(11, 0.6 * (len(RPMS) + 3)))
    ax.axis("off")
    col_labels = (["RPM", "omega\n[rad/s]", "Re", "motor P\n[W]"]
                  + [CASES[c]["table_col"] for c in CASES])
    rows = []
    by = {(r["case"], r["rpm"]): r for r in results}
    for rpm in RPMS:
        row = [f"{rpm}", f"{rpm_to_omega(rpm):.2f}", f"{reynolds(rpm):.0f}",
               f"{motor_power_const(rpm_to_omega(rpm)):.2f}"]
        for case_name in CASES:
            r = by.get((case_name, rpm))
            if r and r["ok"]:
                row.append(f"{r['conv_steady']:.4f}")
            elif r and r.get("diverged"):
                row.append(f"{r['conv_steady']:.0e}\n(diverged)")
            else:
                row.append("--")
        rows.append(row)
    tbl = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.0, 1.8)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#30323a")
        tbl[0, j].set_text_props(color="w", fontweight="bold")
    ax.set_title("Final steady-state conversion by angular velocity", pad=16, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "summary_table.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def analyze(results):
    print("\nGenerating figures + table ...")
    write_summary_csv(results)
    print_headline_table(results)
    plot_timeseries(results)
    plot_conversion_vs_omega(results)
    plot_conversion_vs_power(results)
    plot_summary_table_image(results)
    print(f"\nAll outputs in: {RESULTS_DIR}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help=f"parallel episodes, 1 core each (default {DEFAULT_WORKERS})")
    ap.add_argument("--smoke", action="store_true",
                    help="quick pipeline test: short episodes so the whole flow runs in ~2 min")
    ap.add_argument("--analyze-only", action="store_true",
                    help="regenerate plots/table from existing results/ without running CFD")
    ap.add_argument("--cases", default=None,
                    help="comma-separated subset of cases to RUN (default: all). e.g. "
                         "--cases side_outlet_cat_wallmodel re-runs only the wall model; "
                         "the final table still includes any other case already in results/.")
    ap.add_argument("--rpms", default=None,
                    help="comma-separated subset of RPMs to RUN (default: all in RPMS). "
                         "Like --cases, the final table/plots still include any rpm already "
                         "on disk in results/. e.g. --rpms 2500 runs ONLY the 2500 rpm "
                         "episode and then re-plots every speed (the finished runs are reused, "
                         "not recomputed).")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.analyze_only:
        analyze(load_results_from_logs())
        return

    if args.smoke:
        for c in CASES.values():
            c["duration"] = 6.0
        print("[SMOKE MODE] durations set to 6 s -- pipeline test only, NOT physical steady state.")

    if shutil.which("pimpleFoam") is None or shutil.which("foamDictionary") is None:
        sys.exit("ERROR: OpenFOAM not on PATH (need pimpleFoam, foamDictionary). "
                 "Source your OpenFOAM environment first.")

    if args.cases:
        run_cases = [c.strip() for c in args.cases.split(",") if c.strip()]
        unknown = [c for c in run_cases if c not in CASES]
        if unknown:
            sys.exit(f"ERROR: unknown case(s) {unknown}. Known: {list(CASES)}")
    else:
        run_cases = list(CASES)

    if args.rpms:
        try:
            run_rpms = [int(x.strip()) for x in args.rpms.split(",") if x.strip()]
        except ValueError:
            sys.exit(f"ERROR: --rpms must be integers, got {args.rpms!r}")
        unknown = [r for r in run_rpms if r not in RPMS]
        if unknown:
            sys.exit(f"ERROR: --rpms {unknown} not in RPMS={RPMS}. Add them to RPMS "
                     f"(in the sweep wrapper) first so the table/plots include them.")
    else:
        run_rpms = list(RPMS)

    episodes = [(cn, rpm, CASES[cn]["duration"]) for cn in run_cases for rpm in run_rpms]
    print(f"Cases dir : {CASES_DIR}")
    print(f"Results   : {RESULTS_DIR}")
    print(f"Running   : {len(episodes)} episodes  ({', '.join(run_cases)} x {run_rpms} rpm)")
    print(f"Durations : " + ", ".join(f"{cn}={CASES[cn]['duration']:.0f}s" for cn in run_cases))
    print(f"Workers   : {args.workers} (1 CPU core / episode)\n")

    # 1) Compile each run case's coded sources once (serial), so clones start instantly.
    print("Compiling coded sources (one throwaway step per case) ...")
    templates = {}
    for case_name in run_cases:
        t0 = time.time()
        templates[case_name] = prepare_template(case_name)
        print(f"  {case_name}: template ready ({time.time()-t0:.0f}s)")
    print()

    # 2) Run episodes in parallel (each pimpleFoam is serial -> 1 core).
    t_start = time.time()
    print("Running episodes ...")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_episode, cn, templates[cn], rpm, dur): (cn, rpm)
                for (cn, rpm, dur) in episodes}
        for fut in as_completed(futs):
            fut.result()
    print(f"\nAll episodes done in {(time.time()-t_start)/60:.1f} min "
          f"(wall clock; ran {args.workers}-wide).")

    # 3) Tables + figures -- rebuilt from ALL logs on disk, so a subset re-run
    #    (--cases) still produces a complete table including previously-run cases.
    analyze(load_results_from_logs())


if __name__ == "__main__":
    main()
