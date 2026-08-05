#!/usr/bin/env python3
"""Recreate Lopez-Guajardo Fig. 7 (pulsed vs constant rotation) on the GRADED
Sc=1075 side-outlet case.

WHAT THIS ANSWERS
-----------------
Does pulsating modulation beat constant rotation at the same NOMINAL speed w_b in
our resolved-gradient reactor -- the paper's Fig. 7 claim -- and what reward would
each waveform earn if this were a TD3 training episode?

For each nominal speed w_b in {100, 200, 300, 400, 500} rpm we run TWO independent
60 s episodes on side_outlet_case_sc1075_graded (25 um wall cell, Sc=1075):

  pulsed    -- the paper's duty-cycle square wave (Fig. 3 / Eq. 8):
               Delta_w = w_0 = w_b / (2 D), idle level 0, so the wave jumps between
               0 and w_b / D. With D = 0.2, T = 25 s: bursts at 5*w_b rpm for 5 s,
               idle for 20 s. Peaks span {500 .. 2500} rpm -- all previously shown
               stable on this mesh (constant-2500 rpm sweep ran 150 s clean).
               Burst-first phase, 0.05 s ramped edges (the env's ramp_time), same
               square builder as TaylorCouetteWaveformEnv.
  constant  -- D = 1.0: the wall holds w_b for the whole episode (the paper's
               "constant" baseline; identical drive path, degenerate table).

NB 60 s / 25 s = 2.4 periods (burst-first), so the realized MEAN omega of a pulsed
episode is 1.25*w_b, not w_b -- inherent to the requested (60 s, T=25 s) spec; the
Fig. 7 x-axis is the NOMINAL w_b, as in the paper.

Omega is driven by an inline tabulated Function1 written into 0/U (the proven
post-omega-freeze-fix path; a single continuous run reads the table once at solver
start, so the omegaCoeffs re-serialisation shadowing bug cannot occur). The
rotationalPower FO measures the ACTUAL wall Omega every 0.1 s, and omega_traces.png
overlays commanded vs measured omega(t) as an explicit freeze-bug guard.

REWARD ("as if this were a TD3 run")
------------------------------------
Per episode:  X = mean outlet conversion over the LAST FULL PERIOD (t in [35,60] s;
same 25 s window for both modes), P = episode-average motor electrical power on the
commanded omega(t) (paper Eqs 18-23 via motor_power.py, densified at 100 Hz exactly
like TaylorCouetteWaveformEnv), P_max = 31.94 W (= motor power at 2500 rpm, the
w_b=500 pulse peak). Both weights 1. The table reports BOTH signs:
    R_minus = X - P/P_max   (the trainer's convention: energy is a PENALTY)
    R_plus  = X + P/P_max   (the formula as literally given)

DATA / PLOTS (mirrors steady_state_conversion/results_graded_sc1075)
--------------------------------------------------------------------
results/: conversion_vs_time__{pulsed,constant}.png, fig7_conversion_vs_wb.png,
conversion_vs_power.png, omega_traces.png, summary_table.{csv,png}, per-episode
{mode}_wb{N}_timeseries.csv / _power.csv / _waveform_points.csv, and the full case
dirs with time folders every 1 s (ParaView animation). Wall-clock per episode is
recorded (TD3 cost estimation).

USAGE
-----
    nohup python3 -u fig7_sweep.py > fig7.log 2>&1 &     # full run (~3 h, 10 cores)
    python3 fig7_sweep.py --smoke                        # ~10 min pipeline test
    python3 fig7_sweep.py --analyze-only                 # replot from results/
    python3 fig7_sweep.py --modes pulsed --wbs 500       # subset re-run

Requires OpenFOAM v2506 on PATH (pimpleFoam, foamDictionary) + numpy, matplotlib.
"""

import argparse
import csv
import importlib.util
import math
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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

CASE_NAME = "side_outlet_case_sc1075_graded"

WBS = [100, 200, 300, 400, 500]      # nominal angular speed w_b [rpm]
MODES = {"pulsed": 0.2, "constant": 1.0}   # mode -> duty cycle D
PERIOD = 25.0                        # square-wave period T [s]
EPISODE = 60.0                       # episode length [s]
RAMP = 0.05                          # edge ramp width [s] -- the env's ramp_time
P_MAX = 31.94                        # W; motor power at 2500 rpm (w_b=500 pulse peak)

# WARMED-template mode (default OFF -> pristine IC, all pre-2026-08 sweeps
# unchanged). Set WARMUP_RPM in a variant script to spin the template once at
# that constant speed for WARMUP_S seconds and PROMOTE the final state to 0/,
# so every episode starts from the steady operating state at t=0 (waveform
# tables and plot time axes keep their t=0 origin). This is the
# continuous-manufacturing benchmark IC matching the duty-cycle RL runs
# (experiments/modulation_rl/parallel_train_duty.py); warmed pulsed episodes
# must be compared against WARMED constant baselines, never the pristine ones.
WARMUP_RPM = None
WARMUP_S = 60.0                      # ~2.3 residence times (tau = V/Q ~ 26 s)

RPM = 2.0 * math.pi / 60.0           # rpm -> rad/s

# Paper motor power model (Eqs 18-23), loaded in isolation like the steady-state
# sweep does (imports only numpy; avoids dragging in the gym package).
_MP_PATH = os.path.normpath(os.path.join(HERE, "..", "..", "taylor_couette_mixing", "motor_power.py"))
_spec = importlib.util.spec_from_file_location("motor_power", _MP_PATH)
motor_power = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(motor_power)

CONV_RE = re.compile(
    r"SIDE_OUTLET_CONVERSION\s+t=(?P<t>[-+\d.eE]+)\s+cOut=(?P<cout>[-+\d.eE]+)"
    r"\s+conversion=(?P<conv>[-+\d.eE]+)")
POWER_RE = re.compile(
    r"ROTATIONAL_POWER\s+t=(?P<t>[-+\d.eE]+)\s+Omega=(?P<omega>[-+\d.eE]+)"
    r"\s+Mz_wedge=(?P<mz>[-+\d.eE]+)\s+P_full=(?P<pfull>[-+\d.eE]+)")


def peak_rpm(wb, duty):
    """Paper Eq. 8: Delta_w = w_0 = w_b/(2D), idle 0 -> peak = w_b/D."""
    return wb / duty


# Copied from taylor_couette_mixing/envs/taylor_couette_waveform.py
# (square_wave_points) so this script stays self-contained like the other sweep
# scripts. Keep in sync with the env -- the future TD3 run drives the wall with
# the exact same table builder.
def square_wave_points(t0, duration, w_hi, w_lo, period, duty, ramp, phase0=0.0):
    """Duty-cycle square wave as a ramped tabulated Function1 over [t0, t0+duration].
    Phase in [0,1): <duty -> w_hi (burst), >=duty -> w_lo (idle). Returns
    (points, phase_out) with absolute, strictly-increasing times and a final hold
    point past the end so the BC never evaluates out of range."""
    period = max(float(period), 1e-6)
    duty = min(max(float(duty), 1e-6), 1.0)

    def level(ph):
        return w_hi if (ph % 1.0) < duty else w_lo

    phase = phase0 % 1.0
    cur = level(phase)
    pts = [(t0, cur)]
    t = t0
    t_end = t0 + duration
    for _ in range(1000000):
        nb = duty if phase < duty else 1.0      # next level-change boundary
        t_sw = t + (nb - phase) * period
        if t_sw >= t_end - 1e-9:
            break
        new_phase = nb % 1.0
        nxt = level(new_phase)
        if nxt != cur:                          # emit a ramped transition
            ta = max(t_sw - 0.5 * ramp, pts[-1][0] + 1e-6)
            tb = max(ta + ramp, ta + 1e-6)
            pts.append((ta, cur))
            pts.append((tb, nxt))
            cur = nxt
        t, phase = t_sw, new_phase
    pts.append((t_end + 1.0, cur))
    return pts, (phase0 + duration / period) % 1.0


def waveform_points(mode, wb, duration, period):
    """(t, omega_rad) table for one episode of `mode` at nominal wb rpm."""
    duty = MODES[mode]
    w_hi = peak_rpm(wb, duty) * RPM
    pts, _ = square_wave_points(0.0, duration, w_hi, 0.0, period, duty, RAMP)
    return pts


def densify(pts, duration):
    """Commanded omega(t) on the env's grid (n = duration*100, i.e. 100 Hz) --
    the SAME densification TaylorCouetteWaveformEnv uses for its motor energy,
    so the reward here matches what a TD3 run would compute."""
    tt = np.array([p[0] for p in pts], dtype=float)
    ww = np.array([p[1] for p in pts], dtype=float)
    n = int(min(20000, max(400, duration * 100)))
    grid = np.linspace(0.0, duration, n)
    return grid, np.interp(grid, tt, ww)


def motor_power_avg(pts, duration):
    """Episode-average motor electrical power [W] on the commanded waveform.
    Uniform grid -> plain mean == time average (avoids np.trapezoid, numpy 1/2 safe)."""
    grid, w = densify(pts, duration)
    return float(np.mean(motor_power.electrical_power(grid, w)))


# --------------------------------------------------------------------------- #
# OpenFOAM helpers (same pattern as steady_state_conversion_sweep.py)
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


def prepare_template():
    """Clone the base case into results/_template, normalise controlDict (field
    writes every 1 s for ParaView; coded FOs sampled every 0.1 s), and run
    pimpleFoam for one deltaT so the coded sources COMPILE once. Clones inherit
    dynamicCode/ and start instantly."""
    base = os.path.join(CASES_DIR, CASE_NAME)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Base case not found: {base}")
    tdir = os.path.join(RESULTS_DIR, "_template")
    if os.path.isdir(tdir):
        shutil.rmtree(tdir)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    shutil.copytree(base, tdir)
    clean_run_artifacts(tdir)

    foam_set(tdir, "startFrom", "startTime", "system/controlDict")
    foam_set(tdir, "startTime", "0", "system/controlDict")
    foam_set(tdir, "writeControl", "adjustableRunTime", "system/controlDict")
    foam_set(tdir, "writeInterval", "1", "system/controlDict")     # 1 s time folders
    foam_set(tdir, "purgeWrite", "0", "system/controlDict")
    foam_set(tdir, "runTimeModifiable", "no", "system/controlDict")
    # Dense conversion / omega / power traces: 0.1 s FO cadence (was 1 s).
    for fo in ("sideOutletConversion", "rotationalPower_innerWall"):
        foam_set(tdir, f"functions.{fo}.executeInterval", "0.1", "system/controlDict")
        foam_set(tdir, f"functions.{fo}.writeInterval", "0.1", "system/controlDict")

    # Compile coded sources with a throwaway one-step run (omega VALUE irrelevant).
    foam_set(tdir, "boundaryField.inner_wall.omega", "0", "0/U")
    foam_set(tdir, "endTime", "0.05", "system/controlDict")
    env = dict(os.environ, OMP_NUM_THREADS="1")
    r = subprocess.run(["pimpleFoam"], cwd=tdir, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(
            f"template compile run failed:\n{r.stderr[-2000:]}\n{r.stdout[-1500:]}")
    clean_run_artifacts(tdir)

    if WARMUP_RPM:
        print(f"  warming template: constant {WARMUP_RPM:.0f} rpm x {WARMUP_S:.0f} s "
              f"(one-time; the continuous-ops IC) ...", flush=True)
        t0 = time.time()
        foam_set(tdir, "boundaryField.inner_wall.omega", repr(WARMUP_RPM * RPM), "0/U")
        foam_set(tdir, "endTime", repr(float(WARMUP_S)), "system/controlDict")
        r = subprocess.run(["pimpleFoam"], cwd=tdir, capture_output=True, text=True, env=env)
        with open(os.path.join(tdir, "log.warmup"), "w") as f:
            f.write(r.stdout)
        if r.returncode != 0:
            raise RuntimeError(f"template warmup failed:\n{r.stderr[-2000:]}")
        t_conv, conv = parse_log(r.stdout)[:2]
        if len(conv) and abs(t_conv[-1] - WARMUP_S) > 1.0:
            raise RuntimeError(
                f"template warmup stopped early at t={t_conv[-1]:.1f}s (< {WARMUP_S}s)")
        # Promote the final state to 0/: episodes start warmed AT t=0. Drop
        # 0/uniform/ so the episode's deltaT comes fresh from controlDict, not
        # the warmup's saved dt.
        times = [(float(n), n) for n in os.listdir(tdir)
                 if os.path.isdir(os.path.join(tdir, n)) and _is_float(n) and float(n) > 0]
        latest = max(times)[1]
        shutil.rmtree(os.path.join(tdir, "0"))
        os.rename(os.path.join(tdir, latest), os.path.join(tdir, "0"))
        shutil.rmtree(os.path.join(tdir, "0", "uniform"), ignore_errors=True)
        clean_run_artifacts(tdir)
        print(f"  template warmed in {(time.time()-t0)/60:.1f} min "
              f"(steady X = {conv[-1]:.4f} at {WARMUP_RPM:.0f} rpm)", flush=True)
    return tdir


def _is_float(name):
    try:
        float(name)
        return True
    except ValueError:
        return False


# --------------------------------------------------------------------------- #
# One episode
# --------------------------------------------------------------------------- #
def parse_log(text):
    """pimpleFoam stdout -> (t_conv, conv, t_pw, omega, mz, p_full) arrays."""
    t_conv, conv = [], []
    for m in CONV_RE.finditer(text):
        t_conv.append(float(m.group("t")))
        conv.append(float(m.group("conv")))
    t_pw, om, mz, pf = [], [], [], []
    for m in POWER_RE.finditer(text):
        t_pw.append(float(m.group("t")))
        om.append(float(m.group("omega")))
        mz.append(float(m.group("mz")))
        pf.append(float(m.group("pfull")))
    return (np.array(t_conv), np.array(conv),
            np.array(t_pw), np.array(om), np.array(mz), np.array(pf))


def metrics_from_log(mode, wb, text, duration, period):
    """Episode result dict from a pimpleFoam log. Shared by the live run and
    --analyze-only, so replots recompute reward/power with the CURRENT code."""
    t_conv, conv, t_pw, om, mz, pf = parse_log(text)
    duty = MODES[mode]
    pts = waveform_points(mode, wb, duration, period)

    end_t = float(t_conv[-1]) if len(t_conv) else 0.0
    win = min(period, end_t) if end_t > 0 else period   # last FULL period
    x_win = conv_final = float("nan")
    diverged = False
    if len(conv):
        # Reject unphysical blips (scalar boundedness) before averaging; if almost
        # no physical samples remain in the window the episode diverged.
        phys = (conv >= -0.02) & (conv <= 1.02)
        tcp, ccp = t_conv[phys], conv[phys]
        late = ccp[tcp >= end_t - win] if len(ccp) else np.array([])
        if len(late) < 3:
            diverged = True
            raw_late = conv[t_conv >= end_t - win]
            x_win = float(np.mean(raw_late)) if len(raw_late) else float(conv[-1])
            conv_final = float(conv[-1])
        else:
            x_win = float(np.mean(late))
            conv_final = float(ccp[-1])
    conv_ep_mean = float(np.mean(conv[(conv >= -0.02) & (conv <= 1.02)])) if len(conv) else float("nan")

    # Reward: X = last-full-period mean conversion, P = episode-average motor power
    # on the COMMANDED waveform (env-identical densification), P_max given.
    p_motor = motor_power_avg(pts, duration)
    p_norm = p_motor / P_MAX
    r_minus = x_win - p_norm      # trainer convention (energy penalized)
    r_plus = x_win + p_norm       # the formula as literally specified

    # CFD cross-checks from the measured wall: viscous power over the same window,
    # measured peak/mean omega (freeze-bug guard: pulsed peak should be ~5*w_b).
    if len(t_pw):
        sel = t_pw >= t_pw[-1] - win
        p_visc_win = float(np.mean(pf[sel]))
        om_peak_rpm = float(np.max(np.abs(om)) / RPM)
        om_mean_rpm = float(np.mean(np.abs(om)) / RPM)
    else:
        p_visc_win = om_peak_rpm = om_mean_rpm = float("nan")

    return dict(
        mode=mode, wb=wb, duty=duty, peak_rpm=peak_rpm(wb, duty),
        duration=duration, period=period,
        ok=(len(conv) > 0 and not diverged), diverged=diverged,
        x_win=x_win, conv_final=conv_final, conv_ep_mean=conv_ep_mean,
        p_motor=p_motor, p_norm=p_norm, r_minus=r_minus, r_plus=r_plus,
        p_visc_win=p_visc_win, om_peak_rpm=om_peak_rpm, om_mean_rpm=om_mean_rpm,
        wall_s=float("nan"),
        t_conv=t_conv, conv=conv, t_pw=t_pw, om=om, mz=mz, pf=pf, pts=pts,
    )


def episode_tag(mode, wb):
    return f"{mode}_wb{wb}"


def run_episode(template, mode, wb, duration, period):
    tag = episode_tag(mode, wb)
    workdir = os.path.join(RESULTS_DIR, tag)
    if os.path.isdir(workdir):
        shutil.rmtree(workdir)
    shutil.copytree(template, workdir)          # inherits compiled dynamicCode/
    clean_run_artifacts(workdir)

    pts = waveform_points(mode, wb, duration, period)
    table = "table (" + " ".join(f"({t:.6f} {w:.6f})" for t, w in pts) + ")"
    foam_set(workdir, "boundaryField.inner_wall.omega", table, "0/U")
    foam_set(workdir, "endTime", repr(float(duration)), "system/controlDict")

    env = dict(os.environ, OMP_NUM_THREADS="1")
    t0 = time.time()
    r = subprocess.run(["pimpleFoam"], cwd=workdir, capture_output=True, text=True, env=env)
    wall = time.time() - t0

    with open(os.path.join(workdir, "log.pimpleFoam"), "w") as f:
        f.write(r.stdout)
    with open(os.path.join(workdir, "log.err"), "w") as f:
        f.write(r.stderr)

    res = metrics_from_log(mode, wb, r.stdout, duration, period)
    res["ok"] = res["ok"] and (r.returncode == 0)
    res["wall_s"] = wall
    save_episode_csvs(res)

    status = "OK" if res["ok"] and len(res["conv"]) else ("FAILED" if r.returncode != 0 else "NO-DATA")
    print(f"  [{status:7s}] {tag:16s} peak={res['peak_rpm']:4.0f} rpm  "
          f"X={res['x_win']:.4f}  P_motor={res['p_motor']:.3f} W  "
          f"R-={res['r_minus']:.4f}  R+={res['r_plus']:.4f}  "
          f"omega_meas_peak={res['om_peak_rpm']:.0f} rpm  "
          f"({wall/60:.1f} min, reached t={(res['t_conv'][-1] if len(res['t_conv']) else 0):.0f}s)",
          flush=True)
    if r.returncode != 0:
        print(f"           stderr tail: {r.stderr.strip()[-300:]}", flush=True)
    return res


def save_episode_csvs(res):
    tag = episode_tag(res["mode"], res["wb"])
    with open(os.path.join(RESULTS_DIR, f"{tag}_timeseries.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "conversion"])
        for t, c in zip(res["t_conv"], res["conv"]):
            w.writerow([f"{t:.6g}", f"{c:.8g}"])
    with open(os.path.join(RESULTS_DIR, f"{tag}_power.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "omega_meas_rad_s", "Mz_wedge_Nm", "P_visc_full_W"])
        for t, o, m, p in zip(res["t_pw"], res["om"], res["mz"], res["pf"]):
            w.writerow([f"{t:.6g}", f"{o:.8g}", f"{m:.8g}", f"{p:.8g}"])
    with open(os.path.join(RESULTS_DIR, f"{tag}_waveform_points.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "omega_cmd_rad_s"])
        for t, o in res["pts"]:
            w.writerow([f"{t:.6f}", f"{o:.6f}"])


# --------------------------------------------------------------------------- #
# Analysis / plotting
# --------------------------------------------------------------------------- #
def load_results_from_logs(duration, period):
    results = []
    for mode in MODES:
        for wb in WBS:
            log = os.path.join(RESULTS_DIR, episode_tag(mode, wb), "log.pimpleFoam")
            if not os.path.isfile(log):
                continue
            with open(log, errors="replace") as f:
                results.append(metrics_from_log(mode, wb, f.read(), duration, period))
    return results


def write_summary_csv(results):
    path = os.path.join(RESULTS_DIR, "summary_table.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "wb_rpm", "duty", "peak_rpm",
                    "X_conv_lastperiod", "conv_final", "conv_episode_mean",
                    "P_motor_W", "P_over_Pmax",
                    "R_minus (X - P/Pmax)", "R_plus (X + P/Pmax)",
                    "P_visc_cfd_W", "omega_meas_peak_rpm", "omega_meas_mean_rpm",
                    "diverged", "wall_minutes"])
        for r in sorted(results, key=lambda x: (x["mode"], x["wb"])):
            wall_min = r.get("wall_s", float("nan"))
            wall_min = wall_min / 60 if wall_min == wall_min else float("nan")
            w.writerow([
                r["mode"], r["wb"], f"{r['duty']:.2f}", f"{r['peak_rpm']:.0f}",
                f"{r['x_win']:.6g}", f"{r['conv_final']:.6g}", f"{r['conv_ep_mean']:.6g}",
                f"{r['p_motor']:.6g}", f"{r['p_norm']:.6g}",
                f"{r['r_minus']:.6g}", f"{r['r_plus']:.6g}",
                f"{r['p_visc_win']:.6g}", f"{r['om_peak_rpm']:.1f}", f"{r['om_mean_rpm']:.1f}",
                int(bool(r["diverged"])), f"{wall_min:.2f}"])
    return path


def print_headline(results):
    print("\n" + "=" * 78)
    print("LOPEZ-GUAJARDO FIG. 7 RECREATION -- graded mesh, Sc=1075")
    print("=" * 78)
    by = {(r["mode"], r["wb"]): r for r in results}
    print(f"  {'w_b':>5s} | {'constant X':>10s} | {'pulsed X':>10s} | "
          f"{'const R-':>9s} | {'pulsed R-':>9s} | {'const R+':>9s} | {'pulsed R+':>9s}")
    print("  " + "-" * 74)
    for wb in WBS:
        c, p = by.get(("constant", wb)), by.get(("pulsed", wb))
        def g(r, k):
            return f"{r[k]:.4f}" if (r and r["ok"]) else "--"
        print(f"  {wb:>5d} | {g(c,'x_win'):>10s} | {g(p,'x_win'):>10s} | "
              f"{g(c,'r_minus'):>9s} | {g(p,'r_minus'):>9s} | "
              f"{g(c,'r_plus'):>9s} | {g(p,'r_plus'):>9s}")
    print("=" * 78 + "\n")


def plot_timeseries(results):
    for mode, duty in MODES.items():
        rs = [r for r in results if r["mode"] == mode and r["ok"]]
        if not rs:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        cmap = plt.get_cmap("viridis")
        for r in sorted(rs, key=lambda x: x["wb"]):
            frac = r["wb"] / max(WBS)
            ax.plot(r["t_conv"], r["conv"], color=cmap(frac), lw=1.6,
                    label=f"$w_b$={r['wb']} rpm (peak {r['peak_rpm']:.0f})")
        sv = [r["x_win"] for r in rs if abs(r["x_win"]) < 1.0]
        if sv:
            ax.set_ylim(-0.01, max(max(sv) * 1.9, 0.03))
        ax.set_xlabel("time [s]")
        ax.set_ylabel("outlet conversion  (1 - cup c / c0)")
        desc = (f"pulsed square, D={duty:.0%}, T={rs[0]['period']:.3g} s"
                if mode == "pulsed" else "constant rotation")
        ax.set_title(f"Conversion vs time -- {desc}\nGraded mesh, Sc=1075 (25 um wall cell)")
        ax.grid(True, alpha=0.3)
        ax.legend(title="nominal speed", loc="best", fontsize=9)
        fig.tight_layout()
        out = os.path.join(RESULTS_DIR, f"conversion_vs_time__{mode}.png")
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"  wrote {out}")


def plot_fig7(results):
    """The Fig. 7 recreation: conversion vs nominal w_b, pulsed (dotted, open
    triangles) vs constant (solid, open circles), paper styling."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    styles = {"pulsed": dict(ls=":", marker="^", label="Pulsating"),
              "constant": dict(ls="-", marker="o", label="Constant")}
    for mode, st in styles.items():
        rs = sorted([r for r in results if r["mode"] == mode and r["ok"]],
                    key=lambda x: x["wb"])
        if not rs:
            continue
        ax.plot([r["wb"] for r in rs], [r["x_win"] for r in rs],
                color="k", lw=1.5, ms=10, mfc="white", mec="k", **st)
    ax.set_xlabel("Nominal angular speed $w_b$ [rpm]")
    ax.set_ylabel("Conversion  (mean over last period)")
    ax.set_title("Lopez-Guajardo Fig. 7 recreation\n"
                 f"Graded mesh Sc=1075, {EPISODE:.0f} s episodes, "
                 f"D={MODES['pulsed']:.0%}, T={PERIOD:.3g} s")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Rotation", loc="best")
    ax.set_xticks(WBS)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "fig7_conversion_vs_wb.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_conversion_vs_power(results):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    styles = {"pulsed": ("^:", "#d1495b"), "constant": ("o-", "#2e6f95")}
    for mode, (style, color) in styles.items():
        rs = sorted([r for r in results if r["mode"] == mode and r["ok"]],
                    key=lambda x: x["p_motor"])
        if not rs:
            continue
        ax.plot([r["p_motor"] for r in rs], [r["x_win"] for r in rs], style,
                color=color, lw=2, ms=8, label=mode)
        for r in rs:
            ax.annotate(f"{r['wb']}", (r["p_motor"], r["x_win"]),
                        textcoords="offset points", xytext=(6, 5), fontsize=8, color=color)
    ax.set_xlabel("episode-average motor electrical power [W]  (Eqs 18-23, commanded omega(t))")
    ax.set_ylabel("conversion (mean over last period)")
    ax.set_title("Conversion vs motor power (labels = nominal $w_b$ rpm)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "conversion_vs_power.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_omega_traces(results):
    """Commanded vs CFD-measured wall omega for every episode -- the explicit
    guard against the omega-freeze bug (measured must track the table)."""
    rs = sorted([r for r in results if len(r["t_pw"])],
                key=lambda x: (x["mode"], x["wb"]))
    if not rs:
        return
    n = len(rs)
    ncol = 2
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 2.1 * nrow),
                             sharex=True, squeeze=False)
    for i, r in enumerate(rs):
        ax = axes[i // ncol][i % ncol]
        grid, wcmd = densify(r["pts"], r["duration"])
        ax.plot(grid, wcmd / RPM, "-", color="#999999", lw=1.2, label="commanded")
        ax.plot(r["t_pw"], np.abs(r["om"]) / RPM, ".", color="#d1495b", ms=2.5,
                label="CFD wall (measured)")
        ax.set_ylabel("rpm", fontsize=8)
        ax.set_ylim(-0.08 * r["peak_rpm"], 1.15 * r["peak_rpm"])
        ax.ticklabel_format(useOffset=False, axis="y")
        ax.set_title(f"{r['mode']}  $w_b$={r['wb']} rpm", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("time [s]", fontsize=8)
    fig.suptitle("Commanded vs measured inner-wall omega (freeze-bug guard)", y=1.0)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "omega_traces.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_summary_table_image(results):
    by = {(r["mode"], r["wb"]): r for r in results}
    fig, ax = plt.subplots(figsize=(13, 0.55 * (2 * len(WBS) + 3)))
    ax.axis("off")
    cols = ["mode", "$w_b$\n[rpm]", "peak\n[rpm]", "X (conv,\nlast period)",
            "P motor\n[W]", "P/Pmax", "R = X - P/Pmax", "R = X + P/Pmax",
            "wall\n[min]"]
    rows = []
    for mode in ("constant", "pulsed"):
        for wb in WBS:
            r = by.get((mode, wb))
            if r is None:
                continue
            wall_min = r.get("wall_s", float("nan"))
            wall_min = f"{wall_min/60:.0f}" if wall_min == wall_min else "--"
            if r["ok"]:
                rows.append([mode, f"{wb}", f"{r['peak_rpm']:.0f}", f"{r['x_win']:.4f}",
                             f"{r['p_motor']:.2f}", f"{r['p_norm']:.3f}",
                             f"{r['r_minus']:.4f}", f"{r['r_plus']:.4f}", wall_min])
            else:
                rows.append([mode, f"{wb}", f"{r['peak_rpm']:.0f}",
                             "DIVERGED" if r["diverged"] else "failed",
                             "--", "--", "--", "--", wall_min])
    tbl = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.7)
    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#30323a")
        tbl[0, j].set_text_props(color="w", fontweight="bold")
    ax.set_title("Fig. 7 recreation -- conversion & TD3-style reward "
                 "(X = last-period mean conv, P_max = 31.94 W)",
                 pad=16, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "summary_table.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def print_timing(results):
    timed = [r for r in results if r.get("wall_s", float("nan")) == r.get("wall_s")]
    if not timed:
        return
    print("\nCOMPUTE COST (per 60 s episode, 1 core each):")
    for r in sorted(timed, key=lambda x: x["wall_s"]):
        print(f"  {episode_tag(r['mode'], r['wb']):16s} {r['wall_s']/60:7.1f} min "
              f"({r['wall_s']/r['duration']:.0f} s CPU per sim-second)")
    tot = sum(r["wall_s"] for r in timed)
    print(f"  total CPU time: {tot/3600:.1f} core-hours; slowest episode "
          f"{max(r['wall_s'] for r in timed)/60:.0f} min (= wall clock when fully parallel)")


def analyze(results):
    print("\nGenerating figures + table ...")
    write_summary_csv(results)
    print_headline(results)
    plot_timeseries(results)
    plot_fig7(results)
    plot_conversion_vs_power(results)
    plot_omega_traces(results)
    plot_summary_table_image(results)
    print_timing(results)
    print(f"\nAll outputs in: {RESULTS_DIR}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=len(MODES) * len(WBS),
                    help="parallel episodes, 1 core each (default 10)")
    ap.add_argument("--smoke", action="store_true",
                    help="pipeline test: 6 s episodes, 2.5 s period (~10 min)")
    ap.add_argument("--analyze-only", action="store_true",
                    help="regenerate plots/table from existing results/ without CFD")
    ap.add_argument("--modes", default=None,
                    help="comma-separated subset of modes to RUN (pulsed,constant)")
    ap.add_argument("--wbs", default=None,
                    help="comma-separated subset of w_b values to RUN")
    args = ap.parse_args()

    duration, period = EPISODE, PERIOD
    if args.smoke:
        duration, period = 6.0, 2.5
        print("[SMOKE MODE] 6 s episodes, 2.5 s period -- pipeline test only.")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.analyze_only:
        analyze(load_results_from_logs(duration, period))
        return

    if shutil.which("pimpleFoam") is None or shutil.which("foamDictionary") is None:
        sys.exit("ERROR: OpenFOAM not on PATH (need pimpleFoam, foamDictionary).")

    run_modes = ([m.strip() for m in args.modes.split(",") if m.strip()]
                 if args.modes else list(MODES))
    unknown = [m for m in run_modes if m not in MODES]
    if unknown:
        sys.exit(f"ERROR: unknown mode(s) {unknown}. Known: {list(MODES)}")
    run_wbs = ([int(x) for x in args.wbs.split(",") if x.strip()]
               if args.wbs else list(WBS))
    unknown = [w for w in run_wbs if w not in WBS]
    if unknown:
        sys.exit(f"ERROR: --wbs {unknown} not in WBS={WBS}.")

    episodes = [(m, wb) for m in run_modes for wb in run_wbs]
    print(f"Case      : {CASE_NAME}")
    print(f"Results   : {RESULTS_DIR}")
    print(f"Episodes  : {len(episodes)}  ({run_modes} x {run_wbs} rpm), "
          f"{duration:.0f} s each, T={period:.3g} s, D={{{', '.join(f'{m}:{d:.0%}' for m, d in MODES.items())}}}")
    print(f"Workers   : {args.workers} (1 CPU core / episode)\n")

    print("Compiling coded sources (one throwaway step) ...")
    t0 = time.time()
    template = prepare_template()
    print(f"  template ready ({time.time()-t0:.0f}s)\n")

    t_start = time.time()
    print("Running episodes ...")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_episode, template, m, wb, duration, period): (m, wb)
                for (m, wb) in episodes}
        for fut in as_completed(futs):
            fut.result()
    print(f"\nAll episodes done in {(time.time()-t_start)/60:.1f} min "
          f"(wall clock; ran {args.workers}-wide).")

    # Rebuilt from ALL logs on disk, so subset re-runs still give a full table.
    # Live wall_s survives only for episodes run THIS invocation; re-parsed ones
    # keep their printed wall time in the run log.
    fresh = load_results_from_logs(duration, period)
    wall_by_tag = {}
    for fut, (m, wb) in futs.items():
        try:
            wall_by_tag[episode_tag(m, wb)] = fut.result()["wall_s"]
        except Exception:
            pass
    for r in fresh:
        r["wall_s"] = wall_by_tag.get(episode_tag(r["mode"], r["wb"]), float("nan"))
    analyze(fresh)


if __name__ == "__main__":
    main()
