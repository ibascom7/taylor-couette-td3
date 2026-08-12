#!/usr/bin/env python3
"""FINALS baseline grid, SHORT WEDGE cell (Gamma = 6, 5 deg wedge).

The paper's baseline table, regime-placed per the Lopez-Guajardo Re* map
(Re = 1.5707 * rpm, Re_cr = 96 from Esser-Grossmann at eta = 0.8, so
Re* = rpm / 61.12). Bands (nominal, infinite-cylinder): LCF < 1, TVF 1-6,
WVF 6-15, MWVF 15-33, turbulent cells ~33+.

THE FINALS BASELINE TABLE on side_outlet_case_sc1075_graded (the fig7 sweep
case -- identical physics to the RL grad twin, same established reward
convention). Episodes are 5*tau = 130 s (tau = V/Q ~ 26 s).

  CONSTANT rows (always run):  w_b in {300, 750, 1500} rpm -- one operating
      point per nominal regime (TVF Re*=4.9 / WVF 12.3 / MWVF 24.5).
  PULSED rows (--champion-duty D*, in PERCENT, from the Fig-6-replication
      duty sweep -- run_duty_sweep.py -- which selects the best static pulse
      Lopez-Guajardo-style): the champion square wave (T = t_visc = d^2/nu
      = 3.75 s, w_low = 0, duty D*) REPLAYED AT EACH BAND MEAN, the paper's
      Fig-7 pattern, so each pulsed row is power-matched to its constant row.
      The 2500 rpm envelope binds through a per-band duty floor
      D_eff = max(D*, w_b/2500) -- {12, 30, 60}% at {300, 750, 1500} -- so a
      low champion D* runs CAP-CONSTRAINED at the upper bands (logged, never
      silent).

Conventions (methods section):
  * X = mean outlet conversion over the LAST RESIDENCE TIME [4*tau, 5*tau]
    (waveform-agnostic window = the finals TD3 final-block reward window;
    with T = 3.75 s it spans ~6.9 pulse periods).
  * P = episode-average motor electrical power on the commanded omega(t)
    (paper Eqs 18-23); R- = X - P/31.94 (P_max = motor power at 2500 rpm).
  * Pristine IC (no warmup), burst-first phase, 0.05 s ramped edges -- the
    proven post-freeze-fix tabulated-Function1 drive; omega_traces.png is the
    freeze-bug guard (measured wall omega must track the table).

Reuses experiments/modulation_vs_constant/fig7_sweep.py for all OpenFOAM
plumbing (template compile, foamDictionary, log parsing, motor power).

USAGE
    python3 run_short_wedge_baselines.py                        # constants only
    python3 run_short_wedge_baselines.py --champion-duty 40     # + pulsed rows
    python3 run_short_wedge_baselines.py --smoke                # pipeline test
    python3 run_short_wedge_baselines.py --index 4 --champion-duty 40   # array task
    python3 run_short_wedge_baselines.py --index 2 --resume     # continue after timeout
    python3 run_short_wedge_baselines.py --list --champion-duty 40      # index map
    python3 run_short_wedge_baselines.py --analyze-only --champion-duty 40
(--champion-duty must accompany --analyze-only/--list for the pulsed rows to
be included; without it only the constant rows exist.)
"""

import argparse
import csv
import importlib.util
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- fig7_sweep as a library (OpenFOAM plumbing + motor power) -------------
_F_PATH = os.path.normpath(os.path.join(
    HERE, "..", "modulation_vs_constant", "fig7_sweep.py"))
_spec = importlib.util.spec_from_file_location("fig7_sweep", _F_PATH)
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)

# ---- cell configuration (overridden by the long-wedge wrapper) -------------
CASE_NAME = "side_outlet_case_sc1075_graded"   # (Gamma=6, wedge) cell
TAU = 26.0                                     # residence time V/Q [s]
PERIOD = TAU                                   # pulse period = one tau = TD3 block
EPISODE = 5.0 * TAU                            # 5 residence times = 130 s
WRITE_INTERVAL = 1.0                           # s between saved time folders
RESULTS_DIR = os.path.join(HERE, "results")
CELL_TAG = "Gamma=6 wedge (short)"

RAMP = 0.05
P_MAX = 31.94                                  # W, motor power at 2500 rpm
RPM = 2.0 * math.pi / 60.0

# ---- Re* map (Lopez-Guajardo Table 2 / Esser-Grossmann, eta = 0.8) ---------
RE_PER_RPM = 1.5707                            # Re_rot per rpm (their Table 2)
RE_CR = 96.0
BANDS = [(1.0, "LCF"), (6.0, "TVF"), (15.0, "WVF"), (33.0, "MWVF"),
         (float("inf"), "TTVF")]


def re_star(rpm):
    return rpm * RE_PER_RPM / RE_CR


def regime(rs):
    for hi, name in BANDS:
        if rs < hi:
            return name
    return "TTVF"


# ---- the run grid ----------------------------------------------------------
# mean = D*w_hi + (1-D)*w_lo (commanded); equal within each band by design.
# Pulse period for every run = the module-level PERIOD (= tau).
RUNS = [
    dict(tag="constant_tvf",  kind="constant", band="TVF",  w_lo=300,  w_hi=300,  duty=1.0),
    dict(tag="constant_wvf",  kind="constant", band="WVF",  w_lo=750,  w_hi=750,  duty=1.0),
    dict(tag="constant_mwvf", kind="constant", band="MWVF", w_lo=1500, w_hi=1500, duty=1.0),
    dict(tag="inband_tvf",    kind="inband",   band="TVF",  w_lo=250,  w_hi=350,  duty=0.5),
    dict(tag="inband_wvf",    kind="inband",   band="WVF",  w_lo=625,  w_hi=875,  duty=0.5),
    dict(tag="inband_mwvf",   kind="inband",   band="MWVF", w_lo=1250, w_hi=1750, duty=0.5),
    dict(tag="cross_tvf",     kind="cross",    band="TVF",  w_lo=0,    w_hi=600,  duty=0.5),
    dict(tag="cross_wvf",     kind="cross",    band="WVF",  w_lo=0,    w_hi=1500, duty=0.5),
    dict(tag="cross_mwvf",    kind="cross",    band="MWVF", w_lo=0,    w_hi=1800, duty=5.0 / 6.0),
]


def mean_rpm(run):
    return run["duty"] * run["w_hi"] + (1.0 - run["duty"]) * run["w_lo"]


def waveform_pts(run, duration, period=None):
    T = period if period is not None else PERIOD
    pts, _ = F.square_wave_points(0.0, duration, run["w_hi"] * RPM,
                                  run["w_lo"] * RPM, T, run["duty"], RAMP)
    return pts


# ---- metrics (fig7 conventions, generalized to (w_lo, w_hi, D)) ------------
def metrics_from_log(run, text, duration, period=None):
    T = period if period is not None else PERIOD
    t_conv, conv, t_pw, om, mz, pf = F.parse_log(text)
    pts = waveform_pts(run, duration, T)

    end_t = float(t_conv[-1]) if len(t_conv) else 0.0
    win = min(T, end_t) if end_t > 0 else T           # last FULL period
    x_win = conv_final = float("nan")
    diverged = False
    if len(conv):
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

    grid, w = F.densify(pts, duration)
    p_motor = float(np.mean(F.motor_power.electrical_power(grid, w)))
    p_norm = p_motor / P_MAX

    if len(t_pw):
        sel = t_pw >= t_pw[-1] - win
        p_visc_win = float(np.mean(pf[sel]))
        om_peak_rpm = float(np.max(np.abs(om)) / RPM)
        om_mean_rpm = float(np.mean(np.abs(om)) / RPM)
    else:
        p_visc_win = om_peak_rpm = om_mean_rpm = float("nan")

    m = mean_rpm(run)
    return dict(
        run, mean_rpm=m, period=T,
        re_lo=re_star(run["w_lo"]), re_hi=re_star(run["w_hi"]),
        re_mean=re_star(m),
        regime_lo=regime(re_star(run["w_lo"])), regime_hi=regime(re_star(run["w_hi"])),
        duration=duration, ok=(len(conv) > 0 and not diverged), diverged=diverged,
        x_win=x_win, conv_final=conv_final,
        p_motor=p_motor, p_norm=p_norm,
        r_minus=x_win - p_norm, r_plus=x_win + p_norm,
        p_visc_win=p_visc_win, om_peak_rpm=om_peak_rpm, om_mean_rpm=om_mean_rpm,
        wall_s=float("nan"),
        t_conv=t_conv, conv=conv, t_pw=t_pw, om=om, pts=pts,
    )


# ---- one episode -----------------------------------------------------------
def _time_dirs(workdir):
    out = []
    for n in os.listdir(workdir):
        if os.path.isdir(os.path.join(workdir, n)) and F._is_float(n):
            out.append((float(n), n))
    return sorted(out)


def run_episode(template, run, duration, period=None, resume=False):
    """One pimpleFoam episode. resume=True continues a partial run from its
    latest saved time folder (for episodes longer than one walltime). The
    omega TABLE is re-stamped into the restart-time U file explicitly --
    rotatingWallVelocity re-serialization must never be trusted to preserve
    the Function1 table (the omega-freeze bug family)."""
    tag = run["tag"]
    workdir = os.path.join(RESULTS_DIR, tag)
    log_path = os.path.join(workdir, "log.pimpleFoam")

    pts = waveform_pts(run, duration, period)
    table = "table (" + " ".join(f"({t:.6f} {w:.6f})" for t, w in pts) + ")"

    latest_t, latest = 0.0, "0"
    if resume and os.path.isdir(workdir) and os.path.isfile(log_path):
        tds = _time_dirs(workdir)
        if tds:
            latest_t, latest = tds[-1]

    env = dict(os.environ, OMP_NUM_THREADS="1")
    if resume and latest_t >= duration - 0.5:
        print(f"  [RESUME ] {tag}: already at t={latest_t:.0f}s >= {duration:.0f}s "
              "-- parsing existing logs, no solver run.", flush=True)
        wall = 0.0
        returncode = 0
        with open(log_path, errors="replace") as f:
            text = f.read()
    elif resume and latest_t > 0.0:
        print(f"  [RESUME ] {tag}: continuing from t={latest_t:.0f}s "
              f"(target {duration:.0f}s)", flush=True)
        F.foam_set(workdir, "boundaryField.inner_wall.omega", table, f"{latest}/U")
        F.foam_set(workdir, "startFrom", "latestTime", "system/controlDict")
        F.foam_set(workdir, "endTime", repr(float(duration)), "system/controlDict")
        t0 = time.time()
        r = subprocess.run(["pimpleFoam"], cwd=workdir, capture_output=True,
                           text=True, env=env)
        wall = time.time() - t0
        returncode = r.returncode
        with open(log_path, "a") as f:
            f.write(f"\n// ---- RESUMED from t={latest_t:.6g}s ----\n")
            f.write(r.stdout)
        with open(os.path.join(workdir, "log.err"), "a") as f:
            f.write(r.stderr)
        with open(log_path, errors="replace") as f:
            text = f.read()
    else:
        if os.path.isdir(workdir):
            shutil.rmtree(workdir)
        shutil.copytree(template, workdir)
        F.clean_run_artifacts(workdir)
        F.foam_set(workdir, "boundaryField.inner_wall.omega", table, "0/U")
        F.foam_set(workdir, "endTime", repr(float(duration)), "system/controlDict")
        F.foam_set(workdir, "writeInterval", repr(float(WRITE_INTERVAL)),
                   "system/controlDict")
        t0 = time.time()
        r = subprocess.run(["pimpleFoam"], cwd=workdir, capture_output=True,
                           text=True, env=env)
        wall = time.time() - t0
        returncode = r.returncode
        with open(log_path, "w") as f:
            f.write(r.stdout)
        with open(os.path.join(workdir, "log.err"), "w") as f:
            f.write(r.stderr)
        text = r.stdout

    res = metrics_from_log(run, text, duration, period)
    res["ok"] = res["ok"] and (returncode == 0)
    res["wall_s"] = wall
    save_episode_csvs(res)

    status = "OK" if res["ok"] and len(res["conv"]) else (
        "FAILED" if returncode != 0 else "NO-DATA")
    print(f"  [{status:7s}] {tag:14s} {run['w_lo']:4.0f}->{run['w_hi']:4.0f} rpm "
          f"D={run['duty']:.2f}  X={res['x_win']:.4f}  P={res['p_motor']:.2f} W  "
          f"R-={res['r_minus']:.4f}  om_meas_peak={res['om_peak_rpm']:.0f} rpm  "
          f"({wall / 60:.1f} min, reached t="
          f"{(res['t_conv'][-1] if len(res['t_conv']) else 0):.0f}s)", flush=True)
    if returncode != 0:
        err = os.path.join(workdir, "log.err")
        if os.path.isfile(err):
            with open(err, errors="replace") as f:
                print(f"           stderr tail: {f.read().strip()[-300:]}", flush=True)
    return res


def save_episode_csvs(res):
    tag = res["tag"]
    with open(os.path.join(RESULTS_DIR, f"{tag}_timeseries.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "conversion"])
        for t, c in zip(res["t_conv"], res["conv"]):
            w.writerow([f"{t:.6g}", f"{c:.8g}"])
    with open(os.path.join(RESULTS_DIR, f"{tag}_waveform_points.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "omega_cmd_rad_s"])
        for t, o in res["pts"]:
            w.writerow([f"{t:.6f}", f"{o:.6f}"])


# ---- analysis --------------------------------------------------------------
def load_results_from_logs(duration, period=None):
    out = []
    for run in RUNS:
        log = os.path.join(RESULTS_DIR, run["tag"], "log.pimpleFoam")
        if not os.path.isfile(log):
            continue
        with open(log, errors="replace") as f:
            out.append(metrics_from_log(run, f.read(), duration, period))
    return out


def write_summary(results):
    path = os.path.join(RESULTS_DIR, "summary_table.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "kind", "band", "w_lo_rpm", "w_hi_rpm", "duty",
                    "period_s", "mean_cmd_rpm", "Re*_lo", "Re*_hi", "Re*_mean",
                    "X_conv_lastperiod", "conv_final", "P_motor_W", "P_over_Pmax",
                    "R_minus", "R_plus", "P_visc_cfd_W",
                    "omega_meas_peak_rpm", "omega_meas_mean_rpm",
                    "diverged", "wall_minutes"])
        for r in results:
            wall_min = r.get("wall_s", float("nan"))
            wall_min = wall_min / 60 if wall_min == wall_min else float("nan")
            w.writerow([
                r["tag"], r["kind"], r["band"], r["w_lo"], r["w_hi"],
                f"{r['duty']:.4f}", f"{r['period']:.3g}", f"{r['mean_rpm']:.0f}",
                f"{r['re_lo']:.2f}", f"{r['re_hi']:.2f}", f"{r['re_mean']:.2f}",
                f"{r['x_win']:.6g}", f"{r['conv_final']:.6g}",
                f"{r['p_motor']:.6g}", f"{r['p_norm']:.6g}",
                f"{r['r_minus']:.6g}", f"{r['r_plus']:.6g}",
                f"{r['p_visc_win']:.6g}",
                f"{r['om_peak_rpm']:.1f}", f"{r['om_mean_rpm']:.1f}",
                int(bool(r["diverged"])), f"{wall_min:.2f}"])
    return path


def print_headline(results):
    by = {r["tag"]: r for r in results}
    print("\n" + "=" * 74)
    print(f"FINALS BASELINE GRID -- {CELL_TAG}, case {CASE_NAME}")
    print(f"X = last-full-period mean conversion; R- = X - P/{P_MAX} W")
    print("=" * 74)
    print(f"  {'band':>5s} | {'constant':>18s} | {'inband pulse':>18s} | {'cross pulse':>18s}")
    print("  " + "-" * 70)
    for band in ("TVF", "WVF", "MWVF"):
        cells = []
        for kind in ("constant", "inband", "cross"):
            r = by.get(f"{kind}_{band.lower()}")
            cells.append(f"X={r['x_win']:.3f} R-={r['r_minus']:.3f}"
                         if (r and r["ok"]) else "--")
        print(f"  {band:>5s} | {cells[0]:>18s} | {cells[1]:>18s} | {cells[2]:>18s}")
    print("=" * 74 + "\n")


KIND_STYLE = {"constant": ("-", "o"), "inband": ("--", "s"), "cross": (":", "^")}
BAND_COLOR = {"TVF": "#2e6f95", "WVF": "#d1495b", "MWVF": "#66a182"}


def plot_timeseries(results):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for r in results:
        if not len(r["t_conv"]):
            continue
        ls, _ = KIND_STYLE[r["kind"]]
        ax.plot(r["t_conv"], r["conv"], ls=ls, color=BAND_COLOR[r["band"]],
                lw=1.5, label=f"{r['tag']} ({r['w_lo']:.0f}-{r['w_hi']:.0f} rpm)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("outlet conversion")
    ax.set_title(f"Conversion vs time -- {CELL_TAG}\n"
                 "color = band, style = waveform kind")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "conversion_vs_time.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_conversion_vs_power(results):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for r in results:
        if not r["ok"]:
            continue
        _, marker = KIND_STYLE[r["kind"]]
        ax.plot(r["p_motor"], r["x_win"], marker, color=BAND_COLOR[r["band"]],
                ms=10, mec="k", mew=0.5)
        ax.annotate(r["tag"], (r["p_motor"], r["x_win"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.set_xlabel("episode-average motor electrical power [W]")
    ax.set_ylabel("conversion (last-period mean)")
    ax.set_title(f"Conversion vs power -- {CELL_TAG}\n"
                 "circle constant / square inband / triangle cross; color = band")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "conversion_vs_power.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_omega_traces(results):
    rs = [r for r in results if len(r["t_pw"])]
    if not rs:
        return
    ncol, n = 3, len(rs)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.2 * nrow),
                             sharex=True, squeeze=False)
    for i, r in enumerate(rs):
        ax = axes[i // ncol][i % ncol]
        grid, wcmd = F.densify(r["pts"], r["duration"])
        ax.plot(grid, wcmd / RPM, "-", color="#999999", lw=1.1, label="commanded")
        ax.plot(r["t_pw"], np.abs(r["om"]) / RPM, ".", ms=2.0,
                color=BAND_COLOR[r["band"]], label="CFD wall")
        for edge_rpm in (61.1, 366.7, 916.8, 2016.6):
            if edge_rpm < 1.2 * r["w_hi"] + 50:
                ax.axhline(edge_rpm, color="k", lw=0.5, alpha=0.35)
        ax.set_title(r["tag"], fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("time [s]", fontsize=8)
    fig.suptitle("Commanded vs measured wall omega (freeze-bug guard); "
                 "thin lines = nominal regime boundaries", y=1.0)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "omega_traces.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


def analyze(results):
    if not results:
        print("No episode logs found -- nothing to analyze.")
        return
    print("\nGenerating summary + figures ...")
    write_summary(results)
    print_headline(results)
    plot_timeseries(results)
    plot_conversion_vs_power(results)
    plot_omega_traces(results)
    print(f"All outputs in: {RESULTS_DIR}")


# ---- main ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=len(RUNS))
    ap.add_argument("--smoke", action="store_true",
                    help="6 s episodes, T=2.5 s -- pipeline test")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--only", default=None,
                    help="comma-separated run tags to execute")
    ap.add_argument("--index", type=int, default=None,
                    help="run RUNS[index] only (slurm array mode)")
    ap.add_argument("--resume", action="store_true",
                    help="continue partial episodes from their latest saved "
                         "time folder instead of restarting from scratch")
    ap.add_argument("--list", action="store_true",
                    help="print index <-> tag map and exit")
    args = ap.parse_args()

    if args.list:
        for i, run in enumerate(RUNS):
            print(f"{i}: {run['tag']:14s} {run['w_lo']:4.0f}->{run['w_hi']:4.0f} rpm "
                  f"D={run['duty']:.3f} mean={mean_rpm(run):.0f} "
                  f"(Re* {re_star(run['w_lo']):.1f}-{re_star(run['w_hi']):.1f})")
        return

    duration, period = EPISODE, None
    if args.smoke:
        duration, period = 6.0, 2.5
        print("[SMOKE MODE] 6 s episodes, T=2.5 s -- pipeline test only.")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.analyze_only:
        analyze(load_results_from_logs(duration, period))
        return

    if shutil.which("pimpleFoam") is None or shutil.which("foamDictionary") is None:
        sys.exit("ERROR: OpenFOAM not on PATH (need pimpleFoam, foamDictionary).")

    todo = list(RUNS)
    if args.index is not None:
        todo = [RUNS[args.index]]
    elif args.only:
        names = {t.strip() for t in args.only.split(",") if t.strip()}
        unknown = names - {r["tag"] for r in RUNS}
        if unknown:
            sys.exit(f"ERROR: unknown tags {sorted(unknown)}")
        todo = [r for r in RUNS if r["tag"] in names]

    # Template: unique parent per single-episode invocation so concurrent slurm
    # array tasks never race on the same _template dir.
    F.CASE_NAME = CASE_NAME
    tpl_parent = RESULTS_DIR if len(todo) > 1 else os.path.join(
        RESULTS_DIR, f"_tpl_{todo[0]['tag']}")
    os.makedirs(tpl_parent, exist_ok=True)
    F.RESULTS_DIR = tpl_parent

    print(f"Cell      : {CELL_TAG}")
    print(f"Case      : {CASE_NAME}")
    print(f"Results   : {RESULTS_DIR}")
    print(f"Episodes  : {len(todo)} x {duration:.0f} s "
          f"({', '.join(r['tag'] for r in todo)})")
    print(f"Workers   : {args.workers}\n")

    print("Compiling coded sources (one throwaway step) ...")
    t0 = time.time()
    template = F.prepare_template()
    print(f"  template ready ({time.time() - t0:.0f}s)\n")

    t_start = time.time()
    print("Running episodes ...")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_episode, template, run, duration, period,
                          args.resume): run
                for run in todo}
        for fut in as_completed(futs):
            fut.result()
    print(f"\nEpisodes done in {(time.time() - t_start) / 60:.1f} min wall.")

    if tpl_parent != RESULTS_DIR:
        shutil.rmtree(tpl_parent, ignore_errors=True)

    if len(todo) == len(RUNS):
        fresh = load_results_from_logs(duration, period)
        wall_by_tag = {}
        for fut, run in futs.items():
            try:
                wall_by_tag[run["tag"]] = fut.result()["wall_s"]
            except Exception:
                pass
        for r in fresh:
            r["wall_s"] = wall_by_tag.get(r["tag"], float("nan"))
        analyze(fresh)
    else:
        print("Subset run complete. Regenerate the combined table with "
              "--analyze-only once all episodes exist.")


if __name__ == "__main__":
    main()
