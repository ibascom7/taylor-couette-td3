#!/usr/bin/env python3
"""B + E -- static waveforms on the FULL-HEIGHT (Gamma = 30) reactor.

One engine, two run groups, chosen with --group:

  B  CHAMPION-FAMILY STATICS (fair comparator for the zero-shot transfer).
     The transfer eval (modulation_rl/results/full_tc_eval) compares the
     learned policy against constant-300 (X 0.787) and the D=80 %/T=10 s
     pulse (X 0.876) -- but the short-reactor champion family was never run
     on this reactor. Three fixed-mean-300 waveforms, 300 s from the pristine
     IC exactly like results_full_tc and the eval:
        champ_D90_T5     D=0.90 T=5 s   w_hi 333 rpm  (the n=1 short champion)
        champ_D80_T2p5   D=0.80 T=2.5 s w_hi 375 rpm  (the 50 s "rapid shimmy")
        shimmy_D63_T5    D=0.63 T=5 s   w_hi 476 rpm  (the transfer policy's
                                                        own plateau waveform)

  E  LOPEZ-GUAJARDO'S OPTIMUM ON OUR REACTOR, AT TWO SCHMIDT NUMBERS.
     Their Fig. 6 optimum (D=20 %, T=30 s at a 500 rpm mean -> 2500 rpm
     bursts), their local minimum (D=80 %, T=5 s) and constant 500, each run
     at our Sc=1075 AND at Sc=1e4 (the authors' emailed value; D = nu/Sc set
     in constant/transportProperties and the scalarTransport FO). If the
     ordering flips toward low duty as Sc rises, the disagreement with their
     paper is a Schmidt-number effect and becomes a finding; if it does not,
     it is the mesh/stabilization corner (2500 rpm bursts) or their shorter
     reactor... which we now share, so geometry is ruled out.
     NB at Sc=1e4 the film is ~0.5x thinner: 1.5 cells across it at 300 rpm,
     < 1 at 2500 -- that under-resolution is THEIR regime and is part of the
     comparison; state it.

Conventions (match results_full_tc + eval_full_tc.py, so rows are comparable):
    episode 300 s, pristine pre-filled IC, burst-first phase, 0.05 s ramps,
    tabulated Function1 omega in 0/U; X_last10 = mean conversion over the
    last 10 s (= the eval's final block = the fig7 last-period window),
    X_lastT = last full waveform period, X_tau = last residence time
    [170, 300] s; P = episode-average motor electrical power on the commanded
    omega(t) (paper Eqs 18-23); R- = X_last10 - P/31.94.

COST: ~230-260 CPU-s per simulated second at <= 500 rpm (16,650 cells) ->
~22 h per 300 s episode; 2500 rpm bursts Courant-shrink dt, budget ~1.7x.
One episode per slurm array task; --resume continues a walltime-killed
episode from its latest saved time folder (the omega table is re-stamped
into the restart U file -- never trust the re-serialized BC).

USAGE
    python3 gamma30_statics.py --group B --list
    python3 gamma30_statics.py --group E --index 3 --resume     # one array task
    python3 gamma30_statics.py --group B --smoke                 # 6 s pipeline test (~25 min)
    python3 gamma30_statics.py --group E --analyze-only
    # local, all of a group in parallel (1 core each):
    nohup python3 -u gamma30_statics.py --group B --workers 3 > logs/B_local.log 2>&1 &
"""

import argparse
import csv
import importlib.util
import json
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
GYM_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))

# fig7_sweep as a library: template compile, foamDictionary, log parsing,
# waveform builder, motor power (exactly the results_full_tc plumbing).
_F_PATH = os.path.join(GYM_ROOT, "experiments", "modulation_vs_constant", "fig7_sweep.py")
_spec = importlib.util.spec_from_file_location("fig7_sweep", _F_PATH)
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)

CASE_NAME = "full_tc_cat_case"
EPISODE = 300.0
TAU = 130.0
RAMP = 0.05
P_MAX = 31.94
RPM = 2.0 * math.pi / 60.0
NU = 1.0752688172043011e-05          # transportProperties nu (silicone oil)
SC_CASE = 1075.0                     # the case's own D = 1e-8 -> Sc = 1075
WRITE_INTERVAL = 10.0                # 30 time folders per episode (~1 GB)
RESULTS_ROOT = os.path.join(HERE, "results")

REF_FULL_TC = dict(constant300_X=0.786883, pulsed_D80T10_300_X=0.87645,
                   constant500_X=0.828111, pulsed_D80T10_500_X=0.876873,
                   transfer_stretch_X_last=0.901486, transfer_wrap_X_last=0.881342)
REF_LG_PAPER = dict(constant500_X=0.427, pulsed_D20T30_500_X=0.572,
                    note="Lopez-Guajardo 2024 (CEJ 489:151174), Sc~1e5 per text, ~1e4 per authors")


def _run(tag, w_hi, duty, period, sc=SC_CASE, mean=None, note=""):
    return dict(tag=tag, w_lo=0.0, w_hi=float(w_hi), duty=float(duty), period=float(period),
                sc=float(sc), mean=float(mean if mean is not None else w_hi * duty), note=note)


GROUPS = {
    "B": dict(
        desc="champion-family statics, fixed mean 300 rpm (transfer comparators)",
        runs=[
            _run("champ_D90_T5",   300.0 / 0.90, 0.90, 5.0,  note="short-reactor n=1 champion"),
            _run("champ_D80_T2p5", 300.0 / 0.80, 0.80, 2.5,  note="50 s rapid-shimmy champion"),
            _run("shimmy_D63_T5",  300.0 / 0.63, 0.63, 5.0,  note="transfer policy plateau waveform"),
        ]),
    "E": dict(
        desc="Lopez-Guajardo optimum / local minimum / constant at 500 rpm mean, Sc 1075 and 1e4",
        runs=[
            _run("lg_opt_D20_T30_sc1075", 2500.0, 0.20, 30.0, sc=1075.0, note="their Fig. 6 optimum"),
            _run("lg_min_D80_T5_sc1075",   625.0, 0.80,  5.0, sc=1075.0, note="their Fig. 6 local minimum"),
            _run("constant_500_sc1075",    500.0, 1.00, 10.0, sc=1075.0, note="constant 500"),
            _run("lg_opt_D20_T30_sc1e4",  2500.0, 0.20, 30.0, sc=1.0e4, note="their Fig. 6 optimum, their Sc"),
            _run("lg_min_D80_T5_sc1e4",    625.0, 0.80,  5.0, sc=1.0e4, note="their local minimum, their Sc"),
            _run("constant_500_sc1e4",     500.0, 1.00, 10.0, sc=1.0e4, note="constant 500, their Sc"),
        ]),
}


# --------------------------------------------------------------------------- #
def waveform_pts(run, duration, period=None):
    T = period if period is not None else run["period"]
    if run["duty"] >= 1.0:
        return [(0.0, run["w_hi"] * RPM), (duration + 1.0, run["w_hi"] * RPM)]
    pts, _ = F.square_wave_points(0.0, duration, run["w_hi"] * RPM, run["w_lo"] * RPM,
                                  T, run["duty"], RAMP)
    return pts


def _window_mean(t, c, t0, t1):
    sel = (t >= t0) & (t <= t1)
    return float(np.mean(c[sel])) if sel.sum() >= 3 else float("nan")


def metrics_from_log(run, text, duration, period=None):
    T = period if period is not None else run["period"]
    t_conv, conv, t_pw, om, mz, pf = F.parse_log(text)
    pts = waveform_pts(run, duration, T)
    end_t = float(t_conv[-1]) if len(t_conv) else 0.0
    diverged = False
    x10 = xT = xtau = conv_final = float("nan")
    if len(conv):
        phys = (conv >= -0.02) & (conv <= 1.02)
        tcp, ccp = t_conv[phys], conv[phys]
        if len(ccp) < 3 or (tcp[-1] < end_t - 1.0):
            diverged = True
        else:
            x10 = _window_mean(tcp, ccp, end_t - min(10.0, end_t), end_t)
            xT = _window_mean(tcp, ccp, end_t - min(T, end_t), end_t)
            xtau = _window_mean(tcp, ccp, end_t - min(TAU, end_t), end_t)
            conv_final = float(ccp[-1])
    grid, w = F.densify(pts, duration)
    p_motor = float(np.mean(F.motor_power.electrical_power(grid, w)))
    p_norm = p_motor / P_MAX
    if len(t_pw):
        om_peak_rpm = float(np.max(np.abs(om)) / RPM)
        om_mean_rpm = float(np.mean(np.abs(om)) / RPM)
    else:
        om_peak_rpm = om_mean_rpm = float("nan")
    return dict(run, period_used=T, duration=duration, reached_t=end_t,
                ok=(len(conv) > 0 and not diverged), diverged=diverged,
                x_last10=x10, x_lastT=xT, x_tau=xtau, conv_final=conv_final,
                p_motor=p_motor, p_norm=p_norm, r_minus=x10 - p_norm,
                om_peak_rpm=om_peak_rpm, om_mean_rpm=om_mean_rpm, wall_s=float("nan"),
                t_conv=t_conv, conv=conv, t_pw=t_pw, om=om, pts=pts)


def _time_dirs(workdir):
    out = []
    for n in os.listdir(workdir):
        if os.path.isdir(os.path.join(workdir, n)) and F._is_float(n):
            out.append((float(n), n))
    return sorted(out)


def set_schmidt(workdir, sc):
    """D = nu/Sc in BOTH places the case reads it."""
    if abs(sc - SC_CASE) < 1e-9:
        return
    dval = NU / sc
    F.foam_set(workdir, "D", f"D [0 2 -1 0 0 0 0] {dval:.6e}", "constant/transportProperties")
    F.foam_set(workdir, "functions.scalarTransport.D", f"{dval:.6e}", "system/controlDict")


def run_episode(template, run, results_dir, duration, period=None, resume=False):
    tag = run["tag"]
    workdir = os.path.join(results_dir, tag)
    log_path = os.path.join(workdir, "log.pimpleFoam")
    pts = waveform_pts(run, duration, period)
    table = "table (" + " ".join(f"({t:.6f} {w:.6f})" for t, w in pts) + ")"
    env = dict(os.environ, OMP_NUM_THREADS="1")

    latest_t, latest = 0.0, "0"
    if resume and os.path.isdir(workdir) and os.path.isfile(log_path):
        tds = _time_dirs(workdir)
        if tds:
            latest_t, latest = tds[-1]

    if resume and latest_t >= duration - 0.5:
        print(f"  [RESUME ] {tag}: already at t={latest_t:.0f}s -- parsing logs only", flush=True)
        wall, rc = 0.0, 0
        with open(log_path, errors="replace") as f:
            text = f.read()
    elif resume and latest_t > 0.0:
        print(f"  [RESUME ] {tag}: continuing from t={latest_t:.0f}s to {duration:.0f}s", flush=True)
        F.foam_set(workdir, "boundaryField.inner_wall.omega", table, f"{latest}/U")
        subprocess.run(["foamDictionary", "-entry", "boundaryField.inner_wall.omegaCoeffs",
                        "-remove", f"{latest}/U"], cwd=workdir, check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        F.foam_set(workdir, "startFrom", "latestTime", "system/controlDict")
        F.foam_set(workdir, "endTime", repr(float(duration)), "system/controlDict")
        t0 = time.time()
        r = subprocess.run(["pimpleFoam"], cwd=workdir, capture_output=True, text=True, env=env)
        wall, rc = time.time() - t0, r.returncode
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
        set_schmidt(workdir, run["sc"])
        F.foam_set(workdir, "boundaryField.inner_wall.omega", table, "0/U")
        F.foam_set(workdir, "endTime", repr(float(duration)), "system/controlDict")
        F.foam_set(workdir, "writeInterval", repr(float(WRITE_INTERVAL)), "system/controlDict")
        with open(os.path.join(workdir, "run.json"), "w") as f:
            json.dump(dict(run, duration=duration, period_used=period or run["period"]), f, indent=2)
        t0 = time.time()
        r = subprocess.run(["pimpleFoam"], cwd=workdir, capture_output=True, text=True, env=env)
        wall, rc = time.time() - t0, r.returncode
        with open(log_path, "w") as f:
            f.write(r.stdout)
        with open(os.path.join(workdir, "log.err"), "w") as f:
            f.write(r.stderr)
        text = r.stdout

    res = metrics_from_log(run, text, duration, period)
    res["ok"] = res["ok"] and (rc == 0)
    res["wall_s"] = wall
    save_episode_csvs(res, results_dir)
    status = "OK" if res["ok"] else ("FAILED" if rc != 0 else "DIVERGED/NO-DATA")
    print(f"  [{status:8s}] {tag:24s} Sc={run['sc']:.0f} 0->{run['w_hi']:.0f} rpm D={run['duty']:.2f} "
          f"T={res['period_used']:.3g}s | X10={res['x_last10']:.4f} Xtau={res['x_tau']:.4f} "
          f"P={res['p_motor']:.2f} W R-={res['r_minus']:.4f} | om_meas_peak={res['om_peak_rpm']:.0f} "
          f"({wall/3600:.2f} h, reached t={res['reached_t']:.0f}s)", flush=True)
    if rc != 0:
        err = os.path.join(workdir, "log.err")
        if os.path.isfile(err):
            with open(err, errors="replace") as f:
                print(f"           stderr tail: {f.read().strip()[-300:]}", flush=True)
    return res


def save_episode_csvs(res, results_dir):
    tag = res["tag"]
    with open(os.path.join(results_dir, f"{tag}_timeseries.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "conversion"])
        for t, c in zip(res["t_conv"], res["conv"]):
            w.writerow([f"{t:.6g}", f"{c:.8g}"])
    with open(os.path.join(results_dir, f"{tag}_waveform_points.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "omega_cmd_rad_s"])
        for t, o in res["pts"]:
            w.writerow([f"{t:.6f}", f"{o:.6f}"])


# --------------------------------------------------------------------------- #
def load_results(group, results_dir, duration, period=None):
    out = []
    for run in GROUPS[group]["runs"]:
        log = os.path.join(results_dir, run["tag"], "log.pimpleFoam")
        if not os.path.isfile(log):
            continue
        with open(log, errors="replace") as f:
            out.append(metrics_from_log(run, f.read(), duration, period))
    return out


def write_summary(group, results, results_dir):
    path = os.path.join(results_dir, "summary_table.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "note", "Sc", "w_hi_rpm", "duty", "period_s", "mean_cmd_rpm",
                    "X_last10", "X_lastT", "X_tau", "conv_final", "P_motor_W", "P_over_Pmax",
                    "R_minus", "omega_meas_peak_rpm", "omega_meas_mean_rpm", "reached_t_s",
                    "diverged", "wall_hours"])
        for r in results:
            wall_h = r["wall_s"] / 3600 if r["wall_s"] == r["wall_s"] else float("nan")
            w.writerow([r["tag"], r["note"], f"{r['sc']:.0f}", f"{r['w_hi']:.1f}", f"{r['duty']:.3f}",
                        f"{r['period_used']:.3g}", f"{r['mean']:.1f}",
                        f"{r['x_last10']:.6g}", f"{r['x_lastT']:.6g}", f"{r['x_tau']:.6g}",
                        f"{r['conv_final']:.6g}", f"{r['p_motor']:.6g}", f"{r['p_norm']:.6g}",
                        f"{r['r_minus']:.6g}", f"{r['om_peak_rpm']:.1f}", f"{r['om_mean_rpm']:.1f}",
                        f"{r['reached_t']:.1f}", int(bool(r["diverged"])), f"{wall_h:.2f}"])
    refs = dict(full_tc=REF_FULL_TC, lopez_guajardo=REF_LG_PAPER)
    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(dict(group=group, desc=GROUPS[group]["desc"], refs=refs,
                       rows=[{k: v for k, v in r.items()
                              if k not in ("t_conv", "conv", "t_pw", "om", "pts")}
                             for r in results]), f, indent=2, default=float)
    return path


def print_headline(group, results):
    print("\n" + "=" * 80)
    print(f"{group}: {GROUPS[group]['desc']}  --  {CASE_NAME}, {EPISODE:.0f} s, pristine IC")
    print("X10 = last-10 s mean conversion (eval convention); Xtau = last residence time")
    print("=" * 80)
    for r in results:
        st = "ok" if r["ok"] else ("DIVERGED" if r["diverged"] else "failed")
        print(f"  {r['tag']:24s} Sc={r['sc']:>6.0f}  D={r['duty']:.2f} T={r['period_used']:>4.3g}s "
              f"peak={r['w_hi']:>5.0f}  X10={r['x_last10']:.4f}  Xtau={r['x_tau']:.4f}  "
              f"P={r['p_motor']:.2f}W  R-={r['r_minus']:.4f}  [{st}]")
    if group == "B":
        print(f"  refs (results_full_tc / transfer eval): constant-300 X={REF_FULL_TC['constant300_X']:.4f}, "
              f"pulsed D80/T10 X={REF_FULL_TC['pulsed_D80T10_300_X']:.4f}, "
              f"policy stretch X_last={REF_FULL_TC['transfer_stretch_X_last']:.4f}, "
              f"wrap {REF_FULL_TC['transfer_wrap_X_last']:.4f}")
    else:
        print(f"  refs: ours Sc=1075 constant-500 X={REF_FULL_TC['constant500_X']:.4f}; "
              f"Lopez-Guajardo paper: constant-500 {REF_LG_PAPER['constant500_X']:.3f}, "
              f"D20/T30 {REF_LG_PAPER['pulsed_D20T30_500_X']:.3f}")
    print("=" * 80 + "\n")


def plot_group(group, results, results_dir):
    if not results:
        return
    fig, ax = plt.subplots(figsize=(10, 5.5))
    cmap = plt.get_cmap("tab10")
    for i, r in enumerate(results):
        if not len(r["t_conv"]):
            continue
        keep = (r["conv"] >= -0.02) & (r["conv"] <= 1.02)
        ax.plot(r["t_conv"][keep], r["conv"][keep], lw=1.3, color=cmap(i % 10),
                ls="--" if r["sc"] > 2000 else "-",
                label=f"{r['tag']} (X10={r['x_last10']:.3f})")
    if group == "B":
        ax.axhline(REF_FULL_TC["pulsed_D80T10_300_X"], color="k", ls=":", lw=1, label="pulsed D80/T10 @300 (ref)")
        ax.axhline(REF_FULL_TC["constant300_X"], color="gray", ls=":", lw=1, label="constant 300 (ref)")
        ax.axhline(REF_FULL_TC["transfer_stretch_X_last"], color="#d1495b", ls=":", lw=1, label="policy, stretch clock")
    else:
        ax.axhline(REF_FULL_TC["constant500_X"], color="gray", ls=":", lw=1, label="constant 500, Sc=1075 (ref)")
        ax.axhline(REF_LG_PAPER["pulsed_D20T30_500_X"], color="k", ls="-.", lw=1, label="L-G paper: D20/T30 57.2 %")
        ax.axhline(REF_LG_PAPER["constant500_X"], color="k", ls=":", lw=1, label="L-G paper: constant 500 42.7 %")
    for k in range(1, int(EPISODE // TAU) + 1):
        ax.axvline(k * TAU, color="k", lw=0.4, alpha=0.25)
    ax.set_xlabel("time [s]   (thin lines: residence times)")
    ax.set_ylabel("outlet conversion")
    ax.set_title(f"{group}: {GROUPS[group]['desc']}\n{CASE_NAME}, pristine IC "
                 f"(dashed = Sc 1e4)" if group == "E" else
                 f"{group}: {GROUPS[group]['desc']}\n{CASE_NAME}, pristine IC")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "conversion_vs_time.png"), dpi=140)
    plt.close(fig)

    rs = [r for r in results if len(r["t_pw"])]
    if rs:
        n = len(rs)
        fig, axes = plt.subplots(n, 1, figsize=(11, 2.0 * n), sharex=True, squeeze=False)
        for i, r in enumerate(rs):
            ax = axes[i][0]
            grid, wcmd = F.densify(r["pts"], r["duration"])
            ax.plot(grid, wcmd / RPM, "-", color="#999", lw=1.0, label="commanded")
            ax.plot(r["t_pw"], np.abs(r["om"]) / RPM, ".", ms=1.8, color="#d1495b", label="CFD wall")
            ax.set_ylabel("rpm", fontsize=8)
            ax.set_title(r["tag"], fontsize=9)
            ax.set_xlim(0, min(60.0, r["duration"]))
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(fontsize=7, loc="upper right")
        axes[-1][0].set_xlabel("time [s] (first 60 s shown)")
        fig.suptitle("Commanded vs measured wall omega (freeze-bug guard)", y=1.0)
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, "omega_traces.png"), dpi=140)
        plt.close(fig)

    okr = [r for r in results if r["ok"]]
    if okr:
        fig, ax = plt.subplots(figsize=(8, 5))
        for r in okr:
            ax.plot(r["p_motor"], r["x_last10"], "s" if r["sc"] > 2000 else "o", ms=10, mec="k",
                    label=f"{r['tag']}")
            ax.annotate(r["tag"], (r["p_motor"], r["x_last10"]), fontsize=7,
                        xytext=(5, 4), textcoords="offset points")
        ax.set_xlabel("episode-average motor power [W]")
        ax.set_ylabel("X (last 10 s)")
        ax.set_title(f"{group}: conversion vs power (square = Sc 1e4)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, "conversion_vs_power.png"), dpi=140)
        plt.close(fig)


def analyze(group, results_dir, duration, period=None):
    results = load_results(group, results_dir, duration, period)
    if not results:
        print("No episode logs found -- nothing to analyze.")
        return
    write_summary(group, results, results_dir)
    print_headline(group, results)
    plot_group(group, results, results_dir)
    print(f"All outputs in: {results_dir}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", choices=sorted(GROUPS), required=True)
    ap.add_argument("--index", type=int, default=None, help="run RUNS[index] only (slurm array)")
    ap.add_argument("--only", default=None, help="comma-separated tags")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="6 s episodes, T=2.5 s (D20 rows use T=2.5)")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--results_dir", default=None)
    args = ap.parse_args()

    runs = GROUPS[args.group]["runs"]
    results_dir = args.results_dir or os.path.join(RESULTS_ROOT, args.group + ("_smoke" if args.smoke else ""))
    if args.list:
        for i, r in enumerate(runs):
            print(f"{i}: {r['tag']:24s} Sc={r['sc']:.0f} 0->{r['w_hi']:.0f} rpm D={r['duty']:.2f} "
                  f"T={r['period']:g}s mean={r['mean']:.0f}  ({r['note']})")
        return

    duration, period = EPISODE, None
    if args.smoke:
        duration, period = 6.0, 2.5
        print("[SMOKE] 6 s episodes, T = 2.5 s -- pipeline test only", flush=True)
    os.makedirs(results_dir, exist_ok=True)
    if args.analyze_only:
        analyze(args.group, results_dir, duration, period)
        return
    if shutil.which("pimpleFoam") is None or shutil.which("foamDictionary") is None:
        sys.exit("ERROR: OpenFOAM not on PATH (need pimpleFoam, foamDictionary).")

    todo = list(runs)
    if args.index is not None:
        todo = [runs[args.index]]
    elif args.only:
        names = {t.strip() for t in args.only.split(",") if t.strip()}
        todo = [r for r in runs if r["tag"] in names]
    workers = args.workers or len(todo)

    # unique template parent per single-episode invocation: no array-task races
    F.CASE_NAME = CASE_NAME
    tpl_parent = results_dir if len(todo) > 1 else os.path.join(results_dir, f"_tpl_{todo[0]['tag']}")
    os.makedirs(tpl_parent, exist_ok=True)
    F.RESULTS_DIR = tpl_parent

    print(f"Group    : {args.group} -- {GROUPS[args.group]['desc']}")
    print(f"Case     : {CASE_NAME}   results -> {results_dir}")
    print(f"Episodes : {len(todo)} x {duration:.0f} s ({', '.join(r['tag'] for r in todo)}); "
          f"workers {workers}\n", flush=True)
    t0 = time.time()
    template = F.prepare_template()
    print(f"  template ready ({time.time()-t0:.0f}s)\n", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_episode, template, run, results_dir, duration, period, args.resume): run
                for run in todo}
        for fut in as_completed(futs):
            fut.result()
    if tpl_parent != results_dir:
        shutil.rmtree(tpl_parent, ignore_errors=True)
    analyze(args.group, results_dir, duration, period)


if __name__ == "__main__":
    main()
