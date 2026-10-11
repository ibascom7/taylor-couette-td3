#!/usr/bin/env python3
"""D -- quick grid- and time-step-independence check on the short (Gamma = 6) reactor.

Built 2026-09-30 for the Rice RTG-NASC poster (2026-10-02) and the GCURS version
two weeks later. Question: do the headline comparisons survive mesh refinement,
and how large is the discretisation error on the production mesh?

MESH FAMILY (the 2026-09-11 plan, self-similar in the grading):
    keep the radial grading (simpleGrading 0.0459982 -> outer-wall cell ~25 um/f),
    Nr = round(37 f), Nz_main = round(89 f); the bottom side-outlet band keeps its
    fixed 0.4233 mm height with 1 cell (f < 1.4) or 2 cells (f >= 1.4).
        f0.71 : 26 x  64 ->  1,664 cells, outer-wall cell ~35 um
        f1.00 : 37 x  90 ->  3,330 cells (= production mesh), 25 um
        f1.41 : 52 x 127 ->  6,604 cells, ~18 um
    Consecutive grids differ by ~sqrt(2) per direction (r ~ 1.41) -> a Celik et al.
    (2008) GCI triplet.

WAVEFORMS (pristine pre-filled IC like results_full_tc / endgame B+E, 150 s ~ 5.8
residence times, burst-first, 0.05 s ramps, tabulated Function1 omega):
    const300       constant 300 rpm
    champ_D90_T5   0 <-> 333 rpm, D 0.90, T 5 s    (mean 300; the search champion)
    const500       constant 500 rpm
    lg_D20_T30     0 <-> 2500 rpm, D 0.20, T 30 s  (mean 500; Lopez-Guajardo optimum,
                                                    where the wall film is thinnest)
TIME-STEP CHECK on the production mesh: const300 / champ / lg at maxCo 0.9 (half of
the production 1.8).

METRICS
    X_win  = mean outlet conversion over the last 30 s [120,150] (one full LG period,
             six champion periods, > 1 residence time)
    X_prev = mean over [90,120] (stationarity check; drift = X_win - X_prev)
    P      = episode-average motor electrical power on the COMMANDED omega(t)
             (paper Eqs 18-23) -- mesh-independent by construction
    r      = X_win - P/31.94
    P_cfd  = mean inner-wall torque power from the CFD over the window (mesh-dependent,
             secondary check)

USAGE (from taylorCouetteGym/, OpenFOAM sourced)
    python3 experiments/endgame/d_grid_check.py --list
    python3 experiments/endgame/d_grid_check.py --smoke          # 2 s pipeline test, ~10 min
    setsid nohup nice -n 10 python3 -u experiments/endgame/d_grid_check.py --resume \
        > experiments/endgame/logs/D_local.log 2>&1 &
    python3 experiments/endgame/d_grid_check.py --analyze-only   # any time, partial OK

COST (7800X3D, 1 core each): f0.71 < 1 h; f1.00 ~2 h (const) / ~4 h (LG bursts);
f1.41 ~5-6 h / ~10-12 h; maxCo-0.9 rows ~2x the f1.00 rows. 15 episodes in parallel.
"""

import argparse
import csv
import importlib.util
import json
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

HERE = os.path.dirname(os.path.abspath(__file__))
GYM_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))

_F_PATH = os.path.join(GYM_ROOT, "experiments", "modulation_vs_constant", "fig7_sweep.py")
_spec = importlib.util.spec_from_file_location("fig7_sweep", _F_PATH)
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)

BASE_CASE = os.path.join(GYM_ROOT, "taylor_couette_mixing", "cases", "side_outlet_case_sc1075_graded")
RESULTS_ROOT = os.path.join(HERE, "results")

EPISODE = 150.0
WINDOW = 30.0
RAMP = 0.05
P_MAX = 31.94
RPM = 2.0 * math.pi / 60.0
WRITE_INTERVAL = 10.0
PROD_MAXCO = 1.8
GAP_M = 0.00635
GRADING = 0.0459982
NR0, NZ0 = 37, 89

MESHES = [("f0.71", 0.71), ("f1.00", 1.00), ("f1.41", 1.41)]      # the GCI triplet
PROD_MESH = "f1.00"
# Outlet-band control: f1.41 with the band left at 1 cell (the family uses 2 there to keep the
# band refined; the 2 s smoke showed the band count moves early-time X by ~0.03-0.05).
BAND_OVERRIDE = {"f1.41b1": (1.41, 1)}
BAND_WAVES = ["const300", "champ_D90_T5"]
MESH_F = dict(MESHES, **{k: v[0] for k, v in BAND_OVERRIDE.items()})

WAVES = {
    "const300":     dict(w_hi=300.0,         duty=1.00, period=10.0, mean=300.0),
    "champ_D90_T5": dict(w_hi=300.0 / 0.90,  duty=0.90, period=5.0,  mean=300.0),
    "const500":     dict(w_hi=500.0,         duty=1.00, period=10.0, mean=500.0),
    "lg_D20_T30":   dict(w_hi=2500.0,        duty=0.20, period=30.0, mean=500.0),
}
TIMESTEP_WAVES = ["const300", "champ_D90_T5", "lg_D20_T30"]
HALF_MAXCO = 0.9


def mesh_dims(f, mesh=None):
    nr = int(round(NR0 * f))
    nz_main = int(round(NZ0 * f))
    nband = 1 if f < 1.4 else 2
    if mesh in BAND_OVERRIDE:
        nband = BAND_OVERRIDE[mesh][1]
    return nr, nz_main, nband


def wall_cell_um(nr):
    """Outer-wall (finest) radial cell for simpleGrading R = last/first over the gap."""
    ratio = GRADING ** (1.0 / (nr - 1))
    first = GAP_M * (ratio - 1.0) / (ratio ** nr - 1.0)
    return first * GRADING * 1e6


def all_runs():
    runs = []
    for mesh, f in MESHES:
        for wave in WAVES:
            runs.append(dict(mesh=mesh, f=f, wave=wave, maxco=PROD_MAXCO,
                             tag=wave, **WAVES[wave]))
    f_prod = dict(MESHES)[PROD_MESH]
    for wave in TIMESTEP_WAVES:
        runs.append(dict(mesh=PROD_MESH, f=f_prod, wave=wave, maxco=HALF_MAXCO,
                         tag=f"{wave}_co0p9", **WAVES[wave]))
    for mesh, (f, _) in BAND_OVERRIDE.items():
        for wave in BAND_WAVES:
            runs.append(dict(mesh=mesh, f=f, wave=wave, maxco=PROD_MAXCO, tag=wave, **WAVES[wave]))
    return runs


# --------------------------------------------------------------------------- #
def build_mesh_case(cases_root, mesh, f):
    """Copy the production case, rewrite the two block counts, blockMesh + checkMesh."""
    dst = os.path.join(cases_root, mesh)
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(BASE_CASE, dst)
    shutil.rmtree(os.path.join(dst, "constant", "polyMesh"), ignore_errors=True)
    bmd = os.path.join(dst, "system", "blockMeshDict")
    with open(bmd) as fh:
        text = fh.read()
    nr, nz_main, nband = mesh_dims(f, mesh)
    band_old = r"hex \(0 1 2 3 4 5 6 7\) \(37 1 1\)"
    main_old = r"hex \(4 5 6 7 8 9 10 11\) \(37 1 89\)"
    if not (re.search(band_old, text) and re.search(main_old, text)):
        raise RuntimeError("blockMeshDict block lines not found -- base case changed?")
    text = re.sub(band_old, f"hex (0 1 2 3 4 5 6 7) ({nr} 1 {nband})", text)
    text = re.sub(main_old, f"hex (4 5 6 7 8 9 10 11) ({nr} 1 {nz_main})", text)
    with open(bmd, "w") as fh:
        fh.write(text)
    for app in ("blockMesh", "checkMesh"):
        r = subprocess.run([app], cwd=dst, capture_output=True, text=True)
        with open(os.path.join(dst, f"log.{app}"), "w") as fh:
            fh.write(r.stdout)
        if r.returncode != 0:
            raise RuntimeError(f"{app} failed for {mesh}:\n{r.stderr[-1500:]}")
    m = re.search(r"cells:\s+(\d+)", open(os.path.join(dst, "log.checkMesh")).read())
    ncells = int(m.group(1)) if m else nr * (nz_main + nband)
    ok = "Mesh OK." in open(os.path.join(dst, "log.checkMesh")).read()
    print(f"  mesh {mesh}: {nr} x {nz_main}+{nband} -> {ncells} cells, outer-wall cell "
          f"{wall_cell_um(nr):.1f} um, checkMesh {'OK' if ok else 'WARNINGS (see log)'}", flush=True)
    return dst, ncells


def prepare_templates(results_dir, meshes):
    """One compiled template per mesh (fig7 engine: copies case, compiles coded FOs)."""
    cases_root = os.path.join(results_dir, "_cases")
    os.makedirs(cases_root, exist_ok=True)
    out = {}
    for mesh, f in meshes:
        _, ncells = build_mesh_case(cases_root, mesh, f)
        F.CASES_DIR = cases_root
        F.CASE_NAME = mesh
        F.RESULTS_DIR = os.path.join(results_dir, mesh)
        t0 = time.time()
        tpl = F.prepare_template()
        print(f"  template {mesh} compiled ({time.time()-t0:.0f}s)", flush=True)
        out[mesh] = dict(template=tpl, ncells=ncells)
    with open(os.path.join(results_dir, "meshes.json"), "w") as fh:
        json.dump({m: dict(f=f, ncells=out[m]["ncells"], nr=mesh_dims(f, m)[0],
                           nz_main=mesh_dims(f, m)[1], nband=mesh_dims(f, m)[2],
                           wall_cell_um=wall_cell_um(mesh_dims(f, m)[0]))
                   for m, f in meshes}, fh, indent=2)
    return out


# --------------------------------------------------------------------------- #
def waveform_pts(run, duration, period):
    if run["duty"] >= 1.0:
        return [(0.0, run["w_hi"] * RPM), (duration + 1.0, run["w_hi"] * RPM)]
    pts, _ = F.square_wave_points(0.0, duration, run["w_hi"] * RPM, 0.0, period, run["duty"], RAMP)
    return pts


def _wmean(t, y, t0, t1):
    sel = (t >= t0) & (t <= t1)
    return (float(np.mean(y[sel])), float(np.std(y[sel]))) if sel.sum() >= 3 else (float("nan"),) * 2


def metrics(run, text, duration, period, window):
    t_conv, conv, t_pw, om, mz, pf = F.parse_log(text)
    pts = waveform_pts(run, duration, period)
    end_t = float(t_conv[-1]) if len(t_conv) else 0.0
    diverged = False
    x_win = x_std = x_prev = float("nan")
    if len(conv):
        phys = (conv >= -0.02) & (conv <= 1.02)
        tc, cc = t_conv[phys], conv[phys]
        if len(cc) < 3 or tc[-1] < end_t - 1.0 or phys.mean() < 0.98:
            diverged = True
        if len(cc) >= 3:
            x_win, x_std = _wmean(tc, cc, duration - window, duration)
            x_prev, _ = _wmean(tc, cc, duration - 2 * window, duration - window)
    grid, w = F.densify(pts, duration)
    p_motor = float(np.mean(F.motor_power.electrical_power(grid, w)))
    p_cfd = float("nan")
    if len(t_pw):
        sel = (t_pw >= duration - window) & (t_pw <= duration)
        if sel.sum() >= 3:
            p_cfd = float(np.mean(np.abs(pf[sel])))
    complete = end_t >= duration - 0.5
    return dict(run, duration=duration, period_used=period, reached_t=end_t,
                complete=complete, diverged=diverged,
                ok=bool(len(conv)) and complete and not diverged,
                x_win=x_win, x_std=x_std, x_prev=x_prev, drift=x_win - x_prev,
                p_motor=p_motor, r=x_win - p_motor / P_MAX, p_cfd=p_cfd,
                t_conv=t_conv, conv=conv)


def _time_dirs(workdir):
    return sorted((float(n), n) for n in os.listdir(workdir)
                  if os.path.isdir(os.path.join(workdir, n)) and F._is_float(n))


def run_episode(template, run, results_dir, duration, period_override, window, resume):
    period = period_override or run["period"]
    workdir = os.path.join(results_dir, run["mesh"], run["tag"])
    log_path = os.path.join(workdir, "log.pimpleFoam")
    pts = waveform_pts(run, duration, period)
    table = "table (" + " ".join(f"({t:.6f} {w:.6f})" for t, w in pts) + ")"
    env = dict(os.environ, OMP_NUM_THREADS="1")
    label = f"{run['mesh']}/{run['tag']}"

    latest_t, latest = 0.0, "0"
    if resume and os.path.isfile(log_path):
        tds = _time_dirs(workdir)
        if tds:
            latest_t, latest = tds[-1]

    t0 = time.time()
    if resume and latest_t >= duration - 0.5:
        print(f"  [RESUME ] {label}: already at t={latest_t:.0f}s -- parsing only", flush=True)
        rc = 0
        text = open(log_path, errors="replace").read()
    elif resume and latest_t > 0.0:
        print(f"  [RESUME ] {label}: continuing from t={latest_t:.0f}s", flush=True)
        F.foam_set(workdir, "boundaryField.inner_wall.omega", table, f"{latest}/U")
        subprocess.run(["foamDictionary", "-entry", "boundaryField.inner_wall.omegaCoeffs",
                        "-remove", f"{latest}/U"], cwd=workdir, check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        F.foam_set(workdir, "startFrom", "latestTime", "system/controlDict")
        F.foam_set(workdir, "endTime", repr(float(duration)), "system/controlDict")
        r = subprocess.run(["pimpleFoam"], cwd=workdir, capture_output=True, text=True, env=env)
        rc = r.returncode
        with open(log_path, "a") as fh:
            fh.write(f"\n// ---- RESUMED from t={latest_t:.6g}s ----\n" + r.stdout)
        with open(os.path.join(workdir, "log.err"), "a") as fh:
            fh.write(r.stderr)
        text = open(log_path, errors="replace").read()
    else:
        if os.path.isdir(workdir):
            shutil.rmtree(workdir)
        shutil.copytree(template, workdir)
        F.clean_run_artifacts(workdir)
        F.foam_set(workdir, "boundaryField.inner_wall.omega", table, "0/U")
        F.foam_set(workdir, "endTime", repr(float(duration)), "system/controlDict")
        F.foam_set(workdir, "writeInterval", repr(float(WRITE_INTERVAL)), "system/controlDict")
        F.foam_set(workdir, "maxCo", repr(float(run["maxco"])), "system/controlDict")
        with open(os.path.join(workdir, "run.json"), "w") as fh:
            json.dump(dict(run, duration=duration, period_used=period), fh, indent=2)
        r = subprocess.run(["pimpleFoam"], cwd=workdir, capture_output=True, text=True, env=env)
        rc = r.returncode
        with open(log_path, "w") as fh:
            fh.write(r.stdout)
        with open(os.path.join(workdir, "log.err"), "w") as fh:
            fh.write(r.stderr)
        text = r.stdout
    res = metrics(run, text, duration, period, window)
    res["ok"] = res["ok"] and rc == 0
    res["wall_h"] = (time.time() - t0) / 3600.0
    st = "OK" if res["ok"] else ("FAILED" if rc != 0 else "DIVERGED/INCOMPLETE")
    print(f"  [{st:8s}] {label:26s} maxCo={run['maxco']:.1f} | X_win={res['x_win']:.4f} "
          f"(sd {res['x_std']:.4f}) drift={res['drift']:+.4f} P={res['p_motor']:.2f} W "
          f"r={res['r']:.4f} | {res['wall_h']:.2f} h, t={res['reached_t']:.0f}s", flush=True)
    return res


# --------------------------------------------------------------------------- #
def gci_triplet(phi1, phi2, phi3, r21, r32):
    """Celik et al. 2008 (J. Fluids Eng. 130:078001). 1 = finest. Returns observed
    order p, extrapolated value, and the fine-grid GCI as an ABSOLUTE band."""
    out = dict(p=float("nan"), phi_ext=float("nan"), gci_fine_abs=float("nan"),
               oscillatory=False, note="")
    e21, e32 = phi2 - phi1, phi3 - phi2
    if not all(np.isfinite([phi1, phi2, phi3])):
        out["note"] = "missing grid"
        return out
    if abs(e21) < 1e-12 or abs(e32) < 1e-12:
        out["note"] = "identical grids"
        return out
    s = float(np.sign(e32 / e21))
    out["oscillatory"] = s < 0
    p = 2.0
    for _ in range(200):
        q = math.log((r21 ** p - s) / (r32 ** p - s)) if (r21 ** p - s) > 0 and (r32 ** p - s) > 0 else 0.0
        p_new = abs(math.log(abs(e32 / e21)) + q) / math.log(r21)
        if abs(p_new - p) < 1e-8:
            p = p_new
            break
        p = 0.5 * p + 0.5 * p_new
    out["p"] = p
    if p > 1e-6:
        out["phi_ext"] = (r21 ** p * phi1 - phi2) / (r21 ** p - 1.0)
        out["gci_fine_abs"] = 1.25 * abs(phi1 - phi2) / (r21 ** p - 1.0)
    # conservative fallback when the observed order is unreliable (noise / oscillation)
    out["gci_fine_abs_p2_fs3"] = 3.0 * abs(phi1 - phi2) / (r21 ** 2 - 1.0)
    if out["oscillatory"]:
        out["note"] = "oscillatory convergence -- trust the p=2/Fs=3 band"
    elif not (0.5 <= p <= 4.0):
        out["note"] = "observed order outside [0.5,4] -- trust the p=2/Fs=3 band"
    return out


def load_all(results_dir, duration, period_override, window):
    out = []
    for run in all_runs():
        log = os.path.join(results_dir, run["mesh"], run["tag"], "log.pimpleFoam")
        if os.path.isfile(log):
            period = period_override or run["period"]
            res = metrics(run, open(log, errors="replace").read(), duration, period, window)
            res["wall_h"] = float("nan")
            out.append(res)
    return out


def analyze(results_dir, duration, period_override, window):
    rows = load_all(results_dir, duration, period_override, window)
    if not rows:
        print("No episode logs yet.")
        return
    mj = os.path.join(results_dir, "meshes.json")
    meshes = json.load(open(mj)) if os.path.isfile(mj) else {}
    get = {(r["mesh"], r["tag"]): r for r in rows}

    with open(os.path.join(results_dir, "summary_table.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["mesh", "f", "cells", "wall_cell_um", "tag", "maxCo", "mean_rpm", "w_hi_rpm",
                    "duty", "period_s", "X_win", "X_win_sd", "X_prev", "drift", "P_motor_W",
                    "r", "P_cfd_torque_W", "reached_t", "complete", "diverged"])
        for r in rows:
            m = meshes.get(r["mesh"], {})
            w.writerow([r["mesh"], r["f"], m.get("ncells", ""), f"{m.get('wall_cell_um', float('nan')):.1f}",
                        r["tag"], r["maxco"], r["mean"], f"{r['w_hi']:.1f}", r["duty"], r["period_used"],
                        f"{r['x_win']:.6f}", f"{r['x_std']:.6f}", f"{r['x_prev']:.6f}",
                        f"{r['drift']:+.6f}", f"{r['p_motor']:.4f}", f"{r['r']:.6f}",
                        f"{r['p_cfd']:.6g}", f"{r['reached_t']:.1f}", int(r["complete"]),
                        int(r["diverged"])])

    # GCI per waveform and for the headline gaps (production maxCo only)
    names = [m for m, _ in MESHES]                     # coarse -> fine
    cells = [meshes.get(m, {}).get("ncells", float("nan")) for m in names]
    report = dict(meshes=meshes, gci={}, gaps={}, timestep={})

    def series(fn):
        return [fn(m) for m in names]

    def xw(m, tag):
        r = get.get((m, tag))
        return r["x_win"] if (r and r["complete"] and not r["diverged"]) else float("nan")

    quantities = {f"X[{t}]": (lambda m, t=t: xw(m, t)) for t in WAVES}
    quantities["gap champ-const300"] = lambda m: xw(m, "champ_D90_T5") - xw(m, "const300")
    quantities["gap lg-const500"] = lambda m: xw(m, "lg_D20_T30") - xw(m, "const500")
    quantities["gap champ-const500"] = lambda m: xw(m, "champ_D90_T5") - xw(m, "const500")
    if all(np.isfinite(cells)):
        h = [(1.0 / c) ** 0.5 for c in cells]         # 2D representative size
        r32, r21 = h[0] / h[1], h[1] / h[2]
        for qn, fn in quantities.items():
            coarse, med, fine = series(fn)
            g = gci_triplet(fine, med, coarse, r21, r32)
            report["gci"][qn] = dict(coarse=coarse, medium=med, fine=fine, **g)
    for qn in ("gap champ-const300", "gap lg-const500", "gap champ-const500"):
        report["gaps"][qn] = {m: quantities[qn](m) for m in names}
    for t in TIMESTEP_WAVES:
        a, b = get.get((PROD_MESH, t)), get.get((PROD_MESH, f"{t}_co0p9"))
        if a and b and a["complete"] and b["complete"]:
            report["timestep"][t] = dict(maxCo1p8=a["x_win"], maxCo0p9=b["x_win"],
                                         diff=b["x_win"] - a["x_win"])
    report["outlet_band"] = {}
    for t in BAND_WAVES:
        a, b = get.get(("f1.41", t)), get.get(("f1.41b1", t))
        if a and b and a["complete"] and b["complete"]:
            report["outlet_band"][t] = dict(band2=a["x_win"], band1=b["x_win"], diff=b["x_win"] - a["x_win"])
    with open(os.path.join(results_dir, "grid_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=float)

    # ---- print ------------------------------------------------------------ #
    print("\n" + "=" * 92)
    print(f"D: grid check, short reactor Sc 1075, pristine IC, {duration:.0f} s, "
          f"X over last {window:.0f} s")
    print("=" * 92)
    print(f"  {'':22s}" + "".join(f"{m:>12s}" for m in names))
    print(f"  {'cells':22s}" + "".join(f"{c:>12}" for c in cells))
    for t in WAVES:
        print(f"  X {t:20s}" + "".join(f"{xw(m, t):>12.4f}" for m in names))
    for qn in report["gaps"]:
        print(f"  {qn:22s}" + "".join(f"{v:>+12.4f}" for v in report["gaps"][qn].values()))
    if report["gci"]:
        print("\n  GCI (Celik 2008), fine-grid band in absolute X units:")
        for qn, g in report["gci"].items():
            print(f"    {qn:22s} p={g['p']:.2f}  GCI={g['gci_fine_abs']:.4f}  "
                  f"(p=2,Fs=3: {g['gci_fine_abs_p2_fs3']:.4f})  ext={g['phi_ext']:.4f}  {g['note']}")
    if report["timestep"]:
        print("\n  Time step (production mesh), X_win at maxCo 1.8 vs 0.9:")
        for t, d in report["timestep"].items():
            print(f"    {t:16s} {d['maxCo1p8']:.4f} vs {d['maxCo0p9']:.4f}  diff {d['diff']:+.4f}")
    if report["outlet_band"]:
        print("\n  Outlet band on the finest mesh, X_win with 2 vs 1 band cells:")
        for t, d in report["outlet_band"].items():
            print(f"    {t:16s} {d['band2']:.4f} vs {d['band1']:.4f}  diff {d['diff']:+.4f}")
    print("  (noise reference: within-window sd per run is in summary_table.csv; the "
          "flutter floor is ~0.005)")
    print("=" * 92 + "\n")

    plot(results_dir, rows, meshes, names, xw, report)


def plot(results_dir, rows, meshes, names, xw, report):
    cells_k = [meshes.get(m, {}).get("ncells", np.nan) / 1000.0 for m in names]
    labels = {"const300": "constant 300 rpm", "champ_D90_T5": "best fixed pulse (mean 300)",
              "const500": "constant 500 rpm", "lg_D20_T30": "Lopez-Guajardo pulse (mean 500)"}
    style = {"const300": ("o", "-", "#1f5f8b"), "champ_D90_T5": ("*", "-", "#c8102e"),
             "const500": ("s", "--", "#6b8fb3"), "lg_D20_T30": ("^", "--", "#7a7a7a")}
    plt.rcParams.update({"font.size": 13})
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for t in WAVES:
        ys = [xw(m, t) for m in names]
        mk, ls, col = style[t]
        ax.plot(cells_k, ys, ls=ls, marker=mk, ms=11 if mk == "*" else 8, color=col, lw=2,
                label=labels[t])
        for m in names:
            r = next((q for q in rows if q["mesh"] == m and q["tag"] == t), None)
            if r and np.isfinite(r["x_std"]):
                ax.errorbar(meshes.get(m, {}).get("ncells", np.nan) / 1000.0, r["x_win"],
                            yerr=r["x_std"], color=col, capsize=3, lw=1, alpha=0.6)
    prod = meshes.get(PROD_MESH, {}).get("ncells")
    if prod:
        ax.axvline(prod / 1000.0, color="k", ls=":", lw=1.2)
        ax.annotate("production mesh", (prod / 1000.0, ax.get_ylim()[0]), xytext=(4, 6),
                    textcoords="offset points", fontsize=10)
    ax.set_xlabel("mesh size [thousand cells]")
    ax.set_ylabel("outlet conversion (last 30 s)")
    ax.set_title("Grid check: conversion vs mesh size")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "grid_conversion_vs_cells.png"), dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for qn, col in (("gap champ-const300", "#c8102e"), ("gap lg-const500", "#7a7a7a")):
        ys = [report["gaps"][qn][m] for m in names]
        ax.plot(cells_k, ys, "-o", color=col, lw=2, ms=8,
                label={"gap champ-const300": "best fixed pulse minus constant 300",
                       "gap lg-const500": "Lopez-Guajardo pulse minus constant 500"}[qn])
    ax.axhline(0, color="k", lw=1)
    if prod:
        ax.axvline(prod / 1000.0, color="k", ls=":", lw=1.2)
    ax.set_xlabel("mesh size [thousand cells]")
    ax.set_ylabel("conversion gain over constant")
    ax.set_title("Does the pulsing gain survive refinement?")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "grid_gap_vs_cells.png"), dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for r in rows:
        if not len(r["t_conv"]) or r["maxco"] != PROD_MAXCO:
            continue
        keep = (r["conv"] >= -0.02) & (r["conv"] <= 1.02)
        mk, ls, col = style[r["wave"]]
        lw = {"f0.71": 0.8, "f1.00": 1.3, "f1.41": 1.8}.get(r["mesh"], 1.0)
        alpha = {"f0.71": 0.45, "f1.00": 0.7, "f1.41": 1.0}.get(r["mesh"], 1.0)
        ax.plot(r["t_conv"][keep], r["conv"][keep], color=col, lw=lw, alpha=alpha,
                label=f"{labels[r['wave']]}, {r['mesh']}")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("outlet conversion")
    ax.set_title("Conversion traces on each mesh (thicker = finer)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "grid_traces.png"), dpi=200)
    plt.close(fig)
    print(f"Figures + tables in {results_dir}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="2 s episodes, pulsed T=1 s")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", default=None, help="comma list of mesh/tag, e.g. f1.41/lg_D20_T30")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    runs = all_runs()
    if args.only:
        keep = {s.strip() for s in args.only.split(",")}
        runs = [r for r in runs if f"{r['mesh']}/{r['tag']}" in keep]
    duration, period_override, window = EPISODE, None, WINDOW
    results_dir = os.path.join(RESULTS_ROOT, "D")
    if args.smoke:
        duration, period_override, window = 2.0, 1.0, 0.5
        results_dir = os.path.join(RESULTS_ROOT, "D_smoke")

    if args.list:
        for r in runs:
            nr, nz, nb = mesh_dims(r["f"], r["mesh"])
            print(f"{r['mesh']}/{r['tag']:18s} {nr}x{nz}+{nb} cells, wall {wall_cell_um(nr):.1f} um, "
                  f"0<->{r['w_hi']:.0f} rpm D={r['duty']:.2f} T={r['period']:g}s maxCo={r['maxco']}")
        return
    os.makedirs(results_dir, exist_ok=True)
    if args.analyze_only:
        analyze(results_dir, duration, period_override, window)
        return
    for app in ("pimpleFoam", "foamDictionary", "blockMesh", "checkMesh"):
        if shutil.which(app) is None:
            sys.exit(f"ERROR: {app} not on PATH (source the OpenFOAM bashrc).")

    need = sorted({r["mesh"] for r in runs})
    mesh_list = [(m, MESH_F[m]) for m in [n for n, _ in MESHES] + list(BAND_OVERRIDE) if m in need]
    print(f"D grid check -> {results_dir}\n  {len(runs)} episodes x {duration:.0f} s on meshes "
          f"{', '.join(need)}\n", flush=True)
    t0 = time.time()
    tpls = prepare_templates(results_dir, mesh_list)
    print(f"  templates ready ({time.time()-t0:.0f}s)\n", flush=True)

    workers = args.workers or len(runs)
    # longest first so the tail is short
    cost = {"f0.71": 0.35, "f1.00": 1.0, "f1.41": 4.0, "f1.41b1": 4.0}
    runs.sort(key=lambda r: -cost[r["mesh"]] * (1.8 if r["w_hi"] > 1000 else 1.0)
              * (2.0 if r["maxco"] < PROD_MAXCO else 1.0))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_episode, tpls[r["mesh"]]["template"], r, results_dir, duration,
                          period_override, window, args.resume) for r in runs]
        for fut in as_completed(futs):
            fut.result()
    analyze(results_dir, duration, period_override, window)


if __name__ == "__main__":
    main()
