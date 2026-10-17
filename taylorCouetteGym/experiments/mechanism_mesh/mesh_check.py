#!/usr/bin/env python3
"""Does the Taylor-cell mechanism survive mesh refinement?

Built 2026-10-12 after the mechanism experiment (experiments/mechanism) found
that pulsing helps by making the restart re-form MORE Taylor cells (8 vs 4 at
constant 300 rpm on the production mesh), that the best stop lengths are the
ones whose restart lands in the 8-cell state, and that the extra cells merge
back within ~10 s under steady rotation. Cell-count selection between nearly
equal states can depend on the mesh, so this repeats the key runs on a coarser
and a finer member of the endgame-D self-similar mesh family:

    f0.71 : 26 x  64 (+1 outlet-band cell)  ~1.7k cells, outer-wall cell ~35 um
    f1.00 : 37 x  89 (+1)                   ~3.3k  (production; REUSED from
                                                    experiments/mechanism/results/main)
    f1.41 : 52 x 127 (+2)                   ~6.6k  ~18 um

Per mesh (nominal viscosity, warmed 60 s at constant 300, maxCo 0.9, 130 s,
the mechanism experiment's exact protocol and scoring):
    stops 0 (constant), 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.2 s  (run 4.5 s, mean 300)
    hold: pulse (stop 0.5 s) until 60 s, then constant 300 -> cell-merge time

The questions, in order:
  1. Is constant 300 a 4-cell state on every mesh?
  2. Does each stop length give the same burst cell count on every mesh, i.e.
     are the 6/8-cell windows (8 at 0.4-0.6 s, 6 at 0.3 and 0.7 s) mesh-stable?
  3. Does conversion still step with cell count, and is the best stop in the
     same window?
  4. Do the extra cells merge back on the same ~10 s time after pulsing stops?

The f1.41 outlet band has 2 cells (the family rule). The endgame-D check
found that this lets fluid flow back in through the outlet, so the
outflow-only outlet conversion over-reads there: use X_budget (wall-flux
budget) as the primary conversion on every mesh. The cell count is an
interior quantity and is unaffected.

USAGE (from taylorCouetteGym/, OpenFOAM on PATH)
    python3 experiments/mechanism_mesh/mesh_check.py --list
    python3 experiments/mechanism_mesh/mesh_check.py --smoke          # 6 s pipeline test
    python3 experiments/mechanism_mesh/mesh_check.py --resume         # the real thing (Carya)
    python3 experiments/mechanism_mesh/mesh_check.py --analyze-only
"""

import argparse
import csv
import importlib.util
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
GYM_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(GYM_ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load("mechanism_sweep", "experiments/mechanism/mechanism_sweep.py")   # protocol + diagnostics
G = _load("d_grid_check", "experiments/endgame/d_grid_check.py")            # mesh family builder

RESULTS_ROOT = os.path.join(HERE, "results")
REF_RESULTS = os.path.join(GYM_ROOT, "experiments", "mechanism", "results", "main")
MESHES = [("f0.71", 0.71), ("f1.41", 1.41)]
STOPS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.2]
SMOKE_STOPS = [0.0, 0.4]
HOLD_STOP = 0.5
MESH_COST = {"f0.71": 0.3, "f1.41": 4.7}        # relative to f1.00 (2026-10-12 smoke-test warm-ups)
MESH_COL = {"f0.71": "#6b8fb3", "f1.00": "#222222", "f1.41": "#c8102e"}


def runs_for(mesh, smoke):
    cfg = M.SMOKE if smoke else M.FULL
    out = []
    for s in (SMOKE_STOPS if smoke else STOPS):
        r = M.make_run("e1", "nu1", s, cfg=cfg)
        out.append(dict(r, mesh=mesh, tag=f"{mesh}_s{M.stag(s)}"))
    hs = cfg["e3_stop"] if smoke else HOLD_STOP
    r = M.make_run("e5", "nu1", hs, cfg=cfg, kind="hold")
    out.append(dict(r, mesh=mesh, tag=f"{mesh}_hold"))
    return out


def all_runs(smoke):
    return [r for mesh, _ in MESHES for r in runs_for(mesh, smoke)]


def reference_runs():
    """The production-mesh (f1.00) rows from the mechanism experiment."""
    out = []
    want = {f"e1_nu1_s{M.stag(s)}": s for s in STOPS}
    want["e5_nu1_hold"] = None
    for tag in want:
        wd = os.path.join(REF_RESULTS, "runs", tag)
        rj = os.path.join(wd, "run.json")
        if not os.path.isfile(rj):
            continue
        cfg = json.load(open(rj))["cfg"]
        run = (M.make_run("e5", "nu1", HOLD_STOP, cfg=cfg, kind="hold") if want[tag] is None
               else M.make_run("e1", "nu1", want[tag], cfg=cfg))
        out.append(dict(M.metrics(run, cfg, wd), mesh="f1.00"))
    return out


def prepare_mesh(mesh, f, work_root, cfg):
    """Build the mesh case, compile a template on it, warm it at constant 300."""
    cases_root = os.path.join(work_root, "_cases")
    os.makedirs(cases_root, exist_ok=True)
    tpl_root = os.path.join(work_root, "_templates", mesh)
    os.makedirs(tpl_root, exist_ok=True)
    if not os.path.isfile(os.path.join(tpl_root, "base", "COMPILED_OK")):
        G.build_mesh_case(cases_root, mesh, f)
    M.BASE_CASE = os.path.join(cases_root, mesh)     # prepare_base copies M.BASE_CASE
    base = M.prepare_base(tpl_root)
    return base, tpl_root


def load_rows(results_dir, smoke):
    rows = []
    for run in all_runs(smoke):
        wd = os.path.join(results_dir, "runs", run["tag"])
        rj = os.path.join(wd, "run.json")
        if os.path.isfile(rj):
            cfg = json.load(open(rj))["cfg"]
            rows.append(dict(M.metrics(run, cfg, wd), mesh=run["mesh"]))
    return rows


def hold_trace(row):
    m = M.read_mech(row["workdir"])
    if m is None:
        return None
    return m["t"], m["nVort"], m["wall"]


def merge_time(row, cfg_switch):
    """Seconds after the switch to constant until the cell count first reaches
    the constant-rotation count (4) and stays there for 10 s."""
    tr = hold_trace(row)
    if tr is None:
        return float("nan")
    t, n, _ = tr
    sel = t >= cfg_switch
    t, n = t[sel], n[sel]
    for i in range(len(t)):
        if n[i] <= 4 and np.all(n[(t >= t[i]) & (t <= t[i] + 10.0)] <= 4):
            return float(t[i] - cfg_switch)
    return float("nan")


def analyze(results_dir, smoke):
    rows = load_rows(results_dir, smoke)
    ref = [] if smoke else reference_runs()
    allr = ref + rows
    if not allr:
        print("No runs yet.")
        return
    cols = ["mesh", "tag", "stop", "kind", "complete", "ok", "x_out", "x_budget", "closure",
            "nvort_burst", "wall_burst", "omega_err_rpm"]
    with open(os.path.join(results_dir, "mesh_table.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in allr:
            w.writerow([(f"{r.get(k):.6g}" if isinstance(r.get(k), float) else r.get(k)) for k in cols])

    meshes = [m for m in ("f0.71", "f1.00", "f1.41") if any(r["mesh"] == m for r in allr)]
    switch = (M.SMOKE if smoke else M.FULL)["e5_switch"]
    report = {}
    print("\n" + "=" * 86)
    print("MESH CHECK of the Taylor-cell mechanism (X = wall-flux budget conversion)")
    print("=" * 86)
    stops = sorted({r["stop"] for r in allr if r.get("kind", "wave") == "wave"})
    print(f"  {'stop [s]':10s}" + "".join(f"{m:>18s}" for m in meshes))
    for s in stops:
        line = f"  {('constant' if s == 0 else f'{s:.2f}'):10s}"
        for m in meshes:
            r = next((q for q in allr if q["mesh"] == m and q.get("kind", "wave") == "wave"
                      and abs(q["stop"] - s) < 1e-9), None)
            line += (f"{r['x_budget']:>10.4f} ({r['nvort_burst']:.0f} c)" if r and r["ok"]
                     else f"{'--':>18s}")
        print(line)
    for m in meshes:
        h = next((q for q in allr if q["mesh"] == m and q.get("kind") == "hold"), None)
        mt = merge_time(h, switch) if (h and h["complete"]) else float("nan")
        mr = [q for q in allr if q["mesh"] == m and q.get("kind", "wave") == "wave" and q["ok"]]
        const = next((q for q in mr if q["stop"] == 0), None)
        best = max((q for q in mr if q["stop"] > 0), key=lambda q: q["x_budget"], default=None)
        report[m] = dict(
            const_cells=const["nvort_burst"] if const else None,
            const_x_budget=const["x_budget"] if const else None,
            best_stop=best["stop"] if best else None, best_cells=best["nvort_burst"] if best else None,
            best_gain=(best["x_budget"] - const["x_budget"]) if (best and const) else None,
            cells_by_stop={f"{q['stop']:.2f}": q["nvort_burst"] for q in mr},
            hold_merge_s=mt, hold_x_budget=h["x_budget"] if h else None)
        print(f"  {m}: constant {report[m]['const_cells']} cells; best stop {report[m]['best_stop']} s "
              f"({report[m]['best_cells']} cells, gain {report[m]['best_gain'] if report[m]['best_gain'] is None else round(report[m]['best_gain'], 4)}); "
              f"hold: cells back to 4 after {mt:.1f} s, X_budget {report[m]['hold_x_budget']}")
    print("  (c = median Taylor-cell count during bursts)")
    bad = [r["tag"] for r in rows if not r["ok"]]
    if bad:
        print(f"  NOT OK: {', '.join(bad)}")
    print("=" * 86 + "\n")
    json.dump(report, open(os.path.join(results_dir, "mesh_report.json"), "w"), indent=2, default=float)
    plot(results_dir, allr, meshes, switch)


def plot(results_dir, allr, meshes, switch):
    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    for m in meshes:
        rs = sorted([q for q in allr if q["mesh"] == m and q.get("kind", "wave") == "wave" and q["ok"]],
                    key=lambda q: q["stop"])
        const = next((q for q in rs if q["stop"] == 0), None)
        pul = [q for q in rs if q["stop"] > 0]
        if not pul:
            continue
        ax[0].plot([q["stop"] for q in pul], [q["x_budget"] for q in pul], "-", color=MESH_COL[m], lw=1.5,
                   label=m + (" (production)" if m == "f1.00" else ""))
        for q in pul:
            ax[0].annotate(f"{q['nvort_burst']:.0f}", (q["stop"], q["x_budget"]), xytext=(0, 6),
                           textcoords="offset points", ha="center", fontsize=9, color=MESH_COL[m])
            ax[0].plot(q["stop"], q["x_budget"], "o", color=MESH_COL[m], ms=7)
        if const:
            ax[0].axhline(const["x_budget"], color=MESH_COL[m], ls=":", lw=1.2)
        h = next((q for q in allr if q["mesh"] == m and q.get("kind") == "hold"), None)
        if h and h["complete"]:
            t, n, _ = hold_trace(h)
            sel = t >= switch - 10
            ax[1].plot(t[sel] - switch, n[sel], "-", color=MESH_COL[m], lw=1, label=m)
    ax[0].set_xscale("log")
    ax[0].set_xticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0, 1.2])
    ax[0].set_xticklabels(["0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "1", "1.2"])
    ax[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[0].set_xlabel("stop length [s]  (run 4.5 s, mean 300 rpm)")
    ax[0].set_ylabel("conversion (wall-flux budget)")
    ax[0].set_title("Numbers = Taylor cells during bursts; dotted = constant 300")
    ax[0].legend(fontsize=10)
    ax[0].grid(True, which="both", alpha=0.3)
    ax[1].axvline(0, color="k", lw=0.8)
    ax[1].set_xlabel("time after the switch from pulsing to constant 300 [s]")
    ax[1].set_ylabel("Taylor-cell count at mid-gap")
    ax[1].set_title("Hold run: how fast the extra cells merge")
    ax[1].legend(fontsize=10)
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "mesh_check.png"), dpi=200)
    plt.close(fig)
    print(f"Figure + tables in {results_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--results_dir", default=None)
    ap.add_argument("--work_root", default=None,
                    help="run the cases here (node-local disk) and copy outputs back to results_dir")
    args = ap.parse_args()

    cfg = dict(M.SMOKE if args.smoke else M.FULL)
    results_dir = args.results_dir or os.path.join(RESULTS_ROOT, "smoke" if args.smoke else "main")
    runs = all_runs(args.smoke)
    if args.list:
        for r in runs:
            nr, nz, nb = G.mesh_dims(dict(MESHES)[r["mesh"]], r["mesh"])
            print(f"{r['tag']:14s} mesh {nr}x{nz}+{nb}  kind {r['kind']:5s} stop {r['stop'] or r.get('kick_stop', 0):.2f} s  "
                  f"w_hi {r.get('pulse_w_hi', r['w_hi']):.1f} rpm")
        print(f"{len(runs)} runs (+ {len(STOPS) + 1} production-mesh rows reused from experiments/mechanism)")
        return
    os.makedirs(results_dir, exist_ok=True)
    if args.analyze_only:
        analyze(results_dir, args.smoke)
        return
    for app in ("pimpleFoam", "foamDictionary", "blockMesh", "checkMesh"):
        if shutil.which(app) is None:
            sys.exit(f"ERROR: {app} not on PATH (source the OpenFOAM bashrc).")

    work_root = args.work_root or results_dir
    print(f"mechanism mesh check -> {results_dir}\n  {len(runs)} runs x {cfg['episode']:g} s on "
          f"{', '.join(m for m, _ in MESHES)}, warm {cfg['warm']:g} s, maxCo {M.MAXCO}\n", flush=True)
    t0 = time.time()
    todo = [r for r in runs if not (args.resume and os.path.isfile(
        os.path.join(results_dir, "runs", r["tag"], "COMPLETE")))]
    need = [(m, f) for m, f in MESHES if any(r["mesh"] == m for r in todo)]
    bases = {}
    for m, f in need:                          # sequential: prepare_base reads M.BASE_CASE
        bases[m] = prepare_mesh(m, f, work_root, cfg)
        print(f"  {m}: mesh built, template compiled ({time.time() - t0:.0f} s)", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, len(need))) as ex:
        futs = {m: ex.submit(M.warm_template, bases[m][0], bases[m][1], "nu1", cfg) for m in bases}
        tpls = {m: fu.result() for m, fu in futs.items()}
    for m, d in tpls.items():
        rec = os.path.join(results_dir, "_templates", m)
        os.makedirs(rec, exist_ok=True)
        for fn in ("WARM_OK.json", "log.warmup"):
            if os.path.isfile(os.path.join(d, fn)) and os.path.abspath(d) != os.path.abspath(rec):
                shutil.copy2(os.path.join(d, fn), os.path.join(rec, fn))
    print(f"  warm templates ready ({(time.time() - t0) / 60:.1f} min)\n", flush=True)

    runs.sort(key=lambda r: (-MESH_COST[r["mesh"]], -r.get("pulse_w_hi", r["w_hi"])))   # slowest first
    with ThreadPoolExecutor(max_workers=args.workers or len(runs)) as ex:
        futs = [ex.submit(M.run_episode, r, cfg, {"nu1": tpls.get(r["mesh"])}, results_dir, work_root,
                          args.resume) for r in runs]
        for fut in as_completed(futs):
            fut.result()
    print(f"\nall runs done in {(time.time() - t0) / 3600:.2f} h", flush=True)
    analyze(results_dir, args.smoke)


if __name__ == "__main__":
    main()
