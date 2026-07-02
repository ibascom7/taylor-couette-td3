#!/usr/bin/env python3
"""
Conversion-vs-Power ATLAS: place every SUCCESSFUL TD3 run on one
conversion (y) vs mean motor power [W] (x) plane, to scope the region of
operating points RL discovers -- and to expose a bistable flow-regime transition.

Power is RECOMPUTED from each run's omega_per_step.npy via the paper's electric-
motor model (motor_power.average_power, Eqs 18-23), NOT read from a log. So every
run lands on ONE uniform power basis -- including runs trained before per-step
power logging existed (e.g. so_parallel_freeform_s0 / job 7596936). Conversion
comes from conv_per_step.npy. Neither requires re-simulating.

Operating point per run = the CONVERGED tail: over the last --last-n completed
episodes, average conversion and motor power over the final --tail-frac of each
episode (skips the residence-time fill transient). One point per run, colored by
energy_weight. --cloud adds the faint per-step (power, conv) trajectory; a policy
that MODULATES traces a loop whose time-mean point sits inside it. --hysteresis
overlays the up/down conv-vs-power branches from experiments/hysteresis as a
regime backbone: the bistable band is the gap between the branches, and the RL
points reveal whether the learned policies exploit it.

WHICH runs: a curated MANIFEST you bless one at a time (idle-collapse / diverged
runs stay out), not an auto-scan of results/.

  # catalog a run (energy_weight/seed parsed from the tag if not passed):
  python plot_conversion_vs_power.py --add ../../results/td3/so_parallel_freeform_ew0.2_s0
  # (re)draw the atlas from everything cataloged so far:
  python plot_conversion_vs_power.py
  # with the regime backbone + per-step clouds:
  python plot_conversion_vs_power.py --hysteresis ../hysteresis/results/h90 --cloud
  # preview candidate runs under a results tree WITHOUT cataloging them:
  python plot_conversion_vs_power.py --scan ../../results/td3
"""
import argparse
import csv
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# marker per initial-condition (two-IC probe): idle vs spin, so a bistable ew reads as a
# same-color circle-low / square-high pair. Unknown IC -> circle.
IC_MARKER = {"idle": "o", "spin": "s", None: "o"}

# The motor model (Lopez-Guajardo Eqs 18-23). Prefer the package import used by the
# other experiment scripts; fall back to loading the numpy-only module directly, since
# the package __init__ pulls in gymnasium/OpenFOAM that this pure post-processing tool
# does not need (lets the atlas run anywhere with just numpy + matplotlib).
GYM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if GYM_ROOT not in sys.path:
    sys.path.insert(0, GYM_ROOT)
try:
    from taylor_couette_mixing import motor_power  # noqa: E402
except Exception:
    import importlib.util
    _mp = os.path.join(GYM_ROOT, "taylor_couette_mixing", "motor_power.py")
    _spec = importlib.util.spec_from_file_location("motor_power", _mp)
    motor_power = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(motor_power)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MANIFEST = os.path.join(HERE, "manifest.csv")
MANIFEST_FIELDS = ["run_dir", "label", "energy_weight", "seed",
                   "dt", "tail_frac", "last_n", "notes"]
RPM_TO_RADS = 2.0 * np.pi / 60.0

# Operating-point defaults (a manifest row may override any of these per run).
DEF_DT = 1.0        # seconds per control step (freeform_dt)
DEF_TAIL = 0.5      # use the final 50% of each episode (past the fill transient)
DEF_LAST_N = 5      # average over the last 5 completed episodes


# --------------------------------------------------------------------------- #
# small parsers / helpers
# --------------------------------------------------------------------------- #
def _num(x, default, cast=float):
    """Cast a manifest cell to a number, falling back to default on ''/None/bad."""
    if x is None or x == "":
        return default
    try:
        return cast(x)
    except (TypeError, ValueError):
        return default


def parse_ic(name):
    """Pull the two-IC label ('idle'/'spin') out of a tag like '..._icidle_s0'; None if
    absent (runs from before the two-IC probe)."""
    n = (name or "").lower()
    if "icidle" in n:
        return "idle"
    if "icspin" in n:
        return "spin"
    return None


def parse_tag(run_dir):
    """Pull (energy_weight, seed) out of a tag like '..._ew0.2_s0'; None if absent."""
    base = os.path.basename(os.path.normpath(run_dir))
    ew = re.search(r"ew([0-9]*\.?[0-9]+)", base)
    sd = re.search(r"_s([0-9]+)", base)
    return (float(ew.group(1)) if ew else None,
            int(sd.group(1)) if sd else None,
            base)


def resolve(run_dir, extra_bases=()):
    """Find a run dir whether the manifest stored it absolute, relative to CWD, to the
    manifest's own directory, or to the repo root (the two-checkout /home vs /project
    split means stored paths are best kept relative)."""
    for base in ("", *extra_bases, GYM_ROOT, os.getcwd()):
        cand = run_dir if base == "" else os.path.join(base, run_dir)
        if os.path.isdir(cand):
            return os.path.abspath(cand)
    return None


def resolve_file(path, extra_bases=()):
    """Like resolve() but for a file (e.g. a baseline_sweep.npz)."""
    for base in ("", *extra_bases, GYM_ROOT, os.getcwd()):
        cand = path if base == "" else os.path.join(base, path)
        if os.path.isfile(cand):
            return os.path.abspath(cand)
    return None


# --------------------------------------------------------------------------- #
# operating point: recompute power from omega, read conversion from log
# --------------------------------------------------------------------------- #
def _completed_rows(grid, n_complete):
    """[ep, step] grid for COMPLETED episodes only (drop the in-progress last row
    that save_logs appends so periodic saves capture the latest steps)."""
    g = np.asarray(grid, float)
    if g.ndim == 1:
        g = g[None, :]
    if n_complete is not None and 0 < n_complete <= g.shape[0]:
        g = g[:n_complete]
    return g


def _tail(row, tail_frac):
    """Trailing tail of one episode row: strip NaN pad, keep the last tail_frac."""
    r = row[~np.isnan(row)]
    if r.size == 0:
        return r
    k = max(1, int(round(r.size * tail_frac)))
    return r[-k:]


def operating_point(run_dir, dt=DEF_DT, tail_frac=DEF_TAIL, last_n=DEF_LAST_N):
    """Converged (power_W, conversion) point for a run from its per-step logs, or
    None if the run lacks omega/conv logs. Power is recomputed from omega via the
    motor model, so this works uniformly on every run (no re-sim, no power log)."""
    op = os.path.join(run_dir, "omega_per_step.npy")
    cp = os.path.join(run_dir, "conv_per_step.npy")
    if not (os.path.exists(op) and os.path.exists(cp)):
        return None
    omega = np.load(op)            # rpm, [ep, step]
    conv = np.load(cp)             # conversion, [ep, step]
    rp = os.path.join(run_dir, "episode_returns.npy")
    n_complete = int(np.load(rp).shape[0]) if os.path.exists(rp) else None
    omega = _completed_rows(omega, n_complete)
    conv = _completed_rows(conv, n_complete)
    n_ep = min(omega.shape[0], conv.shape[0])
    if n_ep == 0:
        return None
    sel = range(max(0, n_ep - last_n), n_ep)

    p_means, c_means, cloud_p, cloud_c = [], [], [], []
    for e in sel:
        w_tail = _tail(omega[e], tail_frac) * RPM_TO_RADS   # rad/s
        c_tail = _tail(conv[e], tail_frac)
        m = min(w_tail.size, c_tail.size)
        if m < 2:                          # motor model needs a >=2-point trace
            continue
        w_tail, c_tail = w_tail[-m:], c_tail[-m:]
        t = np.arange(m) * dt
        p_means.append(motor_power.average_power(t, w_tail))
        c_means.append(float(np.nanmean(c_tail)))
        cloud_p.append(motor_power.electrical_power(t, w_tail))  # pointwise W
        cloud_c.append(c_tail)
    if not p_means:
        return None
    return dict(
        power_mean=float(np.nanmean(p_means)), power_std=float(np.nanstd(p_means)),
        conv_mean=float(np.nanmean(c_means)), conv_std=float(np.nanstd(c_means)),
        cloud_power=np.concatenate(cloud_p), cloud_conv=np.concatenate(cloud_c),
        n_eps=len(p_means),
    )


# --------------------------------------------------------------------------- #
# manifest (the curated catalog of blessed runs)
# --------------------------------------------------------------------------- #
def read_manifest(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_manifest(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in MANIFEST_FIELDS})


def add_run(path, run_dir, label, energy_weight, seed, dt, tail_frac, last_n, notes):
    """Append (or replace, keyed by run_dir) one blessed run in the manifest.
    Only explicitly-passed dt/tail_frac/last_n are stored; blanks let the plot use
    its global defaults."""
    run_dir = os.path.normpath(run_dir)
    if resolve(run_dir) is None:
        print(f"  WARNING: '{run_dir}' not found from here -- cataloging anyway, but "
              f"the plot will SKIP it unless the path is right (it should be relative "
              f"to this directory, or absolute).")
    ew_tag, sd_tag, base = parse_tag(run_dir)
    ew = energy_weight if energy_weight is not None else ew_tag
    sd = seed if seed is not None else sd_tag
    row = dict(
        run_dir=run_dir,
        label=label or base,
        energy_weight="" if ew is None else ew,
        seed="" if sd is None else sd,
        dt="" if dt is None else dt,
        tail_frac="" if tail_frac is None else tail_frac,
        last_n="" if last_n is None else last_n,
        notes=notes or "",
    )
    rows = [r for r in read_manifest(path)
            if os.path.normpath(r["run_dir"]) != run_dir]   # replace if present
    rows.append(row)
    write_manifest(path, rows)
    return row


# --------------------------------------------------------------------------- #
# hysteresis backbone overlay (optional)
# --------------------------------------------------------------------------- #
def load_hysteresis(run_dir):
    """up/down branches from a hysteresis_sweep run (experiments/hysteresis), in the
    same conv-vs-motor_P convention analyze_hysteresis.py uses. None if absent."""
    path = os.path.join(run_dir, "hysteresis_branches.csv")
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k != "branch":
                try:
                    r[k] = float(v)
                except (TypeError, ValueError):
                    pass
    up = [r for r in rows if r.get("branch") == "up"]
    down = [r for r in rows if r.get("branch") == "down"]
    if not up or not down or "motor_P" not in up[0] or "conv" not in up[0]:
        return None
    return dict(up=up, down=down)


def load_baselines(npz_path):
    """Constant + pulsating conv-vs-power baselines from compare_catalysis.py's
    baseline_sweep.npz. const_motor/const_conv (and puls_*) are avg motor power [W]
    vs windowed conversion -- the SAME motor_power basis as the atlas, so directly
    overlayable. Returns {const:(W,conv), puls:(W,conv)} or None."""
    if not os.path.exists(npz_path):
        return None
    try:
        z = np.load(npz_path, allow_pickle=True)
    except Exception:
        return None
    out = {}
    if "const_motor" in z.files and "const_conv" in z.files:
        cm = np.asarray(z["const_motor"], float)
        cc = np.asarray(z["const_conv"], float)
        o = np.argsort(cm)                       # sort by power for a clean line
        out["const"] = (cm[o], cc[o])
    if "puls_motor" in z.files and "puls_conv" in z.files:
        out["puls"] = (np.asarray(z["puls_motor"], float),
                       np.asarray(z["puls_conv"], float))
    return out or None


# --------------------------------------------------------------------------- #
# plot / scan
# --------------------------------------------------------------------------- #
def _color_fn(pts):
    """A color-by-energy_weight function + optional colorbar norm."""
    ews = [p["energy_weight"] for p in pts if p["energy_weight"] is not None]
    cmap = plt.get_cmap("viridis")
    if ews:
        lo, hi = min(ews), max(ews)
        norm = mcolors.Normalize(lo, hi if hi > lo else lo + 1.0)
    else:
        norm = None

    def color_of(p):
        if p["energy_weight"] is None or norm is None:
            return "0.35"
        return cmap(norm(p["energy_weight"]))

    return color_of, cmap, norm


def plot(manifest_path, out, dt, tail_frac, last_n, cloud, hysteresis, baselines=None):
    rows = read_manifest(manifest_path)
    if not rows:
        print(f"manifest empty: {manifest_path}\n  add runs with:  --add <run_dir>")
        return
    # Anchor relative run_dirs to the manifest's own directory too, so a manifest
    # stays valid no matter which directory the plot is invoked from.
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    pts = []
    for r in rows:
        rd = resolve(r["run_dir"], extra_bases=[manifest_dir])
        if rd is None:
            print(f"  skip (dir not found): {r['run_dir']}")
            continue
        op = operating_point(
            rd,
            dt=_num(r.get("dt"), dt),
            tail_frac=_num(r.get("tail_frac"), tail_frac),
            last_n=_num(r.get("last_n"), last_n, cast=int),
        )
        if op is None:
            print(f"  skip (no usable logs): {r['run_dir']}")
            continue
        op["label"] = r.get("label") or os.path.basename(rd)
        op["energy_weight"] = _num(r.get("energy_weight"), None)
        op["ic"] = parse_ic(r["run_dir"]) or parse_ic(op["label"])
        pts.append(op)
    if not pts:
        print("no plottable runs in manifest.")
        return

    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    # constant/pulsating baseline backbone (compare_catalysis.py's baseline_sweep.npz)
    bl = load_baselines(resolve_file(baselines, extra_bases=[manifest_dir])) \
        if baselines else None
    if baselines and bl is None:
        print(f"  baselines overlay skipped (unreadable/absent npz): {baselines}")
    if bl:
        if "const" in bl:
            cm, cc = bl["const"]
            ax.plot(cm, cc, "-D", color="k", lw=1.8, ms=5, zorder=3,
                    label="constant $\\omega$ sweep")
        if "puls" in bl:
            pm, pc = bl["puls"]
            ax.plot(pm, pc, "P", color="0.35", ms=9, mec="k", mew=0.5,
                    ls="none", zorder=3, label="pulsating (squarewave)")

    # regime backbone first (behind the RL points)
    for i, hdir in enumerate(hysteresis or []):
        hd = resolve(hdir, extra_bases=[manifest_dir])
        br = load_hysteresis(hd) if hd else None
        if br is None:
            print(f"  hysteresis overlay skipped (no hysteresis_branches.csv): {hdir}")
            continue
        ax.plot([x["motor_P"] for x in br["up"]], [x["conv"] for x in br["up"]],
                "-", color="0.55", lw=1.7, zorder=1,
                label=("hysteresis up" if i == 0 else None))
        ax.plot([x["motor_P"] for x in br["down"]], [x["conv"] for x in br["down"]],
                "--", color="0.55", lw=1.7, zorder=1,
                label=("hysteresis down" if i == 0 else None))

    color_of, cmap, norm = _color_fn(pts)

    if cloud:
        for p in pts:
            ax.scatter(p["cloud_power"], p["cloud_conv"], s=6, alpha=0.08,
                       color=color_of(p), linewidths=0, zorder=2)

    for p in pts:
        c = color_of(p)
        ax.errorbar(p["power_mean"], p["conv_mean"],
                    xerr=p["power_std"], yerr=p["conv_std"],
                    fmt=IC_MARKER.get(p.get("ic"), "o"), ms=9, color=c, ecolor=c,
                    elinewidth=1, capsize=3, mec="k", mew=0.6, zorder=4)
        ax.annotate(p["label"], (p["power_mean"], p["conv_mean"]),
                    textcoords="offset points", xytext=(7, 4),
                    fontsize=7.5, color="0.15", zorder=5)

    if norm is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax).set_label("energy_weight (reward knob)")

    ax.set_xlabel("mean motor power  [W]   (Lopez-Guajardo motor model, Eqs 18-23)")
    ax.set_ylabel("converged conversion")
    ax.set_title(f"Conversion vs power -- {len(pts)} successful TD3 run(s)\n"
                 f"converged tail: last {last_n} ep x final {int(tail_frac * 100)}% of steps")
    ax.grid(alpha=0.3)
    # legend = baseline/hysteresis auto-labels + a marker key for the two-IC probe
    handles, labels = ax.get_legend_handles_labels()
    for k in sorted(ic for ic in {p.get("ic") for p in pts} if ic):
        handles.append(Line2D([0], [0], marker=IC_MARKER[k], color="0.3", ls="none",
                              mec="k", mew=0.6, ms=9))
        labels.append(f"{k} IC")
    if handles:
        ax.legend(handles, labels, loc="best", fontsize=8)
    fig.tight_layout()
    out = out or os.path.join(os.path.dirname(manifest_path) or ".",
                              "conversion_vs_power.png")
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")
    print("\n  power[W]    conv    n_ep  label")
    for p in sorted(pts, key=lambda q: q["power_mean"]):
        print(f"  {p['power_mean']:8.2f}  {p['conv_mean']:.4f}  {p['n_eps']:>4}  {p['label']}")


def scan(root, dt, tail_frac, last_n):
    """Preview every run dir under ROOT with per-step logs -- no cataloging."""
    root = resolve(root) or root
    found = []
    for dirpath, _dirs, files in os.walk(root):
        if "omega_per_step.npy" in files and "conv_per_step.npy" in files:
            found.append(dirpath)
    print(f"scanning {root}: {len(found)} run dir(s) with per-step logs\n")
    print("  power[W]    conv    n_ep  run_dir")
    for rd in sorted(found):
        op = operating_point(rd, dt=dt, tail_frac=tail_frac, last_n=last_n)
        if op is None:
            print(f"      --        --      --  {rd}  (no usable logs)")
            continue
        print(f"  {op['power_mean']:8.2f}  {op['conv_mean']:.4f}  {op['n_eps']:>4}  {rd}")
    print("\ncatalog a good one with:  --add <run_dir>")


def main():
    ap = argparse.ArgumentParser(
        description="Conversion-vs-power atlas of successful TD3 runs.")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST,
                    help=f"catalog CSV (default: {DEFAULT_MANIFEST})")
    ap.add_argument("--add", metavar="RUN_DIR", help="catalog a run, then exit")
    ap.add_argument("--scan", metavar="ROOT",
                    help="list candidate runs under ROOT (does not catalog)")
    ap.add_argument("--out", default=None, help="output png (default: beside manifest)")
    # operating-point knobs (None => use DEF_*; a manifest row can override per run)
    ap.add_argument("--dt", type=float, default=None,
                    help=f"seconds per control step / freeform_dt (default {DEF_DT})")
    ap.add_argument("--tail-frac", type=float, default=None,
                    help=f"final fraction of each episode to average (default {DEF_TAIL})")
    ap.add_argument("--last-n", type=int, default=None,
                    help=f"average over the last N completed episodes (default {DEF_LAST_N})")
    ap.add_argument("--cloud", action="store_true",
                    help="also draw faint per-step (power, conv) clouds")
    ap.add_argument("--hysteresis", action="append", default=[], metavar="RUN_DIR",
                    help="hysteresis run dir to overlay as regime backbone (repeatable)")
    ap.add_argument("--baselines", metavar="NPZ", default=None,
                    help="compare_catalysis.py baseline_sweep.npz to overlay the "
                         "constant-omega (+pulsating) conv-vs-power backbone")
    # --add metadata
    ap.add_argument("--label")
    ap.add_argument("--energy-weight", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--notes")
    args = ap.parse_args()

    dt = args.dt if args.dt is not None else DEF_DT
    tail = args.tail_frac if args.tail_frac is not None else DEF_TAIL
    last_n = args.last_n if args.last_n is not None else DEF_LAST_N

    if args.add:
        row = add_run(args.manifest, args.add, args.label, args.energy_weight,
                      args.seed, args.dt, args.tail_frac, args.last_n, args.notes)
        print(f"cataloged -> {args.manifest}\n  {row}")
        return
    if args.scan:
        scan(args.scan, dt, tail, last_n)
        return
    plot(args.manifest, args.out, dt, tail, last_n, args.cloud, args.hysteresis,
         baselines=args.baselines)


if __name__ == "__main__":
    main()
