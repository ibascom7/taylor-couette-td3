#!/usr/bin/env python3
"""
Aggregate several hysteresis_sweep runs (different --settle-seconds) into the
RATE-DEPENDENCE diagnostic that distinguishes equilibrium hysteresis from a merely
dynamic (transient) loop:

  * loop AREA vs settle time, in both the conv-vs-Re and the conv-vs-power planes.
      area -> a nonzero plateau as settle grows  => TRUE (equilibrium) hysteresis
      area -> 0          as settle grows          => DYNAMIC hysteresis (lag only)
  * an overlay of every run's up/down branches so you can see the loop shrink (or
    not) as the ramp is made more quasi-static.

The loop area is computed as the closed-polygon (shoelace) area of the up branch
followed by the reversed down branch, interpolated onto the shared Re grid -- the
signed area's magnitude, in (conversion x Reynolds) and (conversion x Watt) units.

Usage:
  python analyze_hysteresis.py results/h15 results/h40 results/h90
  python analyze_hysteresis.py results/h*          # shell-expanded
"""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(run_dir):
    path = os.path.join(run_dir, "hysteresis_branches.csv")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k != "branch":
                r[k] = float(v)
    up = [r for r in rows if r["branch"] == "up"]
    down = [r for r in rows if r["branch"] == "down"]
    # settle time is encoded in the dir name hNN (fallback: span/levels)
    base = os.path.basename(os.path.normpath(run_dir))
    settle = float(base[1:]) if base.startswith("h") and base[1:].replace(".", "").isdigit() \
        else float("nan")
    return dict(dir=run_dir, settle=settle, up=up, down=down)


def loop_area(up, down, xkey, ykey):
    """Magnitude of the closed-loop area between the up and down branches in the
    (xkey, ykey) plane. Interpolate both onto a shared x grid, integrate the gap."""
    xu = np.array([r[xkey] for r in up]); yu = np.array([r[ykey] for r in up])
    xd = np.array([r[xkey] for r in down]); yd = np.array([r[ykey] for r in down])
    if len(xu) < 2 or len(xd) < 2:
        return float("nan")
    ou, od = np.argsort(xu), np.argsort(xd)
    xu, yu = xu[ou], yu[ou]; xd, yd = xd[od], yd[od]
    lo, hi = max(xu.min(), xd.min()), min(xu.max(), xd.max())
    if hi <= lo:
        return float("nan")
    g = np.linspace(lo, hi, 300)
    return float(np.trapezoid(np.abs(np.interp(g, xu, yu) - np.interp(g, xd, yd)), g))


def main():
    run_dirs = sys.argv[1:]
    if not run_dirs:
        sys.exit("usage: analyze_hysteresis.py <run_dir> [<run_dir> ...]")
    runs = sorted((load(d) for d in run_dirs), key=lambda r: r["settle"])
    out = os.path.commonpath([os.path.dirname(os.path.normpath(d)) for d in run_dirs]) \
        if len(run_dirs) > 1 else os.path.dirname(os.path.normpath(run_dirs[0]))

    # ---- loop area vs settle time --------------------------------------
    settles = [r["settle"] for r in runs]
    area_re = [loop_area(r["up"], r["down"], "Re_rot", "conv") for r in runs]
    area_pw = [loop_area(r["up"], r["down"], "motor_P", "conv") for r in runs]

    print("settle[s]   loop area (conv*Re)   loop area (conv*W)")
    for s, a1, a2 in zip(settles, area_re, area_pw):
        print(f"  {s:6.0f}      {a1:14.3f}      {a2:12.4f}")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(settles, area_re, "o-", color="#1f77b4", label="conv-vs-Re loop")
    ax2 = ax.twinx()
    ax2.plot(settles, area_pw, "s--", color="#d62728", label="conv-vs-power loop")
    ax.set_xlabel("settle time per level [s]  (more quasi-static ->)")
    ax.set_ylabel("loop area  (conversion x Re)", color="#1f77b4")
    ax2.set_ylabel("loop area  (conversion x W)", color="#d62728")
    ax.set_title("Hysteresis loop area vs settle time\n"
                 "plateau => equilibrium bistability;  -> 0 => dynamic (transient) only")
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(out, "fig_loop_area_vs_rate.png"), dpi=140,
                bbox_inches="tight"); plt.close(fig)

    # ---- overlay of all branches in the conv-vs-power plane ------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    cmap = plt.cm.viridis
    for k, r in enumerate(runs):
        c = cmap(k / max(1, len(runs) - 1))
        up, down = r["up"], r["down"]
        ax.plot([x["motor_P"] for x in up], [x["conv"] for x in up], "o-", color=c,
                label=f"settle={r['settle']:.0f}s")
        ax.plot([x["motor_P"] for x in down], [x["conv"] for x in down], "s--", color=c)
    ax.set_xlabel("mean motor power [W]"); ax.set_ylabel("conversion (1 - cup C)")
    ax.set_title("Conversion vs power: up (solid) / down (dashed) at each settle time\n"
                 "loop shrinking with settle = dynamic; persistent = bistable")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.savefig(os.path.join(out, "fig_branches_overlay.png"), dpi=140,
                bbox_inches="tight"); plt.close(fig)

    print(f"\nwrote fig_loop_area_vs_rate.png + fig_branches_overlay.png to {out}")


if __name__ == "__main__":
    main()
