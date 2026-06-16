#!/usr/bin/env python3
"""
Score & compare mixing for the oscillating-vs-constant omega runs.

Reads the METRICS lines that rlMetrics prints (constant.log, squarewave.log),
each of the form:
    METRICS t=<s> Mz_kin=<..> C0=.. C1=.. .. C19=.. Vz0=.. .. Vz19=..
C0..C19 are the dye concentration in 20 radial bins at the bottom outlet.

Mixing measure: radial uniformity of that outlet profile.
  unmixedness(t) = std_dev(C0..C19)           (0 = perfectly mixed)
  mixing_index(t) = 1 - std/std_inlet         (1 = perfectly mixed, 0 = as
                    injected; std_inlet = 0.4330 for the inner-1/4 dye band)

Prints the time-averaged mixing index over the last `--avg-window` seconds
(steady-ish tail) for each run and says which mixes better, and writes
mixing_comparison.png.

Usage:  python analyze.py results [--avg-window 60]
"""
import argparse
import glob
import math
import os
import re

STD_INLET = math.sqrt(0.1875)   # std of the 5/20-on inner-quarter dye band = 0.4330

TOK = re.compile(r"(\w+)=(-?[\d.eE+-]+)")


def parse_log(path):
    """-> (times[list], bins[list of 20-float lists])."""
    times, bins = [], []
    with open(path) as f:
        for line in f:
            if not line.startswith("METRICS"):
                continue
            kv = dict(TOK.findall(line))
            try:
                t = float(kv["t"])
                row = [float(kv[f"C{b}"]) for b in range(20)]
            except (KeyError, ValueError):
                continue
            times.append(t)
            bins.append(row)
    return times, bins


def std(row):
    n = len(row)
    m = sum(row) / n
    return math.sqrt(sum((x - m) ** 2 for x in row) / n)


def summarize(name, times, bins, avg_window):
    if not times:
        print(f"  {name:<11s}: no METRICS data")
        return None
    unmix = [std(r) for r in bins]
    midx = [1.0 - s / STD_INLET for s in unmix]
    t_end = times[-1]
    tail = [m for t, m in zip(times, midx) if t >= t_end - avg_window]
    tail_unmix = [s for t, s in zip(times, unmix) if t >= t_end - avg_window]
    mean_midx = sum(tail) / len(tail)
    mean_unmix = sum(tail_unmix) / len(tail_unmix)
    print(f"  {name:<11s}: tail-avg mixing_index={mean_midx:+.4f}  "
          f"(unmixedness={mean_unmix:.4f})  over last {avg_window:.0f}s of {t_end:.0f}s")
    return dict(name=name, times=times, midx=midx, unmix=unmix, mean_midx=mean_midx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--avg-window", type=float, default=60.0,
                    help="seconds at the end to time-average over")
    args = ap.parse_args()

    runs = {}
    for path in sorted(glob.glob(os.path.join(args.results_dir, "*.log"))):
        name = os.path.splitext(os.path.basename(path))[0]
        runs[name] = parse_log(path)

    if not runs:
        raise SystemExit(f"no *.log found in {args.results_dir}")

    print(f"Mixing comparison (higher mixing_index = better mixed):")
    summaries = []
    for name, (t, b) in runs.items():
        s = summarize(name, t, b, args.avg_window)
        if s:
            summaries.append(s)

    if {"constant", "squarewave"} <= {s["name"] for s in summaries}:
        c = next(s for s in summaries if s["name"] == "constant")
        q = next(s for s in summaries if s["name"] == "squarewave")
        d = q["mean_midx"] - c["mean_midx"]
        verdict = "squarewave mixes BETTER" if d > 0 else "constant mixes better"
        print(f"\n  -> {verdict} by {abs(d):.4f} mixing-index "
              f"({'+' if d>0 else ''}{d/abs(c['mean_midx'] or 1)*100:.1f}% vs constant)")

    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib unavailable -- skipping plot)")
        return

    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for s in summaries:
        ax[0].plot(s["times"], s["midx"], label=s["name"])
        ax[1].plot(s["times"], s["unmix"], label=s["name"])
    ax[0].set_ylabel("mixing index  (1 = mixed)")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].set_ylabel("unmixedness  std(C bins)")
    ax[1].set_xlabel("time [s]")
    ax[1].grid(alpha=0.3)
    ax[0].set_title("Outlet radial mixing: oscillating vs constant omega")
    out = os.path.join(args.results_dir, "mixing_comparison.png")
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
