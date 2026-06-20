#!/usr/bin/env python3
"""
Score & compare mixing for the oscillating-vs-constant omega runs.

Reads the METRICS lines that rlMetrics prints (constant.log, squarewave.log),
each of the form:
    METRICS t=<s> Mz_kin=<..> C0=.. C1=.. .. C19=.. Vz0=.. .. Vz19=..
C0..C19 are the dye concentration in 20 radial bins at the bottom outlet.

Mixing measure: intensity of segregation of the outlet radial profile.
  sigma(t) = std_dev(C0..C19)                 (0 = perfectly mixed)
  I_mix(t) = sigma^2 / sigma_max^2            (0 = perfectly mixed, 1 = as
             injected; sigma_max = 0.4330 for the inner-1/4 dye band)
LOWER I_mix = better mixed.

Prints the time-averaged I_mix over the last `--avg-window` seconds
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
    imix = [(s / STD_INLET) ** 2 for s in unmix]   # intensity of segregation
    t_end = times[-1]
    tail = [m for t, m in zip(times, imix) if t >= t_end - avg_window]
    tail_unmix = [s for t, s in zip(times, unmix) if t >= t_end - avg_window]
    mean_imix = sum(tail) / len(tail)
    mean_unmix = sum(tail_unmix) / len(tail_unmix)
    print(f"  {name:<11s}: tail-avg I_mix={mean_imix:.4f}  "
          f"(sigma={mean_unmix:.4f})  over last {avg_window:.0f}s of {t_end:.0f}s")
    return dict(name=name, times=times, imix=imix, unmix=unmix, mean_imix=mean_imix)


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

    print("Mixing comparison (LOWER I_mix = sigma^2/sigma_max^2 = better mixed):")
    summaries = []
    for name, (t, b) in runs.items():
        s = summarize(name, t, b, args.avg_window)
        if s:
            summaries.append(s)

    if {"constant", "squarewave"} <= {s["name"] for s in summaries}:
        c = next(s for s in summaries if s["name"] == "constant")
        q = next(s for s in summaries if s["name"] == "squarewave")
        d = q["mean_imix"] - c["mean_imix"]          # <0 => squarewave more mixed
        verdict = "squarewave mixes BETTER" if d < 0 else "constant mixes better"
        print(f"\n  -> {verdict} by {abs(d):.4f} in I_mix "
              f"(squarewave I_mix {q['mean_imix']:.4f} vs constant {c['mean_imix']:.4f})")

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
        ax[0].plot(s["times"], s["imix"], label=s["name"])
        ax[1].plot(s["times"], s["unmix"], label=s["name"])
    ax[0].set_ylabel(r"$I_{mix}=\sigma^2/\sigma_{\max}^2$  (0 = mixed)")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].set_ylabel(r"$\sigma$ = std(C bins)")
    ax[1].set_xlabel("time [s]")
    ax[1].grid(alpha=0.3)
    ax[0].set_title("Outlet radial mixing: oscillating vs constant omega")
    out = os.path.join(args.results_dir, "mixing_comparison.png")
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
