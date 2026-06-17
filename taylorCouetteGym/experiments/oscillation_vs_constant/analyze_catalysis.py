#!/usr/bin/env python3
"""
Score the catalysis runs the way the paper does: CONVERSION vs ENERGY.

Reads catalysis METRICS logs (fields: t, Mz_kin, conv, cupC, wallFlux, C0..C19),
reconstructs omega(t) per controller to get power P = rho*Mz_kin*omega, and reports
for each controller:
  - time-averaged conversion over the comparison window (higher = better),
  - cumulative input energy (lower = cheaper),
  - conversion-per-energy (the efficiency the paper's "more conversion, less power"
    headline is about).

Writes fig_conversion_ts.png (conversion vs time) and fig_conversion_vs_energy.png
(the 2-axis comparison: where does each controller sit?).

Usage: python analyze_catalysis.py results_catalysis [--period 30] [--duty 0.2]
                                    [--mean-rpm 500] [--window TMIN TMAX]
"""
import argparse, glob, math, os, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RPM = 2 * math.pi / 60
RHO = 930.0                      # silicone oil density (energy is proportional anyway)
TOK = re.compile(r"(\w+)=(-?[\d.eE+-]+)")
COLORS = {"constant": "#1f77b4", "squarewave": "#d62728"}


def parse(path):
    t, mz, conv = [], [], []
    for ln in open(path):
        if not ln.startswith("METRICS"):
            continue
        kv = dict(TOK.findall(ln))
        try:
            t.append(float(kv["t"])); mz.append(float(kv["Mz_kin"])); conv.append(float(kv["conv"]))
        except (KeyError, ValueError):
            continue
    return np.array(t), np.array(mz), np.array(conv)


def omega_rad(name, t, mean_rpm, duty, period):
    """Reconstruct the prescribed omega(t) [rad/s] for power."""
    if name == "constant":
        return np.full_like(t, mean_rpm * RPM)
    hi = (mean_rpm / duty) * RPM                 # active value
    phase = np.mod(t, period)
    return np.where(phase < duty * period, hi, 0.0)


def active_spans(t1, period, duty):
    return [(k * period, k * period + duty * period)
            for k in range(int(t1 // period) + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--period", type=float, default=30.0)
    ap.add_argument("--duty", type=float, default=0.2)
    ap.add_argument("--mean-rpm", type=float, default=500.0)
    ap.add_argument("--window", type=float, nargs=2, default=None,
                    help="TMIN TMAX to average over (default: last full period both reached)")
    args = ap.parse_args()

    runs = {}
    for p in sorted(glob.glob(os.path.join(args.results_dir, "*.log"))):
        name = os.path.splitext(os.path.basename(p))[0]
        d = parse(p)
        if len(d[0]):
            runs[name] = d
    if not runs:
        raise SystemExit(f"no usable *.log in {args.results_dir}")
    order = [n for n in ("constant", "squarewave") if n in runs] or list(runs)
    tcommon = min(runs[n][0].max() for n in order)
    w0, w1 = args.window if args.window else (max(0.0, tcommon - args.period), tcommon)

    print(f"\nCATALYSIS comparison over t in [{w0:.1f}, {w1:.1f}] s "
          f"(higher conversion + lower energy = better):\n")
    summary = []
    for name in order:
        t, mz, conv = runs[name]
        w = (t >= w0) & (t <= w1)
        om = omega_rad(name, t, args.mean_rpm, args.duty, args.period)
        power = RHO * mz * om                        # instantaneous input power (~)
        energy = np.trapezoid(np.abs(power[w]), t[w])  # input energy over the window (np 2.0)
        mconv = conv[w].mean()
        eff = mconv / energy if energy > 0 else float("nan")
        summary.append(dict(name=name, mconv=mconv, energy=energy, eff=eff))
        print(f"  {name:11s}  conversion={mconv:6.3f}   energy={energy:10.4g}   "
              f"conv/energy={eff:10.4g}")

    if len(summary) == 2:
        c = next(s for s in summary if s["name"] == "constant")
        q = next(s for s in summary if s["name"] == "squarewave")
        dconv = (q["mconv"] - c["mconv"]) / abs(c["mconv"]) * 100 if c["mconv"] else float("nan")
        dener = (q["energy"] - c["energy"]) / c["energy"] * 100 if c["energy"] else float("nan")
        print(f"\n  squarewave vs constant:  conversion {dconv:+.1f}%   energy {dener:+.1f}%")
        verdict = ("RECREATES the paper (more conversion AND less energy)"
                   if dconv > 0 and dener < 0 else
                   "more conversion but more energy" if dconv > 0 else
                   "does NOT beat constant on conversion")
        print(f"  -> {verdict}")

    # ---- conversion vs time ----
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sq = runs.get("squarewave")
    if sq is not None:
        for i, (a, b) in enumerate(active_spans(sq[0].max(), args.period, args.duty)):
            ax.axvspan(a, b, color="0.88", label="active (2500 rpm)" if i == 0 else None)
    for name in order:
        t, _, conv = runs[name]
        ax.plot(t, conv, color=COLORS.get(name), lw=1.3, label=name)
    ax.axvspan(w0, w1, color="gold", alpha=0.15, label="averaging window")
    ax.set_xlabel("time [s]"); ax.set_ylabel("conversion  (1 - cup outlet C)")
    ax.grid(alpha=0.3); ax.legend(loc="best")
    ax.set_title("Catalytic conversion vs time")
    fig.savefig(os.path.join(args.results_dir, "fig_conversion_ts.png"),
                dpi=140, bbox_inches="tight"); plt.close(fig)

    # ---- conversion vs energy (the paper's 2 axes) ----
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for s in summary:
        ax.scatter(s["energy"], s["mconv"], s=160, color=COLORS.get(s["name"]),
                   zorder=3, label=s["name"])
        ax.annotate(s["name"], (s["energy"], s["mconv"]),
                    textcoords="offset points", xytext=(8, 6))
    ax.set_xlabel("input energy over window  (~ J, proportional)")
    ax.set_ylabel("time-avg conversion")
    ax.grid(alpha=0.3)
    ax.set_title("Conversion vs energy — upper-left is better\n"
                 "(RL's target: beat both baselines toward the upper-left)")
    fig.savefig(os.path.join(args.results_dir, "fig_conversion_vs_energy.png"),
                dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"\nwrote {args.results_dir}/fig_conversion_ts.png and fig_conversion_vs_energy.png")


if __name__ == "__main__":
    main()
