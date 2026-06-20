#!/usr/bin/env python3
"""
Presentation figures for the oscillating-vs-constant omega mixing study.

Reads the METRICS logs (constant.log, squarewave.log) in a results dir and
writes four PNGs telling the story honestly:

  fig_spacetime.png   space-time map of the outlet radial dye profile C(bin, t),
                      one panel per case -- shows the modulation dynamics vividly.
  fig_mixing_ts.png   intensity of segregation I_mix=sigma^2/sigma_max^2 vs time,
                      both cases, active phases shaded -- the quantitative compare
                      (squarewave SPIKES UP during idle = more segregated).
  fig_radial_profile.png  time-averaged outlet radial profile vs the injected
                      step -- shows BOTH cases are ~flat (why the metric saturates).
  fig_torque.png      inner-cylinder torque vs time -- proves the square-wave
                      actuation is doing what it should.

Usage:
  python make_figures.py <results_dir> [--period 30] [--duty 0.2]
                         [--prof-window TMIN TMAX]
"""
import argparse
import glob
import math
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STD_INLET = math.sqrt(0.1875)   # std of the inner-1/4 injected dye band
NBINS = 20
TOK = re.compile(r"(\w+)=(-?[\d.eE+-]+)")
COLORS = {"constant": "#1f77b4", "squarewave": "#d62728"}


def parse(path):
    t, mz, C, Vz = [], [], [], []
    with open(path) as f:
        for ln in f:
            if not ln.startswith("METRICS"):
                continue
            kv = dict(TOK.findall(ln))
            try:
                t.append(float(kv["t"]))
                mz.append(float(kv["Mz_kin"]))
                C.append([float(kv[f"C{b}"]) for b in range(NBINS)])
                Vz.append([float(kv[f"Vz{b}"]) for b in range(NBINS)])
            except (KeyError, ValueError):
                continue
    return (np.array(t), np.array(mz), np.array(C), np.array(Vz))


def active_spans(t0, t1, period, duty):
    """[(start,end), ...] active (high-omega) intervals overlapping [t0,t1]."""
    spans = []
    k = int(t0 // period)
    while k * period < t1:
        a, b = k * period, k * period + duty * period
        if b > t0 and a < t1:
            spans.append((max(a, t0), min(b, t1)))
        k += 1
    return spans


def time_bin(t, y, nb=1200):
    """Average y onto nb uniform time bins (y may be 1-D or 2-D in bin axis)."""
    edges = np.linspace(t.min(), t.max(), nb + 1)
    idx = np.clip(np.digitize(t, edges) - 1, 0, nb - 1)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    if y.ndim == 1:
        out = np.full(nb, np.nan)
        for b in range(nb):
            m = idx == b
            if m.any():
                out[b] = y[m].mean()
    else:
        out = np.full((nb, y.shape[1]), np.nan)
        for b in range(nb):
            m = idx == b
            if m.any():
                out[b] = y[m].mean(axis=0)
    return ctr, out


def unmixed(C):
    return C.std(axis=1)            # per-time std across the 20 radial bins


def shade(ax, spans, label="active (2500 rpm)"):
    for i, (a, b) in enumerate(spans):
        ax.axvspan(a, b, color="0.85", zorder=0,
                   label=label if i == 0 else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--period", type=float, default=30.0)
    ap.add_argument("--duty", type=float, default=0.2)
    ap.add_argument("--prof-window", type=float, nargs=2, default=None,
                    help="TMIN TMAX to average the radial profile over "
                         "(default: last full period present in both runs)")
    ap.add_argument("--tmax", type=float, default=None,
                    help="clip ALL runs to t<=TMAX so the curves end together "
                         "(default: the shorter run's end time; use 'full' via a "
                         "big number to keep each run's own length)")
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

    # Runs that timed out reach different end times (e.g. squarewave is costlier
    # so it covers less sim-time in the same walltime). Clip all to a common
    # end so the figures are symmetric and the comparison is over equal spans.
    clip = args.tmax if args.tmax else min(runs[n][0].max() for n in order)
    for n in list(runs):
        t, mz, C, Vz = runs[n]
        m = t <= clip
        runs[n] = (t[m], mz[m], C[m], Vz[m])
    tmax_all = min(runs[n][0].max() for n in order)
    print(f"clipping all runs to t <= {clip:.1f} s (common end)")

    # default profile window = last full period both runs reached
    if args.prof_window:
        pw0, pw1 = args.prof_window
    else:
        pw1 = tmax_all
        pw0 = max(0.0, pw1 - args.period)

    # ---- 1. space-time map C(bin, t) ---------------------------------------
    fig, axes = plt.subplots(len(order), 1, figsize=(10, 2.6 * len(order)),
                             sharex=True, squeeze=False)
    for ax, name in zip(axes[:, 0], order):
        t, _, C, _ = runs[name]
        ctr, Cb = time_bin(t, C, nb=700)
        im = ax.imshow(Cb.T, aspect="auto", origin="lower",
                       extent=[ctr[0], ctr[-1], 0, 1], cmap="viridis",
                       vmin=0, vmax=max(0.5, np.nanpercentile(Cb, 99)))
        ax.set_ylabel(f"{name}\ngap pos (0=in)")
        for a, b in active_spans(t.min(), t.max(), args.period, args.duty):
            ax.plot([a, a], [0, 1], "w:", lw=0.6, alpha=0.6)
            ax.plot([b, b], [0, 1], "w:", lw=0.6, alpha=0.6)
    axes[-1, 0].set_xlabel("time [s]")
    fig.colorbar(im, ax=axes[:, 0], label="dye conc C", shrink=0.8)
    fig.suptitle("Outlet radial dye profile vs time "
                 "(dotted = active-phase edges)")
    _save(fig, args.results_dir, "fig_spacetime.png")

    # ---- 2. intensity of segregation vs time -------------------------------
    # I_mix = sigma^2 / sigma_max^2 (Danckwerts intensity of segregation):
    # 1 = as-injected (fully segregated), 0 = perfectly mixed.  LOWER = better.
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sq = runs.get("squarewave")
    if sq is not None:
        shade(ax, active_spans(sq[0].min(), sq[0].max(), args.period, args.duty))
    for name in order:
        t, _, C, _ = runs[name]
        ctr, u = time_bin(t, unmixed(C), nb=1500)
        ax.plot(ctr, (u / STD_INLET) ** 2, color=COLORS.get(name), lw=1.6, label=name)
    ax.set_xlabel("time [s]")
    ax.set_ylabel(r"intensity of segregation  $I_{mix}=\sigma^2/\sigma_{\max}^2$"
                  "   (0 = fully mixed)")
    ax.set_ylim(bottom=0.0); ax.grid(alpha=0.3); ax.legend(loc="upper right")
    ax.set_title("Outlet intensity of segregation vs time")
    _save(fig, args.results_dir, "fig_mixing_ts.png")

    # ---- 3. time-averaged radial profile -----------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.8))
    x = (np.arange(NBINS) + 0.5) / NBINS
    inj = np.where(x < 0.25, 1.0, 0.0)
    ax.step(x, inj, where="mid", color="0.5", ls="--", label="injected (inlet)")
    print(f"\nradial profile averaged over t in [{pw0:.1f}, {pw1:.1f}] s")
    for name in order:
        t, _, C, _ = runs[name]
        m = (t >= pw0) & (t <= pw1)
        prof = C[m].mean(axis=0) if m.any() else np.full(NBINS, np.nan)
        u = prof.std(); Imix = (u / STD_INLET) ** 2
        ax.plot(x, prof, "o-", color=COLORS.get(name), ms=4,
                label=f"{name} ($I_{{mix}}$ {Imix:.4f})")
        print(f"  {name:11s} sigma={u:.4f}  I_mix={Imix:.4f}")
    ax.set_xlabel("normalized gap position  (0 = inner wall, 1 = outer)")
    ax.set_ylabel("time-avg dye conc C")
    ax.grid(alpha=0.3); ax.legend()
    ax.set_title(f"Time-averaged outlet radial profile  (t∈[{pw0:.0f},{pw1:.0f}]s)")
    _save(fig, args.results_dir, "fig_radial_profile.png")

    # ---- 4. torque / actuation ---------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.0))
    if sq is not None:
        shade(ax, active_spans(sq[0].min(), sq[0].max(), args.period, args.duty))
    for name in order:
        t, mz, _, _ = runs[name]
        ctr, m = time_bin(t, np.abs(mz), nb=1500)
        ax.plot(ctr, m, color=COLORS.get(name), lw=1.4, label=name)
    ax.set_xlabel("time [s]"); ax.set_ylabel("|inner-wall torque|  [m^2/s^2]")
    ax.grid(alpha=0.3); ax.legend(loc="upper right")
    ax.set_title("Inner-cylinder torque vs time (square-wave actuation)")
    _save(fig, args.results_dir, "fig_torque.png")


def _save(fig, d, name):
    out = os.path.join(d, name)
    fig.savefig(out, dpi=140, bbox_inches="tight")   # bbox_inches handles colorbars
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
