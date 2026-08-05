#!/usr/bin/env python3
"""Clean (annotation-free) figures for a duty_v3_refs run directory.

Reads {run_dir}/timeseries.csv and writes into the same directory:
    conversion_vs_time.png  -- dense outlet-conversion trace + 26 s window means
    wallflux_vs_time.png    -- wall flux over the full episode
    wallflux_zoom.png       -- wall flux over a two-period window (the
                               film-renewal spikes at each stop)

USAGE: python3 plot_v3_ref.py results/duty_v3_refs/champion_D0.90 [zoom_t0 zoom_t1]
"""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(run_dir, zoom=(100.0, 115.0)):
    t, c, wf = [], [], []
    with open(os.path.join(run_dir, "timeseries.csv")) as f:
        for row in csv.DictReader(f):
            tv, cv = float(row["time_s"]), float(row["conversion"])
            if -0.02 <= cv <= 1.02:
                t.append(tv)
                c.append(cv)
                wf.append(float(row["wallFlux"]))
    t, c, wf = np.array(t), np.array(c), np.array(wf)
    name = os.path.basename(os.path.normpath(run_dir))
    dec = max(1, len(t) // 20000)          # ~20k points for the full-length plots

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.plot(t[::dec], c[::dec], color="forestgreen", lw=0.8)
    for w0 in np.arange(0.0, t[-1] - 1.0, 26.0):
        m = (t >= w0) & (t <= w0 + 26.0)
        ax.hlines(c[m].mean(), w0 + 0.5, w0 + 25.5, color="black", ls="--", lw=1.4)
    ax.set_xlabel("time since warmed start  [s]")
    ax.set_ylabel("outlet conversion  X")
    ax.set_title(f"{name} — outlet conversion (dashes: 26 s window means)",
                 fontsize=11)
    lo, hi = np.quantile(c, 0.001), np.quantile(c, 0.999)
    ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "conversion_vs_time.png"), dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.plot(t[::dec], wf[::dec] * 1e9, color="tab:blue", lw=0.8)
    ax.set_xlabel("time since warmed start  [s]")
    ax.set_ylabel("wall flux  [$10^{-9}$ mol m$^{-2}$ s$^{-1}$]")
    ax.set_title(f"{name} — catalytic wall flux", fontsize=11)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "wallflux_vs_time.png"), dpi=170)
    plt.close(fig)

    z = (t >= zoom[0]) & (t <= zoom[1])
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.plot(t[z], wf[z] * 1e9, color="tab:blue", lw=1.0)
    ax.set_xlabel("time since warmed start  [s]")
    ax.set_ylabel("wall flux  [$10^{-9}$ mol m$^{-2}$ s$^{-1}$]")
    ax.set_title(f"{name} — wall flux, {zoom[0]:g}–{zoom[1]:g} s", fontsize=11)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "wallflux_zoom.png"), dpi=170)
    plt.close(fig)
    print(f"wrote 3 figures -> {run_dir}")


if __name__ == "__main__":
    args = sys.argv[1:]
    zoom = (float(args[1]), float(args[2])) if len(args) >= 3 else (100.0, 115.0)
    main(args[0], zoom)
