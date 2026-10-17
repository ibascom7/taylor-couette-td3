#!/usr/bin/env python3
"""The best fixed waveform from the Bayesian-optimization search (endgame A,
episode 219: burst 4.68 s, stop 0.5 s, peak 331 rpm, 300 rpm average), drawn
exactly like the poster's RL-waveform figure (fig4_learned_waveform): five
26 s blocks, each restarting burst-first, so the two plots compare directly.

Run from anywhere:  python3 reports/gcurs_talk_2026-10/make_bo_waveform.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EVALS = os.path.join(HERE, "..", "..", "taylorCouetteGym", "experiments", "endgame", "results", "A",
                     "evaluations.csv")
RED, DARK = "#C8102E", "#222222"
plt.rcParams.update({"font.size": 16, "axes.titlesize": 18, "axes.labelsize": 17,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "font.family": "DejaVu Sans"})

best = max((r for r in csv.DictReader(open(EVALS)) if r["status"] == "ok"),
           key=lambda r: float(r["r_last"]))                       # observed best episode
tp, tm, wh = float(best["t_plus_s"]), float(best["t_minus_s"]), float(best["w_hi_rpm"])

t_all, w_all = [], []
for i in range(5):                                                 # same block rebuild as fig4
    t0 = 26.0 * i
    t = t0
    while t < t0 + 26.0 - 1e-9:
        t_on_end = min(t + tp, t0 + 26.0)
        t_all += [t, t_on_end]; w_all += [wh, wh]
        t_off_end = min(t_on_end + tm, t0 + 26.0)
        if t_off_end > t_on_end:
            t_all += [t_on_end, t_off_end]; w_all += [0.0, 0.0]
        t = t_off_end

fig, ax = plt.subplots(figsize=(11, 4.6))
ax.plot(t_all, w_all, color=RED, lw=1.6)
ax.axhline(300, color=DARK, ls=":", lw=1.5)
ax.text(1, 345, "average fixed at 300 rpm", ha="left", fontsize=13)
ax.set_xlim(0, 130); ax.set_ylim(-10, 370)
ax.set_xlabel("time [s]"); ax.set_ylabel("inner cylinder speed [rpm]")
ax.set_title(f"Best fixed waveform (Bayesian optimization): stop {tm:.1f} s every {tp + tm:.1f} s")
ax.grid(alpha=0.3)
fig.tight_layout()
out = os.path.join(HERE, "figures", "bo_best_waveform.png")
fig.savefig(out); plt.close(fig)
print("episode", best["ep"], "burst", tp, "stop", tm, "peak", round(wh, 1), "->", out)
