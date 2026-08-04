#!/usr/bin/env python3
"""10-seed learning-curve figure for the full v5 campaign (s0-s9), in the
style of results/td3/v5_learning_curves.png (episode return = SUM of the 5
block rewards; scatter + running mean, shaded 200-episode random phase,
last-40 annotation). Verdicts are computed from the data: PASS = last-30-ep
mean FINAL-BLOCK reward > 0.270 (the pre-registered criterion); 'trapped' =
late mean w_low > 150 (the constant-300 trap signature).

USAGE: python3 plot_v5_all10.py   -> results/td3/v5_learning_curves_all10.png
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
TD3_DIR = os.path.join(HERE, "results", "td3")
RANDOM_EPS = 200          # START_TIMESTEPS=1000 steps / 5 blocks
RUN_MEAN_W = 15
PASS_THRESH = 0.270

fig, axes = plt.subplots(5, 2, figsize=(13, 15), sharex=True, sharey=True)
fig.suptitle("mod_wb300_v5 — TD3 with UTD=64: learning curves, all 10 seeds\n"
             "(episode return = sum of the 5 block rewards $X_b - P_b/P_{max}$)",
             fontsize=14, x=0.5, y=0.995)

for s in range(10):
    ax = axes[s // 2, s % 2]
    d = os.path.join(TD3_DIR, f"mod_wb300_v5_s{s}")
    er = np.load(os.path.join(d, "episode_returns.npy"))
    rw = np.load(os.path.join(d, "reward_per_step.npy"))
    pr = np.load(os.path.join(d, "params_per_step.npy"))
    lastblk = np.array([row[~np.isnan(row)][-1] for row in rw])
    rfin = lastblk[-30:].mean()
    wlo = float(np.nanmean(pr[-30:, :, 1]))
    D = float(np.nanmean(pr[-30:, :, 0]))
    T = float(np.nanmean(pr[-30:, :, 2]))
    if rfin > PASS_THRESH:
        verdict = f"PASS  $R_{{fin}}$={rfin:.3f}"
    elif wlo > 150:
        verdict = f"trapped (w_low={wlo:.0f})  $R_{{fin}}$={rfin:.3f}"
    else:
        verdict = f"near-miss  $R_{{fin}}$={rfin:.3f}"

    ep = np.arange(1, len(er) + 1)
    run = np.convolve(er, np.ones(RUN_MEAN_W) / RUN_MEAN_W, mode="valid")
    rmean = er[:RANDOM_EPS].mean()
    color = "tab:green" if rfin > PASS_THRESH else \
            ("tab:red" if wlo > 150 else "tab:orange")

    ax.axvspan(0, RANDOM_EPS, color="0.92", zorder=0)
    ax.scatter(ep, er, s=6, alpha=0.35, color=color, lw=0)
    ax.plot(ep[RUN_MEAN_W - 1:], run, color=color, lw=1.8)
    ax.axhline(rmean, color="0.4", ls="--", lw=0.9)
    last40 = er[-40:].mean()
    ax.annotate(f"last 40: {last40:.3f}", xy=(len(er), run[-1]),
                xytext=(0.62, 0.9), textcoords="axes fraction", fontsize=9,
                arrowprops=dict(arrowstyle="-", color="0.4", lw=0.8))
    ax.set_title(f"seed {s} — {verdict}   (late D={D:.2f}, w_low={wlo:.0f}, "
                 f"T={T:.1f})", fontsize=10, loc="left")
    ax.grid(alpha=0.25)
    ax.set_ylim(0.70, 1.12)
    if s % 2 == 0:
        ax.set_ylabel("episode return")
    if s // 2 == 4:
        ax.set_xlabel("episode")

axes[0, 0].text(RANDOM_EPS * 0.5, 0.72, "random exploration", ha="center",
                fontsize=8, color="0.4")
axes[0, 0].text(RANDOM_EPS + 40, 0.72, "policy", fontsize=8, color="0.4")
fig.text(0.5, 0.005,
         "PASS = last-30-episode mean final-block reward > 0.270 (static champion); "
         "dashed line = that seed's random-phase mean; tally: see title colors "
         "(green PASS / orange near-miss / red constant-300 trap)",
         ha="center", fontsize=9, color="0.35")
fig.tight_layout(rect=[0, 0.01, 1, 0.98])
out = os.path.join(TD3_DIR, "v5_learning_curves_all10.png")
fig.savefig(out, dpi=140)
print(f"wrote {out}")
