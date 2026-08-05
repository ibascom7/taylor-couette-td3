#!/usr/bin/env python3
"""Figures for the duty_v1 (D-only) campaign, three outputs:

1. results/warmed_eval/duty_eval_omega_vs_time.png -- the commanded waveform
   of each seed's deterministic warmed eval, in the style of
   results/td3/eval_omega_vs_time.png (one panel per seed; T=5 s pinned,
   w_low=0, w_hi=300/D, phase resets each 10 s block).
2. results/td3/duty_v1_learning_curves.png -- per-seed training reward over
   time (every 10 s block, in episode-completion order) with the commanded
   duty D underneath on a shared time axis, so collapse-to-D=1 is visible
   against the reward it earns.
3. results/duty_diag/duty_diag_explainer.png -- what the diagnostic sweep
   showed: (a) the true sustained-reward-vs-D landscape at T=5 with the three
   actors placed on it, (b) the idle-duration law (reward vs (1-D)*T across
   periods), (c) the 50 s conversion transient for constant vs the static
   champion vs the one modulating actor.

USAGE: python3 plot_duty_v1.py
"""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
GYM_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if GYM_ROOT not in sys.path:
    sys.path.insert(0, GYM_ROOT)
from taylor_couette_mixing.envs.taylor_couette_waveform import square_wave_points  # noqa: E402

TD3_DIR = os.path.join(HERE, "results", "td3")
EVAL_DIR = os.path.join(HERE, "results", "warmed_eval")
DIAG_DIR = os.path.join(HERE, "results", "duty_diag")

RPM = 2.0 * np.pi / 60.0
PERIOD, BLOCK_DT, N_BLOCKS, RAMP = 5.0, 10.0, 5, 0.05
REF_CONST = 0.2437      # warmed constant-300 sustained (last-block) reward
REF_CHAMP = 0.2886      # warmed static champion T=5/D=0.90
REF_V5S2 = 0.2908       # 3-knob v5-s2 warmed deterministic eval (upper bound)
START_TIMESTEPS = 500   # random-action env steps before the policy acts


def read_blocks(path):
    with open(path, newline="") as f:
        return [{k: float(v) for k, v in row.items()}
                for row in csv.DictReader(f)]


# ---------------------------------------------------------------- figure 1 --
def fig_eval_omega():
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, sharey=True)
    fig.suptitle("duty_v1 policies (deterministic, warmed) — commanded waveform, "
                 "50 s episode\n(T = 5 s and $w_{low}$ = 0 pinned; burst "
                 "$w_{hi} = 300/D$; phase resets each 10 s block)",
                 fontsize=13, x=0.5, y=0.985)
    verdicts = {}
    for s, ax in zip(range(3), axes):
        rows = read_blocks(os.path.join(EVAL_DIR, f"duty_v1_s{s}", "blocks.csv"))
        tt, ww = [], []
        for r in rows:
            t0 = (r["block"] - 1) * BLOCK_DT
            pts, _ = square_wave_points(t0, BLOCK_DT, r["w_hi_rpm"] * RPM, 0.0,
                                        PERIOD, r["duty"], RAMP, phase0=0.0)
            kept = [(t, w / RPM) for t, w in pts if t <= t0 + BLOCK_DT + 1e-9]
            # hold the last level to the block boundary so panels never show a
            # spurious diagonal into the next block's opening level
            kept.append((t0 + BLOCK_DT, kept[-1][1]))
            tt.extend(p[0] for p in kept)
            ww.extend(p[1] for p in kept)
        ax.plot(tt, ww, color="tab:blue", lw=1.8)
        ax.axhline(300.0, color="0.35", ls="--", lw=1.1)
        for x in (10, 20, 30, 40):
            ax.axvline(x, color="0.88", lw=1.0, zorder=0)
        for r in rows:
            xc = (r["block"] - 0.5) * BLOCK_DT
            ax.text(xc, 425, f"b{int(r['block'])}\nD={r['duty']:.2f}\n"
                    f"$w_{{hi}}$={r['w_hi_rpm']:.0f}",
                    ha="center", va="top", fontsize=8.5, color="0.35")
        last_r = rows[-1]["reward"]
        modulated = any(r["duty"] < 0.995 for r in rows)
        verdict = ("modulates weakly (mean D≈0.97)" if modulated
                   else "collapsed to constant 300 (D=1 every block)")
        verdicts[s] = verdict
        ax.set_title(f"seed {s} — last-block r = {last_r:+.4f} — {verdict}",
                     loc="left", fontsize=11)
        ax.set_ylabel("inner-wall angular velocity  [rpm]")
        ax.grid(alpha=0.25)
    axes[0].text(0.6, 310, "$w_b$ = 300 rpm (commanded block mean)",
                 ha="left", fontsize=8.5, color="0.35")
    axes[-1].set_xlabel("time since warmed start  [s]")
    axes[0].set_ylim(-15, 430)
    axes[0].set_xlim(-0.5, 50.5)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(EVAL_DIR, "duty_eval_omega_vs_time.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------- figure 2 --
def fig_learning_curves():
    fig = plt.figure(figsize=(13, 13))
    outer = fig.add_gridspec(3, 1, hspace=0.34, left=0.075, right=0.985,
                             top=0.90, bottom=0.075)
    fig.suptitle("duty_v1 — TD3 on the D-only duty env: training reward and "
                 "commanded duty, every 10 s block\n(warmed episodes, "
                 "3–7 blocks each, flattened in episode order; "
                 "first 500 steps = uniform-random actions)",
                 fontsize=13, x=0.5, y=0.97)
    W = 41
    det = {0: (0.2437, "collapsed"), 1: (0.2822, "modulating"),
           2: (0.2437, "collapsed")}
    lbl = dict(fontsize=8, color="0.35",
               bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1))
    for s in range(3):
        d = os.path.join(TD3_DIR, f"duty_v1_s{s}")
        rw = np.load(os.path.join(d, "reward_per_step.npy"))
        pr = np.load(os.path.join(d, "params_per_step.npy"))
        mask = ~np.isnan(rw)
        r_flat = rw[mask]
        d_flat = pr[:, :, 0][mask]
        x = np.arange(1, len(r_flat) + 1)
        inner = outer[s].subgridspec(2, 1, height_ratios=[3, 1.3], hspace=0.08)
        axr = fig.add_subplot(inner[0])
        axd = fig.add_subplot(inner[1], sharex=axr)

        axr.axvspan(0, START_TIMESTEPS, color="0.92", zorder=0)
        axr.scatter(x, r_flat, s=5, alpha=0.3, color="tab:blue", lw=0)
        run = np.convolve(r_flat, np.ones(W) / W, mode="valid")
        axr.plot(x[W - 1:], run, color="tab:blue", lw=1.8)
        axr.axhline(REF_CHAMP, color="0.4", ls=":", lw=1.1)
        axr.axhline(REF_CONST, color="0.4", ls="--", lw=1.1)
        r_eval, tag = det[s]
        axr.set_title(f"seed {s} — deterministic warmed eval r = {r_eval:+.4f} "
                      f"({tag})   [late-100 block-r mean = "
                      f"{r_flat[-100:].mean():.4f}]", loc="left", fontsize=11)
        axr.set_ylabel("block reward")
        axr.set_ylim(0.14, 0.32)
        axr.grid(alpha=0.25)
        plt.setp(axr.get_xticklabels(), visible=False)
        if s == 0:
            axr.text(len(x), REF_CHAMP + 0.003,
                     "static champion D=0.90 (0.2886)", ha="right", **lbl)
            axr.text(len(x), REF_CONST - 0.010,
                     "constant-300 sustained (0.2437)", ha="right", **lbl)
            axr.text(START_TIMESTEPS / 2, 0.155, "random exploration",
                     ha="center", fontsize=8, color="0.4")

        axd.axvspan(0, START_TIMESTEPS, color="0.92", zorder=0)
        axd.scatter(x, d_flat, s=4, alpha=0.3, color="tab:purple", lw=0)
        rund = np.convolve(d_flat, np.ones(W) / W, mode="valid")
        axd.plot(x[W - 1:], rund, color="tab:purple", lw=1.6)
        axd.axhline(0.90, color="0.4", ls=":", lw=1.0)
        axd.set_ylabel("duty D")
        axd.set_ylim(0.15, 1.05)
        axd.grid(alpha=0.25)
        if s == 2:
            axd.set_xlabel("training env step (10 s block)")
    fig.text(0.5, 0.012,
             "reward = $X_{block} - P_{block}/P_{max}$, exploration noise on "
             "(σ=0.2 raw ≈ 0.08 duty);  dashed/dotted = WARMED sustained "
             "references;  dotted line in D panels = static optimum D=0.90",
             ha="center", fontsize=9, color="0.35")
    out = os.path.join(TD3_DIR, "duty_v1_learning_curves.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------- figure 3 --
def fig_diag_explainer():
    stat = {}          # tag -> summary row (last_r, x_last, ...)
    for row in csv.DictReader(open(os.path.join(DIAG_DIR, "summary.csv"))):
        stat[row["tag"]] = {k: float(v) for k, v in row.items() if k != "tag"}
    probe = list(csv.DictReader(open(os.path.join(DIAG_DIR, "probe2_summary.csv"))))
    probe = [{**{k: float(v) for k, v in r.items() if k != "tag"}, "tag": r["tag"]}
             for r in probe]

    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(16.5, 5.4))
    fig.suptitle("duty_diag — what the static sweep + deterministic actor evals showed "
                 "(all runs: warmed IC, 5 × 10 s blocks, $w_b$=300, last block = "
                 "sustained-operation value)", fontsize=13, x=0.5, y=0.98)

    # (a) sustained reward vs D at T = 5 -----------------------------------
    d_grid = [0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0]
    curve = {d: stat[f"static_D{d:.2f}"]["last_r"] for d in d_grid}
    for r in probe:
        if r["period"] == 5.0:
            curve[r["duty"]] = r["last_r"]
    ds = sorted(curve)
    axa.plot(ds, [curve[d] for d in ds], "-o", color="tab:blue", ms=5, lw=1.8)
    axa.axhline(REF_V5S2, color="0.4", ls=":", lw=1.1)
    axa.text(0.21, REF_V5S2 + 0.0008, "3-knob v5-s2 warmed eval (0.2908)",
             fontsize=8, color="0.35")
    axa.annotate("static optimum\nD=0.90, r=0.2886", xy=(0.90, curve[0.90]),
                 xytext=(0.52, 0.283), fontsize=9, color="0.3",
                 arrowprops=dict(arrowstyle="->", color="0.4", lw=0.9))
    axa.annotate("cliff: D=1 is\nconstant-300\n(r=0.2437)", xy=(1.0, curve[1.0]),
                 xytext=(0.84, 0.252), fontsize=9, color="0.3",
                 arrowprops=dict(arrowstyle="->", color="0.4", lw=0.9))
    a1 = stat["actor_s1"]
    axa.plot([0.966], [a1["last_r"]], "*", ms=15, color="tab:green", zorder=5)
    axa.text(0.998, a1["last_r"] - 0.003, "actor s1\n(mean D=0.97)",
             ha="right", va="top", fontsize=8.5, color="tab:green")
    axa.plot([1.0], [stat["actor_s0"]["last_r"]], "*", ms=15, color="tab:red",
             zorder=5)
    axa.text(0.995, stat["actor_s0"]["last_r"] - 0.004, "actors s0, s2\n(D=1 always)",
             ha="right", va="top", fontsize=8.5, color="tab:red")
    axa.set_xlabel("duty D  (fraction of each 5 s period at $w_{hi}=300/D$)")
    axa.set_ylabel("last-block reward  $X - P/P_{max}$")
    axa.set_title("(a) the true 1-D landscape at T = 5 s", loc="left", fontsize=11)
    axa.grid(alpha=0.25)

    # (b) the idle-duration law --------------------------------------------
    series = {2.5: ([], []), 5.0: ([], []), 10.0: ([], [])}
    for d in d_grid:
        series[5.0][0].append((1 - d) * 5.0)
        series[5.0][1].append(curve[d])
    for r in probe:
        series[r["period"]][0].append(r["idle_s"])
        series[r["period"]][1].append(r["last_r"])
    # T=2.5 and T=10 each have ONE probe point at idle=0.5 s; the T=2.5 value
    # (0.2885) sits within 0.0002 of the T=5 peak (0.2886), so filled markers
    # would be invisible under it -- draw the singles as large open shapes.
    styles = {2.5: dict(marker="s", ms=12, ls="none", mfc="none", mew=2.0,
                        color="tab:purple", zorder=6),
              5.0: dict(marker="o", ms=6, ls="-", lw=1.5, color="tab:blue"),
              10.0: dict(marker="D", ms=9, ls="none", mfc="none", mew=2.0,
                         color="tab:orange", zorder=6)}
    for T in (2.5, 5.0, 10.0):
        xs, ys = series[T]
        o = np.argsort(xs)
        axb.plot(np.array(xs)[o], np.array(ys)[o], label=f"T = {T:g} s",
                 **styles[T])
    axb.axvspan(0.45, 0.55, color="0.9", zorder=0)
    axb.annotate("idle ≈ 0.5 s (≈ swirl-decay time):\nnear-equal reward at "
                 "T = 2.5, 5, 10\n→ enhancement set by IDLE LENGTH,\n"
                 "not period or duty per se", xy=(0.55, 0.2855),
                 xytext=(1.35, 0.272), fontsize=9, color="0.3",
                 arrowprops=dict(arrowstyle="->", color="0.4", lw=0.9))
    axb.legend(loc="lower right", fontsize=9, framealpha=0.9)
    axb.set_xlabel("idle duration per period  $(1-D)\\,T$  [s]")
    axb.set_ylabel("last-block reward")
    axb.set_title("(b) the idle-duration (film-renewal) law", loc="left", fontsize=11)
    axb.grid(alpha=0.25)

    # (c) the 50 s conversion transients -----------------------------------
    picks = [("static_D1.00", "constant 300 (D=1)", "0.35", "--"),
             ("static_D0.90", "static champion D=0.90", "tab:blue", "-"),
             ("actor_s1", "actor s1 (deterministic)", "tab:green", "-")]
    for tag, label, color, ls in picks:
        rows = read_blocks(os.path.join(DIAG_DIR, tag, "blocks.csv"))
        t = [(r["block"] - 0.5) * BLOCK_DT for r in rows]
        axc.plot(t, [r["X_block"] for r in rows], ls, marker="o", ms=5,
                 color=color, lw=1.7, label=label)
    axc.legend(loc="upper left", fontsize=9, framealpha=0.9)
    axc.set_xlabel("block midpoint time  [s]")
    axc.set_ylabel("block-mean outlet conversion  $X$")
    axc.set_title("(c) conversion transient from the warmed IC", loc="left",
                  fontsize=11)
    axc.grid(alpha=0.25)

    fig.text(0.5, 0.005,
             "outlet X lags the wall by τ ≈ 26 s ≈ 2.6 blocks, which is why the "
             "landscape uses LAST-block reward (sustained value) and why the "
             "transients in (c) are still rising",
             ha="center", fontsize=9, color="0.35")
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    out = os.path.join(DIAG_DIR, "duty_diag_explainer.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


# ------------------------------------------------- clean single panels -----
def fig_diag_clean():
    """Annotation-free standalone versions of the explainer's panels (a) and
    (b) -> results/duty_diag/{landscape_reward_vs_D,idle_law}.png."""
    stat = {}
    for row in csv.DictReader(open(os.path.join(DIAG_DIR, "summary.csv"))):
        stat[row["tag"]] = {k: float(v) for k, v in row.items() if k != "tag"}
    probe = [{**{k: float(v) for k, v in r.items() if k != "tag"}, "tag": r["tag"]}
             for r in csv.DictReader(open(os.path.join(DIAG_DIR, "probe2_summary.csv")))]

    d_grid = [0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0]
    curve = {d: stat[f"static_D{d:.2f}"]["last_r"] for d in d_grid}
    for r in probe:
        if r["period"] == 5.0:
            curve[r["duty"]] = r["last_r"]
    ds = sorted(curve)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(ds, [curve[d] for d in ds], "-o", color="tab:blue", ms=6, lw=1.8)
    ax.set_xlabel("duty D")
    ax.set_ylabel("last-block reward  $X - P/P_{max}$")
    ax.set_title("Sustained reward vs duty (T = 5 s, warmed)", fontsize=11)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = os.path.join(DIAG_DIR, "landscape_reward_vs_D.png")
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"wrote {out}")

    series = {2.5: ([], []), 5.0: ([], []), 10.0: ([], [])}
    for d in d_grid:
        series[5.0][0].append((1 - d) * 5.0)
        series[5.0][1].append(curve[d])
    for r in probe:
        series[r["period"]][0].append(r["idle_s"])
        series[r["period"]][1].append(r["last_r"])
    styles = {2.5: dict(marker="s", ms=11, ls="none", mfc="none", mew=2.0,
                        color="tab:purple", zorder=6),
              5.0: dict(marker="o", ms=6, ls="-", lw=1.8, color="tab:blue"),
              10.0: dict(marker="D", ms=9, ls="none", mfc="none", mew=2.0,
                         color="tab:orange", zorder=6)}
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for T in (2.5, 5.0, 10.0):
        xs, ys = series[T]
        o = np.argsort(xs)
        ax.plot(np.array(xs)[o], np.array(ys)[o], label=f"T = {T:g} s", **styles[T])
    ax.set_xlabel("idle duration per period  $(1-D)\\,T$  [s]")
    ax.set_ylabel("last-block reward  $X - P/P_{max}$")
    ax.set_title("Sustained reward vs idle duration (warmed)", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = os.path.join(DIAG_DIR, "idle_law.png")
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig_eval_omega()
    fig_learning_curves()
    fig_diag_explainer()
    fig_diag_clean()
