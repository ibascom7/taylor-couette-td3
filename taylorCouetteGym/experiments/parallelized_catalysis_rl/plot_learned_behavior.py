#!/usr/bin/env python3
"""Visualize WHAT A TRAINED TD3 AGENT LEARNED, purely from its training .npy logs
(no CFD, no torch) -- so it runs anywhere numpy+matplotlib exist. Two figures:

  <out>/learned_behavior.png   learning curve (episode return) + the converged-episode
                               waveform: omega(t), motor power(t), conversion(t), reward(t).
  <out>/beats_constant.png     (optional, --baseline-npz) the agent's operating point
                               (mean power vs mean conversion over the reward window)
                               dropped onto the cached constant + pulsating baseline curves.

The per-step logs are the TRAINING rollout of the last episode(s): they include the
exploration noise TD3 adds during collection, so the waveform is the *near-converged*
behavior, not the clean deterministic eval. For the publication-clean waveform + the
true beats-constant point, run compare_catalysis.py on Carya (needs OpenFOAM). This is
the fast, CFD-free "what did it learn / did it spin or idle" look for a meeting.

Usage:
  python plot_learned_behavior.py --run results/td3/so_parallel_freeform_ew0.3_s0 \
      --dt 1.0 --window 30 \
      --baseline-npz results/comparison/so_parallel_freeform_ew0.8_s0_early/baseline_sweep.npz
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _row(grid, ep=-1):
    """One episode row of a [ep, step] NaN-padded grid, trailing NaNs trimmed."""
    g = np.asarray(grid, float)
    if g.ndim == 1:
        g = g[None, :]
    r = g[ep]
    return r[~np.isnan(r)]


def _last_k_mean(grid, k, step_len):
    """Mean +/- std across the last k episodes, aligned to the first `step_len` steps."""
    g = np.asarray(grid, float)
    if g.ndim == 1:
        g = g[None, :]
    tail = g[-k:, :step_len]
    return np.nanmean(tail, axis=0), np.nanstd(tail, axis=0)


def load_run(d):
    L = lambda n: np.load(os.path.join(d, n))
    return dict(
        ret=L("episode_returns.npy").astype(float),
        omega=L("omega_per_step.npy"),
        power=L("power_per_step.npy"),
        conv=L("conv_per_step.npy"),
        reward=L("reward_per_step.npy"),
    )


def fig_behavior(run, dt, k, title, out):
    om = _row(run["omega"]); pw = _row(run["power"])
    cv = _row(run["conv"]);  rw = _row(run["reward"])
    n = min(len(om), len(pw), len(cv), len(rw))
    om, pw, cv, rw = om[:n], pw[:n], cv[:n], rw[:n]
    t = np.arange(n) * dt
    om_m, om_s = _last_k_mean(run["omega"], k, n)

    fig = plt.figure(figsize=(12, 7.2))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.15, 1.0], hspace=0.35, wspace=0.25)

    # learning curve (spans left column, top two rows)
    axL = fig.add_subplot(gs[0:2, 0])
    ep = np.arange(1, len(run["ret"]) + 1)
    axL.plot(ep, run["ret"], color="0.7", lw=1, label="episode return")
    if len(ep) >= 5:
        w = min(10, len(ep))
        rm = np.convolve(run["ret"], np.ones(w) / w, mode="valid")
        axL.plot(ep[w - 1:], rm, color="#1f77b4", lw=2.2, label=f"{w}-ep mean")
    axL.set_xlabel("episode"); axL.set_ylabel("return")
    axL.set_title("Learning curve"); axL.grid(alpha=0.3); axL.legend(fontsize=8)

    # summary text (left column, bottom two rows)
    axT = fig.add_subplot(gs[2:4, 0]); axT.axis("off")
    win = max(1, int(round(min(30, n * dt) / dt)))
    lines = [
        f"episodes trained : {len(run['ret'])}",
        f"final return     : {run['ret'][-1]:.2f}",
        f"best return      : {np.nanmax(run['ret']):.2f}",
        "",
        f"converged episode ({n} steps, dt={dt}s):",
        f"  omega  mean {om.mean():6.0f}  std {om.std():5.0f} rpm  [{om.min():.0f}, {om.max():.0f}]",
        f"  power  mean {pw.mean():6.3f}  std {pw.std():5.3f} W",
        f"  conv   mean {cv.mean():6.4f}  final {cv[-1]:.4f}",
        f"last {win}s window:  omega {om[-win:].mean():.0f} rpm   "
        f"power {pw[-win:].mean():.3f} W   conv {cv[-win:].mean():.4f}",
        "",
        "verdict: " + ("IDLE/COAST (omega ~0 in window; rides warmed IC)"
                       if om[-win:].mean() < 60 else "SPINNING / MODULATING"),
    ]
    axT.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=10,
             family="monospace", transform=axT.transAxes)

    # waveform stack (right column)
    specs = [("$\\omega$ [rpm]", om, "#2ca02c", (om_m, om_s)),
             ("motor power [W]", pw, "#d62728", None),
             ("conversion", cv, "#1f77b4", None),
             ("reward", rw, "#9467bd", None)]
    for i, (lab, y, c, band) in enumerate(specs):
        ax = fig.add_subplot(gs[i, 1])
        ax.plot(t, y, "-", lw=1.6, color=c)
        if band is not None:
            m, s = band
            ax.fill_between(t[:len(m)], m - s, m + s, color=c, alpha=0.18,
                            label=f"last {k}-ep mean$\\pm$sd")
            ax.legend(fontsize=7, loc="best")
        if lab.startswith("motor"):
            ax.axhline(0, ls=":", lw=0.8, color="0.5")
        ax.set_ylabel(lab, fontsize=9); ax.grid(alpha=0.3)
        if i < 3:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("time [s]")
    fig.add_subplot(gs[0, 1]).set_visible(False) if False else None
    fig.suptitle(title, fontsize=13, y=0.98)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return dict(n=n, om_win=om[-win:].mean(), pw_win=pw[-win:].mean(), cv_win=cv[-win:].mean())


def fig_beats(run, dt, k, npz_path, title, out):
    z = np.load(npz_path, allow_pickle=True)
    cc_conv, cc_pw = z["const_conv"], z["const_motor"]
    pc_conv, pc_pw = z["puls_conv"], z["puls_motor"]
    cv = _row(run["conv"]); pw = _row(run["power"])
    n = min(len(cv), len(pw)); win = max(1, int(round(min(30, n * dt) / dt)))
    a_conv, a_pw = cv[-win:].mean(), pw[-win:].mean()

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    oc = np.argsort(cc_conv); op = np.argsort(pc_conv)
    ax.plot(cc_conv[oc], cc_pw[oc], "o-", color="#d62728", lw=2, ms=6, label="constant $\\omega$ sweep")
    ax.plot(pc_conv[op], pc_pw[op], "s-", color="#1f77b4", lw=2, ms=6, label="pulsating (square) sweep")
    ax.plot(a_conv, a_pw, "*", color="#2ca02c", ms=22, mec="k", mew=0.8,
            label=f"TD3 agent ({a_conv:.3f}, {a_pw:.2f} W)", zorder=5)
    ax.set_xlabel("conversion"); ax.set_ylabel("avg motor power [W]")
    ax.set_title(title); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax.annotate("lower = cheaper\nat equal conversion", xy=(0.02, 0.98),
                xycoords="axes fraction", ha="left", va="top", fontsize=8, color="0.4")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return dict(a_conv=a_conv, a_pw=a_pw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="training run dir with the .npy logs")
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--last-k", type=int, default=10, help="episodes to average for the settled band")
    ap.add_argument("--baseline-npz", default=None, help="wall-model baseline_sweep.npz for the overlay")
    ap.add_argument("--out", default=None, help="output dir (default: the run dir)")
    ap.add_argument("--label", default=None, help="title label (default: run basename)")
    args = ap.parse_args()

    run = load_run(args.run)
    out = args.out or args.run
    os.makedirs(out, exist_ok=True)
    lab = args.label or os.path.basename(os.path.normpath(args.run))

    b = fig_behavior(run, args.dt, args.last_k, f"Learned behavior — {lab}",
                     os.path.join(out, "learned_behavior.png"))
    print(f"learned_behavior.png  steps={b['n']}  last30s: omega={b['om_win']:.0f}rpm "
          f"power={b['pw_win']:.3f}W conv={b['cv_win']:.4f}")
    if args.baseline_npz and os.path.exists(args.baseline_npz):
        c = fig_beats(run, args.dt, args.last_k, args.baseline_npz,
                      f"Power vs conversion — {lab}", os.path.join(out, "beats_constant.png"))
        print(f"beats_constant.png    agent=({c['a_conv']:.4f} conv, {c['a_pw']:.3f} W)")
    print(f"wrote -> {out}")


if __name__ == "__main__":
    main()
