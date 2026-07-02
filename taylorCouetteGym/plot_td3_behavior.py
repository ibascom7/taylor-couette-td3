#!/usr/bin/env python3
"""
Visualize a TD3 run's behavior on the reactor: learning curves (return AND energy
cost) + the policy it actually executes (omega per step across training).

Reads a run dir written by train.py:
    episode_returns.npy   total return per completed episode
    omega_per_step.npy    chosen omega [rpm], shape [episode, step] (NaN-padded)
    reward_per_step.npy   reward,          shape [episode, step]
    power_per_step.npy    step-average power [W] (optional; catalysis env). If
                          absent, power is RECOMPUTED from omega via the motor model
                          -- so the energy panel works on every run either way.

Writes <run_dir>/td3_behavior.png with four panels (2x2):
  1. learning curve (return vs episode, + trailing mean) -- is it improving?
  2. energy cost (mean motor power [W] vs episode, + trailing mean) -- what does the
     learned policy COST, and did the agent trade power for return as it learned?
  3. omega heatmap (episode x step) -- how the policy evolves; does it settle
     to a constant or keep modulating?
  4. omega trajectories for the first vs last episodes -- early exploration vs
     converged behavior.

Usage:  python plot_td3_behavior.py results/td3/full3d_s0
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Motor model, for recomputing power from omega on runs with no power log. Prefer the
# package import; fall back to the numpy-only module directly (the package __init__
# pulls in gymnasium/OpenFOAM this plot doesn't need). motor_power=None => skip panel.
GYM_ROOT = os.path.dirname(os.path.abspath(__file__))
if GYM_ROOT not in sys.path:
    sys.path.insert(0, GYM_ROOT)
try:
    from taylor_couette_mixing import motor_power
except Exception:
    try:
        import importlib.util
        _mp = os.path.join(GYM_ROOT, "taylor_couette_mixing", "motor_power.py")
        _spec = importlib.util.spec_from_file_location("motor_power", _mp)
        motor_power = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(motor_power)
    except Exception:
        motor_power = None

RPM_TO_RADS = 2.0 * np.pi / 60.0


def episode_mean_power(run_dir, om, n_ep, dt):
    """Mean electrical power [W] per episode. Prefer the logged power_per_step.npy;
    else recompute from omega via the motor model. Returns (array[n_ep], source-str)
    or (None, None) if power is unavailable."""
    p_path = os.path.join(run_dir, "power_per_step.npy")
    if os.path.exists(p_path):
        pw = np.load(p_path).astype(float)
        if pw.ndim == 1:
            pw = pw[None, :]
        pw = pw[:n_ep]
        with np.errstate(invalid="ignore"):
            per_ep = np.nanmean(pw, axis=1)
        if np.isfinite(per_ep).any():
            return per_ep, "logged"
    if motor_power is None:
        return None, None
    per_ep = np.full(n_ep, np.nan)
    for e in range(n_ep):
        w = om[e][~np.isnan(om[e])] * RPM_TO_RADS
        if w.size < 2:
            continue
        per_ep[e] = motor_power.average_power(np.arange(w.size) * dt, w)
    return (per_ep, "recomputed from ω") if np.isfinite(per_ep).any() else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="e.g. results/td3/full3d_s0")
    ap.add_argument("--last-n", type=int, default=3, help="how many final episodes to overlay")
    ap.add_argument("--dt", type=float, default=1.0,
                    help="seconds per control step (freeform_dt); used to recompute power")
    args = ap.parse_args()
    d = args.run_dir

    ret = np.load(os.path.join(d, "episode_returns.npy"))
    om = np.load(os.path.join(d, "omega_per_step.npy")).astype(float)  # [ep, step]
    if om.ndim == 1:
        om = om[None, :]
    n_ep = len(ret)
    om = om[:n_ep]                      # align to completed episodes
    steps = om.shape[1]
    w = min(10, n_ep)                   # trailing-mean window, shared by both curves
    pw_ep, pw_src = episode_mean_power(d, om, n_ep, args.dt)

    fig = plt.figure(figsize=(12.5, 8))
    gs = fig.add_gridspec(2, 2)

    # ---- 1. learning curve (return) ----
    ax = fig.add_subplot(gs[0, 0])
    mov = np.array([ret[max(0, i - w + 1):i + 1].mean() for i in range(n_ep)])
    ax.plot(np.arange(n_ep), ret, "-o", ms=3, lw=1, alpha=0.5, label="per-episode")
    ax.plot(np.arange(n_ep), mov, "-", lw=2.2, color="C3", label=f"trailing mean ({w})")
    ax.set_xlabel("episode"); ax.set_ylabel("return  (conversion $-$ energy)")
    ax.set_title(f"Learning curve ({n_ep} episodes)")
    ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=8)

    # ---- 2. energy cost across training (mean motor power per episode) ----
    axp = fig.add_subplot(gs[0, 1])
    if pw_ep is not None:
        movp = np.array([np.nanmean(pw_ep[max(0, i - w + 1):i + 1]) for i in range(n_ep)])
        axp.plot(np.arange(n_ep), pw_ep, "-o", ms=3, lw=1, alpha=0.5, color="C0",
                 label="per-episode")
        axp.plot(np.arange(n_ep), movp, "-", lw=2.2, color="C1",
                 label=f"trailing mean ({w})")
        axp.set_xlabel("episode"); axp.set_ylabel("mean motor power  [W]")
        axp.set_title(f"Energy cost across training ({pw_src})")
        axp.grid(alpha=0.3); axp.legend(loc="best", fontsize=8)
    else:
        axp.text(0.5, 0.5, "no power data\n(need omega log or motor_power)",
                 ha="center", va="center", transform=axp.transAxes, color="0.4")
        axp.set_axis_off()

    # ---- 3. omega heatmap across training ----
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(om, aspect="auto", origin="lower", cmap="viridis",
                   extent=[0, steps, 0, n_ep], interpolation="nearest")
    ax.set_xlabel("step within episode"); ax.set_ylabel("episode")
    ax.set_title("Policy: chosen $\\omega$ across training")
    fig.colorbar(im, ax=ax, label="$\\omega$ [rpm]")

    # ---- 4. early vs late omega trajectories ----
    ax = fig.add_subplot(gs[1, 1])
    x = np.arange(steps)
    ax.plot(x, om[0], color="0.6", lw=1.5, label="episode 1 (explore)")
    for k in range(max(1, n_ep - args.last_n), n_ep):
        ax.plot(x, om[k], lw=1.6, label=f"episode {k+1}")
    ax.set_xlabel("step within episode"); ax.set_ylabel("$\\omega$ [rpm]")
    ax.set_title("Behavior: early vs final episodes")
    ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    out = os.path.join(d, "td3_behavior.png")
    fig.savefig(out, dpi=150)
    print(f"episodes={n_ep}  mean_return={ret.mean():+.3f}  "
          f"last{w}_mean={ret[-w:].mean():+.3f}")
    # describe the converged policy
    final = om[-args.last_n:]
    final = final[~np.isnan(final)]
    if final.size:
        print(f"final-episode omega: mean={final.mean():.0f} rpm  "
              f"min={final.min():.0f}  max={final.max():.0f}  "
              f"spread(std)={final.std():.0f} rpm  "
              f"-> {'~CONSTANT policy' if final.std() < 25 else 'MODULATING policy'}")
    if pw_ep is not None:
        fp = pw_ep[-w:]
        fp = fp[np.isfinite(fp)]
        if fp.size:
            print(f"final power ({pw_src}): last{w}_mean={fp.mean():.2f} W")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
