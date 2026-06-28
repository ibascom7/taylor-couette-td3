#!/usr/bin/env python3
"""CMA-ES search for the best burst-and-idle inner-cylinder waveform on the
catalysis wedge -- the sample-efficient alternative to TD3 for the waveform
problem (Lopez-Guajardo modulation).

WHY THIS INSTEAD OF TD3 FOR THE WAVEFORM:
  The waveform is just THREE numbers -- (peak omega, duty D, period T). That is a
  tiny continuous optimization, not a sequential-decision problem, so TD3 (which
  needs thousands of transitions + a Markov state it doesn't really have here) is
  the wrong tool. CMA-ES is the gold standard for low-dimensional, expensive,
  noisy black-box objectives: it keeps a Gaussian "cloud" over the search space,
  each GENERATION samples a small population of candidates, evaluates them, then
  moves/reshapes the cloud toward the good ones. ~50-150 evaluations vs TD3's
  thousands.

WHAT IT OPTIMIZES (the thesis: "less power at the SAME conversion"):
  Each candidate is an action a in [-1,1]^3, decoded by the SAME env._decode used
  in training into (peak, D, T). The waveform env (per_episode=True) runs that
  waveform for --episode_duration s from the warmed IC and returns the windowed
  conversion and the motor energy. Default objective (constrained): MINIMIZE motor
  power SUBJECT TO conversion >= a target (penalty for missing it). The target
  defaults to the conversion a CONSTANT --ref_rpm achieves, so the result is
  directly "the modulated waveform that hits the same conversion as constant
  --ref_rpm, at lower power." A constant-omega sweep (duty=1) is run first to (a)
  set that target and (b) draw the constant power-vs-conversion curve the winning
  waveform is plotted against.

  Reuses TaylorCouetteWaveformEnv unchanged -- the env IS the objective, so there
  is no new physics/CFD here. Env flags MUST match the TD3 train/compare slurms.

Run:  see run_carya_cma_waveform.slurm   (needs `pip install cma` in the venv).
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GYM_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if GYM_DIR not in sys.path:
    sys.path.insert(0, GYM_DIR)

from taylor_couette_mixing.envs.taylor_couette_waveform import TaylorCouetteWaveformEnv
import cma


RPM = 2.0 * np.pi / 60.0


def build_env(args):
    """Waveform env in per-episode (bandit) mode -- one (peak,D,T) per evaluation.
    Mirrors how train.py constructs it so CMA and TD3 see the identical physics."""
    return TaylorCouetteWaveformEnv(
        case_path=args.case_path,
        per_episode=True,
        episode_duration=args.episode_duration,
        period_min=args.period_min,
        period_max=args.period_max,
        duty_min=args.duty_min,
        duty_max=args.duty_max,
        r_in=args.r_in,
        r_out=args.r_out,
        E_max_per_step=args.e_max_per_step,
        warmup_duration=args.warmup_duration,
        warmup_omega_rpm=args.warmup_omega_rpm,
        energy_model=args.energy_model,
        max_steps=1,
    )


def evaluate(env, action, episode_duration, seed):
    """Run one waveform episode for `action` (in [-1,1]^3) and return its metrics.
    A hard reset before each eval makes every candidate start from the same warmed
    IC, so (conversion, power) reflect the waveform alone."""
    env.reset(seed=seed, options={"reset_mode": "hard"})
    _, reward, _, _, info = env.step(np.asarray(action, dtype=float))
    conv = float(info["mixing_index"])              # windowed conversion in [0,1]
    energy = float(info["energy_consumption"])      # motor electrical energy [J]
    power = energy / episode_duration               # average electrical power [W]
    return {
        "conv": conv, "power": power, "energy": energy, "reward": float(reward),
        "peak": float(env.peak), "duty": float(env.duty), "period": float(env.period),
    }


def action_for_constant_rpm(env, rpm):
    """Action whose decode is a CONSTANT speed: duty -> duty_max (=1 if duty_max=1),
    peak -> rpm, period irrelevant. Used for the constant-omega reference sweep."""
    a0 = 2.0 * (rpm - env.omega_min) / (env.omega_max - env.omega_min) - 1.0
    return [float(np.clip(a0, -1.0, 1.0)), 1.0, 0.0]


def main():
    ap = argparse.ArgumentParser()
    # ---- env flags (MUST match run_carya_train_catalysis_td3.slurm) ----
    ap.add_argument("--case_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--r_in", type=float, default=25.4)
    ap.add_argument("--r_out", type=float, default=31.75)
    ap.add_argument("--e_max_per_step", type=float, default=0.0011017031875434)
    ap.add_argument("--warmup_duration", type=float, default=100.0)
    ap.add_argument("--warmup_omega_rpm", type=float, default=500.0)
    ap.add_argument("--episode_duration", type=float, default=120.0)
    ap.add_argument("--period_min", type=float, default=5.0)
    ap.add_argument("--period_max", type=float, default=30.0)
    ap.add_argument("--duty_min", type=float, default=0.1)
    ap.add_argument("--duty_max", type=float, default=1.0)
    ap.add_argument("--energy_model", default="motor")
    ap.add_argument("--seed", type=int, default=0)
    # ---- CMA-ES knobs ----
    ap.add_argument("--popsize", type=int, default=8,
                    help="candidates per generation (lambda). 8 is good for 3-D.")
    ap.add_argument("--max_evals", type=int, default=96,
                    help="total waveform evaluations (~popsize x generations).")
    ap.add_argument("--sigma0", type=float, default=0.35,
                    help="initial CMA step size over the [-1,1] action box.")
    # ---- objective ----
    ap.add_argument("--objective", choices=["constrained", "weighted"],
                    default="constrained",
                    help="constrained: min power s.t. conv>=target (the thesis). "
                         "weighted: max conv - power_weight*power_norm.")
    ap.add_argument("--conv_target", type=float, default=None,
                    help="conversion floor for 'constrained' (default: the constant "
                         "--ref_rpm conversion, so it's an EQUAL-conversion comparison).")
    ap.add_argument("--ref_rpm", type=float, default=500.0,
                    help="constant speed whose conversion sets the default target.")
    ap.add_argument("--penalty", type=float, default=10.0,
                    help="penalty weight on (target - conv)_+ in 'constrained'.")
    ap.add_argument("--power_weight", type=float, default=1.0,
                    help="lambda in the 'weighted' objective.")
    ap.add_argument("--const_sweep", default="0,300,500,800,1200,1600,2000,2500",
                    help="constant-omega rpm references (duty=1) for the curve/target.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    env = build_env(args)
    # Reference power = motor power running constantly at the warmup speed (W),
    # to normalize the objective to ~O(1). (env.motor_e_norm is energy over one
    # episode at warmup_omega; /time_step -> average power.)
    power_ref = (env.motor_e_norm / env.time_step) or 1.0

    # Incremental CSV: a wall-timeout still preserves every completed evaluation
    # (CMA-ES has no mid-run checkpoint of its own).
    FIELDS = ["eval", "kind", "gen", "peak_rpm", "duty", "period_s",
              "conv", "power_W", "power_norm", "objective"]
    csv_f = open(os.path.join(args.out, "cma_log.csv"), "w", newline="")
    csv_w = csv.DictWriter(csv_f, fieldnames=FIELDS)
    csv_w.writeheader(); csv_f.flush()
    log_rows = []          # every evaluation (constant refs + CMA candidates)
    eval_idx = [0]

    def record(kind, gen, r, obj=None):
        eval_idx[0] += 1
        row = {"eval": eval_idx[0], "kind": kind, "gen": gen,
               "peak_rpm": r["peak"], "duty": r["duty"], "period_s": r["period"],
               "conv": r["conv"], "power_W": r["power"],
               "power_norm": r["power"] / power_ref, "objective": obj}
        log_rows.append(row)
        csv_w.writerow(row); csv_f.flush()
        print(f"[cma] {kind:6s} gen={gen} eval={eval_idx[0]:3d} "
              f"peak={r['peak']:7.1f} D={r['duty']:.2f} T={r['period']:5.1f} "
              f"conv={r['conv']:.4f} power={r['power']:.3e}W obj={obj}",
              flush=True)

    # ---- 1) constant-omega reference sweep (duty=1) ---------------------
    const_pts = []   # (rpm, conv, power)
    for rpm in [float(x) for x in args.const_sweep.split(",") if x.strip()]:
        r = evaluate(env, action_for_constant_rpm(env, rpm), args.episode_duration, args.seed)
        record("const", -1, r)
        const_pts.append((rpm, r["conv"], r["power"]))
    const_pts.sort(key=lambda p: p[1])          # sort by conversion for interpolation
    cc_conv = np.array([p[1] for p in const_pts])
    cc_pow = np.array([p[2] for p in const_pts])

    def constant_power_at(conv):
        """Interpolate constant-speed power at a given conversion (the curve the
        modulated waveform must beat). NaN if outside the swept range."""
        if conv < cc_conv.min() or conv > cc_conv.max():
            return float("nan")
        return float(np.interp(conv, cc_conv, cc_pow))

    # Default target = conversion of the constant ref_rpm (equal-conversion test).
    if args.conv_target is None:
        ref = evaluate(env, action_for_constant_rpm(env, args.ref_rpm),
                       args.episode_duration, args.seed)
        record("ref", -1, ref)
        target = ref["conv"]
    else:
        target = args.conv_target
    print(f"[cma] conversion target = {target:.4f} "
          f"({'constant %g rpm' % args.ref_rpm if args.conv_target is None else 'user-set'})",
          flush=True)

    def objective(r):
        pn = r["power"] / power_ref
        if args.objective == "constrained":
            return pn + args.penalty * max(0.0, target - r["conv"])
        return -(r["conv"] - args.power_weight * pn)

    # ---- 2) CMA-ES over the action box [-1,1]^3 -------------------------
    es = cma.CMAEvolutionStrategy(
        3 * [0.0], args.sigma0,
        {"bounds": [3 * [-1.0], 3 * [1.0]], "popsize": args.popsize,
         "maxfevals": args.max_evals, "seed": args.seed, "verbose": -9},
    )
    best = None
    gen = 0
    while not es.stop():
        cand = es.ask()
        costs = []
        for a in cand:
            r = evaluate(env, a, args.episode_duration, args.seed)
            c = objective(r)
            costs.append(c)
            record("cma", gen, r, c)
            if best is None or c < best["objective"]:
                best = {**r, "objective": c, "action": list(map(float, a))}
        es.tell(cand, costs)
        gen += 1
        if best is not None:   # best-so-far each generation, so a timeout still yields the winner
            with open(os.path.join(args.out, "best.json"), "w") as bf:
                json.dump({"peak_rpm": best["peak"], "duty": best["duty"],
                           "period_s": best["period"], "conv": best["conv"],
                           "power_W": best["power"], "objective": best["objective"],
                           "n_evals": eval_idx[0]}, bf, indent=2)

    # ---- 3) report: equal-conversion power saving -----------------------
    const_at_best = constant_power_at(best["conv"])
    saving_W = const_at_best - best["power"]
    saving_pct = 100.0 * saving_W / const_at_best if const_at_best == const_at_best and const_at_best > 0 else float("nan")

    summary = {
        "objective": args.objective,
        "conv_target": target,
        "best_waveform": {
            "peak_rpm": best["peak"], "duty": best["duty"], "period_s": best["period"],
            "conv": best["conv"], "power_W": best["power"],
        },
        "equal_conversion_comparison": {
            "constant_power_at_best_conv_W": const_at_best,
            "waveform_power_W": best["power"],
            "power_saving_W": saving_W,
            "power_saving_pct": saving_pct,
        },
        "constant_sweep": [{"rpm": p[0], "conv": p[1], "power_W": p[2]} for p in const_pts],
        "n_evals": eval_idx[0],
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    csv_f.close()   # cma_log.csv was written incrementally during the run

    print("\n=== CMA-ES result ===")
    print(f"best waveform: peak={best['peak']:.0f} rpm  D={best['duty']:.2f}  "
          f"T={best['period']:.1f} s  -> conv={best['conv']:.4f}  power={best['power']:.3e} W")
    if const_at_best == const_at_best:   # not NaN
        print(f"constant speed at the same conversion ({best['conv']:.3f}) needs "
              f"{const_at_best:.3e} W")
        print(f"==> modulation saves {saving_W:.3e} W ({saving_pct:.1f}%) at equal conversion"
              if saving_W > 0 else
              f"==> modulation did NOT beat constant here ({saving_W:.3e} W).")
    else:
        print("best conversion is outside the constant sweep range -- widen --const_sweep.")

    _maybe_plot(args.out, const_pts, log_rows, best, target)


def _maybe_plot(out, const_pts, log_rows, best, target):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                       # plotting is optional
        print(f"[cma] (skip plot: {e})")
        return
    cma_rows = [r for r in log_rows if r["kind"] == "cma"]
    fig, ax = plt.subplots(figsize=(7, 5))
    cp = sorted(const_pts, key=lambda p: p[1])
    ax.plot([p[1] for p in cp], [p[2] for p in cp], "o-", color="#1f77b4",
            label="constant speed", zorder=2)
    if cma_rows:
        ax.scatter([r["conv"] for r in cma_rows], [r["power_W"] for r in cma_rows],
                   s=14, c="#aaaaaa", label="CMA candidates", zorder=1)
    ax.scatter([best["conv"]], [best["power"]], s=120, marker="*", c="#2ca02c",
               label="best waveform", zorder=3)
    ax.axvline(target, ls="--", c="k", lw=0.8, label=f"conv target {target:.2f}")
    ax.set_xlabel("conversion"); ax.set_ylabel("motor power [W]")
    ax.set_title("Modulated waveform vs constant speed (lower-right = better)")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out, "power_vs_conversion.png"), dpi=130)
    print(f"[cma] wrote {os.path.join(out, 'power_vs_conversion.png')}")


if __name__ == "__main__":
    main()
