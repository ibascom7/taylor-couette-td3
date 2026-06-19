# Catalysis RL — TD3 vs constant vs square-wave on conversion

Train a **TD3 agent** to drive the inner-cylinder speed on the **catalytic-wall**
Taylor–Couette wedge, then compare it against the two prescribed baselines —
**constant omega** and **square-wave omega** — on the quantity the paper
(López-Guajardo et al., *Chem. Eng. J.* 489 (2024) 151174) actually reports:
**conversion** (`1 − cup-mixing outlet C`), with input energy as the second axis.

This is the RL counterpart of `experiments/oscillation_vs_constant` (which only
runs the two *prescribed* controllers). Here the agent is free to invent its own
ω(t); the baselines are special cases it can reproduce.

## The setup

The catalytic case feeds reactant at `c0 = 1` from the top inlet and consumes it
at the **outer (catalytic) wall** (a `C = 0` sink). Fast inner-cylinder rotation
thins the wall concentration boundary layer → more diffusive flux into the wall →
more conversion, but it costs power. The paper's claim is that *modulating* ω
delivers **a given conversion at lower power** than a constant ω — i.e. the win
shows up when you compare **at equal conversion**, not at equal mean speed (see
"Energy / power model and equal conversion" below).

**Env** (`taylor_couette_mixing/envs/taylor_couette_catalysis.py`,
`TaylorCouetteCatalysisEnv`): a stepwise env where each simulated second the
agent picks an **absolute** ω in `[0, 2500] rpm` — wide enough to reproduce the
constant (500 rpm) and square-wave (0/2500 rpm) baselines or anything in between.

**Reward** per step:

```
reward = conv_weight · conversion  −  energy_weight · E_norm
```

`E_norm` is the per-step input energy, normalized so it's ≈ 1 at the mean speed.

**Energy model (`--energy_model`, default `motor`) — this is what makes the
"dream" reachable.** There are two ways to score energy:

- **`motor`** — the paper's electric-motor power model (Eqs. 18–23,
  `taylor_couette_mixing/motor_power.py`): inertia + **bearing dry friction**
  (linear in ω) + drag, through the motor's electrical circuit, with
  **regenerative braking**. Idle is ~free, braking *returns* energy, and the cost
  is dominated by the *linear* bearing term — so brief high-speed bursts at a low
  mean are cheap. This is the regime where modulation can win, so it's the only
  setting under which the agent can **discover** a paper-style ω(t).
- **`mechanical`** — viscous-drag work from the CFD (`ρ·Mz_kin·ω`), the original
  metric. Convex in ω (∝ ω³), so it *punishes* bursts and the agent collapses to a
  near-constant speed. Kept for contrast.

Why it matters: with the mechanical model, even the prescribed square wave costs
~9× the constant at equal mean (Jensen's inequality on a convex cost); with the
motor model it's the *opposite* once you compare fairly (see below). If the reward
penalizes bursts, the agent will never learn to burst.

**`energy_weight`** is the second knob:

| `energy_weight` | what the agent learns |
|---|---|
| too **high** | idles ω → 0 (cheap, but low conversion) |
| too **low**  | pins ω → 2500 (max conversion, max power) |
| intermediate | the paper's "more conversion at less energy" regime |

Start at `0.1` (the slurm default) and inspect the learned waveform in
`results/td3/<tag>/omega_per_step.npy` — if it collapses to a rail, adjust.

## Run

```bash
cd taylorCouetteGym
# 1) train (saves checkpoints + ParaView frames for episodes 1, 10, final)
sbatch experiments/catalysis_rl/run_carya_train_catalysis_td3.slurm
#    use --array=0-2 in the slurm for 3 seeds

# 2) after a td3_tc_final checkpoint exists, compare vs both baselines
sbatch experiments/catalysis_rl/run_carya_compare_catalysis.slurm
```

Both slurms configure the case with `make_case.py --catalysis --mode constant`
(reused from `oscillation_vs_constant`), then delete the stale *mixing*
`0.orig/`/`0.warmed/` so the env re-warms a true **catalytic** initial condition.

## Outputs

Training → `results/td3/catalysis_s<seed>/`:
- `td3_tc_*_actor` / `_critic` … — checkpoints (`td3_tc_final` is the trained agent)
- `episode_returns.npy`, `omega_per_step.npy`, `reward_per_step.npy` — training logs
- `frames/ep0001/`, `frames/ep0010/`, `frames/ep<final>/` — per-second OpenFOAM
  time dirs of those episodes, each with a `.foam` file → open in **ParaView**

Compare → `results/comparison_s<seed>/` (the compare runs **constant- and
pulsating-speed sweeps** plus the agent, so the plots are real curves, not single
points — and reports the **equal-conversion power gap** the way the paper's
Fig. 11 does):
- `fig_conversion_vs_speed.png` — **Fig. 7 style**: conversion vs *mean* angular
  speed, constant vs pulsating. Shows you must spin a *constant* shaft faster to
  match a slower *pulsating* one. The agent is placed at its mean ω.
- `fig_power_vs_conversion.png` — **Fig. 11 style**: avg **motor** power vs
  conversion, constant vs pulsating, with the equal-conversion gap annotated
  (pulsating should sit *below* constant over a conversion band). The agent's
  point shows whether it beats both at its own conversion.
- `fig_omega_ts.png` — **Fig. 3 style**: the ω(t) waveform (with ramps) each
  controller ran — including the agent's **learned** waveform (the dream plot).
- `fig_conversion_ts.png` — conversion vs time for the canonical three.
- `fig_duty.png` — **Fig. 6 echo** (only with `--duty-sweep`): conversion vs duty
  cycle at fixed mean speed; low duty (~20%) should win.
- `summary.txt` — the sweep tables, the equal-conversion gaps, and the TD3 verdict.
- `baseline_sweep.npz` — cached baseline sweeps (policy-independent), so re-running
  after retraining the agent only re-runs the agent (`--refresh-baselines` to redo).
- `frames_constant/`, `frames_squarewave/`, `frames_td3/` — canonical trajectories
  for ParaView.

## Energy / power model and "equal conversion"

The paper's headline (modulation uses ~25% less power) holds only when you (a)
compare at **equal conversion** — not equal mean speed — and (b) use the
**electric-motor power**, not viscous-drag work. Both are baked into the compare:

- **Equal conversion:** the constant strategy is swept over speed to build a
  conversion-vs-speed curve; the pulsating point's power is then compared to the
  constant power *interpolated at the same conversion*. (A pulsating mean-500 reaches
  the same conversion as a constant ~689 rpm, but at lower mean speed → less power.)
- **Motor power:** see `motor_power.py`. Validated numerically — constant 500 rpm
  ≈ 6.1 W, square mean-500 ≈ 6.5 W (only +6% at *equal mean*), constant-689 ≈ 8.4 W,
  so pulsating beats constant-at-equal-conversion by ~23% (paper: 25%).

## Numerical stability (ω ramps)

The case uses **adaptive timestepping** (`adjustTimeStep yes; maxCo 0.8`, written
by `make_case.py`) so the solver shrinks `dt` to cap the Courant number at high ω
— this is why ω can reach 2500 rpm here (a *fixed* `deltaT` is what crashed the
old mixing RL case around 1000 rpm). On top of that, the agent's ω changes are
**ramped over `--ramp_time` s (default 0.05)** rather than applied instantly: each
step writes a tabulated `omega` Function1 on the wall that ramps from the previous
speed to the new one, then holds — the same trick `make_case.py` uses for the
square wave. This keeps the wall acceleration finite so a big action can't spike
the Courant number on the first timestep. Set `--ramp_time 0` for instant jumps.

## Notes / fairness

- All three controllers run through the **same env**, from the **same warmed IC**,
  at the **same 1 s control cadence**, with the **same ω ramps**, so the
  comparison is apples-to-apples. The baselines are realized by inverting the
  env's action→ω map.
- The square-wave is a 1 s-resolution staircase (active 2500 rpm for the first
  `duty·period` s of each period, else 0) — the same cadence the agent controls
  at, by design.
- Geometry/normalizer flags (`--r_in 25.4 --r_out 31.75
  --e_max_per_step 0.0011017031875434`) are the wedge values; the compare slurm
  passes the **same** ones used in training (they must match).
- This is the 2D-axisymmetric wedge (faithful to the paper's own model). The
  diffusivity `--scalar-D 1e-8` keeps the wall boundary layer resolvable
  (paper ~1e-10).
