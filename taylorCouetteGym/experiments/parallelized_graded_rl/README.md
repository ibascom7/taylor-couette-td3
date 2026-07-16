# parallelized_graded_rl

Parallelized TD3 training on the **graded-mesh, resolved-gradient** side-outlet
case — the same N-workers → one-shared-replay-buffer → one-learner harness as
[`parallelized_catalysis_rl`](../parallelized_catalysis_rl/README.md), but the
environment CFD is `side_outlet_grad_case`: an RL-drivable clone of the
steady-state sweep's `side_outlet_case_sc1075_graded`. **No Sherwood wall
model.** The catalytic outer wall is a plain `c = 0` sink and the 25 µm graded
wall cells *resolve* the Sc = 1075 concentration film, so the wall flux the
reward is built on is real `-D·∂c/∂n` physics, not the ETW correlation.

## Why this experiment exists

The wall-model runs (`parallelized_catalysis_rl`, post omega-freeze-fix round 2)
showed TD3 discovering **pulsed modulation** that beats constant rotation — but
the wall flux there *responds to rotation by construction* (`Sh = a·Re^b·Sc^c`).
Yuhe's steady-state sweep then showed the graded resolved-gradient case also has
a real, monotone conversion–omega dependence (0.146 at 0 rpm → 0.568 at
2500 rpm, `steady_state_conversion/results_graded_sc1075`). The natural next
question:

> **Does TD3 still find (pulsed) modulation when the film response is resolved
> rather than modeled?** If brief bursts genuinely thin a *resolved* film, the
> modulation result stops depending on the wall closure.

## The case: `side_outlet_grad_case`

`taylor_couette_mixing/cases/side_outlet_grad_case` is byte-identical physics to
`side_outlet_case_sc1075_graded` (mesh: 37×1×90 wedge, 25 µm outer-wall cell;
`nu = 1.075e-5`, `D = 1e-8` ⇒ Sc = 1075; `c0 = 50` feed; 100 mL/min; settable
`rotatingWallVelocity` inner wall). Only the bookkeeping differs, to meet the RL
env's step-loop contract (the sweep case is left untouched so its
`--analyze-only` reproducibility is preserved):

- `0.orig/` snapshot (the fan-out/`reset` IC source expected by
  `parallel_train.py` and `helpers.reset_case`);
- `system/controlDict`: `startFrom latestTime`, `writeInterval 1`
  (adjustableRunTime), `timePrecision 9` — the env restarts pimpleFoam from the
  latest written time dir every step;
- the sweep's `.dat`-writing function objects are replaced by one `rlMetrics`
  coded FO emitting the `METRICS t= Mz_kin= conv= cupC= wallFlux=` line
  `helpers._parse_metrics` consumes. `wallFlux` is the **resolved** outer-wall
  consumption `Σ(−D·∂c/∂n·dA)/c0` in m³/s **per unit c0** — the same units and
  mass-balance identity (`wallFlux = Q·conversion` at steady state) as
  `side_outlet_cat_case`'s modeled flux, so the env's `wallflux_max`
  normalization carries over unchanged.

## Reward calibration (from the steady sweep + motor model)

`WALLFLUX_MAX = Q_wedge · conv(2500 rpm) = 2.315e-8 × 0.568 = 1.32e-8` m³/s per
unit c0 (mass balance on the sweep's steady point), so `wf_index ≈ 1` at the
omega_max operating point — the same semantics as the wall-model runs. The
steady constant-omega landscape:

| rpm | 0 | 250 | 500 | 750 | 1000 | 2500 |
|---|---|---|---|---|---|---|
| conversion | 0.146 | 0.347 | 0.416 | 0.459 | 0.487 | 0.568 |
| wf_index | 0.256 | 0.611 | 0.732 | 0.809 | 0.857 | 1.000 |
| E_index (motor) | 0 | 0.099 | 0.191 | 0.285 | 0.380 | 1.000 |

⇒ the best *constant* omega\* by regime: ew < 0.23 pegs 2500 rpm, 0.23–0.51 →
1000 rpm, 0.51–0.82 → 750, 0.82–1.30 → 500, 1.30–3.59 → 250, and idle only wins
above ew ≈ 3.6. Note `wf_index = 0.26` at 0 rpm — the resolved film converts
14.6 % with no rotation, unlike the wall-model case, so "idle collapse" is soft
here.

The `ENERGY_WEIGHTS=(0.3 0.6 0.8 1.3 1.8)` grid covers one task per distinct
constant-ω regime rather than piling onto the low end: 0.3/0.6 probe the
1000/750-rpm regimes where burst+regen modulation has the most to gain (measured
transients: a burst from a developed-idle film harvests wf_index ≈ 1.9 for
E_index ≈ 1.2, and braking refunds ~0.12 while flux is still ~0.9, putting the
pulse-vs-slow-constant breakeven around ew ≈ 1.5–2); 0.8/1.3 are shared with the
wall-model round-2 sweep for cross-case comparison; 1.8 sits at the estimated
breakeven — the most discriminating single point.

## Cost (the catch)

The graded mesh's 25 µm wall cells Courant-limit dt. Measured on the
steady-state sweep (single core, 7800X3D): **~41–51 s per simulated second at
0–750 rpm, 66 at 1000 rpm, 179 at 2500 rpm** — about 3–10× the wall-model case.
A 120 s episode is ~1.4–6 h; 150 episodes across 44 workers ≈ 10–20 h + ~1 h
one-time warmup per task (hence `-t 48:0:0`).

## Files

- `run_carya_parallel_graded.slurm` — training: a job ARRAY over `energy_weight`
  (5 tasks × 1 full 48-core node, 44 workers each, 150 episodes × 120 s).
  **Reuses `../parallelized_catalysis_rl/parallel_train.py`** — the trainer is
  case-agnostic (everything case-specific is a flag) and its threading /
  divergence-recovery logic stays single-source; this folder deliberately adds
  no fork of it.
- `run_carya_compare.slurm` — EVAL: drives
  `experiments/catalysis_rl/compare_catalysis.py` on each swept policy vs the
  constant/pulsating baselines → the beats-constant figure
  (`fig_power_vs_conversion.png`) + `summary.txt` verdict per tag. Serial 1-node
  job; computes the baselines ONCE and reuses them for all 5 policies. Run after
  training. Budget note: this case is expensive, the baseline sweep alone is
  ~12–15 h.
- Results land in `results/td3/<tag>/` and `results/comparison/<tag>/`
  (`tag = sog_parallel_freeform_ew<ew>_s<seed>`).

For the CFD-free "what did it learn" look at a finished run, use the sibling's
plotter:

```bash
python experiments/parallelized_catalysis_rl/plot_learned_behavior.py \
    --run experiments/parallelized_graded_rl/results/td3/sog_parallel_freeform_ew0.8_s0 \
    --dt 1.0 --window 30
```

## Usage

```bash
# 1. Train: energy_weight sweep (5 nodes, 44 workers each, 150 episodes/120 s):
sbatch experiments/parallelized_graded_rl/run_carya_parallel_graded.slurm

# 2. Eval (after training): beats-constant figure per swept policy:
sbatch experiments/parallelized_graded_rl/run_carya_compare.slurm
#    then: grep -H 'vs constant at equal conversion' experiments/parallelized_graded_rl/results/comparison/*/summary.txt

# Local smoke test (2 workers, 3 s warmup, tiny budget -- validates the whole path):
python experiments/parallelized_catalysis_rl/parallel_train.py --smoke \
    --case_path taylor_couette_mixing/cases/side_outlet_grad_case \
    --wallflux_max 1.32e-8 \
    --worker_root /tmp/$USER/sog_smoke/workers --results_dir /tmp/$USER/sog_smoke/results
```

Output format is identical to `parallelized_catalysis_rl` (`episode_returns.npy`,
`omega_per_step.npy`, `reward_per_step.npy`, `conv_per_step.npy`,
`power_per_step.npy`, `td3_tc_t<N>_actor/_critic` checkpoints + `td3_tc_final`).
