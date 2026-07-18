# modulation_rl — TD3 over Lopez-Guajardo waveform parameters (design)

TD3 learns block-wise waveform modulation of the inner cylinder on the graded
Sc=1075 side-outlet reactor, at a FIXED nominal (mean) speed. The design below
was settled in the modulation_vs_constant campaign (2026-07-16..18); that
experiment's README holds the full evidence trail. This file is the build spec —
the trainer + slurm job are implemented separately against it.

## Question the run answers

The static grid already found the best fixed waveform at w_b = 300 rpm:
**D = 80 %, T ≤ 2.5 s ("rapid shimmy"), X ≈ 0.386, R ≈ 0.270** — beating
constant-300 (X = 0.3402) by +0.046 at equal power. TD3's job is NOT to
rediscover that point; it is to answer: **can state-dependent, block-varying
modulation beat the best static waveform?**

- R > 0.28 sustained → genuine discovery (feedback/phasing beats any fixed (D,T)).
- R ≈ 0.27 converged → clean validation (RL recovers the hand-tuned optimum).
- The environment noise floor is ±0.01 in episode reward (measured: chaotic
  wavy-regime sensitivity); differences below that are not resolvable.

## Environment

- Case: `taylor_couette_mixing/cases/side_outlet_case_sc1075_graded`
  (25 um wall cells, D = 1e-8 → Sc = 1075, nu = 1.075e-5, rho = 930, c0 = 50,
  feed 100 mL/min → tau ≈ 26 s). Episodes 50 s from the pre-filled IC.
- Control step = **10 s block**; 5 decisions per episode.
- Inner-wall omega driven by a tabulated Function1 (`omega table (...)`) built
  per block — the proven post-freeze-bug path. CRITICAL: after every
  foamDictionary `-set` of the table, delete any `omegaCoeffs` sub-dict left in
  the BC or the wall freezes at the first block's waveform (see
  `envs/helpers.py:_clear_omega_coeffs`). Reuse `square_wave_points()` from
  `envs/taylor_couette_waveform.py` (ramp = 0.05 s, phase0 support).
- Wave phase: **reset at each block boundary** (every block opens with a
  burst). Simpler than carrying phase, fully determined by (action, t) — no
  hidden state. Note it in the paper as a structural prior.

## Action space (3-D, all in [-1, 1], decoded per block)

| dim | decodes to | range | note |
| --- | --- | --- | --- |
| a0 | duty D | [0.6, 1.0] | linear map |
| a1 | idle speed ω_low | [0, 300] rpm | linear map |
| a2 | period T | [0.5, 5] s | **log map**: T = exp(ln0.5 + (a2+1)/2·ln10) |

Burst speed is SOLVED from the fixed-mean constraint, never chosen:
**ω_hi = (w_b − (1−D)·ω_low) / D**, with w_b = 300 rpm. This guarantees every
block's time-average speed is exactly w_b (equal bearing power ≈ equal motor
power across ALL admissible actions — the reward is therefore effectively pure
conversion; kept as X − P/P_max anyway for comparability across future runs
with different w_b). With D ≥ 0.6 the burst speed is capped at 500 rpm — free
safety bound, film well-resolved.

T floor is 0.5 s (not lower): below that the gap-scale vortices (~0.5–0.6 s
response) low-pass the forcing into effective constant speed, the 0.05 s ramps
consume the waveform, and CFD timesteps resolve the cycle only marginally. The
measured payoff plateau is T ≤ 2.5 s; the step sits between 5 and 2.5 s (idle
crossing the 0.6 s swirl-decay time — `results_td3_prep/period_scan_wb300_D80.png`).

Known degeneracies (tolerated): D = 1 makes a1, a2 meaningless; ω_low = 300
makes D, T meaningless (both decode to constant-300).

## Observation space (7-D, at each block boundary)

1. ∫wallFlux over the last block (block-average consumption; the film-state
   signal — responds in seconds, unlike X which lags by tau)
2. ΔX over the last block (the sag/tongue detector)
3. **X, current, block-averaged** — REQUIRED: the reward's dominant term must
   be observable or the critic is non-Markov
4. episode clock t/50
5–7. previous action (3-D, raw)

ALL flow observables are **block/step AVERAGES, never point samples**: 1 Hz
point sampling aliases the ±0.02 outlet flutter into fake slow waves
(`results_td3_prep/aliasing_check__constant500.png`). Averaging suppresses it 6×.

## Reward (per block)

**R_block = X_block − P_block / P_max**, P_max = 31.94 W (motor power at
2500 rpm), both weights 1. X_block = block-averaged outlet conversion
(flux-weighted cup mean, 1 − c_out/50). P_block = paper motor model
(Eqs 18–23, `taylor_couette_mixing/motor_power.py`) on the commanded omega(t),
densified at 100 Hz like `TaylorCouetteWaveformEnv._waveform_motor_energy`.

## Benchmarks to load into every results plot (50 s, X over [40,50] s)

| policy | X | P [W] | R |
| --- | ---: | ---: | ---: |
| constant 300 | 0.3402 | 3.74 | 0.223 |
| D=80 T=10 | 0.3744 | 3.70 | 0.258 |
| D=80 T=5 | 0.3741 | 3.71 | 0.258 |
| **D=80 T=2.5 (static champion)** | **0.3860** | 3.72 | **0.270** |
| D=80 T=1.25 | 0.3850 | 3.74 | 0.268 |
| constant 400 (off-family reference) | 0.3986 | 4.92 | 0.245 |

Raw traces for all of these: `../modulation_vs_constant/results_td3_prep/` and
`results_frag_T{5,2p5,1p25}/` (0.1 s cadence CSVs + 1 s ParaView folders).

## Training mechanics

- Trainer: reuse the 16-worker threaded TD3 + shared replay buffer from
  `experiments/parallelized_catalysis_rl/parallel_train.py` (warm-once-and-
  fan-out, divergence recovery). 5 transitions per episode.
- Cost (measured, 7800X3D): 50 s episode ≈ 40–53 min on one core → 16 workers
  ≈ 300–450 episodes/day ≈ 1500–2250 transitions/day. Budget ≥ 2–3 days for a
  1000-episode run.
- Optional warm start: seed the buffer with transitions synthesized from the
  grid episodes above (omega histories + traces are on disk; recompute the obs
  from the saved 0.1 s CSVs).
- Exploration noise: per-dim on the raw [-1,1] actions; remember a2 is
  log-mapped, so uniform noise explores octaves of T evenly.
- Step-loop contract: the env continues from the latest written time dir —
  writeInterval must equal the control step boundary spacing (1 s writes are
  safe; see the openfoam-step-loop-contract note).
- Determinism caveat: episode metrics in wavy regimes carry ±0.01
  reproducibility noise (1-ulp omega-table differences amplify chaotically).
  Do not tune on differences below that; average evaluation over ≥3 episodes.

## Fixed for this run / variable across future runs

Fixed: w_b = 300, episode 50 s, block 10 s, reward weights (1, 1), P_max.
Future runs change w_b (the reward stays comparable because P_max is global);
the tall-reactor variant would swap the case for `full_tc_cat_case`
(Gamma = 30, tau ≈ 130 s — needs longer episodes, ~300 s; see
`../modulation_vs_constant/fig7_sweep_full_tc.py`).

---

## Implementation (2026-07-18)

- Env: `taylor_couette_mixing/envs/taylor_couette_modulation.py`
  (`TaylorCouetteModulationEnv`) — flat 7-D obs / 3-D action per the spec
  above; reuses `square_wave_points`, `Helpers.do_simulation_table` (with its
  `_clear_omega_coeffs` freeze-bug guard), and `motor_power` at 100 Hz.
- Trainer: `parallel_train.py` — the parallelized_catalysis_rl harness
  (threaded collectors → one shared replay buffer → one TD3 learner,
  divergence recovery), env swapped and warmup REMOVED (see below).
- Slurm: `run_carya_modulation.slurm` — 1 node, 44 workers, 1000 episodes
  (`tag=mod_wb300_s0`), ~23–27 h, `-t 48:0:0`.

Implementation notes (deviations from the spec text, both deliberate):

1. **Case is `side_outlet_grad_case`**, not `side_outlet_case_sc1075_graded`
   literally: byte-identical physics (verified: same mesh dict, transport
   properties, ICs), but the grad twin carries the `rlMetrics` FO emitting the
   `METRICS t= Mz_kin= conv= cupC= wallFlux=` line the env helpers parse, plus
   `startFrom latestTime` / `writeInterval 1` (the step-loop contract). The
   sweep case has neither and cannot be driven by the env. Same substitution
   parallelized_graded_rl made.
2. **No warmup**: episodes start from the pristine pre-filled IC (`0.orig`,
   c = 50 everywhere, fluid at rest) — exactly how the benchmark episodes in
   the table above were run (`fig7_sweep_td3_prep`), so learned rewards are
   directly comparable. Template prep only compiles the coded FOs once
   (0.05 s throwaway run) and fans the compiled case out to the workers.

Known caveat (measured): when T does not divide the 10 s block, the truncated
last period is burst-first, so the REALIZED block mean sits above w_b — up to
~14 % at the worst corner (D ≈ 0.6, T ≈ 3.9 → mean ≈ 342 rpm). The reward
stays honest because P_block is computed on the commanded wave as-built
(that corner pays P/P_max = 0.132 vs 0.117), but quote block means from
`params_per_step.npy` when writing this up, not w_b. At the static champion
(D = 0.8, T = 2.5) the realized mean is exact and P_block = 3.70 W matches
the benchmark table.

Because every block of a STATIC policy is the same wave, the 5th block's
reward in `reward_per_step.npy[:, 4]` is the benchmark-convention
R = X_[40,50] − P/P_max — compare it directly against the table above
(champion 0.270, discovery threshold 0.28).

### Usage

```bash
# local smoke test (2 workers, 3 x 2 s blocks, real pimpleFoam, ~15 min):
python experiments/modulation_rl/parallel_train.py --smoke \
    --worker_root /tmp/$USER/mod_smoke/workers \
    --results_dir /tmp/$USER/mod_smoke/results

# Carya (from the /project clone after git pull):
sbatch experiments/modulation_rl/run_carya_modulation.slurm
```

Outputs in `results/td3/mod_wb300_s0/`: the shared log set
(`episode_returns.npy`, `reward_per_step.npy`, `conv_per_step.npy` = X_block,
`power_per_step.npy` = block-avg motor W, `omega_per_step.npy` = burst speed
w_hi rpm) plus `params_per_step.npy` `[episode, step, (duty, w_low_rpm,
period_s, w_hi_rpm)]` and `td3_tc_t<N>` / `td3_tc_final` checkpoints.
Warm-starting the buffer from the static-grid episodes (spec §Training
mechanics) is NOT implemented yet — v1 trains from scratch.
