# endgame — the gap-closing experiments for the October 2 talk / paper

Four experiments that turn the existing results into defensible claims.
Everything here reuses the proven pipelines (duty_v3 env, fig7 sweep engine,
eval_full_tc) and follows the same metric conventions, so new rows drop
straight into the existing tables. Plan and rationale: the 2026-09-13 paper
spine (thesis: short-stop film-renewal pulsing beats Lopez-Guajardo's D=20 %
pulse; TD3 discovers it from scratch and transfers zero-shot).

| | question | script | slurm | cost |
|---|---|---|---|---|
| **A** | What is the true best *fixed* waveform in the RL's own family, with a noise floor? (the honest comparator for duty_v3; replaces the n=1 hand-picked champion and Lopez-Guajardo's Fig. 6) | `a_champion_search.py` | `run_carya_A_champion_search.slurm` | ~20 h, 44 Carya workers |
| **B** | Does the short-reactor champion family beat the D80/T10 pulse on Γ=30? (the honest comparator for the zero-shot transfer) | `gamma30_statics.py --group B` | `run_carya_B_gamma30_champion.slurm` | 3 × ~25 h, 1 core each |
| **C** | Does the transfer policy's end-of-run gain (+0.025 X) survive 3 more residence times, or was it a harvest? | `c_continue_transfer.py` | `run_carya_C_continue_transfer.slurm` | 2 × ~27 h |
| **E** | Lopez-Guajardo's optimum (D=20 %, T=30 s @500) vs their local minimum vs constant, on our Γ=30 twin at Sc=1075 **and** Sc=1e4: is the duty disagreement a Schmidt-number effect? | `gamma30_statics.py --group E` | `run_carya_E_lopez_optimum.slurm` | 6 × 25–60 h |

(D, the grid-independence study, lives in its own plan and is not here.)

## Conventions (do not change without re-running the comparators)

* **A** — `TaylorCouetteDutyV2Env` exactly as `run_carya_duty_v3.slurm`:
  warmed constant-300 start, 5 × 26 s τ-blocks, action held for the whole
  episode, objective = last-block reward `r5 = X_[104,130] − P/31.94`.
  Comparable to `modulation_rl/results/duty_v3_refs` (constant 0.2441,
  n=1 champion 0.2948) and the five deterministic duty_v3 evals
  (0.273–0.284). The optimizer is a numpy-only GP (Matern-5/2, EI,
  Kriging-believer for in-flight points), asynchronous over the workers,
  resumable from `evaluations.csv`.
* **B, E** — `full_tc_cat_case`, 300 s from the pristine IC, burst-first,
  0.05 s ramps, tabulated omega in `0/U`; `X_last10` = last-10 s mean
  (= the transfer eval's final block = the fig7 last-period window),
  plus `X_lastT` and `X_tau` (last 130 s); `R− = X_last10 − P/31.94`.
  E sets `D = ν/Sc` in both `constant/transportProperties` and the
  `scalarTransport` function object.
* **C** — the modulation env is rebuilt on a minimal copy of the kept 300 s
  case **without reset()**; its state is reconstructed from the source
  `blocks.csv`; the clock obs is held at the last value the policy saw
  (stretch) or keeps wrapping (wrap). `--mode frozen` repeats the final
  action open-loop instead of running the policy.

## Running

Carya (submit **from `taylorCouetteGym/`** — paths are submit-dir relative):

```bash
sbatch experiments/endgame/run_carya_A_champion_search.slurm
sbatch experiments/endgame/run_carya_B_gamma30_champion.slurm
sbatch experiments/endgame/run_carya_C_continue_transfer.slurm
sbatch experiments/endgame/run_carya_E_lopez_optimum.slurm
```

Local (B + C + E = 11 cores, A needs Carya):

```bash
cd taylorCouetteGym && bash experiments/endgame/launch_local.sh
```

Every job is resumable after a walltime kill by resubmitting the same
script (A: `--resume` reloads `evaluations.csv`; B/E: the episode continues
from its latest time folder; C: from its own `blocks.csv`).

Smoke tests (real pimpleFoam, minutes):

```bash
python3 experiments/endgame/a_champion_search.py --smoke --worker_root /tmp/$USER/endgame_A_smoke
python3 experiments/endgame/gamma30_statics.py --group E --index 3 --smoke
python3 experiments/endgame/c_continue_transfer.py --source s2_stretch --clock hold --smoke
```

Analysis without CFD: add `--analyze-only` to any script.

## Outputs → figures

* `results/A/A_surface.png` — the static (T+, T−) landscape with the
  posterior argmax, the anchors, and the T−=0.5 s line: **Fig. 2** of the
  paper (replaces a Lopez-Guajardo-style Fig. 6). `A_idle_collapse.png` is
  the off-diagonal test of the idle-duration hypothesis. `summary.json`
  holds the champion, its GP uncertainty, and the replicate noise floor.
* `results/B/summary_table.csv` + `results/C/<tag>/conversion_vs_time.png`
  — the transfer figure (**Fig. 6**): policy trace to 300 s, continuation,
  static references including the champion family.
* `results/E/summary_table.csv` + `conversion_vs_time.png` — the
  Lopez-Guajardo comparison rows for **Fig. 4** and the Sc discussion.
