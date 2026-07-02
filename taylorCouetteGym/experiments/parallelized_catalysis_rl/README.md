# parallelized_catalysis_rl

Parallelized TD3 training for the **side-outlet catalysis** case. The goal is to
collect experience from **N environments at once** — each running its own
`pimpleFoam` rollout — into a **single shared replay buffer** feeding **one**
actor/critic learner. This is the AlphaZero pattern (many self-play games → one
replay buffer → one network), specialized to an expensive CFD environment.

> Status: **implemented + smoke-tested locally.** `parallel_train.py` +
> `run_carya_parallel_catalysis.slurm` are ready. See "Usage" below.

---

## Why this works (and why it's the biggest available speedup)

- **TD3 is off-policy.** Transitions are just `(s, a, s', r, done)` tuples; the
  learner does not care which behavior policy produced them. So we can run K
  actors in parallel, dump every transition into one buffer, and train the
  critic/actor on the pooled data. Nothing about TD3 changes — only *where the
  transitions come from*.
- **The bottleneck is CFD, not the network.** Each `env.step()` shells out to
  `pimpleFoam` (`helpers.do_simulation`), which on the side-outlet case costs
  ~15–22 s of wall-clock **per simulated second** at 2500 rpm. The 400×300
  actor/critic MLP is microseconds. So wall-clock ≈ (CFD seconds) / (parallelism).
  Going 1 → 8–16 workers is a near-linear ~10× speedup: a ~24 h run becomes ~2–3 h.
- **Threads suffice — no multiprocessing needed.** `env.step()` blocks on an
  external subprocess; Python releases the GIL during `subprocess` I/O wait, so N
  threads genuinely run N `pimpleFoam` processes concurrently. This sidesteps
  pickling the env / OpenFOAM handles across processes.

## The one real cost: N case directories on disk

OpenFOAM is **file-stateful** and the env's step-loop contract restarts
`pimpleFoam` from the latest written time dir each step (see the
`openfoam-step-loop-contract` memory). Two workers **cannot** share one case dir —
they would clobber each other's time directories and `foamDictionary` edits.

So each worker gets its **own case copy** (`case_worker00/`, `case_worker01/`, …),
each `blockMesh`ed once, each warmed once (or all seeded from a single shared
`0.warmed/` copied in — warmup is identical across workers, so warm once and
distribute). Budget cores: `N_workers × (cores per pimpleFoam)`; `pimpleFoam` runs
serial here, so 1 core each → N cores for N workers.

## Not the same as the existing job array

`run_carya_train_catalysis_td3*.slurm` already runs a Slurm **array** — but those
are *independent agents on independent seeds*, each with its own buffer and policy.
That parallelizes *experiments*, not *learning*. Here we want N workers → **one**
buffer → **one** policy within a single job, so the learner sees N× the experience
per wall-clock second.

---

## Architecture (proposed)

```
                 ┌─────────── one process ───────────┐
  worker 0  ─┐   │                                    │
  (case_00)  │   │   shared ReplayBuffer (thread-safe)│
  worker 1  ─┼──▶│            │                       │
  (case_01)  │   │            ▼                        │
   ...       │   │   TD3.train(buffer)  × G grad steps │
  worker K-1─┘   │   (one actor + twin critics)        │
  (case_K1)      │            │                        │
                 │            ▼ (params broadcast)      │
                 │   each worker's select_action uses  │
                 │   the latest actor + expl noise     │
                 └────────────────────────────────────┘
```

- **Collection:** a thread pool of K workers, each looping
  `reset → step×max_steps → reset`, pushing every transition into the shared
  buffer. Each worker holds a `TaylorCouetteCatalysisEnv` on its own `case_path`.
- **Learning:** the main thread runs `policy.train(buffer, batch_size)`, scaling
  the number of gradient steps `G` to the number of *new* transitions collected
  (keep the gradient-steps-per-env-step ratio near the serial baseline of 1, so
  the critic doesn't overfit stale data).
- **Sync vs async:** start **synchronous** (barrier: all K workers step, then G
  grad steps, repeat) — simplest and deterministic. Move to async only if worker
  step-time variance (idle vs 2500 rpm steps differ a lot in cost) wastes too much
  wall-clock at the barrier.

## Reuses from the existing code (don't reinvent)

- `TaylorCouetteCatalysisEnv` (`taylor_couette_mixing/envs/taylor_couette_catalysis.py`)
  — unchanged; just instantiate K of them with distinct `case_path`.
- `ReplayBuffer`, `TD3.TD3`, `obs_to_state`, `make_policy` from `train.py`
  — the buffer needs a lock around `add`/`sample` for thread safety (or a
  per-worker staging queue drained by the learner).
- `helpers.Helpers` step-loop, ramp, warmup, snapshot — already per-case.
- The side-outlet case `taylor_couette_mixing/cases/side_outlet_cat_case`.

## Files

- `parallel_train.py` — the implementation: one-time warm-a-template-and-fan-out
  (`prepare_worker_cases`), N threaded collectors (`collector_loop`) feeding a
  thread-safe `ThreadSafeReplayBuffer`, and a single `learner_loop` that paces
  gradient steps to collected transitions. Reuses `ReplayBuffer`,
  `make_obs_to_state`, `make_policy`, `save_logs`, `TD3` from the top-level
  `train.py`, so state/obs/logging/checkpoints stay identical to serial runs.
- `run_carya_parallel_catalysis.slurm` — one job, `-c 18` (16 workers + learner
  headroom), points at the pre-configured `side_outlet_cat_case`.

## Usage

```bash
# Carya (16 workers):
sbatch experiments/parallelized_catalysis_rl/run_carya_parallel_catalysis.slurm

# Local smoke test (2 workers, 3 s warmup, tiny budget -- validates the whole path):
python experiments/parallelized_catalysis_rl/parallel_train.py --smoke \
    --worker_root /tmp/$USER/so_smoke/workers --results_dir /tmp/$USER/so_smoke/results
```
Outputs land in `results/<algo>/<tag>/`: `episode_returns.npy`, `omega_per_step.npy`,
`reward_per_step.npy`, `conv_per_step.npy`, and `td3_tc_t<N>_actor/_critic` (every
`--save_every` grad steps) + `td3_tc_final`. Set `PT_DEBUG_HANG=20` to dump all
thread stacks every 20 s (hang diagnosis).

## Design choices made

1. **K (worker count):** `--n_workers 16` (= cores). Threaded, so pimpleFoam runs
   truly concurrently.
2. **Async collection** (no sync barrier): each worker loops independently, so a
   cheap idle-omega step never waits on a slow 2500 rpm step in another worker.
3. **Gradient ratio** `--grad_per_step` (default 1.0 = serial-equivalent): the
   learner keeps grad_steps ≈ collected − start_timesteps.
4. **Seeding:** all workers share the one warmed IC; each gets a distinct env seed
   + exploration-noise RNG (`seed + worker_id`) so they don't collect identical
   trajectories.
5. **Reward normalization:** inherited from the catalysis env's dimensionless
   `wallflux_max` (Q·c₀) / `E_max` indices -- nothing extra here.

## Robustness (hardened after the smoke test)

- **Divergence recovery:** a diverged pimpleFoam step raises; the worker logs it,
  hard-resets to the warmed IC, and starts a new episode. It only gives up after
  `--max_fail` (default 3) *consecutive* failures. One bad solve never kills a run.
- **No stranded learner:** each worker decrements a live-worker count on exit; when
  the last one leaves it sets the stop flag, and the learner also exits if it can't
  yet fill a batch -- so the run always finalizes (checkpoints + logs) instead of
  hanging.
