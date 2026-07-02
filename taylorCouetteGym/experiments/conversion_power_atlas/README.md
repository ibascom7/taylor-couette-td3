# conversion_power_atlas

Put every **successful TD3 run** on one **conversion (y) vs mean motor power [W] (x)**
plane, to scope the region of operating points RL discovers and to expose a **bistable
flow-regime transition**. Companion to the per-step **energy logging** added to
training (`power_per_step.npy`).

## Two pieces

### 1. Energy is now tracked through training
The catalysis env already computes the electric-motor energy each step (for the
reward's penalty term); it now also **surfaces** it:

- `taylor_couette_catalysis.py` `_get_info()` adds `energy_step` [J] and
  `power_watt` [W] (`= energy_step / time_step`, the step-average electrical power;
  can be negative on braking/regen).
- `train.py` + `parallel_train.py` accumulate it and `save_logs` writes a new
  **`power_per_step.npy`** `[episode, step]` beside `conv_per_step.npy`.
  Non-catalysis envs write NaN (nan-safe `info.get`), so nothing else breaks.

This is for **live monitoring**. The atlas below does **not** depend on it.

### 2. `plot_conversion_vs_power.py` — the atlas
For each cataloged run it **recomputes** power from `omega_per_step.npy` via the
paper's motor model (`motor_power.average_power`, Lopez-Guajardo Eqs 18-23), **not**
from the log. So every run lands on **one uniform power basis** — including runs
trained before power logging existed (e.g. `so_parallel_freeform_s0` / job 7596936).
No re-simulation.

**Operating point per run** = the *converged tail*: over the last `--last-n` completed
episodes (default 5), average conversion and motor power over the final `--tail-frac`
of each episode (default 0.5 — skips the residence-time fill transient).

## Usage

```bash
# from this directory (paths resolve relative to CWD or the repo root):

# catalog a successful run (energy_weight & seed are parsed from the tag if present):
python plot_conversion_vs_power.py --add ../../results/td3/so_parallel_freeform_ew0.2_s0

# an older run whose tag lacks the knob -- pass it (+ a readable label):
python plot_conversion_vs_power.py --add ../../results/td3/so_parallel_freeform_s0 \
    --energy-weight 0.1 --label "job7596936 (16-worker, ew0.1)"

# (re)draw the atlas from everything cataloged so far -> conversion_vs_power.png:
python plot_conversion_vs_power.py

# add the regime backbone (bistable band = gap between up/down) + per-step clouds:
python plot_conversion_vs_power.py --hysteresis ../hysteresis/results/h90 --cloud

# overlay the CONSTANT-omega (+pulsating) backbone from the compare job's baseline sweep
# (a TD3 point above/left of the constant curve == beats constant at equal conversion):
python plot_conversion_vs_power.py --baselines ../parallelized_catalysis_rl/results/comparison/_baselines/baseline_sweep.npz

# preview candidate runs under a results tree WITHOUT cataloging them:
python plot_conversion_vs_power.py --scan ../../results/td3
```

Reads/writes only numpy + matplotlib (no gymnasium/OpenFOAM), so it runs anywhere.

## The catalog (`manifest.csv`)
A **curated** list — you bless successful runs one at a time; idle-collapse / diverged
runs stay out. Columns: `run_dir,label,energy_weight,seed,dt,tail_frac,last_n,notes`.
`--add` appends (replacing any existing row for the same `run_dir`). Per-run `dt` /
`tail_frac` / `last_n` cells override the plot defaults for that run only.

Store `run_dir` **relative** (the loader tries CWD and the repo root) so the manifest
survives the /home vs /project two-checkout split. The manifest is generated on the
cluster where the runs live — commit it back to the /home clone if you want it
version-controlled.

## Reading the plot for bistability
- Points are colored by `energy_weight` (the reward knob): low ew → spins hard (high
  power, high conversion); high ew → idles (low power, low conversion).
- `--cloud`: a **modulating** policy traces a loop; its time-mean point sits *inside*
  the cloud. A **constant** policy collapses to a dot.
- `--hysteresis`: the up/down branches bracket the **bistable band**. TD3 points that
  land *above the up-branch* (more conversion than a constant sweep reaches at that
  power) are the payoff — modulation exploiting the regime transition.
- `--baselines`: the constant-omega curve (black diamonds) + pulsating points (grey) from
  the compare job's `baseline_sweep.npz`. A TD3 point sitting **above/left** of the
  constant curve beats constant at equal conversion (lower power for the same conversion).
- **Two-IC probe:** runs tagged `_icidle_` vs `_icspin_` are drawn as **circles vs
  squares** at the same `energy_weight` color. A bistable ew shows as a **same-color
  circle-low / square-high pair** that splits across the plane (idle stays idle, spin
  stays spinning); a non-bistable ew has the circle and square land together.
