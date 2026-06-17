# Oscillating vs constant omega — mixing experiment

Partial recreation of the central claim in López-Guajardo et al., *Chem. Eng. J.*
489 (2024) 151174: **modulating the inner-cylinder angular speed mixes better
than a constant speed at the same mean speed.**

This is a plain `pimpleFoam` study — **no RL, no `train.py`**. It comes in two
flavours that share `make_case.py` and `analyze.py`:

| job | case | cells | notes |
|---|---|---|---|
| `run_carya_oscillation_vs_constant.slurm`    | `tc_mixing_case` (2D wedge)      | 12 000 | faithful to the paper's own 2D-axisymmetric model; ~1–1.5 h |
| `run_carya_oscillation_vs_constant_3d.slurm` | `full_tc_mixing_case` (3D, paper geom) | 87 552 | resolves azimuthal structure the wedge can't; ~10 h |

Both use the paper geometry (r_i=25.4, r_o=31.75, H=190.5 mm, Γ=30). The wedge
writes to `results/`, the 3D job to `results_3d/`. Each does two runs, identical
except the inner-wall ω(t):

| run | ω(t) | mean |
|---|---|---|
| `constant`   | 500 rpm steady | 500 rpm |
| `squarewave` | 0 rpm idle / 2500 rpm active, duty D=0.2, period T=30 s, 0.05 s ramps | 500 rpm |

Both inject the same dye band (inner ¼ of the gap = C 1, outer ¾ = C 0) at the
top inlet and are scored on the **radial uniformity of the dye profile at the
bottom outlet** (`rlMetrics` → `METRICS` log lines, 20 radial bins).

## Run

```bash
cd taylorCouetteGym
# 2D wedge (fast):
sbatch experiments/oscillation_vs_constant/run_carya_oscillation_vs_constant.slurm
# 3D paper geometry (slow):
sbatch experiments/oscillation_vs_constant/run_carya_oscillation_vs_constant_3d.slurm
# each is a 2-task array: task 0 = constant, task 1 = squarewave, run concurrently..
squeue -u $USER
```

When **both** tasks of a job finish, point `analyze.py` at that job's results dir:

```bash
PY=/project/mwang/ibascom/envs/tc/bin/python
$PY experiments/oscillation_vs_constant/analyze.py \
    experiments/oscillation_vs_constant/results     --avg-window 60   # wedge
$PY experiments/oscillation_vs_constant/analyze.py \
    experiments/oscillation_vs_constant/results_3d  --avg-window 60   # 3D
```

This prints a tail-averaged `mixing_index` (1 = perfectly mixed, 0 = as injected)
for each run, says which mixes better, and writes `results/mixing_comparison.png`.

## Cost

The binding constraint is the azimuthal Courant number at the **2500 rpm peak**
(wall speed ≈ 6.6 m/s), which holds dt ≈ 3.6×10⁻⁴ s (Co 0.8) in the active phase;
dt relaxes ~5× in the idle phase, and `maxDeltaT` (1e-3) caps the constant run.
**Measured** serial step rates from smoke runs: ~45 ms/step (wedge, 12 k cells),
~208 ms/step (3D, 87 k cells). Giving:

| job | run | wall (≈) |
|---|---|---|
| **wedge** (12 k cells) | `constant` (120 s)   | ~1.5 h |
| **wedge** | `squarewave` (120 s) | ~2 h |
| **3D** (87 k cells) | `constant` (60 s)    | ~3.5 h |
| **3D** | `squarewave` (60 s)  | ~4.5–5 h |

Each job is a 2-task array, so its two runs go **concurrently** → wall ≈ the
squarewave task. Wedge ≈ **2 h**, 3D ≈ **~5 h** (plus a one-time ~1–3 min coded-FO
compile). Both validated by local smoke runs; still, **check the first minute's
`ExecutionTime` to confirm your rate on Carya.**

**Speed knobs:** lower `RUN_SECONDS` (cost ~linear; 60 s = 2 periods still shows
the effect); raise `MAX_CO` toward ~2 (pimpleFoam tolerates it with outer
correctors); coarsen the mesh (`nz` in `blockMeshDict`). For the **3D** job,
parallelizing `pimpleFoam` (decomposePar + `mpirun -np N`) is the big lever —
the slurm has the commands and the safe pre-compile recipe in a comment.

## Interpreting

- Compare the **tail** (last ≥1 period), not the startup transient — dye reaches
  the outlet within seconds but the profile needs a period or two to settle.
- The square-wave outlet profile should be **flatter** (lower `unmixedness`,
  higher `mixing_index`) than the constant one if the paper's mechanism carries
  over to this dye-mixing proxy.
- Note this is a *mixing* proxy (radial dye uniformity), not the paper's
  catalytic-wall conversion, and energy is not tracked here — the qualitative
  trend is what's being reproduced, not the absolute numbers.

## Files
- `make_case.py` — configures a case copy: writes ω(t) on `inner_wall`, seeds a
  clean cold start from `0.orig`, writes the experiment `controlDict`.
- `run_carya_oscillation_vs_constant.slurm` — the 2-task array driver.
- `analyze.py` — parses the `METRICS` logs, scores mixing, plots.
