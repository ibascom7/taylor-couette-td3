# Steady-state conversion vs. angular velocity sweep

## Why this experiment exists

We need to settle one question:

> **Does the outlet conversion actually depend on the inner-wall angular velocity?**

The motivation: earlier the side-outlet catalytic case showed a *flat, low* conversion
(~2%) that barely moved with rotation. That flatness is what pushed us to build the
**Sherwood wall model** (`side_outlet_cat_wallmodel`), an explicit mass-transfer sink
`k_c(Re)·C` whose rate *responds to rotation by construction*. Two things have since
changed our confidence in that story:

1. We found and fixed an **omega-coefficient bug** in the RL env's step loop
   (`envs/helpers.py`), where a sticky `omegaCoeffs` sub-dict froze the wall at the
   first step's speed. That bug could, in principle, have faked an omega-*insensitive*
   result.
2. Yuhe is (reasonably) not convinced the Sherwood/`c_k`-style wall closure is the
   right physics to compare against literature.

So we go back to basics and run a **clean, direct spin sweep** — omega set once per
episode and *verified* in the solver output — on two cases, to see with our own eyes
whether conversion tracks rotation.

## What is compared

Both cases use a **byte-identical wedge mesh** and the **same corrected transport
constants** (`nu = 1.075e-5 m²/s`, `D = 1e-9 m²/s` ⇒ `Sc ≈ 1e4`, Lopez-Guajardo).
They differ **only in the catalytic-wall treatment**:

| case | catalytic wall | meaning |
|------|----------------|---------|
| `side_outlet_case` | **resolved gradient** — `c = 0` at outer wall; consumption = molecular diffusion of `c` to the wall | Yuhe's original physics. At `Sc ≈ 1e4` the wall film is ~microns thick; the coarse mesh cannot resolve it. This is the "does the raw model even feel rotation?" case. |
| `side_outlet_cat_wallmodel` | **Sherwood sink** — implicit `k_c(Re)·C` in wall cells, `Sh = a·Re^b·Sc^c` | Our wall model; conversion responds to rotation by construction. |

`side_outlet_case` is Yuhe's `taylor_couette_reactor_short` case with three minimal,
documented edits (see each file's header comment):

- `constant/transportProperties`, `system/controlDict`: **D 2.8e-8 → 1e-9** so the Schmidt
  number matches the wall-model case (the corrected Lopez-Guajardo value).
- `0/U`: inner wall changed from a **hard-coded modulated waveform** to a settable
  `rotatingWallVelocity omega` (physically the same solid-body rotation, just sweepable —
  and the exact BC the wall-model case uses).
- `0/c`: reactor **pre-filled at feed concentration** (`internalField 50`) instead of
  empty, so each episode starts saturated. Steady state is initial-condition independent;
  a full start just avoids the spurious `conversion = 1` window while an empty reactor
  fills, and (since we expect *small* steady conversion) converges much faster. The
  wall-model case already starts full, so both are consistent.
- `0/U`: feed rate **unified to 100 mL/min** (was Yuhe's 40 mL/min) to match the wall-model
  case — so the **outer-wall catalytic BC is the ONLY difference between the two cases**, a
  clean single controlled variable. It also shrinks the residence time (τ ≈ 26 s) so each
  180/150 s episode reaches steady state with margin.

> The remaining nominal difference — inlet concentration `c0` (Yuhe 50 vs wall-model 1) —
> **does not matter**: the scalar transport is linear/passive, so the whole field scales
> with `c0` and the normalised conversion `1 - cup_c/c0` is `c0`-independent. With the feed
> rate now matched, the two cases are identical apart from the catalytic-wall treatment.

## The sweep

For **each** case, 5 **independent** constant-omega episodes:

- speeds: **0, 250, 500, 750, 1000 rpm**  (`Re ≈ 0, 393, 785, 1178, 1571`)
- duration: **180 s for both cases** (same integration time so they match completely)
- `pimpleFoam`, laminar, in the standard way (omega held constant the whole episode)
- episodes run **in parallel, one CPU core each** (16-core desktop → all 10 at once)

Omega is applied once via `foamDictionary` and the solver echoes it back
(`ROTATIONAL_POWER ... Omega=`, `MODELINFO ... Uwall=`), so we can *confirm* the wall is
turning at the commanded speed — directly ruling the omega bug in or out.

## Run it

```bash
cd taylorCouetteGym/experiments/steady_state_conversion

# quick end-to-end pipeline test (~2 min): short episodes, real plots+table
python3 steady_state_conversion_sweep.py --smoke

# the real sweep (~35-50 min on a 16-core desktop, then plots+table)
python3 steady_state_conversion_sweep.py

# re-make the figures/table from saved data without re-running CFD
python3 steady_state_conversion_sweep.py --analyze-only

# re-run only ONE case (e.g. after changing its duration); the final table still
# includes the other case already sitting in results/
python3 steady_state_conversion_sweep.py --cases side_outlet_cat_wallmodel
```

Needs OpenFOAM v2506 on `PATH` (`pimpleFoam`, `foamDictionary`) and Python `numpy` +
`matplotlib`. The script compiles each case's coded sources once, then clones the
compiled case per episode (no recompiles), and pins each run to a single core
(`OMP_NUM_THREADS=1`).

## Power — one methodology for both cases

Power is reported the **same way for both cases**, using the paper's **electric-motor
model** (`taylor_couette_mixing/motor_power.py`, Lopez-Guajardo Eqs 18–23) — the same
metric the catalysis/`conversion_vs_power` experiments use. Crucially it is a **pure
function of the commanded ω** (bearing friction + drag polynomial + motor electrical
model), so at a given rpm the motor power is **identical for both cases by construction**
— exactly the "compute power the same way for both" requirement. It needs no CFD data, so
changing/adding it required **no CFD re-run** (recomputed by `--analyze-only`).

The summary CSV carries two power columns, both computed identically for the two cases:

- **`power_motor_W`** — the headline motor electrical power P_e(ω) above (full machine, W).
- **`power_visc_cfd_W`** — a secondary cross-check: the CFD viscous-drag power from the
  inner-wall torque, `72·ρ·|Mz_kin|·|ω|` (full device). Because both cases share the same
  momentum field, this should *match between them* at each rpm — and it does (e.g. ~0.0183 W
  vs ~0.0183 W at 250 rpm), which independently confirms the two cases differ only in the
  scalar wall BC. (Yuhe's FO reports the dynamic torque `Mz_wedge = ρ·Mz_kin`; the script
  divides by ρ so both feed the identical formula.)

## Outputs (`results/`)

- **`summary_table.csv`** and **`summary_table.png`** — the headline: for each case, a row
  per rpm with `omega`, `Re`, **motor power**, and **steady-state conversion**. This is the
  thing to look at.
- `conversion_vs_omega.png` — steady conversion vs rpm, both cases on one axis.
- `conversion_vs_power.png` — steady conversion vs **motor power** (same power axis for both
  cases → vertical gaps show which wall treatment converts more at equal motor power).
- `conversion_vs_time__<case>.png` — approach-to-steady-state curves (one line per rpm);
  use these to confirm each episode actually plateaued.
- `rpm_<n>_timeseries.csv` — per-episode conversion(t).
- `rpm_<n>/` — the full working case (logs + fields) for ParaView; `log.pimpleFoam` is the
  source `--analyze-only` re-parses, so power can be recomputed any time from a finished run.

The console also prints the table and, per case, the **conversion range across omega** with
a verdict of *DEPENDS on omega* vs *nearly FLAT in omega*.

## How to read the result

- **`side_outlet_case` conversion rises clearly with rpm** → the resolved-gradient model
  *does* feel rotation once omega is correctly applied. That would mean the earlier flat
  result was (at least partly) the omega bug, and the Sherwood wall model may not be
  strictly necessary for a rotation-dependent conversion.
- **`side_outlet_case` conversion stays flat/tiny across rpm** → the coarse mesh genuinely
  cannot resolve the `Sc ≈ 1e4` wall film, so a wall closure (Sherwood, or a much finer
  near-wall mesh) *is* needed to get rotation-dependent conversion. The omega bug was not
  the (only) culprit.

Either way, watch the `conversion_vs_time` plots: a "steady" value is only meaningful if
the curve has plateaued. With the feed rate unified to 100 mL/min the residence time is
≈ 26 s, so a 180 s episode is ≈ 7 residence times — comfortably steady for both cases —
but the `steady_drift` column is still the guardrail: if any run's drift is not small,
bump `duration` at the top of the script.

---

# Fine-mesh convergence study (`fine_mesh_sweep.py`)

The coarse resolved-gradient result was flat *and* noisy. Is that because the mesh can't
resolve the `Sc ≈ 1e4` concentration film at the outer wall (~17–45 µm), or is the flatness
physical? `fine_mesh_sweep.py` answers this with a **wall-normal mesh-refinement study**:
the same `side_outlet_case` physics on a sequence of **uniform (ungraded)** meshes refined
only in the radial direction (the film's gradient direction; `NZ = 120` fixed):

| case | NR | Δr = gap/NR | cells | est. runtime @150 s (1000 rpm) |
|------|----|-------------|-------|--------------------------------|
| `side_outlet_case_r16`  | 16  | 397 µm | 1936  | ~30 min |
| `side_outlet_case_r32`  | 32  | 198 µm | 3872  | ~2 h |
| `side_outlet_case_r64`  | 64  |  99 µm | 7744  | ~7 h |
| `side_outlet_case_r128` | 128 |  50 µm | 15488 | ~1.2 days (the long pole) |

Each is swept at {0, 250, 500, 750, 1000} rpm for 150 s. It **reuses the engine** in
`steady_state_conversion_sweep.py`, so the table + conversion-vs-omega/time/power plots are
identical in style, **plus** a dedicated `mesh_convergence.png`: steady conversion vs Δr,
one line per rpm, with the Sherwood wall-model value as a dotted reference and the film
thickness shaded. Outputs go to `results_fine/`.

**Honest caveat:** no desktop-feasible *uniform* mesh fully resolves the film — even Δr=50 µm
is coarser than the 17–45 µm film (fully resolving needs ~5 µm cells everywhere → ~1300
radial cells → days-to-weeks per episode). So this is a **refinement trend**: read the slope,
not an absolute converged value. If conversion climbs toward the wall-model dotted lines as
Δr shrinks, the coarse result was under-resolved (and a wall closure or a graded near-wall
mesh is needed); if it plateaus well below them, the flatness is closer to physical.

```bash
cd taylorCouetteGym/experiments/steady_state_conversion

python3 fine_mesh_sweep.py                 # all 4 meshes (~1.2 days; NR=128 dominates)

# overnight 3-point study (drop the ~1-day NR=128 mesh):
python3 fine_mesh_sweep.py --cases side_outlet_case_r16,side_outlet_case_r32,side_outlet_case_r64

python3 fine_mesh_sweep.py --analyze-only  # replot any time, incl. partial progress
```

The episodes run in parallel (1 core each); cheaper meshes finish first, so `--analyze-only`
partway through already shows the 3 coarser points. To change resolutions, edit
`RADIAL_CELLS` in the script and re-run `blockMesh` in the affected `side_outlet_case_r*`
case (its `blockMeshDict` exposes `NR`/`NZ`).

---

# Bottom-cells measurement variant (`bottom_avg_conversion_sweep.py`)

After the presentation, Wang and Yuhe asked for a **different way of measuring the outlet
concentration**: instead of the flux-weighted (cup-mixing) average over the `side_outlet`
patch faces, take the **average concentration of the cells at the bottom of the container**.

- **New case:** `side_outlet_case_bottomavg` — a byte-identical clone of `side_outlet_case`
  (same mesh, physics, BCs) whose **only** change is the conversion functionObject.
  `cOut` is now the **volume-weighted average of `c` over the bottom cell band** — the
  1-cell-tall axial layer adjacent to the closed bottom wall (`z < -0.0186266667`, 15 cells),
  the same band the side outlet drains from. Volume weighting because wedge cell volumes
  grow with radius. Conversion is still `1 - cOut/c0`.
- The functionObject **also logs the old cup-mixing outlet value on the same run**
  (`convOutletCup` in the `BOTTOM_CELLS_CONVERSION` line), so the two measurement styles are
  compared on **identical flow fields**, and the cup value cross-checks against the original
  `results/` sweep.

```bash
python3 bottom_avg_conversion_sweep.py                # full sweep (~35 min, 5 episodes parallel)
python3 bottom_avg_conversion_sweep.py --smoke        # ~2 min pipeline test
python3 bottom_avg_conversion_sweep.py --analyze-only # replot from results_bottomavg/
```

Same speeds (0–1000 rpm), same 180 s episodes, same engine/reporting as the main sweep.
Outputs go to **`results_bottomavg/`**, plus two measurement-comparison extras:

- `conversion_vs_omega.png` — bottom-cells average vs cup-mixing (same run) vs the original
  `results/` sweep, on one axis.
- `conversion_vs_time__bottomavg.png` — the bottom-cells-average conversion time series.
- `measurement_comparison.csv` — the three steady values per rpm.

**Expected result:** the conversion *values* shift (a cell average next to a `c = 0`
catalytic wall reads lower than a flux-weighted outlet average), but the flatness verdict
should not change — the measurement style shouldn't create an omega dependence the
resolved-gradient physics doesn't have.
