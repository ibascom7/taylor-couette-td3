# Mechanism experiments: what sets the best stop length?

Plan written 2026-10-10 for a fresh session. Deadline: GCURS, about 2026-10-16
(confirm the date with Isaiah). Nothing in this folder is built yet.

## Where things stand

- The poster (Rice RTG-NASC, 2026-10-02) went well. Materials are in
  `reports/poster_2026-10-02/` (`make_figures.py` regenerates every figure).
  The same poster will be reused at GCURS, updated with these results if they land.
- Main result: brief full stops at high duty beat constant rotation at equal
  power. The best fixed pulse from the 220-run Bayesian search
  (`experiments/endgame/results/A/`) stops about 0.5 s every about 5 s
  (run 4.6 s, stop 0.5 s, peak about 332 rpm at a 300 rpm average).
- Open question: why about 0.5 s? Two candidate clocks sit near it:

  | Clock | Formula | Nominal value |
  |---|---|---|
  | Swirl decay across the gap | d² / (π² ν) | about 0.38 s |
  | Diffusive re-forming of the wall layer | δ² / D, with δ about 90 µm at 300 rpm | about 0.8 s |

  Here d = 6.35 mm, ν = 1.075e-5 m²/s and D = 1e-8 m²/s, so Sc = 1075.
  The film thickness comes from the Eisenberg-Tobias-Wilke correlation.
  At one viscosity the two clocks cannot be told apart.

## Critical numerics constraint: read before running anything

The grid check (`experiments/endgame/results/D/`, README section "D results")
found an outlet artifact. On the short reactor at the production time step
(maxCo 1.8), a spurious jet of about 0.5 m/s forms in the one-cell side-outlet
band. It inflates outlet conversion by about 0.04 for constant 300 rpm and
about 0.025 for the champion. At maxCo 0.9 the jet disappears. The interior
mesh is converged to about 0.01.

That error is the same size as the effects these experiments look for. So:

- **Run every mechanism episode at maxCo 0.9.** This costs much less than
  expected. In the grid check (`results/D/f1.00`), the champion took 183k time
  steps at maxCo 0.9 against 157k at 1.8 (+17 %), and the wall time was about
  the same (8316 s vs 8653 s for 150 s). Most steps are limited by something
  other than the Courant number.
- **Also compute conversion from the wall-flux budget** as a cross-check on the
  outlet reading. Integrate D·c/(wall distance) over the outermost cell column
  above the outlet band. The wedge's outer-wall faces are flat chords at
  Ro·cos(2.5°), about 30 µm inside Ro, so use that for the wall distance or the
  budget comes out 3.4 times too low.
- **Do not refine the outlet band to 2 cells.** That causes backflow through
  the outlet, and the outflow-only conversion function object then over-reads.
- **The 0.5 s optimum itself was measured with the artifact present.** That is
  why experiment 1 exists.

## Experiments, in priority order

All runs use the short reactor case `taylor_couette_mixing/cases/side_outlet_case_sc1075_graded`.
Each run starts from a warmed state, 60 s at constant 300 rpm, and lasts 130 s,
about five residence times. Each runs **one continuous wave** that does not
restart at block boundaries. Endgame A restarted the wave every 26 s, which
created small artifacts near the peak.

**Score over whole periods.** Score each run over the last whole number of
periods ending at 130 s, the n with n·T closest to 26 s. Do not use a fixed
[104, 130] s window. The champion's outlet conversion swings with a standard
deviation of 0.013 to 0.016 inside the window (`results/D/summary_table.csv`,
X_win_sd). A fixed window holds 4.6 to 5.6 periods depending on stop length,
so where it cuts the last cycle biases X by about 0.002 to 0.005, the size of
the effects being measured. Compute the mean speed and power over the same
window. Phase the wave so a period ends exactly at 130 s.

Report X (primary), r = X − P/31.94 W, and the wall-flux-budget X. Power comes
from the paper motor model on the commanded speed
(`taylor_couette_mixing/motor_power.py`). About 3.5 of its 3.7 W is bearing dry
friction, linear in speed, so power barely depends on waveform: X − r ≈ 0.117
in every endgame A run. r and X therefore peak at the same stop length, and
the model's blindness to viscosity doesn't affect the mechanism test. Use X as
the primary metric.

**Log the clocks in every run** (not only the champion, see experiment 4).
During each stop, record the volume-averaged swirl, the vortex kinetic energy
(radial and axial velocity) and the total wall flux. Fit their decay times per
run. The test then asks which measured clock the optimum follows at each
viscosity, which needs no scaling law. These are interior quantities, so the
outlet jet doesn't affect them.

### 1. Clean stop-length curve at nominal viscosity (15 runs)

Run length fixed at 4.5 s, average 300 rpm, stop level 0. Stop lengths:
0 (constant), 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.85, 1.2 and 2.0 s,
plus replicates of the two points nearest the peak. The replicate noise
measured in endgame A was about 0.002. This confirms or moves the optimum
without the artifact, and it is the control for experiment 2.

**The curve is probably not a smooth hump.** Endgame A runs with a run length
of 3.5 to 5.5 s (`results/A/evaluations.csv`, r_last) show:

| Stop (s) | r_last |
|---|---|
| 0.05 | 0.244 |
| 0.17 to 0.21 | 0.285 to 0.288 |
| 0.24 to 0.34 | 0.279 to 0.283 (a dip) |
| 0.44 to 0.57 | 0.292 to 0.298 (the peak) |
| 0.60 to 0.65 | 0.277 to 0.282 (a cliff) |
| 0.7 to 2.0 | about 0.275 (a plateau) |

The drop of about 0.014 between 0.57 and 0.60 s is too sharp for a diffusive
clock. Either the outlet artifact causes it (it is the same size), or a
threshold does, such as vortices surviving a short stop but dying in a longer
one. These runs also used block restarts and fixed 26 s windows. The old grid
(0.25 / 0.5 / 0.75) would have stepped over all of this. If the cliff survives
at maxCo 0.9, track the **cliff edge** across viscosities as well as the peak,
because a sharp edge pins down a location far better than a broad maximum
does.

### 2. Same sweep at 2× and 0.7× viscosity, diffusivity fixed (23 runs)

Stop lengths, each with run length 4.5 s, average 300 rpm, stop level 0:

- **2× (primary arm, 13 runs):** 0 (constant), 0.1, 0.15, 0.2, 0.25, 0.3, 0.4,
  0.5, 0.6, 0.8, 1.0, 1.3 and 1.8 s. The swirl prediction (0.25 s) needs
  points below it.
- **0.7× (secondary arm, 10 runs):** 0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0
  and 1.3 s.

The 2× arm discriminates better: its predictions are 3.2 times apart
(0.25 vs 0.8 s), against 1.75 times (0.7 vs 0.4 s) at 0.7×. The 0.7× wall
layer is also thinner and less well resolved. If cores run short, cut from 0.7×.

Change ν in `constant/transportProperties` only. Keep D = 1e-8 in both places
the case reads it: `constant/transportProperties` and
`functions.scalarTransport.D` in `system/controlDict`.
`experiments/endgame/gamma30_statics.py::set_schmidt` shows how to set both.
**Re-warm the starting state at each viscosity.** Holding D fixed is
essential: holding Sc fixed instead makes both clocks scale as 1/ν, and the
test cannot discriminate.

**Pre-registered predictions** (write any changes here before launching):

| Viscosity | Best stop if swirl decay sets it | Best stop if wall-layer diffusion sets it |
|---|---|---|
| Nominal | about 0.5 s, measured in experiment 1 | about 0.5 s, measured |
| 2× | about 0.25 s | about 0.8 s |
| 0.7× | about 0.7 s | about 0.4 s |

These come from scaling: swirl decay goes as 1/ν, and δ²/D goes as ν^0.69 at
fixed D. The 0.69 comes from the Eisenberg-Tobias-Wilke correlation, which was
fitted to turbulent Taylor-vortex flow, not laminar vortices at Re 235 to 670.
A laminar-style Sh ∝ Re^0.5 gives about ν^0.33 instead. The magnitude is
uncertain, but the direction is not.

**Primary analysis: one fitted exponent.** Fit t_stop* ∝ ν^a across all three
viscosities and report a with a bootstrap confidence interval.

- Momentum clocks (swirl decay, vortex decay, spin-up) predict a ≈ −1.
- The wall-layer clock predicts a > 0 (about +0.33 to +0.69).
- "Neither" means a ≈ 0, with the interval excluding both.

**The pre-registered call is the sign of a.** Locate t_stop* at each viscosity
without eyeballing. If the curve is smooth near its top, fit a quadratic in
log(t_stop) over the points around the maximum and run a parametric bootstrap
using the replicate noise. If the curve has a cliff like the endgame A data,
locate the cliff edge as the midpoint between the last high and first low
point, with half the gap as its uncertainty, and run the exponent fit on the
edge as well. Choose which feature to track from experiment 1 before looking
at the other viscosities. Then check t_stop* against each viscosity's
**measured** swirl and wall-flux decay times (logged in every run). Use one
noise standard for results that confirm and results that don't (see memory
`idle-law-is-a-hypothesis`). If the peak doesn't move, neither clock sets it.
Report that plainly.

Range limits: Re at 300 rpm is about 470 nominally, 235 at 2× and 670 at 0.7×.
Do not halve viscosity. Re near 940 approaches wavy regimes the 2D wedge
cannot represent. Thinner liquid also thins the wall layer, which the mesh
resolves less well.

### 3. Stop-level sweep at nominal viscosity (4 new runs)

Same 4.5 s run and 0.5 s low period, but the low speed is 30, 60, 100 or
150 rpm (0 rpm is already the 0.5 s point of experiment 1), with the run speed solved for the 300 rpm average. Taylor vortices
appear near 61 rpm in this geometry. If the benefit collapses once the low
speed exceeds onset, vortex reset is the mechanism. If it fades gradually, the
mechanism is smoother, such as a general shear effect.

### e5. Vortex-state tests (2 runs, added 2026-10-10)

Experiment 4 found that the champion's gain over constant 300 rpm is a
sustained higher wall flux through each burst (+0.077 X), not the restart
overshoot (+0.001 X). The saved grid-check fields also differ in steady
Taylor-cell structure: constant 300 has 4 sign changes of u_r at mid-gap, and
the champion has 6. So the pulses may move the flow into a different stable
vortex state. Two runs at nominal viscosity test this:

- **hold:** pulse (stop 0.5 s) until 60 s, then constant 300 rpm.
- **kick:** constant 300 rpm with a single 0.5 s stop at 60 s.

If either keeps the pulse's conversion and cell count through [104, 130] s,
the gain comes from the vortex state, not from repeated stopping. That would
change how the paper states its main claim.

### 4. Measure the clocks from existing data (no new runs)

From the champion's existing traces
(`experiments/modulation_rl/results/duty_v3_refs/champion_D0.90/`, which has
wall-flux plots and data), fit:

- the decay of wall reaction during a stop,
- the overshoot at restart, and
- the fade during the burst.

Also fit the swirl decay time after a stop, using the volume-averaged swirl
from the timeline tooling in `experiments/modulation_vs_constant/`. This turns
the poster's 0.4 s and 0.8 s estimates into measurements.

## Cost and schedule

| Item | Count |
|---|---|
| Experiment 1 | 15 runs (13 stop lengths + 2 replicates) |
| Experiment 2 | 23 runs (13 at 2×, 10 at 0.7×) |
| Experiment 3 | 4 runs |
| e5 (vortex-state tests) | 2 runs |
| Warm templates | 3, one per viscosity |
| **Total** | **44 runs + 3 warms** |

On Carya, endgame A ran 44 workers on one 48-core node (`-c 48`). Its 130 s
episodes took a median of 195 min (range 131 to 258 min) at maxCo 1.8. At
maxCo 0.9 (+17 % steps), expect about 3.8 h median and about 5 h for the
slowest run. The 2026-10-10 smoke test showed **2x viscosity runs about 1.7
times slower** per simulated second, so the 2x arm takes about 6 to 8 h and
its warm-up about 3 h. **All 44 runs fit in one single-node job of about 10 to
12 h**; the script starts the 2x arm first. The job requests `-t 24:0:0`. Launch everything at once with the dense grids
above. Do not wait for experiment 1 before launching experiment 2. Keep a
second, smaller job in reserve to fill in around any peak or cliff edge.

For Carya, commit and push from the /home clone, pull in the /project clone,
and sbatch from `taylorCouetteGym/`. See memory `carya-two-checkouts`.

Schedule to Thursday night (2026-10-15):

| When | What |
|---|---|
| Sat 10-10 to Sun 10-11 | Build the script: continuous wave, whole-period scoring, clock logging, wall-flux budget, viscosity setting. Smoke-test locally with 2 s runs at every viscosity. Commit, push, pull on Carya, sbatch Sunday night. |
| Mon 10-12 | Results land (queue permitting). Run the analysis. Do experiment 4 while waiting. |
| Tue 10-13 to Wed 10-14 | Buffer for one fill-in or rerun job (about 7 h plus queue). Draft the figure. |
| Thu 10-15 | Final figure, poster sentence, update "about 0.5 s" if it moved. |

The main risks are queue wait for a 48-core node, and a bug found after the
job finishes, which costs about one buffer day. A local smoke test at every
viscosity, including the clock-logging and wall-flux outputs, is the cheapest
protection.

## Implementation pointers

- **Sweep engine:** `experiments/modulation_vs_constant/fig7_sweep.py`.
  `WARMUP_RPM` and `WARMUP_S` warm a template once and promote it to `0/`.
  `square_wave_points` builds the ramped table with 0.05 s ramps.
  `prepare_template` compiles the coded function objects once. Import it as a
  library the way `experiments/endgame/gamma30_statics.py` and
  `experiments/endgame/d_grid_check.py` do. `d_grid_check.py` already sets
  maxCo per run and has `--resume`, `--smoke` and `--analyze-only` patterns
  worth copying. Note: it was edited after 2026-09-30, so read the current version.
- **Run speed for a continuous wave:** with stop level w_lo and a whole number
  of periods, the run speed is (300 − (1 − D)·w_lo)/D. For an exact 130 s mean,
  integrate the unit wave numerically, as
  `taylor_couette_mixing/envs/taylor_couette_duty_v2.py::solve_block_wave` does.
- **Cell centres for the wall-flux budget:** `postProcess -func writeCellCentres -time <t>`.
- **Run locally detached and niced:**
  `setsid nohup nice -n 10 python3 -u <script> > logs/... 2>&1 &`.
  OpenFOAM v2506 is on PATH. Plain `python3` has numpy, matplotlib and scipy.
- **Commits:** never commit unprompted. Stage with git add and suggest a
  message (memory `no-unrequested-commits`).

## What to deliver for GCURS

- One figure: best stop length against viscosity, with confidence intervals,
  plus the two predicted lines.
- One sentence for the poster's next-steps box. It either names the clock
  that sets the stop length or says neither clock does.
- An updated champion stop-length curve under clean numerics. If the peak
  moved, update the poster's "about 0.5 s" wherever it appears.
- **Abstract mismatch to prepare for.** The submitted GCURS abstract
  (`GCURS_abstract.txt`) says the policies run "longer intervals ... at less
  expensive low speeds followed by short (and more expensive) rapid bursts."
  The actual result is the opposite. Prepare a one-sentence correction for
  the talk.

## Deferred until after GCURS

- A reduced model fitted to the endgame A surface.
- The slow-restart test.
- RL under changing feed rate or viscosity.
- The sharpness-family waveform search.
- An operator-learning surrogate.
- Fixing the outlet geometry.
