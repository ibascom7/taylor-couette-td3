# Modulation vs constant: Lopez-Guajardo Fig. 7 recreation (graded Sc=1075)

Recreates Figure 7 of Lopez-Guajardo et al. (pulsating vs constant rotation,
conversion vs nominal angular speed) on **our** resolved-gradient reactor:
`side_outlet_case_sc1075_graded` (25 um wall cell, Sc = 1075, no Sherwood wall
model). This is the direct precursor to a TD3 run whose action is the nominal
speed `w_b in [0, 500]` rpm of the paper's duty-cycle waveform.

## Setup

10 independent 60 s episodes, all launched in parallel (1 CPU core each):

| knob | value |
| --- | --- |
| nominal speed `w_b` | {100, 200, 300, 400, 500} rpm |
| pulsed waveform | paper Fig. 3 / Eq. 8: `dw = w0 = w_b/(2D)`, idle 0 → square 0 ↔ `w_b/D` |
| duty cycle / period | D = 20 %, T = 25 s → bursts of 5 s at **5·w_b** (peaks 500–2500 rpm) |
| constant baseline | D = 100 % → wall holds `w_b` all episode |
| edge ramps | 0.05 s (the RL env's `ramp_time`), same `square_wave_points` builder as `TaylorCouetteWaveformEnv` |
| field writes | every 1 s (61 time folders per run → ParaView animation) |
| FO sampling | conversion + wall omega/torque every 0.1 s |

Omega is driven by an inline tabulated `Function1` written into `0/U`
(`omega table ((t w) ...)`), the proven post-omega-freeze-fix path; each episode
is a single continuous pimpleFoam run, so the `omegaCoeffs` re-serialisation bug
cannot bite. `omega_traces.png` overlays commanded vs CFD-measured wall omega for
every episode as an explicit guard.

Caveats baked into the requested spec:

- 60 s / 25 s = **2.4 periods** (burst-first), so a pulsed episode's realized mean
  omega is 1.25·`w_b`. The Fig. 7 x-axis is the *nominal* `w_b`, as in the paper.
- At 60 s the low-`w_b` episodes are not fully steady (the 150 s sweep showed
  250 rpm still drifting at 60 s), so X is a "conversion at end of a 60 s episode"
  metric, not a true steady state.

## Reward (as if a TD3 episode)

- `X` = mean outlet conversion over the **last full period** (t in [35, 60] s;
  same window for both modes).
- `P` = episode-average **motor electrical power** on the commanded omega(t)
  (paper Eqs 18–23, `motor_power.py`, 100 Hz densification — identical to what
  `TaylorCouetteWaveformEnv` computes in training).
- `P_max` = 31.94 W = motor power at 2500 rpm (the `w_b`=500 pulse peak; matches
  the 2500 rpm row of the sc1075 steady-state sweep).
- Both weights 1. The summary reports **both signs** — `R = X - P/P_max` (the
  trainer's convention: energy is a penalty) and `R = X + P/P_max` (the formula
  as literally specified) — pick the column you meant.

## Usage

```bash
nohup python3 -u fig7_sweep.py > fig7.log 2>&1 &   # full run (~3 h wall, 10 cores)
python3 fig7_sweep.py --smoke                      # ~10 min pipeline test
python3 fig7_sweep.py --analyze-only               # replot from results/ (no CFD)
python3 fig7_sweep.py --modes pulsed --wbs 500     # subset re-run
```

## Outputs (`results/`)

- `fig7_conversion_vs_wb.png` — the Fig. 7 recreation (pulsed dotted triangles vs
  constant solid circles).
- `conversion_vs_time__{pulsed,constant}.png` — transient traces, viridis by `w_b`.
- `conversion_vs_power.png` — X vs episode-average motor power (equal-power view).
- `omega_traces.png` — commanded vs measured wall omega, all 10 episodes.
- `summary_table.{csv,png}` — X, P, P/P_max, both rewards, measured omega peaks,
  wall-clock minutes per episode (TD3 cost estimation).
- `{mode}_wb{N}_timeseries.csv`, `_power.csv`, `_waveform_points.csv` — raw traces.
- `{mode}_wb{N}/` — the full OpenFOAM case with 1 s time folders (ParaView).

## Results (run 2026-07-16, 7800X3D, 10 cores, 83 min wall / 9.6 core-h)

**We got the OPPOSITE of the paper's Fig. 7**: constant rotation beats the
pulsed waveform at every nominal speed — on conversion at equal `w_b`, on
conversion at equal motor power, and on the reward `R = X - P/P_max`:

| `w_b` | constant X | pulsed X | constant R− | pulsed R− |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 0.2591 | 0.2306 | 0.2155 | 0.1828 |
| 200 | 0.3055 | 0.2608 | 0.2252 | 0.1657 |
| 300 | 0.3410 | 0.2866 | 0.2239 | 0.1425 |
| 400 | **0.4011** | 0.3040 | **0.2470** | 0.1084 |
| 500 | 0.3886 | 0.3104 | 0.1973 | 0.0604 |

- The drive is verified genuine: measured wall omega tracks every commanded
  square exactly (`omega_traces.png`) — this is physics, not the old freeze bug.
- Why pulsing loses here: the film-thinning burst is only 20 % of the period,
  and the annulus swirl decays within a few seconds of each burst
  (gap²/ν ≈ 3.7 s), so the wall layer re-thickens for most of the 20 s idle;
  the conversion traces visibly sag between bursts. The paper's advantage
  (bearing-friction-dominated motor power making bursts cheap) does not
  compensate, because at equal `w_b` the pulsed waveform here *also* costs
  more motor power (its realized mean omega is 1.25 `w_b` over 2.4 periods,
  plus burst drag at 5·`w_b`).
- Constant `w_b`=400 vs 500 overlap within the vortex-oscillation band at 60 s
  (the 150 s steady sweep puts 500 rpm above 400 at true steady state); the
  60 s window slightly undersells high speeds.
- Caveat: at 2500 rpm the concentration film (~20–35 um) is only marginally
  resolved by the 25 um wall cell, so pulsed `w_b`=500 may undershoot a little.
- Under the literal `R = X + P/P_max`, the "best" policy is simply the highest
  power corner (constant 500: 0.58) — power becomes a virtue, which is why the
  minus convention is almost certainly the intended reward.
- **TD3 cost estimate**: 47–53 min per constant-ish 60 s episode, up to 83 min
  when the policy holds 2500 rpm bursts (~50–85 min/episode/core, 9.6 core-h
  for 10 episodes). With the 16-worker parallel trainer: ~16 episodes/1.2 h →
  a 300-episode run ≈ 1–1.5 days wall; 1000 episodes ≈ 3–5 days.

## Period variant: T = 10 s (run 2026-07-16/17, results_T10/, 69 min wall)

Same D=20%, same `w_b` grid; bursts of 2 s at 5·`w_b`, idle 8 s, exactly 6
periods per episode (realized mean omega = `w_b` exactly). Only the 5 pulsed
episodes were re-run (`fig7_sweep_T10.py`); the period-independent constant
baselines are symlinked from `results/`. Compare with `compare_periods.py` →
`results_T10/fig7_period_comparison.png`, `conversion_vs_power_comparison.png`.

| `w_b` | constant X / R− | pulsed T=25 X / R− | pulsed T=10 X / R− |
| ---: | ---: | ---: | ---: |
| 100 | 0.2591 / 0.2155 | 0.2306 / 0.1828 | 0.2540 / **0.2156** |
| 200 | 0.3055 / 0.2252 | 0.2608 / 0.1657 | 0.2766 / 0.2000 |
| 300 | 0.3410 / 0.2239 | 0.2866 / 0.1425 | 0.3057 / 0.1891 |
| 400 | 0.4011 / 0.2470 | 0.3040 / 0.1084 | 0.3197 / 0.1609 |
| 500 | 0.3886 / 0.1973 | 0.3104 / 0.0604 | 0.3474 / 0.1438 |

- **T=10 strictly dominates T=25**: more conversion (equal-window [50,60] s
  cross-check confirms: +0.008…+0.033) at ~20 % less motor power (mean omega
  `w_b` instead of 1.25·`w_b`; the 2× more accel/decel cycles are nearly free
  under the paper's motor model thanks to regen).
- **Constant still wins** on conversion everywhere and on R− for `w_b` ≥ 200;
  at `w_b` = 100 the T=10 pulse *ties* constant on R− (0.2156 vs 0.2155) at 12 %
  less power — the first equal-power pulsed parity point observed in this case.
- Consistent with the swirl-memory picture: the ~1–2 s burst tail is a fixed
  per-cycle bonus, so more cycles → closer to constant. Shortening T closed
  ~30–50 % of the pulsed-vs-constant gap; extrapolating, T ≲ a few s (idle
  comparable to the 0.6 s swirl decay) should approach the constant curve from
  below — but in this axisymmetric laminar wedge it has no mechanism to CROSS
  it (quasi-steady mass transfer is concave in omega). Period is therefore a
  live optimization knob for TD3, with "constant-like" as the expected ceiling.

## Duty variant: T = 10 s, D = 15% (run 2026-07-17, results_T10_D15/, 71 min wall)

Tests the paper's Fig. 6 claim (conversion rises as D falls at fixed mean
speed). Bursts of 1.5 s at 6.67·`w_b` — peaks {667…3333} rpm, i.e. `w_b`=400
(2667 rpm) and `w_b`=500 (3333 rpm) went past the 2500 rpm mesh ceiling for the
first time. `fig7_sweep_T10_D15.py`; template initial deltaT lowered to 0.002 s.

**Stability: all clean.** Zero scalar-bounding events in any log, including
3333 rpm; max instantaneous Courant 7.3 at a burst edge, immediately absorbed
by adjustTimeStep (maxCo 1.8 steady-state). The 0.05 s ramped edges + tiny
initial dt + implicit momentum solve are sufficient. NB this clears 3333 rpm
in 1.5 s BURSTS; a sustained >2500 rpm constant run is still untested (and at
3333 rpm the ~20–29 um film is under the 25 um wall cell → conversion there
likely reads a bit low).

**Result: lower duty does NOT help on this reactor** — the paper's Fig. 6
direction does not reproduce (same caveats as Fig. 7: resolved film at
Sc=1075, 1/5-height reactor, τ≈26 s, axisymmetric wedge):

| `w_b` | X D=20% | X D=15% | R− D=20% | R− D=15% |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 0.2540 | 0.2510 | 0.2156 | 0.2127 |
| 200 | 0.2766 | 0.2732 | 0.2000 | 0.1957 |
| 300 | 0.3057 | 0.2918 | 0.1891 | 0.1721 |
| 400 | 0.3197 | 0.3201 | 0.1609 | 0.1546 |
| 500 | 0.3474 | 0.3293 | 0.1438 | 0.1136 |

D=15% ≤ D=20% on conversion everywhere (tie at 400) and strictly worse on
reward (taller bursts pay the convex ω² drag). Matches quasi-steady concavity:
duty-averaged transfer ∝ D·(1/D)^0.7 = D^0.3 falls as D falls (0.62→0.57 of
constant), and the idle annulus is dead either way. NB Lopez-Guajardo's CFD is
the SAME 2D-axisymmetric model as ours (same geometry/fluid; their Table 2 Re
matches ours exactly) — the only differences are reactor height (ours 1/5
theirs) + feed (100 vs 40 mL/min), hence τ ≈ 26 s vs ~minutes, plus our
10×-reduced Sc. Their opposite duty trend therefore comes from the τ≫T regime
(idles amortize over ~10 periods per residence) and their Re*/Sc conditions —
NOT from 3D effects unavailable to a wedge.
Ordering so far: constant > pulsed(T10,D20) > pulsed(T10,D15) > pulsed(T25,D20).

## Duty variant: T = 10 s, D = 30% (run 2026-07-17, results_T10_D30/, 66 min wall)

Bursts of 3 s at 3.33·`w_b` (peaks 333–1667 rpm, all tame). **Best pulsed
family yet, and the first waveform to BEAT constant**: at `w_b`=100, pulsed
D=30% gives X=0.2690 vs constant's 0.2591 at 11 % LESS power (1.24 vs 1.39 W)
— strict domination on the conversion–power plane, and R− = 0.2301 vs 0.2155
(now the #2 reward overall, behind constant-400's 0.2470). At `w_b` ≥ 200
constant still wins conversion, but D=30% > D=20% > D=15% at every `w_b`.

So the duty axis at T=10 s has an INTERIOR structure at low speed: at
`w_b`=100 the grid reads D=15% (0.2510) < D=20% (0.2540) < **D=30% (0.2690)**
> D=100% (0.2591). Interpretation: constant 100 rpm sits in the weak-vortex
regime near Taylor onset where conv(ω) is steep/locally convex — bursting to
333 rpm buys strong vortices whose transport outlives the short 7 s idle,
exactly the non-convexity mechanism the hysteresis experiment was designed to
find. At higher `w_b` constant is deep on the concave branch and wins. A
quasi-static chord estimate at `w_b`=100 gives ~0.21; measured 0.269 — the
transient carryover is large.

Caveat: the `w_b`=100 crossing margin (~0.010 conv) is comparable to the
vortex-band oscillation on a single 60 s episode; a 150 s confirmation pair
(pulsed D=30% + constant at `w_b`=100) would settle it, and the ratchet
asymmetry (pulsed still climbing faster at 60 s) suggests the true periodic
margin is LARGER, not smaller.

## Duty variant: T = 10 s, D = 50% (run 2026-07-17, results_T10_D50/, 55 min wall)

Bursts of 5 s at 2·`w_b` (peaks 200–1000 rpm). **The campaign's headline:
pulsed now beats constant at `w_b` ≤ 300, decisively at 200** — the paper's
modulation-beats-constant claim REPRODUCES on the resolved-gradient case with
the right waveform:

| `w_b` | constant X / R− | D=50% X / R− | verdict |
| ---: | ---: | ---: | :-- |
| 100 | 0.2591 / 0.2155 | 0.2731 / 0.2329 | pulsed wins, −8 % power |
| 200 | 0.3055 / 0.2252 | **0.3334 / 0.2563** | pulsed wins big; **best reward of campaign** |
| 300 | 0.3410 / 0.2239 | 0.3439 / 0.2294 | pulsed edges it, −2 % power |
| 400 | 0.4011 / 0.2470 | 0.3577 / 0.2054 | constant wins (conv(ω) peak) |
| 500 | 0.3886 / 0.1973 | 0.3740 / 0.1834 | constant wins narrowly |

Duty is monotone 15 < 20 < 30 < 50 % at every `w_b`, and since D=100 % IS
constant, the duty axis has an INTERIOR OPTIMUM near ~50 % at `w_b` ≤ 300
(exact peak unresolved between 30–100 %).

**Mechanism (measured on `pulsed_wb200`): film renewal, not swirl memory.**
Fluid swirl still dies ~1 s after burst end (vol-avg |u_θ| 0.39 → 0.015 m/s in
1 s), so structure persistence is NOT the story. Instead the inventory-balance
consumption rate shows the burst transiently OUT-TRANSFERS its own speed's
steady state: burst-mean consumption 0.496 µmol/s (peaks ~0.60) vs steady
constant-400's 0.464, because each spin-up drives freshly-renewed, still-thin
concentration films fed by radially-homogenized rich core fluid. The idle keeps
a fat tail (0.294 µmol/s ≈ 84 % of constant-200's steady rate) off the
just-renewed film. Cycle average +12 % consumption vs constant-200 → the +9 %
conversion win. Constant keeps winning at `w_b`=400–500 because conv(ω) peaks
near 400 rpm (0.4011 > 0.3886 at 500) — sitting at the max of a locally
concave curve is unbeatable by averaging around it.

TD3 implication: the (w_b, D, T) reward landscape now has a genuine interior
optimum — best measured point (w_b=200, D=50 %, T=10 s), R− = 0.2563 > best
constant (400 rpm, 0.2470). The waveform env's action space covers it.

## Duty axis completed: D = 60/70/80 % at T = 10 s (run 2026-07-17, results_T10_D{60,70,80}/)

Peaks 1.67/1.43/1.25·`w_b` (all ≤ 833 rpm, ~45–57 min/episode). X (60 s
last-period mean) across the full duty axis:

| `w_b` | D=15 | D=20 | D=30 | D=50 | D=60 | D=70 | D=80 | const (D=100) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 0.2510 | 0.2540 | 0.2690 | 0.2731 | 0.2760 | **0.2774** | 0.2735 | 0.2591 |
| 200 | 0.2732 | 0.2766 | 0.2930 | 0.3334 | 0.3345 | **0.3383** | 0.3357 | 0.3055 |
| 300 | 0.2918 | 0.3057 | 0.3138 | 0.3439 | 0.3478 | 0.3763 | **0.3788** | 0.3410 |
| 400 | 0.3201 | 0.3197 | 0.3397 | 0.3577 | 0.3739 | 0.3767 | 0.3845 | **0.4011** |
| 500 | 0.3293 | 0.3474 | 0.3521 | 0.3740 | 0.3845 | 0.3997 | **0.4054** | 0.3886 |

- The duty curve rises from D=15 % to a MAXIMUM near **D ≈ 70–80 %** at every
  `w_b`, then falls back to constant. Pulsed beats constant everywhere except
  `w_b`=400 (constant's conv(ω) sweet spot). Campaign-best rewards:
  R− = **0.2629** (D=80, `w_b`=300), 0.2600 (D=70, 200), 0.2358 (D=70, 100) —
  all above the best constant (0.2470 at 400 rpm).
- Consistent with the film-renewal mechanism: high duty keeps most of the
  cycle in (renewed-film) on-state while a short idle (2–3 s ≈ a few swirl
  decay times) is enough to reset the film cheaply. Note the D=70/80 idle
  (3/2 s) is comparable to the renewal transient itself.

## Steady-state 400 rpm verdict (150 s, run 2026-07-17)

`graded_mesh_sweep_sc1075.py --rpms 400` (400 added to the steady grid):
**conv_steady(400) = 0.4071** (engine metric: median over the last 30 s of
1 Hz samples). CORRECTED READING (2026-07-17, after the aliasing analysis):
the engine's 30 s-median favors whatever the slow wave is doing in that
window — 500's crest sits there (median 0.4157, mean 0.4131). Long-window
MEANS give 400 = 0.4109 ± 0.0018 vs 500 = 0.4080 ± 0.0058 over [60,150]
(10 s-block SE) → **400 vs 500 is a statistical tie (0.5σ)**; the steady
curve rises monotonically through 250 → 400, PLATEAUS between 400 and 500
(unresolvable without ~10 min runs or ensembles), and resumes rising by 750
(0.4594). The 60 s "constant 400 > 500" reading is a real settling-rate
effect layered on that tie: 400 is nearly converged by 60 s (0.4011 vs
~0.411) while 500 lags (0.3886 vs ~0.408). The 60 s constant runs were
correct, just not steady — and the 1 Hz-sampled steady traces carry ±~0.02
ALIASED flutter (see results_td3_prep/aliasing_check__constant500.png), so
steady medians over 30 samples have ±~0.01 wave-phase noise.

**Caveat on the pulsed crossings**: all pulsed-vs-constant comparisons above
are 60 s-episode metrics — exactly the TD3 reward landscape, but NOT steady
state. Constant's 60 s values undersell steady by 0.006–0.027 (measured at
400/500); the pulsed periodic-steady values are unknown (episodes still
ratcheting at 60 s). A paper-grade steady claim needs 150 s pulsed runs at
the D=70–80 optimum.

## FULL-HEIGHT reference: Gamma=30, 300 s episodes (run 2026-07-18/19, results_full_tc/)

`cases/full_tc_cat_case` = the graded Sc=1075 mesh extended to the paper's
H = 190.5 mm (16,650 cells, D = 1e-8, feed 100 mL/min → τ ≈ 130 s, τ/T = 13).
10 × 300 s episodes (D=80 %/T=10 pulsed + constant), ~23 h wall, 17 GB.
X = mean over the last period ([290,300] s; traces fully plateaued by ~150 s).

| `w_b` | constant X / P [W] | pulsed X / P [W] | gap |
| ---: | ---: | ---: | ---: |
| 100 | 0.6890 / 1.39 | 0.7110 / 1.35 | +0.022 |
| 200 | 0.7617 / 2.57 | 0.8532 / 2.52 | **+0.092** |
| 300 | 0.7869 / 3.74 | 0.8764 / 3.70 | **+0.090** |
| 400 | 0.8116 / 4.92 | 0.8744 / 4.89 | +0.063 |
| 500 | 0.8281 / 6.11 | 0.8769 / 6.09 | +0.049 |

- **Pulsed beats constant at EVERY speed, at slightly less power** — including
  w_b=400, the short reactor's lone constant stronghold. The τ≫T regime
  amortizes idles (each parcel averages ~13 periods) and TRIPLES the w_b=300
  pulsed gap (+0.090 vs +0.034 on the short reactor, same waveform).
- **Conversion magnitudes land in the paper's Fig. 7 range** (their 0.26–0.88;
  ours 0.69–0.88) — the full-height case is a faithful reference geometry.
- The pulsed curve saturates ≈0.875 above w_b=300 (approach to the mass-
  balance ceiling; Da climbing) — so on the tall reactor, LOW nominal speeds
  with modulation are dramatically efficient: pulsed-200 makes 0.853 at 2.52 W,
  more than constant-500's 0.828 at 6.11 W → **2.4× less power for more
  conversion** (R−: 0.774 vs 0.637, best of the tall campaign).
- Cost: 247–276 CPU-s per sim-second, ~21–23 h per 300 s episode.

## TD3-prep dataset: 50 s episodes, D=80% + constant (run 2026-07-17, results_td3_prep/)

50 s is the planned RL episode length. Fresh self-contained runs of BOTH modes
(`fig7_sweep_td3_prep.py`, 10×50 s, 44 min wall), X = mean over [40, 50] s.
Standalone plots only — deliberately NOT in compare_periods.py (other families
are 60 s).

| `w_b` | constant X / R− | pulsed D=80 X / R− |
| ---: | ---: | ---: |
| 100 | 0.2585 / 0.2148 | **0.2718** / 0.2295 |
| 200 | 0.3071 / 0.2268 | **0.3334** / 0.2545 |
| 300 | 0.3402 / 0.2231 | **0.3744** / **0.2585** |
| 400 | **0.3986** / 0.2445 | 0.3743 / 0.2213 |
| 500 | 0.3924 / 0.2012 | **0.3961** / 0.2057 |

Same picture as at 60 s: pulsed wins everywhere except `w_b`=400; best reward
= pulsed 300 (0.2585). NB constant 500 (0.3924) still reads below constant 400
(0.3986) in the [40,50] window — the hoped-for 500-rpm slow-wave crest does
not appear in this window (the crest memory is from the 150 s run's phase).

### Fragmentation probes (results_frag_T5/, results_frag_T2p5/, run 2026-07-18)

Same D=80 %, w_b=300, 50 s, X over [40,50]: T=10 → 0.3744, T=5 → 0.3741,
**T=2.5 → 0.3860, T=1.25 → 0.3850** at identical power (~3.7 W). The
landscape is two shelves with a step between T=5 and T=2.5 — exactly where
the idle (1 s → 0.5 s) crosses the 0.6 s swirl-decay time. CONFIRMED by the
independent T=1.25 episode: once idles are shorter than the swirl decay the
film stays permanently renewed and conversion sits +0.011 above the slow-
pulsing shelf, saturating (no further gain or cost at T=1.25; regen keeps the
8× accel cycles ~free). Best-known waveform at w_b=300: D≈80 %, T ≲ 2.5 s — a
rapid shimmy, +0.045 conversion over constant at equal power (R− ≈ 0.269).
Figure: results_td3_prep/period_scan_wb300_D80.png. TD3 consequence: block
action's fragmentation dim needs ≥4 idle slots per 10 s block to reach the
shelf; finer period resolution below 2.5 s is unnecessary (plateau).

**Reproducibility finding (matters for RL):** the 50 s pulsed episodes match
the first 50 s of the 60 s D=80 runs BIT-EXACTLY (their omega tables are
identical in [0,50]), but the constant episodes differ from Wednesday's by up
to 0.011 in X. Cause: the constant omega `table` hold-point moved (61 s→51 s),
changing the Function1 interpolation denominator → ~1 ulp (1e-16) omega
dither, which chaotic wavy-vortex regimes amplify from t≈0.5 s (100/200/500
rpm) while calm TVF regimes (300/400) stay matched to t≈26–39 s. Implication:
episode metrics carry an irreducible ±~0.01 sensitivity noise floor in the
wavy regimes — the TD3 reward is effectively stochastic at that level even
though the solver is deterministic.
