# Mechanism experiments: what sets the best stop length?

The plan, predictions and analysis rules are in [PLAN.md](PLAN.md). This file covers
how to run the experiments.

| Script | What it does |
|---|---|
| `mechanism_sweep.py` | Experiments 1–3 and e5: 44 CFD runs, and the analysis |
| `exp4_champion_clocks.py` | Experiment 4: clocks and the gain decomposition from the existing duty_v3 traces, no CFD |
| `run_carya_mechanism.slurm` | The Carya job: one 48-core node, about 7 h |

## Run

From `taylorCouetteGym/`, with OpenFOAM v2506 on PATH:

```bash
python3 experiments/mechanism/mechanism_sweep.py --list            # the 44 runs
python3 experiments/mechanism/mechanism_sweep.py --smoke           # 14 runs + 2 e5, 6 s each, about 15 min locally
mkdir -p experiments/mechanism/logs                                # Carya: logs/ is gitignored, slurm needs it
sbatch experiments/mechanism/run_carya_mechanism.slurm             # the real job (Carya, from taylorCouetteGym/)
python3 experiments/mechanism/mechanism_sweep.py --analyze-only    # replot at any time, partial results OK
python3 experiments/mechanism/exp4_champion_clocks.py              # experiment 4
```

e5 is two vortex-state runs at nominal viscosity. In "hold", the flow pulses until 60 s
and then holds constant 300 rpm. In "kick", the speed is constant 300 rpm except for a
single 0.5 s stop at 60 s. They test whether the pulsed flow's Taylor-cell state
survives without further pulsing.

## What each run logs

The `mechClocks` function object writes `postProcessing/mechClocks/0/mech.dat`
every 0.02 s with these columns:

- wall speed
- inlet flow
- reactant inflow (advective and diffusive)
- net outlet outflow
- cup-mixed outlet conversion
- catalytic-wall consumption (−D·snGrad c, using the solver's own face distances)
- reactant inventory
- mean swirl
- secondary-flow energy ⟨u_r² + u_z²⟩
- Taylor-cell count (sign changes of u_r along z at mid-gap)

Fluxes are per wedge and divided by c0. Only the last 3000 solver lines are kept in
`log.pimpleFoam`; full logs would be about 350 MB per run.

## Outputs (`results/main/`)

- `summary_table.csv`: one row per run. It has:
  - X_out, X_budget and the closure residual
  - P and r over the whole-period scoring window
  - the Taylor-cell count and wall flux during bursts
  - the per-stop clocks: swirl, secondary flow and wall flux e-folding times, what fraction of each is left at the end of the stop, the restart overshoot and the burst fade
- `mechanism_report.json`, which holds:
  - the best stop length per viscosity, with its bootstrap interval
  - any cliff edge
  - the fitted exponent a in t_stop* ∝ ν^a, with its interval and whether each prediction is excluded
  - the measured clocks
  - the e3 and e5 tables
- Figures:
  - `stop_curves.png`: gain over constant against stop length, plus Taylor-cell count and burst wall flux
  - `best_stop_vs_viscosity.png`: the GCURS figure
  - `measured_clocks.png`
  - `e3_stop_level.png`
  - `cycle_traces.png`

## Checks built in

- **Closure:** the inventory-balance residual should be about 1e-3 or less. It confirms that the inflow, outflow, wall and inventory accounting adds up.
- **Wall-speed guard:** the measured inner-wall speed must track the commanded table within 1 rpm. This guards against the omega-freeze bug.
- **Viscosity check:** each viscosity template reads back ν and D from both places D is set before warming up.
