# How the mechanism conclusion was reached

These figures come from the Carya mechanism run (job 8814106, 44/44 runs OK) and
from experiment 4. Regenerate them with `python3 experiments/mechanism/explain_figures.py`.
"Cells" means the number of Taylor vortex cells, counted as sign changes of the radial
velocity along the mid-gap line. Conversion is the wall-flux-budget value unless a
figure says otherwise.

| Figure | What it shows | What it rules in or out |
|---|---|---|
| fig1 | Conversion against stop length at each viscosity. Every point is labelled with its cell count. | The pre-registered test expected one smooth peak that moves with viscosity. Instead, conversion jumps between levels that match the cell count: at nominal viscosity the 8-cell stops (0.4–0.6 s) are best, and the 0.7 s "cliff" is a drop to 6 cells. |
| fig2 | Gain over constant against the number of cells the pulse adds over steady rotation, for every run. | Gain appears only when the restart adds cells. At 2× viscosity the constant flow already has 6 cells and most pulses add none, so the gain is close to zero. |
| fig3 | Two periods of wall speed, cell count and wall consumption for constant, a 0.5 s stop and a 0.7 s stop. | Each restart re-forms the vortices. The cell count it settles into sets the wall consumption for the whole burst, so the gain is not a brief spike after restart. A 0.7 s stop lands in 6 or 8 cells from cycle to cycle. |
| fig4 | The hold run (pulse until 60 s, then constant) and the kick run (one stop at 60 s), against constant and continuous pulsing. | Without repeated stops the extra cells merge back to 4 within about 10–15 s, and conversion returns to the constant value. The gain needs the stops to keep repeating. |
| fig5 | The stop-level sweep: speed during a 0.5 s low period from 0 to 150 rpm. | The full gain needs the vortices to nearly die (secondary-flow energy below about 1% of its burst level). Dipping to 60 rpm or more gives fewer cells and less gain. |
| fig6 | Breakdown of the champion's gain from the existing duty_v3 traces (experiment 4). | +0.077 comes from higher wall consumption through each burst, +0.001 from the restart overshoot, and −0.022 is lost during the stops. This rules out the film-renewal-overshoot story. |
| fig7 | The two pre-registered clocks against viscosity. | Swirl decay does follow 1/ν, as predicted, but the best stop follows neither clock. The 0.7× and 2× best stops sit at the edges of their grids because of which cell counts are available, not because of a time scale. Don't quote the automated exponent a = +2.05 from mechanism_report.json. |

**Conclusion.** A brief full stop lets the Taylor vortices die. The restart re-forms
them as more, narrower cells (8 against 4 at constant 300 rpm), which raise wall mass
transfer through the whole burst. Under steady rotation those extra cells merge back
within about 10–15 s, so the stops must keep repeating.

**Caveats.**

- The model is 2D axisymmetric. In a real reactor at this speed (about 5 times vortex onset), the vortices are wavy.
- Mesh dependence of the cell count is being checked in `experiments/mechanism_mesh`.
