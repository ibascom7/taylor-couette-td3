# GCURS talk: speaker notes (Computational & Applied Mathematics section)

Timed to 11:00 for the 15 main slides; the last two are backup slides for questions.

## 1. Brief stops, more vortices

[0:30] Hi, I'm Isaiah Bascom. I work on Taylor–Couette reactors: a liquid fills the gap between two cylinders, the inner one spins, and a catalyst coats the outer wall. Today's story has two parts. First, a reinforcement-learning agent found a surprisingly simple way to run this reactor better: brief full stops. Second, we found out why it works. The picture at the bottom is the punchline: steady rotation gives four vortex cells, and the pulsed rotation gives eight.

## 2. The reaction happens only at the wall

[0:45] The reaction only happens on the catalytic outer wall, so everything depends on how fast reactant gets carried from the bulk to that wall. The Taylor vortices in the gap do that carrying. Earlier work, from López-Guajardo and colleagues in 2024, showed that changing the rotation over time can beat spinning at a constant speed, and their recipe was long slow idles with short, fast bursts. So this is a control problem governed by PDEs: choose the speed as a function of time, with its average held fixed, to maximize how much reactant gets converted. What is the optimal control, and why does it work?

## 3. The model: flow plus reactant transport

[0:45] Here's the model. Incompressible Navier–Stokes for the flow, solved axisymmetrically but keeping the swirl component, and an advection–diffusion equation for the reactant. The inner wall turns at a prescribed speed omega of t. On the catalytic wall the concentration is zero, so the reaction rate is set entirely by how fast reactant diffuses in. It's a finite-volume discretization in OpenFOAM, with the mesh graded down to 25 microns at the wall because the concentration layer there is thin: the Schmidt number is about a thousand. The objective is conversion over the last residence time, with the mean speed fixed so every candidate uses the same power.

## 4. Three ways to search an expensive simulator

[0:45] Each simulated episode costs a few CPU-hours, so I used three complementary tools. Reinforcement learning, with TD3, an actor–critic method, to discover a strategy without being told any physics: it sees only the conversion and the time, and it chooses how long to run and how long to stop. Bayesian optimization, with a Gaussian-process surrogate and expected improvement over 220 simulations, as an independent check on what the agent finds. And designed experiments with hypotheses written down in advance, to explain why the answer works.

## 5. Reinforcement learning found brief full stops

[0:45] Here's what the reinforcement-learning agent did, with the average held at 300 rpm. What it learned, in all five independent training runs, is this: run a little above 300 rpm for about four and a half seconds, then stop completely for about half a second, and repeat. That was not the answer I expected, and it's the opposite of the published recipe.

## 6. The opposite of the published recipe

[0:45] On top, the published recipe: burst to 2500 rpm for six seconds, then sit still for twenty-four. Below, what the agent found. The Bayesian-optimization search over 220 fixed waveforms found the same thing: stop for about half a second every five seconds. The learned policies come within one to two hundredths of conversion of that best fixed waveform, so RL matched the search rather than beating it. The published recipe, run in our model, loses even to steady rotation. One note: my submitted abstract describes long low-speed periods with short bursts. The result turned out to be the opposite, brief full stops at high duty, and this slide is the correction.

## 7. It transfers to a reactor five times taller

[0:45] Then a generalization test: the policy trained on the short reactor, run with no retraining on a reactor five times taller. The blue line is steady rotation at different speeds. The red star is the transferred policy: 0.90 conversion at 3.7 watts, against 0.79 for steady rotation at the same power. It ties the best fixed waveform at 0.91. The grey triangle is the published recipe: 0.76 while using about 1.7 times the power. One caveat in small print: these tall-reactor runs used the original numerical time step, so I'd trust the ranking more than the third decimal.

## 8. Why 0.5 s? Two time scales, two predictions

[0:45] Why half a second? Two time scales sit near it. One is the decay of the swirl: the slowest mode of momentum diffusion across a gap of width d decays on d squared over pi squared nu, about 0.4 seconds. The other is the time for the reactant-depleted layer at the wall to regrow, delta squared over D, about 0.8 seconds. They scale differently with viscosity. Holding the diffusivity fixed, the swirl clock predicts the best stop goes like nu to the minus one, while the wall-layer clock predicts a positive power, between about 0.3 and 0.7. So I varied the viscosity, fit an exponent, and decided in advance that its sign would pick the winner.

## 9. Neither guess sets the best stop

[0:30] On the left, the swirl decays exactly as the eigenmode estimate says, one over viscosity. On the right, the best stop follows neither prediction; a formal exponent fit isn't even meaningful, because the optimum jumps rather than drifting. At 0.7 times the viscosity it's the shortest stop I tried; at twice the viscosity it's the longest. The labels next to the points, six cells and eight cells, are the clue to what's really going on.

## 10. The answer: each restart builds more vortex cells

[1:00] This is the answer. Both panels show the gap between the cylinders, unrolled sideways: the inner wall at the bottom, the catalytic wall at the top. Colour is radial velocity; red means liquid heading toward the catalytic wall. Under steady rotation there are four large vortex cells. Under pulsing, in the middle of a burst, there are eight narrower ones. In the language of dynamical systems, this is multistability: at the same Reynolds number, a short Taylor–Couette annulus has several stable solutions with different numbers of cells, and which one you land in depends on the initial condition. That's classic Taylor–Couette physics; Benjamin and Mullin studied it in the early 1980s. Each brief stop brings the flow nearly to rest, and the restart from rest selects a higher axial wavenumber, eight cells instead of four. Narrower cells carry reactant to the wall faster.

## 11. Conversion follows the cell count

[0:45] Now plot conversion against stop length, and label each point with its cell count. It isn't a smooth curve with one peak. It's a set of levels: steady rotation, four cells, at the bottom; six-cell stops in the middle; and the best stops, from 0.4 to 0.6 seconds, are exactly the ones whose restart lands in eight cells, about 0.04 above steady rotation. The sudden drop at 0.7 seconds, which looked like a strange cliff in the original search, is just the flow falling back to six cells.

## 12. Restarts make the extra cells; steady rotation erases them

[1:00] Two more pieces. On the left, inside each cycle: after every restart the cell count jumps up and settles at eight, and the reaction rate at the wall stays about fifteen percent above steady rotation for the whole burst. It isn't a brief spike after the restart; the restart spike contributes almost nothing. On the right: if I pulse until 60 seconds and then hold steady, or do just one stop, the extra cells merge back to four within ten to fifteen seconds, and conversion drops back to the steady value. So the stops have to keep coming every few seconds. That explains why five-second cycles win and the published thirty-second cycle loses.

## 13. It has to be a full stop

[0:30] One last check on the mechanism. Instead of stopping, I slowed the cylinder to 30, 60, 100 or 150 rpm for the same half second. Thirty is as good as a full stop. At 60 rpm, about where vortices start to form, you get six cells instead of eight, and at 100 and above you lose the gain entirely. The vortices really have to die, down to about one percent of their energy, before the restart can build a new, finer pattern.

## 14. Numerical verification

[0:45] For this audience, the obvious question is whether it's numerics. I repeated the key runs on a coarser and a finer mesh from the same self-similar family. On all three, steady rotation has four cells, the best stops land in eight, and the gain is four to six hundredths. Halving the time step exposed and removed a spurious jet at the outlet worth about four hundredths of conversion, which is why all of these runs use the smaller step. The reactant mass balance closes to a third of a percent, and replicate noise is about 0.002. One thing does move with the mesh: the exact edges of the eight-cell window, by a tenth of a second or two, so I quote "about half a second".

## 15. Three takeaways

[0:45] To wrap up. One: brief full stops beat steady rotation at the same power, and beat the published recipe. Two: reinforcement learning found this with no physics given, matched an independent Bayesian optimization, and generalized to a reactor five times taller. Three: the optimum exploits multistability. Each restart selects a state with more cells that transports reactant faster, and because that state decays back under steady forcing, the stops have to repeat. The main limits are the 2D model, so the mechanism is a hypothesis for real 3D flow, and that RL tied the optimizer rather than beating it. Next: a 3D check, control when the feed changes, and a physics-based surrogate. Thank you, happy to take questions.

## 16. The gain is a higher plateau, not a restart spike

[Backup, use if asked] If someone asks whether this is just the burst right after restart, a "film renewal" spike: no. Splitting the gain over steady rotation, plus 0.077 comes from the reaction rate staying higher through the whole burst, only plus 0.001 from the spike right after the restart, and minus 0.022 is lost while stopped. The parts sum to 0.056 against the measured 0.054; the small difference comes from how the burst plateau is defined.

## 17. Gain needs cells added over steady rotation

[Backup, use if asked about viscosity] Every pulsed run from the viscosity sweep, plotted by how many cells pulsing adds over steady rotation at that viscosity. Runs that add no cells gain almost nothing; runs that add two or four cells gain up to five hundredths. At twice the viscosity, steady rotation already has six cells, so most pulses add none, which is why the gain there is small. The size of the gain also depends on viscosity, so it isn't a single universal curve.
