"""Catalysis variant of TaylorCouetteMixingEnv (stepwise omega control).

The catalysis case feeds reactant at c0=1 from the top inlet and consumes it at
the catalytic OUTER wall (a C=0 sink); the figure of merit is CONVERSION
(1 - cup-mixing outlet C), NOT mixing uniformity. This is the quantity the paper
(Lopez-Guajardo et al., Chem. Eng. J. 489 (2024) 151174) actually reports, and
the regime where *modulating* the inner-cylinder speed is expected to beat a
constant speed at the same mean: brief fast bursts thin the wall concentration
boundary layer -> more diffusive flux into the catalytic wall -> more conversion.

Control. Like the stepwise mixing env the agent acts once per simulated second,
but here action[0] in [-1, 1] selects an ABSOLUTE omega in [omega_min, omega_max]
(default 0..2500 rpm) for the coming second -- wide enough to reproduce the
constant (500 rpm) and square-wave (idle 0 / active 2500 rpm) baselines and
everything in between, so the agent can DISCOVER a waveform rather than be handed
one. (The mixing env instead nudges omega by +-50 rpm/step within +-300 rpm,
which can't reach the catalysis baselines' range.)

Reward (per step):   conv_weight * conversion  -  energy_weight * E_norm
  conversion = results[-1]["conv"] in [0, 1]    (instantaneous outlet conversion)
  E_norm     = E_step / E_max_per_step           (per-step input energy; the
                                                   normalizer is ~1 s @ ~500 rpm,
                                                   so E_norm ~ 1 at 500 rpm and
                                                   grows steeply with omega).
energy_weight is THE knob. Too large -> the agent idles omega->0 (cheap, but low
conversion); too small -> it pins omega->omega_max (max conversion at max power).
The paper's "more conversion at less energy" sits at an intermediate value -- tune
it (start ~0.1 and watch whether the learned omega(t) collapses to a rail).

Case contract. The case MUST already be configured for catalysis: catalytic-wall
0/C (feed c0=1, outer wall C=0 sink), the `catalysis` coded function object in
system/controlDict (emits the `conv` METRICS field), and a SCALAR
inner_wall.omega the env can foamDictionary-set. The training slurm does this by
running experiments/oscillation_vs_constant/make_case.py --catalysis --mode
constant on the case copy, then DELETING the stale mixing 0.orig/ and 0.warmed/
so this env re-warms a true catalytic initial condition.

Logging note. To reuse the shared train.py logging / plotting unchanged,
conversion is carried in the obs/info "mixing_index" slot (both live in [0, 1]);
read that field as conversion for this env.
"""

import os

import numpy as np

from taylor_couette_mixing.envs.taylor_couette_mixing import TaylorCouetteMixingEnv
from taylor_couette_mixing import motor_power


class TaylorCouetteCatalysisEnv(TaylorCouetteMixingEnv):
    def __init__(
        self,
        case_path,
        omega_min=0.0,
        omega_max=2500.0,
        omega_start=None,
        conv_weight=1.0,
        energy_weight=0.1,
        energy_model="motor",
        ramp_time=0.05,
        warmup_omega_rpm=500.0,
        **kwargs,
    ):
        # Each episode starts from the warmed initial condition, where the wall
        # is spinning at warmup_omega_rpm -- so start omega there (not 0) so the
        # first step's ramp and the obs reflect the true wall speed.
        if omega_start is None:
            omega_start = warmup_omega_rpm
        # Wide, one-sided omega range so the agent can reach the catalysis
        # baselines (constant 500 rpm, square-wave peak 2500 rpm). No reverse
        # rotation -- the paper modulates 0/2500, never reverses.
        super().__init__(
            case_path,
            omega_min=omega_min,
            omega_max=omega_max,
            omega_start=omega_start,
            warmup_omega_rpm=warmup_omega_rpm,
            **kwargs,
        )
        # Reuse the parent's reward-weight attributes: alpha now weights
        # CONVERSION (maximized), beta weights energy (penalized).
        self.alpha = conv_weight
        self.beta = energy_weight
        # Seconds to ramp the wall from the previous omega to the new one at the
        # start of each step (finite acceleration -> bounded Courant). 0 disables.
        self.ramp_time = ramp_time

        # Energy model for the reward's penalty term:
        #   "motor"      -- the paper's electric-motor power (Eqs. 18-23, motor_power.py),
        #                   evaluated on the commanded omega(t) including the ramp.
        #                   Dominated by bearing friction (LINEAR in speed) and allows
        #                   regen on braking, so brief high-speed bursts at a low mean
        #                   are CHEAP -- this is what lets the agent discover the paper's
        #                   modulation instead of collapsing to a constant speed.
        #   "mechanical" -- viscous-drag work from the CFD (rho*Mz_kin*omega), the
        #                   original metric. Convex in omega, so it punishes bursts and
        #                   the agent will avoid modulation. Kept for comparison.
        self.energy_model = energy_model
        # Per-step energy normalizer so E_norm ~ 1 at the mean speed (keeps the
        # reward's two terms comparable). Motor: steady motor power at warmup speed
        # x 1 s (~6 J at 500 rpm); mechanical: the passed-in E_max_per_step.
        if energy_model == "motor":
            tau = np.linspace(0.0, self.time_step, 200)
            w_const = np.full_like(tau, (warmup_omega_rpm * 2 * np.pi) / 60)
            self.motor_e_norm = abs(motor_power.energy(tau, w_const)) or 1.0
        # Conversion (carried in the mixing_index slot) starts ~0, not 1.
        self.I_current = 0.0

    def _motor_energy_step(self, prev_rad, new_rad):
        """Electric energy [J] for this 1 s step, from the motor model evaluated on
        the ramp-and-hold omega(t) the env actually commands. Includes the inertial
        spin-up cost and, on decelerating steps, negative (regenerated) energy."""
        tau = np.linspace(0.0, self.time_step, 200)
        w = np.where(
            tau < self.ramp_time,
            prev_rad + (tau / self.ramp_time) * (new_rad - prev_rad),
            new_rad,
        )
        return motor_power.energy(tau, w)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        # Parent resets the mixing_index slot to 1.0 ("unmixed"); for catalysis
        # that slot holds conversion, which starts near 0.
        self.I_current = 0.0
        return self._get_obs(), self._get_info()

    def step(self, action):
        # Wall speed at the end of the previous step (= what the BC is set to now),
        # so the ramp this step starts from the real current speed.
        prev_omega_rad = (self.omega * 2 * np.pi) / 60

        # action[0] in [-1, 1] -> ABSOLUTE omega in [omega_min, omega_max].
        frac = 0.5 * (float(action[0]) + 1.0)
        self.omega = float(
            np.clip(
                self.omega_min + frac * (self.omega_max - self.omega_min),
                self.omega_min,
                self.omega_max,
            )
        )
        omega_rad = (self.omega * 2 * np.pi) / 60

        # Ramp the wall from prev_omega_rad to omega_rad over ramp_time s, then
        # hold, instead of an instantaneous jump (keeps the Courant number sane at
        # high omega). Energy below uses the target omega_rad; the ramp is short
        # (~5% of the step) so the power approximation error is negligible.
        results = self.helpers.do_simulation(
            omega_rad, self.time_step,
            ramp_from=prev_omega_rad, ramp_time=self.ramp_time,
        )

        # Input energy this step.
        if self.energy_model == "motor":
            # Paper's electric-motor model on the commanded ramp profile. Can be
            # NEGATIVE on decelerating steps (regen) -> a reward bonus, which is
            # physically correct and teaches the agent that braking returns energy.
            E = self._motor_energy_step(prev_omega_rad, omega_rad)
            E_norm = E / self.motor_e_norm
        else:
            # Viscous-drag work from the CFD: integral of power = rho * Mz_kin * omega.
            powers, times = [], []
            for result in results:
                Mz = result["Mz_kin"] * 1000       # kinematic torque -> torque (rho=1000)
                powers.append(Mz * omega_rad)
                times.append(result["t"])
            E = -np.trapezoid(powers, times)
            E_norm = E / self.E_max_per_step

        # Conversion at the outlet at the end of this 1 s step.
        conv = float(results[-1]["conv"])

        terminated = False
        truncated = (self.step_count >= self.max_steps)
        reward = self.alpha * conv - self.beta * E_norm

        self.E_current = self.E_current + E
        self.I_current = conv          # conversion carried in the mixing_index slot
        self.step_count += 1

        # Snapshot the whole episode's time dirs for ParaView before train.py's
        # next hard reset wipes them (1-based episode index, same as the parent).
        if self.capture_dir and truncated and self.episode_count in self.capture_episodes:
            dest = os.path.join(self.capture_dir, f"ep{self.episode_count:04d}")
            self.helpers.snapshot_frames(dest)

        return self._get_obs(), reward, terminated, truncated, self._get_info()
