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

Reward (per step):   alpha * wf_index  -  beta * E_index   (alpha=conv_weight,
                                                             beta=energy_weight)
  wf_index = wallFlux / wallflux_max  in ~[0, 1]   (catalytic-wall consumption
             rate at the outer wall, normalized by its steady value at omega_max;
             this is the DRIVER of the reward -- it responds on the boundary-layer
             timescale, unlike outlet conversion which lags by the residence time)
  E_index  = E_step  / E_max          in ~[0, 1]   (per-step input energy over the
             motor energy for one step at omega_max)
Both indices share the ~[0,1] scale, so alpha and beta trade off comparable
quantities (this is the whole point -- before, E_norm was ~1 at the mean speed and
~5 at omega_max while wf was ~0.01, so the energy term dwarfed the flux term and
the reward was monotone-decreasing in omega -> idle collapse). energy_weight is
THE knob: too large -> the agent idles omega->0 (cheap, low conversion); too small
-> it pins omega->omega_max. Conversion is still recorded (mixing_index slot) for
reporting.

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
from gymnasium import spaces

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
        feed_velocity=1.462e-3,   # side-outlet: Umean = Q0(100 mL/min)/annulus area
        azimuthal_fraction=5.0 / 360.0,
        wallflux_max=None,        # None -> auto = Q*c0 (100%-conversion feed ceiling)
        time_step=1.0,
        clock_in_obs=False,
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
        # Control granularity: override the base 1 s step (the freeform agent uses a
        # SHORTER step to paint a fine omega(t)). Set BEFORE motor_e_norm uses it.
        self.time_step = float(time_step)
        # Optional clock in the obs (phase = step_count/max_steps in [0,1]). A
        # deterministic TD3 policy collapses to a CONSTANT at steady state (same
        # state -> same omega); the phase lets it output a TIME-VARYING omega(phase)
        # -- a learned free-form waveform -- while still using conv/wallFlux/energy.
        self.clock_in_obs = bool(clock_in_obs)
        # A field dir must be written at EVERY step boundary (the env continues from
        # the latest written time). writeInterval == time_step: omega mode is 1==1;
        # freeform's 0.5 s step needs writeInterval 0.5 or pimpleFoam writes nothing
        # at 0.5, 1.5, ... and the env "does not advance". (Warmup already ran above
        # at the make_case writeInterval=1, which is fine as long as warmup >= 1 s.)
        self.helpers.set_write_interval(self.time_step)
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
        # Per-step energy normalizer = E_max: the motor energy for ONE step at
        # omega_max (the extreme operating point), so E_index = E_step/E_max lives
        # in ~[0,1] -- the SAME scale as the wall-flux index below. (Previously
        # referenced at the warmup/mean speed 500 rpm, which put E_norm ~1 at the
        # mean and ~5.2 at omega_max, dwarfing wf and making the reward
        # monotone-decreasing in omega -> idle collapse.) Motor model is a pure
        # function of omega(t), so E_max is geometry-independent.
        if energy_model == "motor":
            tau = np.linspace(0.0, self.time_step, 200)
            w_max = np.full_like(tau, (self.omega_max * 2 * np.pi) / 60)
            self.motor_e_norm = abs(motor_power.energy(tau, w_max)) or 1.0
        # Per-step scale for normalizing the cumulative-energy OBSERVATION to ~O(1)
        # (train.py divides E_current by energy_obs_norm * max_steps). It MUST match
        # the active energy model: a motor Joule (~6 J/step) is ~10^4x the mechanical
        # E_max_per_step (~1e-3), so normalizing motor energy by the mechanical scale
        # feeds TD3 an energy state ~1e4 while omega/conversion are ~[-1,1] -> wrecked
        # conditioning. (The reward already uses motor_e_norm; this fixes the obs.)
        self.energy_obs_norm = (self.motor_e_norm if energy_model == "motor"
                                else self.E_max_per_step)
        # Conversion (carried in the mixing_index slot) starts ~0, not 1.
        self.I_current = 0.0

        # ---- wallFlux reward normalizer (dimensionless wall-flux index) ------
        # The reward is driven by wallFlux (catalytic-wall consumption rate, m^3/s
        # per unit c0), which responds on the boundary-layer timescale (seconds)
        # instead of lagging by the ~residence time the way the outlet conversion
        # does. We divide by wallflux_max -- the steady wallFlux at omega_max (the
        # extreme operating point) -- so wf_index = wallFlux/wallflux_max lives in
        # ~[0,1], the SAME scale as E_index, and alpha/beta trade off comparable
        # quantities. wallflux_max is CASE-SPECIFIC and must be MEASURED once from
        # CFD (run the case at omega_max to steady state; see the probe in
        # experiments/parallelized_catalysis_rl). Pass it in via wallflux_max.
        #
        # Fallback if wallflux_max is None (the DEFAULT, and the recommended
        # definition): the analytic feed-rate ceiling Q*c0 = the LARGEST amount of
        # reactant that can be consumed per unit time (100% conversion -- you cannot
        # consume more than you feed in at steady state). By the mass balance
        # wallFlux = Q*c0*conversion this upper-bounds the steady wallFlux, so
        # wf_index = wallFlux/(Q*c0) is exactly the conversion-equivalent in [0,1].
        # Q*c0 = inlet velocity * inlet annular area * c0(=1); r_in/r_out in mm;
        # azimuthal_fraction is the wedge slice (5/360; 1.0 for full 360). Defaults
        # are the side-outlet case -> Q*c0 = 2.315e-8 m^3/s (Q0=100 mL/min). A wedge
        # user must pass feed_velocity/r_in/r_out for their geometry.
        r_in_m = float(kwargs.get("r_in", 25.4)) * 1e-3
        r_out_m = float(kwargs.get("r_out", 31.75)) * 1e-3
        inlet_area = azimuthal_fraction * np.pi * (r_out_m ** 2 - r_in_m ** 2)
        self.wallflux_ref = (feed_velocity * inlet_area) or 1.0   # analytic Q*c0 ceiling
        self.wallflux_max = float(wallflux_max) if wallflux_max else self.wallflux_ref
        self.wf_norm = 0.0   # latest wall-flux index (state + logging)
        # Expose wf_norm in the observation so the policy SEES the quantity it is
        # rewarded on (un-lagged), not just the residence-time-lagged conversion.
        self.observation_space.spaces["wf_norm"] = spaces.Box(
            low=0.0, high=10.0, shape=(1,), dtype=np.float64)
        if self.clock_in_obs:
            self.observation_space.spaces["phase"] = spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float64)

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

    def _get_obs(self):
        # Base obs (omega, conversion-in-mixing_index, energy) plus the wall-flux
        # reward metric, so the agent's state includes the signal it optimizes.
        obs = super()._get_obs()
        obs["wf_norm"] = self.wf_norm
        if self.clock_in_obs:
            # phase in [0,1] = where we are in the episode; lets the policy be time-varying.
            obs["phase"] = self.step_count / max(self.max_steps, 1)
        return obs

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
                Mz = result["Mz_kin"] * 930        # kinematic torque -> torque (rho=930, silicone oil)
                powers.append(Mz * omega_rad)
                times.append(result["t"])
            E = -np.trapezoid(powers, times)
            E_norm = E / self.E_max_per_step

        # Conversion at the outlet at the end of this 1 s step. RECORDED for
        # logging/comparison, but it lags the control by ~the residence time.
        conv = float(results[-1]["conv"])
        # Reward is driven by the wall-consumption rate instead: it responds on
        # the boundary-layer timescale (seconds), so the agent gets an un-lagged
        # gradient. Normalized to a conversion-equivalent (~[0,1]); averaged over
        # the step's sub-results for a smoother signal.
        wf_norm = float(np.mean([r["wallFlux"] for r in results])) / self.wallflux_max

        terminated = False
        truncated = (self.step_count >= self.max_steps)
        reward = self.alpha * wf_norm - self.beta * E_norm

        self.E_current = self.E_current + E
        self.I_current = conv          # conversion carried in the mixing_index slot
        self.wf_norm = wf_norm         # wall-flux reward metric (for logging)
        self.step_count += 1

        # Snapshot the whole episode's time dirs for ParaView before train.py's
        # next hard reset wipes them (1-based episode index, same as the parent).
        if self.capture_dir and truncated and self.episode_count in self.capture_episodes:
            dest = os.path.join(self.capture_dir, f"ep{self.episode_count:04d}")
            self.helpers.snapshot_frames(dest)

        return self._get_obs(), reward, terminated, truncated, self._get_info()
