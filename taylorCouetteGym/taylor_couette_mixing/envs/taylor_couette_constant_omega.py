"""Constant-omega variant of TaylorCouetteMixingEnv.

Alternative formulation of the mixing problem: instead of choosing a
delta_omega every simulated second, the agent chooses ONE constant angular
velocity for an entire episode and is scored on how well that single omega
mixes the dye (final mixing index) versus how much energy it burns (total
energy over the run).

One env.step() IS one episode: it hard-resets the case to the warmed initial
condition, runs pimpleFoam once at the chosen constant omega for
`episode_duration` seconds, then returns a single reward and the end-of-episode
(omega, I_current, E_current) as the observation -- which is carried forward as
the policy's input for the next episode's omega choice.

Because every episode hard-resets, the trials are independent and the reward is
a function of the chosen omega alone -- this is effectively a 1-D black-box
optimization (a bandit). With discount=0 the agent is a clean one-shot omega
optimizer.

Logging compatibility: since each env.step() is a full episode, train.py's
episode_returns / {reward,omega}_per_step logs end up one-value-per-episode, so
plot_comparison.py (reward per episode) and plot_sweep.py (--metric omega/reward
per episode) work unchanged.

Frame capture: pass capture_episodes (1-based episode indices) and capture_dir
to snapshot the OpenFOAM time directories of those episodes into
capture_dir/ep<NNNN>_omega<W>/ for offline ParaView videos (e.g. the 1st, 5th,
and final episode to see initial / learning / converged behavior).
"""

import os

import numpy as np
from gymnasium import spaces

from taylor_couette_mixing.envs.taylor_couette_mixing import TaylorCouetteMixingEnv


class TaylorCouetteConstantOmegaEnv(TaylorCouetteMixingEnv):
    def __init__(
        self,
        case_path,
        episode_duration=60.0,
        capture_episodes=(),
        capture_dir=None,
        **kwargs,
    ):
        # Each gym episode is a single constant-omega trial.
        super().__init__(case_path, max_steps=1, **kwargs)

        self.episode_duration = episode_duration
        self.capture_episodes = {int(e) for e in capture_episodes}
        self.capture_dir = capture_dir

        # Global 1-based episode counter (parent's step_count resets every
        # episode, so it can't index episodes for frame capture).
        self.trial_count = 0
        # Previous episode's terminal (omega, I, E), fed forward as the next
        # observation; None before the first episode -> use the IC.
        self._carry_obs = None

        # E_current is now the TOTAL energy of one constant-omega run (the whole
        # episode), so its upper bound scales with episode_duration rather than
        # with the number of steps. Assign into the underlying mapping directly
        # to avoid relying on Dict.__setitem__ across gymnasium versions.
        self.observation_space.spaces["energy_consumption"] = spaces.Box(
            low=0.0,
            high=self.E_max_per_step * episode_duration,
            shape=(1,),
            dtype=np.float64,
        )

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        # Carry the previous episode's terminal readout forward as the obs the
        # policy conditions on (first episode has none -> parent's IC stands).
        if self._carry_obs is not None:
            self.omega, self.I_current, self.E_current = self._carry_obs
            observation = self._get_obs()
            info = self._get_info()
        return observation, info

    def step(self, action):
        # action[0] in [-1, 1] maps to an ABSOLUTE omega across the full range,
        # not an increment. omega_min/omega_max are symmetric (-/+300) so this
        # is omega = action * omega_max.
        self.omega = float(
            np.clip(action[0] * self.omega_max, self.omega_min, self.omega_max)
        )
        omega_rad = (self.omega * (2 * np.pi)) / 60

        # Independent trial: start every run from the same warmed IC so the
        # measured (I, E) reflects this omega alone, not leftover state.
        self.helpers.reset_case(mode="hard")

        # One pimpleFoam call for the entire episode duration. With
        # writeInterval 1 this yields one METRICS line per simulated second.
        results = self.helpers.do_simulation(omega_rad, self.episode_duration)

        # Total energy over the whole constant-omega run.
        powers = []
        times = []
        for result in results:
            Mz = result["Mz_kin"] * 1000
            powers.append(Mz * omega_rad)
            times.append(result["t"])
        E = -np.trapezoid(powers, times)
        E_norm = E / (self.E_max_per_step * self.episode_duration)

        final_result = results[-1]
        concentrations = [final_result[f"C{i}"] for i in range(20)]
        mixing_index = self.calculate_mixing_index(concentrations)

        reward = -(self.alpha * mixing_index) - (self.beta * E_norm)

        self.trial_count += 1
        # Snapshot this episode's frames BEFORE the next step's hard reset wipes
        # them. capture_episodes is 1-based to match the printed episode index.
        if self.capture_dir and self.trial_count in self.capture_episodes:
            dest = os.path.join(
                self.capture_dir,
                f"ep{self.trial_count:04d}_omega{self.omega:+.1f}",
            )
            self.helpers.snapshot_frames(dest)

        # End-of-episode readout becomes the next observation.
        self.E_current = E
        self.I_current = mixing_index
        self.step_count += 1
        self._carry_obs = (self.omega, self.I_current, self.E_current)

        terminated = False
        truncated = True  # one trial per episode

        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info
