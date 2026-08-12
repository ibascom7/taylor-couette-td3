"""FINALS block-waveform modulation env: TaylorCouetteModulationEnv with the
tau-block conventions of the finals campaign.  Deltas vs the parent
(taylor_couette_modulation.py -- read its docstring first; everything not
listed here is identical, including the 7-D observation, the
X - P/P_max reward, the METRICS parsing, and the pristine-IC contract):

1. TAU BLOCKS. block_dt defaults to 26 s = tau = V/Q on the short cell (the
   duty_v3 granularity: outlet conversion lags the wall by ~tau, so shorter
   blocks mis-bill an action's effect to later blocks). One action per block:
   the agent picks a fresh (D, w_low, T) each residence time.

2. RANDOMIZED HORIZON, drawn IN THE ENV. Each reset draws
   n_blocks ~ U{blocks_min..blocks_max} (default 4..6, mean 5 -> the nominal
   5*tau = 130 s episode, matching the baseline-grid convention). Fixed
   horizons leave late-clock states ungrounded (the terminal-bootstrap
   lesson); the spread keeps every reachable clock value visited by
   experience. reset(options={"n_blocks": n}) overrides the draw --
   deterministic evals pin n_blocks=5.

3. CLOCK OBS = t_elapsed / t_scale with FIXED t_scale (default = block_dt),
   i.e. the obs counts elapsed blocks (0, 1, 2, ...) and NEVER normalizes by
   the episode's own drawn horizon -- normalizing would leak the horizon and
   reintroduce exactly the ungrounded-terminal pathology the randomization
   exists to prevent (duty_v2 convention).

4. PEAK CAP AT 2500 RPM (the finals decision 2026-08-09): the fig7-proven
   envelope AND the definition point of p_max, so P/p_max <= 1 by
   construction again. The cap binds through the duty floor, never by
   clamping (clamping would break the commanded mean): worst case w_low = 0
   gives w_hi = mean_max / D <= cap  <=>  D >= mean_max / cap, and duty_min
   is RAISED to that floor at construction when binding:
        w_b =  300: floor 0.12 (a duty_min >= 0.12 is untouched)
        w_b =  750: floor 0.30
        w_b = 1500: floor 0.60
   Construction fails loudly if the band cannot fit (mean_max / cap >
   duty_max). Pass w_hi_cap_rpm=None to lift the cap, or another value to
   move it; free-mean mode uses nom_max_rpm as mean_max.

The static-baseline family for the finals tables is the constant-action
closure of THIS action box (hold one (D, w_low, T) for the whole episode) --
keep the trainer's box and the static-sweep grid in exact agreement.
"""

import numpy as np

from taylor_couette_mixing.envs.taylor_couette_modulation import (
    TaylorCouetteModulationEnv)


class TaylorCouetteFinalsModulationEnv(TaylorCouetteModulationEnv):

    def __init__(
        self,
        case_path,
        w_b_rpm=300.0,           # fixed-mean is the finals default (band mean)
        block_dt=26.0,           # tau = V/Q on the short cell; 130.0 on the long
        blocks_min=4,            # horizon draw ~ U{blocks_min..blocks_max}
        blocks_max=6,            # mean 5 blocks = the nominal 5*tau episode
        t_scale=None,            # clock normalizer; None -> block_dt (counts blocks)
        w_hi_cap_rpm=2500.0,     # fig7-proven envelope = p_max point; None -> uncapped
        **kwargs,
    ):
        self.blocks_min = int(blocks_min)
        self.blocks_max = int(blocks_max)
        if not (1 <= self.blocks_min <= self.blocks_max):
            raise ValueError(f"bad horizon range [{blocks_min}, {blocks_max}]")
        self.w_hi_cap = None if w_hi_cap_rpm is None else float(w_hi_cap_rpm)

        # Parent wants a fixed episode_duration; give it the nominal (mean)
        # horizon -- reset() re-draws max_steps every episode anyway.
        nominal_blocks = 0.5 * (self.blocks_min + self.blocks_max)
        kwargs.pop("episode_duration", None)
        super().__init__(
            case_path,
            w_b_rpm=w_b_rpm,
            episode_duration=nominal_blocks * float(block_dt),
            block_dt=block_dt,
            **kwargs,
        )
        self.t_scale = float(t_scale) if t_scale is not None else self.block_dt

        # Optional guard rail: raise the duty floor so the mean-preserving
        # burst can never exceed the cap (worst case w_low = 0).
        if self.w_hi_cap is not None:
            mean_max = self.nom_max if self.w_b is None else self.w_b
            duty_floor = mean_max / self.w_hi_cap
            if duty_floor > self.duty_max + 1e-12:
                raise ValueError(
                    f"w_b={mean_max} rpm needs duty >= {duty_floor:.3f} to keep "
                    f"bursts <= {self.w_hi_cap} rpm, but duty_max={self.duty_max}")
            self.duty_min = max(self.duty_min, duty_floor)

    # ------------------------------------------------------------------ #
    def _get_obs(self):
        obs = super()._get_obs()
        # Clock counts elapsed blocks against the FIXED t_scale -- never the
        # episode's own drawn horizon (that would leak it).
        obs[3] = (self.step_count * self.block_dt) / self.t_scale
        return obs

    def _get_info(self):
        info = super()._get_info()
        info["n_blocks"] = self.max_steps
        info["duty_min_effective"] = self.duty_min
        return info

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        n = (options or {}).get("n_blocks")
        if n is not None:
            self.max_steps = int(n)
        else:
            self.max_steps = int(self.np_random.integers(
                self.blocks_min, self.blocks_max + 1))
        self.episode_duration = self.max_steps * self.block_dt
        # Re-emit obs/info: the parent computed them before the horizon draw
        # (obs[3] is 0 at reset either way; info carries the fresh n_blocks).
        return self._get_obs(), self._get_info()
