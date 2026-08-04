"""duty_v2: (T+, T-) feedback env on the WARMED graded Sc=1075 side-outlet
reactor -- the v2 interpretability configuration.

The v1 D-only campaign (taylor_couette_duty.py, results/duty_diag) found the
true T=5 landscape has a sharp interior optimum (D=0.90, r=0.2886) that 2/3
TD3 seeds missed by collapsing to the D=1 corner, and that enhancement obeys
an IDLE-DURATION law: last-block reward is set by the idle time (1-D)T ~ 0.5 s
(~ the swirl-decay time), nearly independent of period. v2 therefore puts the
physics variable ON A RAW ACTION AXIS.

Action (2-D flat Box, raw [-1,1]^2):
    0: T_plus  -- burst (spin-on) duration per period [s], in [t_plus_min,
                  t_plus_max] (default [1, 5]).
    1: T_minus -- idle (spin-off) duration per period [s], in [t_minus_min,
                  t_minus_max] (default [0, 5]).
The period and duty are DERIVED: T = T_plus + T_minus, D = T_plus/T. Every
point of the box is physical (no T_plus <= T constraint to enforce -- the
parametrization builds it in), T_minus = 0 degenerates to constant w_b (a
legitimate strategy), and the idle law predicts the converged policy should
care about T_minus (~0.5 s) and be indifferent to T_plus -- an axis-aligned
optimal set, directly readable off the training scatter and distillable as a
1-D law T_minus = g(X).

FIXED MEAN, EXACTLY: w_low = 0 stays pinned and the burst speed is solved so
the commanded BLOCK mean is w_b even when T does not divide block_dt (the v1
truncated-period caveat is gone): the unit-amplitude waveform of the block
(ramps + partial periods included) is integrated numerically and
w_hi = w_b / on_fraction. With t_plus_min = 1 the deep-idle corner
(T_plus=1, T_minus=5 -> 2 s on-time per 10 s block) solves to exactly
1500 rpm -- the ceiling the v1 random phase already proved on this case, and
inside the 2500 rpm motor-power calibration, so the power model never
extrapolates. (t_plus_min was raised 0.5 -> 1 on 2026-08-04 for exactly this;
the excluded sub-second-burst family is transient-tactics territory, and idle
duration -- the variable the static probes say sets sustained value -- stays
fully free.) w_hi_cap_rpm (default 2500, the calibrated envelope) is a guard
rail the default box cannot reach; if a widened box ever engages it, the
realized mean droops below w_b and is logged as realized_mean_rpm.

Observation, reward modes, warmed-start contract, and the continuing-task
truncation handling are UNCHANGED from v1 (see taylor_couette_duty.py's
docstring for the rationale): obs = (X_block, t_elapsed/t_scale) with
t_scale = 26 s = V/Q; reward "conv" = X_block - P_block/p_max (RAW,
benchmark-comparable -- reward CENTERING is a trainer concern, never the
env's); terminated is ALWAYS False and episodes end by truncation
(reset(options={"n_blocks": n}) overrides the horizon). Wave phase resets at
each block boundary (every block opens with a burst).

Point case_path at a CLONE of the warmed template
(experiments/modulation_rl/results/warmed_grad300/side_outlet_grad_case),
never at cases/side_outlet_grad_case itself.
"""

import os

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from taylor_couette_mixing.envs.helpers import Helpers
from taylor_couette_mixing.envs.taylor_couette_waveform import square_wave_points
from taylor_couette_mixing import motor_power

RPM = 2.0 * np.pi / 60.0


def solve_block_wave(t0, block_dt, t_plus, t_minus, w_b_rad, ramp,
                     w_hi_cap_rad, n_grid=4000):
    """Build the block's commanded (t, omega) table with the burst level solved
    from the EXACT block-mean constraint (ramps and partial periods included).

    Returns (points, duty, period_s, w_hi_rad, realized_mean_rad). Pure
    function of the commanded shape -- unit-testable without a case."""
    period = t_plus + t_minus
    duty = t_plus / period
    if t_minus <= 0.0:
        pts = [(t0, w_b_rad), (t0 + block_dt + 1.0, w_b_rad)]
        return pts, 1.0, period, w_b_rad, w_b_rad
    unit_pts, _ = square_wave_points(t0, block_dt, 1.0, 0.0,
                                     period, duty, ramp, phase0=0.0)
    tt = np.array([p[0] for p in unit_pts], dtype=float)
    ww = np.array([p[1] for p in unit_pts], dtype=float)
    grid = np.linspace(t0, t0 + block_dt, n_grid)
    on_frac = float(np.mean(np.interp(grid, tt, ww)))
    w_hi = min(w_b_rad / max(on_frac, 1e-6), w_hi_cap_rad)
    pts, _ = square_wave_points(t0, block_dt, w_hi, 0.0,
                                period, duty, ramp, phase0=0.0)
    return pts, duty, period, w_hi, w_hi * on_frac


class TaylorCouetteDutyV2Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        case_path,
        w_b_rpm=300.0,           # fixed commanded mean (the warmup's spin speed)
        episode_duration=50.0,
        block_dt=10.0,
        t_plus_min=1.0,          # burst-duration action bounds [s]
        t_plus_max=5.0,
        t_minus_min=0.0,         # idle-duration action bounds [s]
        t_minus_max=5.0,
        w_hi_cap_rpm=2500.0,     # guard rail (default box peaks at exactly 1500)
        ramp_time=0.05,
        p_max_watt=31.94,        # motor power at 2500 rpm (global reward normalizer)
        t_scale=26.0,            # FIXED physical time normalizer: tau = V/Q [s]
        x_init=0.353,            # warmed steady conversion (initial obs stand-in)
        reward_mode="conv",      # "conv" | "flux" (see taylor_couette_duty.py)
        flux_to_conv=4.2e7,      # mass-balance k = X_ss/J_ss from the warmup
        capture_episodes=(),
        capture_dir=None,
    ):
        self.helpers = Helpers(case_path)
        self.w_b = float(w_b_rpm)
        self.block_dt = float(block_dt)
        self.default_max_steps = int(round(float(episode_duration) / self.block_dt))
        self.max_steps = self.default_max_steps
        self.t_plus_min = float(t_plus_min)
        self.t_plus_max = float(t_plus_max)
        self.t_minus_min = float(t_minus_min)
        self.t_minus_max = float(t_minus_max)
        self.w_hi_cap = float(w_hi_cap_rpm)
        self.ramp_time = float(ramp_time)
        self.p_max = float(p_max_watt)
        self.t_scale = float(t_scale)
        self.x_init = float(x_init)
        if reward_mode not in ("conv", "flux"):
            raise ValueError(f"reward_mode must be 'conv' or 'flux', got {reward_mode!r}")
        self.reward_mode = reward_mode
        self.flux_to_conv = float(flux_to_conv)

        self.capture_episodes = {int(e) for e in capture_episodes}
        self.capture_dir = capture_dir
        self.episode_count = 0
        self.step_count = 0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float64)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64)

        self._zero_state()

    # ------------------------------------------------------------------ #
    def _zero_state(self):
        self.x_block = self.x_init
        self.wf_block = 0.0
        self.p_block = 0.0
        # (t_plus, t_minus, duty, w_hi_rpm, realized_mean_rpm)
        self.last_params = (float("nan"),) * 5

    def _decode(self, action):
        """Raw [-1,1]^2 -> (T_plus, T_minus) [s]."""
        a = np.clip(np.asarray(action, dtype=float).ravel(), -1.0, 1.0)
        t_plus = self.t_plus_min + 0.5 * (a[0] + 1.0) * (self.t_plus_max - self.t_plus_min)
        t_minus = self.t_minus_min + 0.5 * (a[1] + 1.0) * (self.t_minus_max - self.t_minus_min)
        return float(t_plus), float(t_minus)

    def _block_motor_power(self, pts, t0):
        """Block-average motor electrical power [W] on the commanded table."""
        tt = np.array([p[0] for p in pts], dtype=float)
        ww = np.array([p[1] for p in pts], dtype=float)
        n = int(min(20000, max(400, self.block_dt * 100)))
        grid = np.linspace(t0, t0 + self.block_dt, n)
        return float(np.mean(motor_power.electrical_power(grid, np.interp(grid, tt, ww))))

    def _get_obs(self):
        return np.array([
            self.x_block,
            (self.step_count * self.block_dt) / self.t_scale,
        ], dtype=np.float64)

    def _get_info(self):
        t_plus, t_minus, duty, w_hi, realized = self.last_params
        return {
            "step_count": self.step_count,
            "mixing_index": self.x_block,          # X_block, shared logging slot
            "wf_block": self.wf_block,
            "power_watt": self.p_block,
            "energy_step": self.p_block * self.block_dt,
            "t_plus_s": t_plus,
            "t_minus_s": t_minus,
            "duty": duty,
            "w_low_rpm": 0.0,
            "period_s": t_plus + t_minus,
            "w_hi_rpm": w_hi,
            "realized_mean_rpm": realized,         # == w_b unless w_hi_cap engaged
        }

    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        opts = options or {}
        self.helpers.reset_case(mode=opts.get("reset_mode", "hard"))
        self.max_steps = int(opts.get("n_blocks", self.default_max_steps))
        self.episode_count += 1
        self.step_count = 0
        self._zero_state()
        return self._get_obs(), self._get_info()

    def step(self, action):
        t_plus, t_minus = self._decode(action)

        # Phase resets at every block boundary: each block opens with a burst.
        t0 = float(self.helpers._get_latest_time())
        pts, duty, period, w_hi_rad, mean_rad = solve_block_wave(
            t0, self.block_dt, t_plus, t_minus, self.w_b * RPM,
            self.ramp_time, self.w_hi_cap * RPM)
        pts = self.helpers.sanitize_table_points(pts)
        results = self.helpers.do_simulation_table(pts, self.block_dt)

        convs = np.array([r["conv"] for r in results], dtype=float)
        wfs = np.array([r["wallFlux"] for r in results], dtype=float)
        phys = (convs >= -0.02) & (convs <= 1.02)
        x_block = float(convs[phys].mean() if phys.any() else convs.mean())
        wf_block = float(wfs[phys].mean() if phys.any() else wfs.mean())

        p_block = self._block_motor_power(pts, t0)
        if self.reward_mode == "flux":
            reward = self.flux_to_conv * wf_block - p_block / self.p_max
        else:
            reward = x_block - p_block / self.p_max

        self.x_block = x_block
        self.wf_block = wf_block
        self.p_block = p_block
        self.last_params = (t_plus, t_minus, duty, w_hi_rad / RPM, mean_rad / RPM)
        self.step_count += 1

        terminated = False                         # continuing task: NEVER done
        truncated = (self.step_count >= self.max_steps)

        if self.capture_dir and truncated and self.episode_count in self.capture_episodes:
            dest = os.path.join(self.capture_dir, f"ep{self.episode_count:04d}")
            self.helpers.snapshot_frames(dest)

        return self._get_obs(), reward, terminated, truncated, self._get_info()
