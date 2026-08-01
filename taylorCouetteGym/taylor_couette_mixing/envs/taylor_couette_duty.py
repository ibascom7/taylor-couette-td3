"""D-only duty-cycle feedback env on the WARMED graded Sc=1075 side-outlet
reactor -- the interpretability configuration (a Zhou & Zhu-style one-knob
state-action map, experiments/modulation_rl README v5 -> D-only reduction).

Everything the settled v5 fixed-mean design left free is PINNED to what the
converged v5-s2 policy chose (params_per_step, last 40 episodes): idle
w_low = 0 (the policy drove it to ~10 rpm on its own), period T = 5 s (the
policy sat at ~4.6 s against the 5 s ceiling; 5 divides block_dt = 10 exactly
so the 3-D env's truncated-period caveat VANISHES -- commanded block mean ==
realized mean), fixed mean w_b = 300. ONE action dim remains: duty D in
[duty_min, duty_max]; the burst is solved from the mean constraint,
w_hi = w_b / D (<= 1500 rpm at D = 0.2, well inside the proven 2500 envelope).
D = 1 degenerates to constant w_b (T meaningless) -- a legitimate strategy.

CONTINUOUS-MANUFACTURING contract: episodes start from the WARMED constant-300
steady operating state (0.warmed cached by
experiments/modulation_rl/warm_template.py: steady X = 0.353, wallFlux =
8.4e-9 after 60 s = 2.3 tau), NOT the pristine startup transient. Point
case_path at a CLONE of the warmed template
(experiments/modulation_rl/results/warmed_grad300/side_outlet_grad_case),
never at cases/side_outlet_grad_case itself -- the base case's pristine-IC
benchmark contract (no cached 0.warmed/) must stay intact.

Observation (2-D flat Box):
    0: X_block  -- block-averaged outlet conversion (the plant-measurable
                   signal; chosen over wallFlux by design decision 2026-07-31)
    1: t_elapsed / t_scale -- seconds since episode start over a FIXED
                   physical timescale (default 26 s = V/Q, the residence
                   time). Deliberately NOT a fraction of episode_duration:
                   the coordinate means the same thing in a 50 s training
                   episode and a 500 s production run. Interpretability
                   check downstream: if the trained policy's dD/dt -> 0 at
                   late t, the policy is stationary once the reactor settles
                   and distillation can drop the time input.

Reward per block (reward_mode):
    "conv" (default): X_block - P_block/p_max -- the 3-D env's form, directly
        comparable to the static benchmarks. CAVEAT: outlet X lags the wall
        by tau ~ 26 s ~ 2.6 blocks, so per-block credit is delayed.
    "flux": flux_to_conv * wf_block - P_block/p_max -- the mass-balance
        conversion equivalent (steady state: X = A_wall*<J>/(Q*c0), so
        k = X_ss/J_ss = 0.3527/8.4e-9 = 4.2e7 from the warmup's steady
        pair). The same objective time-averaged, but each block's reward
        reflects ITS OWN action (no transport delay). wallFlux is logged in
        info either way, so post-hoc analysis has both signals always.

terminated is ALWAYS False; episodes end by TRUNCATION at the horizon. This
is a CONTINUING task: the trainer must BOOTSTRAP at truncation (done=0, the
Pardo time-limit treatment) -- the OPPOSITE of parallel_train.py's
fixed-horizon handling, which was specific to the fraction-clock obs.
reset(options={"n_blocks": n}) overrides the horizon for that episode;
randomizing it varies the truncation time so late-t states also appear as
source states and their Q-values stay grounded.

Same case contract as TaylorCouetteModulationEnv otherwise (rlMetrics FO,
startFrom latestTime, writeInterval 1); wave phase resets at each block
boundary (every block opens with a burst -- policy fully determined by
(action, t), a structural prior noted in the paper).
"""

import os

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from taylor_couette_mixing.envs.helpers import Helpers
from taylor_couette_mixing.envs.taylor_couette_waveform import square_wave_points
from taylor_couette_mixing import motor_power

RPM = 2.0 * np.pi / 60.0


class TaylorCouetteDutyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        case_path,
        w_b_rpm=300.0,           # fixed commanded mean (the warmup's spin speed)
        episode_duration=50.0,
        block_dt=10.0,
        duty_min=0.2,            # w_hi = w_b/D: 1500 rpm at the floor (envelope 2500)
        duty_max=1.0,
        period=5.0,              # pinned; divides block_dt -> exact block means
        ramp_time=0.05,
        p_max_watt=31.94,        # motor power at 2500 rpm (global reward normalizer)
        t_scale=26.0,            # FIXED physical time normalizer: tau = V/Q [s]
        x_init=0.353,            # warmed steady conversion (initial obs stand-in
                                 #  before the first block average exists)
        reward_mode="conv",      # "conv" | "flux" (see module docstring)
        flux_to_conv=4.2e7,      # mass-balance k = X_ss/J_ss from the warmup
        capture_episodes=(),
        capture_dir=None,
    ):
        self.helpers = Helpers(case_path)
        self.w_b = float(w_b_rpm)
        self.block_dt = float(block_dt)
        self.default_max_steps = int(round(float(episode_duration) / self.block_dt))
        self.max_steps = self.default_max_steps
        self.duty_min = float(duty_min)
        self.duty_max = float(duty_max)
        self.period = float(period)
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

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64)

        self._zero_state()

    # ------------------------------------------------------------------ #
    def _zero_state(self):
        """Episode-start values: the warmed IC sits at the constant-300 steady
        state, so the honest initial conversion is x_init (measured), not 0."""
        self.x_block = self.x_init
        self.wf_block = 0.0
        self.p_block = 0.0
        self.last_params = (float("nan"),) * 2   # (duty, w_hi_rpm)

    def _decode(self, action):
        """Raw [-1,1] -> (duty, w_hi_rpm) via the mean constraint w_hi = w_b/D
        (w_low = 0)."""
        a = float(np.clip(np.asarray(action, dtype=float).ravel()[0], -1.0, 1.0))
        duty = self.duty_min + 0.5 * (a + 1.0) * (self.duty_max - self.duty_min)
        w_hi = self.w_b / duty
        return float(duty), float(w_hi)

    def _block_motor_power(self, pts, t0):
        """Block-average motor electrical power [W] on the commanded (t, omega)
        table, densified at 100 Hz (uniform grid -> mean == time average)."""
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
        duty, w_hi = self.last_params
        return {
            "step_count": self.step_count,
            "mixing_index": self.x_block,          # X_block, in the shared logging slot
            "wf_block": self.wf_block,             # raw wallFlux block average (logged
                                                   #  ALWAYS, whichever reward_mode)
            "power_watt": self.p_block,            # block-average motor power [W]
            "energy_step": self.p_block * self.block_dt,
            "duty": duty,
            "w_low_rpm": 0.0,
            "period_s": self.period,
            "w_hi_rpm": w_hi,
        }

    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        opts = options or {}
        mode = opts.get("reset_mode", "hard")
        self.helpers.reset_case(mode=mode)
        self.max_steps = int(opts.get("n_blocks", self.default_max_steps))
        self.episode_count += 1
        self.step_count = 0
        self._zero_state()
        return self._get_obs(), self._get_info()

    def step(self, action):
        duty, w_hi = self._decode(action)

        # Phase resets at every block boundary: each block opens with a burst.
        t0 = float(self.helpers._get_latest_time())
        pts, _ = square_wave_points(
            t0, self.block_dt, w_hi * RPM, 0.0,
            self.period, duty, self.ramp_time, phase0=0.0)
        # Sanitize BEFORE computing motor power so the reward integrates the
        # exact table the BC runs (do_simulation_table re-sanitizes; idempotent).
        pts = self.helpers.sanitize_table_points(pts)
        results = self.helpers.do_simulation_table(pts, self.block_dt)

        # Block averages over ALL METRICS samples, rejecting unphysical
        # scalar-boundedness blips like the fig7 sweep does before averaging.
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
        self.last_params = (duty, w_hi)
        self.step_count += 1

        terminated = False                         # continuing task: NEVER done;
        truncated = (self.step_count >= self.max_steps)   # trainer bootstraps here

        if self.capture_dir and truncated and self.episode_count in self.capture_episodes:
            dest = os.path.join(self.capture_dir, f"ep{self.episode_count:04d}")
            self.helpers.snapshot_frames(dest)

        return self._get_obs(), reward, terminated, truncated, self._get_info()
