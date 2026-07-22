"""Block-wise waveform modulation of the inner cylinder on the graded Sc=1075
side-outlet reactor. TWO action-space modes (experiments/modulation_rl/README.md):

FREE-MEAN mode (w_b_rpm=None; the v5 "interesting action space", 2026-07-21):
    a0 -> duty D        in [duty_min, duty_max]      (linear; e.g. [0.2, 1.0])
    a1 -> nominal w_nom in [nom_min, nom_max] rpm    (linear; default [0, 500])
    a2 -> period T      in [period_min, period_max] s (LOG map; e.g. [0.5, 10])
The agent chooses its LOW/nominal (mean) speed and the env CONVERTS it to the
burst exactly like the paper's Eq. 8 pulse family (and the fixed-mean mode):
w_hi = (w_nom - (1-D)*w_low) / D, idle w_low = idle_min_rpm (default 0). So
every block's commanded mean IS the chosen w_nom -- free-mean simply means the
agent picks w_b per block. Bursts reach w_nom/D <= 500/0.2 = 2500 rpm, the
envelope the fig7 sweep ran clean and the definition of P_max (motor power at
2500 rpm) -> P/P_max <= 1 by construction. The X - P/P_max reward arbitrates
the conversion-vs-power trade natively (measured landscape: interior optimum,
pulsed w_b~200 R=0.270; reward spans ~0.09 across the box vs ~0.015 under the
fixed-mean design). Degeneracies (tolerated, legitimate strategies):
w_nom=0 -> full-stop block (D, T meaningless); D=1 -> constant at w_nom
(T meaningless).

FIXED-MEAN mode (w_b_rpm=<rpm>; the original design, kept for compatibility):
    a0 -> duty D, a1 -> idle w_low in [idle_min, idle_max] rpm, a2 -> period T.
Burst speed is SOLVED from the fixed-mean constraint, never chosen:
w_hi = (w_b - (1-D)*w_low) / D, so every block's commanded time-average equals
w_b (equal motor power across actions). With D >= 0.6 and w_low <= w_b = 300,
w_hi <= 500 rpm -- proven stable on this mesh (the sweep ran 2500 rpm clean).

CAVEAT (both modes): when T does not divide block_dt the last (truncated)
period is burst-first, so the REALIZED block mean can sit above the commanded
mean; the reward stays honest because P_block is computed on the commanded
wave as-built.

Wave phase RESETS at each block boundary (every block opens with a burst):
fully determined by (action, t), no hidden state -- a structural prior, noted
in the paper.

Observation (7-D flat Box; ALL flow observables are BLOCK AVERAGES of the
METRICS samples, never point samples -- 1 Hz point sampling aliases the +-0.02
outlet flutter into fake slow waves):
    0: wallFlux block average / wallflux_max        (the film-state signal)
    1: delta X vs the previous block                (the sag/tongue detector)
    2: X, current block average                     (the reward's dominant term
                                                     MUST be observable or the
                                                     critic is non-Markov)
    3: episode clock t / episode_duration
    4-6: previous action (raw, in [-1, 1])

Reward per block:  R = X_block - P_block / p_max_watt, both weights 1.
X_block = block-averaged outlet conversion (flux-weighted cup mean, 1-c_out/c0,
the case FO's `conv`). P_block = block-average paper motor power (Eqs 18-23,
motor_power.py) on the commanded omega(t), densified at 100 Hz exactly like
TaylorCouetteWaveformEnv._waveform_motor_energy. For a STATIC policy the final
block's reward equals the benchmark R = X_[40,50] - P/P_max, so learned rewards
compare directly against the static-champion table in the experiment README.

Case contract: an RL-drivable side-outlet case (side_outlet_grad_case -- the
byte-identical-physics twin of side_outlet_case_sc1075_graded) with the
rlMetrics FO emitting "METRICS t= Mz_kin= conv= cupC= wallFlux=" lines,
startFrom latestTime, and writeInterval 1 (block boundaries at multiples of
1 s are always written, so the step loop advances). Episodes start from the
PRISTINE pre-filled IC (0.orig: c=c0 everywhere, fluid at rest) with NO
hydrodynamic warmup -- matching the benchmark episodes exactly -- so do not
cache a 0.warmed/ in the case.
"""

import os

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from taylor_couette_mixing.envs.helpers import Helpers
from taylor_couette_mixing.envs.taylor_couette_waveform import square_wave_points
from taylor_couette_mixing import motor_power

RPM = 2.0 * np.pi / 60.0


class TaylorCouetteModulationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        case_path,
        w_b_rpm=None,            # None -> FREE-MEAN mode (a1 chooses w_hi);
                                 # a number -> fixed-mean mode (w_hi solved)
        episode_duration=50.0,
        block_dt=10.0,
        duty_min=0.6,
        duty_max=1.0,
        idle_min_rpm=0.0,
        idle_max_rpm=None,       # fixed-mean: None -> w_b. free-mean: ignored,
                                 # w_low is pinned at idle_min_rpm (no dim left)
        nom_min_rpm=0.0,         # free-mean only: a1 range for the nominal w_nom
        nom_max_rpm=500.0,       # benchmark-grid ceiling; bursts reach nom/duty
                                 # <= 2500 rpm = the proven fig7 envelope = P_max
        period_min=0.5,
        period_max=5.0,
        ramp_time=0.05,
        p_max_watt=31.94,        # motor power at 2500 rpm (global P_max across runs)
        wallflux_max=1.32e-8,    # steady resolved wallFlux at 2500 rpm (obs normalizer)
        capture_episodes=(),
        capture_dir=None,
    ):
        self.helpers = Helpers(case_path)
        self.w_b = None if w_b_rpm is None else float(w_b_rpm)
        self.block_dt = float(block_dt)
        self.max_steps = int(round(float(episode_duration) / self.block_dt))
        self.episode_duration = self.max_steps * self.block_dt
        self.duty_min = float(duty_min)
        self.duty_max = float(duty_max)
        self.idle_min = float(idle_min_rpm)
        if self.w_b is None:
            self.idle_max = self.idle_min      # free-mean: w_low is a constant
        else:
            self.idle_max = self.w_b if idle_max_rpm is None else float(idle_max_rpm)
        self.nom_min = float(nom_min_rpm)
        self.nom_max = float(nom_max_rpm)
        self._logTmin = float(np.log(period_min))
        self._logTmax = float(np.log(period_max))
        self.ramp_time = float(ramp_time)
        self.p_max = float(p_max_watt)
        self.wallflux_max = float(wallflux_max)

        self.capture_episodes = {int(e) for e in capture_episodes}
        self.capture_dir = capture_dir
        self.episode_count = 0
        self.step_count = 0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float64)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64)

        self._zero_state()

    # ------------------------------------------------------------------ #
    def _zero_state(self):
        """Episode-start values: pristine pre-filled IC has c=c0 everywhere, so
        conversion, wall flux, and power all start at exactly 0 (well-defined)."""
        self.x_block = 0.0
        self.delta_x = 0.0
        self.wf_block = 0.0
        self.p_block = 0.0
        self.prev_action = np.zeros(3)
        self.last_params = (float("nan"),) * 4   # (duty, w_low_rpm, period_s, w_hi_rpm)

    def _decode(self, action):
        """Raw [-1,1]^3 -> (duty, w_low_rpm, period_s, w_hi_rpm).

        Both modes convert a LOW speed to the burst with the same mean-
        preserving formula w_hi = (mean - (1-D)*w_low) / D. Free-mean mode
        (w_b is None): a1 chooses the nominal mean w_nom per block, w_low is
        pinned at idle_min. Fixed-mean mode: a1 chooses the idle speed and the
        mean is the fixed w_b."""
        a = np.clip(np.asarray(action, dtype=float).ravel(), -1.0, 1.0)
        duty = self.duty_min + 0.5 * (a[0] + 1.0) * (self.duty_max - self.duty_min)
        period = float(np.exp(
            self._logTmin + 0.5 * (a[2] + 1.0) * (self._logTmax - self._logTmin)))
        if self.w_b is None:
            w_low = self.idle_min
            w_nom = self.nom_min + 0.5 * (a[1] + 1.0) * (self.nom_max - self.nom_min)
        else:
            w_low = self.idle_min + 0.5 * (a[1] + 1.0) * (self.idle_max - self.idle_min)
            w_nom = self.w_b
        w_hi = (w_nom - (1.0 - duty) * w_low) / duty
        return float(duty), float(w_low), period, float(w_hi)

    def _block_motor_power(self, pts, t0):
        """Block-average motor electrical power [W] on the commanded (t, omega_rad)
        table, densified at 100 Hz (uniform grid -> mean == time average)."""
        tt = np.array([p[0] for p in pts], dtype=float)
        ww = np.array([p[1] for p in pts], dtype=float)
        n = int(min(20000, max(400, self.block_dt * 100)))
        grid = np.linspace(t0, t0 + self.block_dt, n)
        return float(np.mean(motor_power.electrical_power(grid, np.interp(grid, tt, ww))))

    def _get_obs(self):
        return np.array([
            self.wf_block / self.wallflux_max,
            self.delta_x,
            self.x_block,
            (self.step_count * self.block_dt) / self.episode_duration,
            self.prev_action[0],
            self.prev_action[1],
            self.prev_action[2],
        ], dtype=np.float64)

    def _get_info(self):
        duty, w_low, period, w_hi = self.last_params
        return {
            "step_count": self.step_count,
            "mixing_index": self.x_block,          # X_block, in the shared logging slot
            "wf_norm": self.wf_block / self.wallflux_max,
            "power_watt": self.p_block,            # block-average motor power [W]
            "energy_step": self.p_block * self.block_dt,
            "duty": duty,
            "w_low_rpm": w_low,
            "period_s": period,
            "w_hi_rpm": w_hi,
        }

    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mode = (options or {}).get("reset_mode", "hard")
        self.helpers.reset_case(mode=mode)
        self.episode_count += 1
        self.step_count = 0
        self._zero_state()
        return self._get_obs(), self._get_info()

    def step(self, action):
        duty, w_low, period, w_hi = self._decode(action)

        # Phase resets at every block boundary: each block opens with a burst.
        t0 = float(self.helpers._get_latest_time())
        pts, _ = square_wave_points(
            t0, self.block_dt, w_hi * RPM, w_low * RPM,
            period, duty, self.ramp_time, phase0=0.0)
        # Sanitize BEFORE computing motor power so the reward integrates the
        # exact table the BC runs (do_simulation_table re-sanitizes; idempotent).
        pts = self.helpers.sanitize_table_points(pts)
        results = self.helpers.do_simulation_table(pts, self.block_dt)

        # Block averages over ALL METRICS samples (~0.1 s cadence at these
        # speeds), rejecting unphysical scalar-boundedness blips like the
        # fig7 sweep does before averaging.
        convs = np.array([r["conv"] for r in results], dtype=float)
        wfs = np.array([r["wallFlux"] for r in results], dtype=float)
        phys = (convs >= -0.02) & (convs <= 1.02)
        x_block = float(convs[phys].mean() if phys.any() else convs.mean())
        wf_block = float(wfs[phys].mean() if phys.any() else wfs.mean())

        p_block = self._block_motor_power(pts, t0)
        reward = x_block - p_block / self.p_max

        self.delta_x = x_block - self.x_block
        self.x_block = x_block
        self.wf_block = wf_block
        self.p_block = p_block
        self.prev_action = np.clip(np.asarray(action, dtype=float).ravel(), -1.0, 1.0)
        self.last_params = (duty, w_low, period, w_hi)
        self.step_count += 1

        terminated = False
        truncated = (self.step_count >= self.max_steps)

        if self.capture_dir and truncated and self.episode_count in self.capture_episodes:
            dest = os.path.join(self.capture_dir, f"ep{self.episode_count:04d}")
            self.helpers.snapshot_frames(dest)

        return self._get_obs(), reward, terminated, truncated, self._get_info()
