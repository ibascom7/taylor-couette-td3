"""Waveform-control variants of the catalysis env: the agent chooses the
PARAMETERS of a duty-cycle square wave instead of an absolute omega.

Where TaylorCouetteCatalysisEnv picks one absolute omega each second, here the
3-D action is (peak omega, duty D, period T) -- the burst-and-idle waveform
Lopez-Guajardo modulate the inner shaft with. Low duty = brief high-speed bursts
that thin the wall layer (more conversion) while sitting idle most of the time
(cheap energy) -- THE regime where the paper's "more conversion at less power"
lives, which a symmetric square cannot reach. Two styles, set by `per_episode`:

  adaptive (per_episode=False): every `control_dt` s the agent updates (peak, D,
      T) and the env runs that square wave for the next control_dt s, carrying the
      wave phase across steps -- a function generator the agent re-tunes in real
      time. Stepwise RL: max_steps control updates per episode.

  episode  (per_episode=True): the agent picks ONE (peak, D, T) per episode; the
      env runs that waveform for `episode_duration` s (one env.step == one
      episode, like TaylorCouetteConstantOmegaEnv), reward = windowed conversion
      - motor energy of the whole waveform. Black-box optimization of the paper's
      waveform parameters.

Action (3-vector in [-1, 1]) -> (peak, D, T), with the idle level fixed at 0:
    peak in [omega_min, omega_max] rpm     (the burst height)
    D    in [duty_min, duty_max]            fraction of each period spent at peak;
                                            D=1 -> constant peak, D->0 -> brief bursts
    T    in [period_min, period_max] s      (log-mapped) -- "how fast it varies"
So the agent reaches constant (D=1), the paper's 0/2500 D=0.2 square (peak=2500,
D=0.2 -> mean 500), and anything between. The MEAN omega (= D*peak) is reported in
the obs "omega" slot, so train.py's obs_to_state / state_dim=3 and omega_per_step
logging work unchanged (peak/D/T are open-loop in the obs for now). Conversion
rides in the mixing_index slot (parent).
"""
import os

import numpy as np
from gymnasium import spaces

from taylor_couette_mixing.envs.taylor_couette_catalysis import (
    TaylorCouetteCatalysisEnv,
)
from taylor_couette_mixing import motor_power

RPM = 2.0 * np.pi / 60.0


def square_wave_points(t0, duration, w_hi, w_lo, period, duty, ramp, phase0=0.0):
    """Duty-cycle square wave as a ramped tabulated Function1 over [t0, t0+duration].
    Phase in [0,1): <duty -> w_hi (burst), >=duty -> w_lo (idle). Returns
    (points, phase_out) with absolute, strictly-increasing times and a final hold
    point past the end so the BC never evaluates out of range. ramp = transition
    width (s). Handles D=1 (constant) and the carried phase for adaptive control."""
    period = max(float(period), 1e-6)
    duty = min(max(float(duty), 1e-6), 1.0)

    def level(ph):
        return w_hi if (ph % 1.0) < duty else w_lo

    phase = phase0 % 1.0
    cur = level(phase)
    pts = [(t0, cur)]
    t = t0
    t_end = t0 + duration
    for _ in range(1000000):
        nb = duty if phase < duty else 1.0      # next level-change boundary
        t_sw = t + (nb - phase) * period
        if t_sw >= t_end - 1e-9:
            break
        new_phase = nb % 1.0
        nxt = level(new_phase)
        if nxt != cur:                          # emit a ramped transition
            ta = max(t_sw - 0.5 * ramp, pts[-1][0] + 1e-6)
            tb = max(ta + ramp, ta + 1e-6)
            pts.append((ta, cur))
            pts.append((tb, nxt))
            cur = nxt
        t, phase = t_sw, new_phase
    pts.append((t_end + 1.0, cur))
    return pts, (phase0 + duration / period) % 1.0


class TaylorCouetteWaveformEnv(TaylorCouetteCatalysisEnv):
    def __init__(self, case_path, per_episode=False, control_dt=10.0,
                 episode_duration=120.0, period_min=5.0, period_max=30.0,
                 duty_min=0.1, duty_max=1.0, capture_episodes=(), capture_dir=None,
                 **kwargs):
        self.per_episode = bool(per_episode)
        self.control_dt = float(control_dt)
        self.episode_duration = float(episode_duration)
        self.period_min = float(period_min)
        self.period_max = float(period_max)
        self.duty_min = float(duty_min)
        self.duty_max = float(duty_max)
        self._logTmin = float(np.log(self.period_min))
        self._logTmax = float(np.log(self.period_max))
        warmup_rpm = float(kwargs.get("warmup_omega_rpm", 500.0))
        # episode mode: one trial per gym episode (like the constant-omega env).
        if self.per_episode:
            kwargs["max_steps"] = 1
        super().__init__(case_path, capture_episodes=capture_episodes,
                         capture_dir=capture_dir, **kwargs)

        # 3-D action: (peak, duty, period). The MEAN omega (= duty*peak) stays in
        # the "omega" obs slot, so observation_space / _get_obs are inherited.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float64)

        # Sim seconds per control step, and energy normalizers matched to it
        # (control_dt adaptive, or the whole episode in per_episode mode).
        self.time_step = self.episode_duration if self.per_episode else self.control_dt
        tau = np.linspace(0.0, self.time_step,
                          int(min(20000, max(400, self.time_step * 100))))
        w_const = np.full_like(tau, warmup_rpm * RPM)
        self.motor_e_norm = abs(motor_power.energy(tau, w_const)) or 1.0
        self.energy_obs_norm = self.motor_e_norm

        self.wave_phase = 0.0
        self.peak = warmup_rpm           # current waveform params (logging + state)
        self.duty = 1.0
        self.period = self.period_max

        # Expose the agent's OWN current (peak, duty, period) in the observation so
        # the waveform policy is no longer blind to the wave it is commanding
        # (without these the adaptive agent cannot condition on its own action).
        self.observation_space.spaces["peak"] = spaces.Box(
            low=self.omega_min, high=self.omega_max, shape=(1,), dtype=np.float64)
        self.observation_space.spaces["duty"] = spaces.Box(
            low=self.duty_min, high=self.duty_max, shape=(1,), dtype=np.float64)
        self.observation_space.spaces["period"] = spaces.Box(
            low=self.period_min, high=self.period_max, shape=(1,), dtype=np.float64)

    def _decode(self, action):
        a = np.clip(np.asarray(action, dtype=float).ravel(), -1.0, 1.0)
        peak = self.omega_min + 0.5 * (a[0] + 1.0) * (self.omega_max - self.omega_min)
        duty = self.duty_min + 0.5 * (a[1] + 1.0) * (self.duty_max - self.duty_min)
        period = float(np.exp(self._logTmin
                              + 0.5 * (a[2] + 1.0) * (self._logTmax - self._logTmin)))
        return float(peak), float(duty), period

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.wave_phase = 0.0
        return obs, info

    def _get_obs(self):
        # Parent adds wf_norm; append the current waveform parameters.
        obs = super()._get_obs()
        obs["peak"] = self.peak
        obs["duty"] = self.duty
        obs["period"] = self.period
        return obs

    def step(self, action):
        self.peak, self.duty, self.period = self._decode(action)
        w_hi = min(self.peak, self.omega_max) * RPM    # burst height (rad/s)
        w_lo = 0.0                                     # idle (the paper's low level)

        if self.per_episode:
            # Independent trial: every episode starts from the warmed IC so the
            # measured (conversion, energy) reflect this waveform alone.
            self.helpers.reset_case(mode="hard")
            self.wave_phase = 0.0

        t0 = float(self.helpers._get_latest_time())
        duration = self.time_step
        pts, self.wave_phase = square_wave_points(
            t0, duration, w_hi, w_lo, self.period, self.duty,
            self.ramp_time, self.wave_phase)
        results = self.helpers.do_simulation_table(pts, duration)

        # Conversion: windowed mean over the last min(period, duration) s for an
        # episode trial (quasi-steady), else end-of-window for adaptive control.
        # conv is RECORDED (mixing_index slot, lags by ~residence time); the
        # wall-consumption rate wallFlux DRIVES the reward (responds in seconds).
        convs = np.array([r["conv"] for r in results], dtype=float)
        wfs = np.array([r["wallFlux"] for r in results], dtype=float)
        times = np.array([r["t"] for r in results], dtype=float)
        if self.per_episode and len(convs) > 1:
            # Episode trial: average over the last min(period,duration) s (quasi-steady).
            w0t = times[-1] - min(self.period, duration)
            mask = times >= w0t
            sel = mask if mask.any() else np.ones_like(times, dtype=bool)
            conv = float(convs[sel].mean())
            wf = float(wfs[sel].mean())
        else:
            # Adaptive control: end-of-window conv, windowed-mean wallFlux (smoother).
            conv = float(convs[-1])
            wf = float(wfs.mean())
        wf_norm = wf / self.wallflux_ref

        # Input energy: the paper's motor model on the ACTUAL waveform omega(t).
        E = self._waveform_motor_energy(pts, t0, duration)
        E_norm = E / self.motor_e_norm

        reward = self.alpha * wf_norm - self.beta * E_norm

        # Report the MEAN omega (= duty*peak) in the omega slot -> obs unchanged.
        self.omega = self.duty * self.peak
        self.E_current = E if self.per_episode else self.E_current + E
        self.I_current = conv
        self.wf_norm = wf_norm
        self.step_count += 1

        terminated = False
        truncated = (self.step_count >= self.max_steps)

        if self.capture_dir and truncated and self.episode_count in self.capture_episodes:
            tag = (f"ep{self.episode_count:04d}_peak{self.peak:.0f}"
                   f"_D{self.duty:.2f}_T{self.period:.0f}")
            self.helpers.snapshot_frames(os.path.join(self.capture_dir, tag))

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _waveform_motor_energy(self, pts, t0, duration):
        """Motor electrical energy [J] over [t0, t0+duration] from the (t, omega_rad)
        table, densified fine enough to resolve the ramps for the inertia term."""
        tt = np.array([p[0] for p in pts], dtype=float)
        ww = np.array([p[1] for p in pts], dtype=float)
        n = int(min(20000, max(400, duration * 100)))
        grid = np.linspace(t0, t0 + duration, n)
        return float(motor_power.energy(grid, np.interp(grid, tt, ww)))
