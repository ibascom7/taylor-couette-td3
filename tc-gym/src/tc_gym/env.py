"""Gymnasium environment for Taylor-Couette mixing driven by OpenFOAM.

Observation: Box(1,) — current inner-cylinder angular velocity (RPM).
Action:      Box(1,) in [-1, 1] — scaled by `action_delta_rpm` and added
             to the current omega (then clipped to [omega_min, omega_max]).
Reward:      -(alpha * mixing_index + beta * energy_consumption_norm)

`mixing_index` is the intensity-of-segregation (variance of the
radially-binned scalar concentration C divided by its theoretical
maximum). `energy_consumption_norm` is the work done on the fluid
during the step, normalized by a reference E_max.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tc_gym.openfoam import OpenFOAMCase


# Geometry of the radial measurement bins at the bottom outlet.
# Must match the rlMetrics function object in system/controlDict.
R_INNER_MM = 25.4
R_OUTER_MM = 31.75
N_BINS = 20

# Energy normalization: work integral over a 1-second pimpleFoam run
# at 52.4 rad/s (=500 RPM) starting from the pristine IC. Used to put
# E and I on comparable scales in the reward.
E_REF_JOULES = 0.0011017031875434

# Water density (kg/m^3): OpenFOAM reports kinematic torque Mz_kin
# (torque / rho); multiplying by RHO recovers torque in N·m.
RHO = 1000.0


def _rpm_to_rad_s(rpm: float) -> float:
    return rpm * 2.0 * np.pi / 60.0


def _mixing_index(concentrations: np.ndarray) -> float:
    """Intensity of segregation on N_BINS radial bins (range ~[0, 1]).

    Bin centers are weighted by their radius (annular area weighting
    assuming bins of equal radial width).
    """
    dr = (R_OUTER_MM - R_INNER_MM) / N_BINS
    r_mids = R_INNER_MM + (np.arange(N_BINS) + 0.5) * dr
    weights = r_mids / r_mids.sum()

    c_bar = np.sum(weights * concentrations)
    sigma2 = np.sum(weights * (concentrations - c_bar) ** 2)
    sigma2_max = c_bar * (1.0 - c_bar) + 1e-16
    return float(sigma2 / sigma2_max)


def _step_energy_joules(
    metrics: list[Dict[str, float]], omega_rad: float
) -> float:
    """Work done on the fluid over a step (J), from per-interval Mz_kin."""
    times = np.array([m["t"] for m in metrics])
    powers = np.array([m["Mz_kin"] * RHO * omega_rad for m in metrics])
    return float(-np.trapezoid(powers, times))


class TaylorCouetteMixingEnv(gym.Env):
    """RL environment for controlling inner-cylinder rotation to mix a passive scalar."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        case_path: str | Path,
        omega_min: float = -300.0,
        omega_max: float = 300.0,
        omega_start: float = 0.0,
        action_delta_rpm: float = 50.0,
        step_duration: float = 1.0,
        max_steps: int = 60,
        warmup_omega_rpm: float = 100.0,
        warmup_duration: float = 10.0,
        alpha: float = 1.0,
        beta: float = 1.0,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.case = OpenFOAMCase(case_path)

        # One-time spin-up so dye reaches the measurement bins. Cached
        # as 0.warmed/; idempotent across env instances pointed at the
        # same case directory.
        self.case.warmup(_rpm_to_rad_s(warmup_omega_rpm), warmup_duration)

        self.omega_min = float(omega_min)
        self.omega_max = float(omega_max)
        self.omega_start = float(omega_start)
        self.action_delta_rpm = float(action_delta_rpm)
        self.step_duration = float(step_duration)
        self.max_steps = int(max_steps)
        self.alpha = float(alpha)
        self.beta = float(beta)

        self.observation_space = spaces.Box(
            low=np.array([self.omega_min], dtype=np.float32),
            high=np.array([self.omega_max], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self.render_mode = render_mode

        # Episode state.
        self.omega: float = self.omega_start
        self.step_count: int = 0
        self.mixing_index: float = 1.0
        self.energy_total: float = 0.0

    # ------------------------------------------------------------------ #
    # Gym API
    # ------------------------------------------------------------------ #

    def _obs(self) -> np.ndarray:
        return np.array([self.omega], dtype=np.float32)

    def _info(self) -> Dict[str, Any]:
        return {
            "step_count": self.step_count,
            "omega": self.omega,
            "mixing_index": self.mixing_index,
            "energy_consumption": self.energy_total,
        }

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        mode = (options or {}).get("reset_mode", "hard")
        self.case.reset(mode=mode)

        self.omega = self.omega_start
        self.step_count = 0
        self.mixing_index = 1.0
        self.energy_total = 0.0

        return self._obs(), self._info()

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        delta_rpm = float(action[0]) * self.action_delta_rpm
        self.omega = float(
            np.clip(self.omega + delta_rpm, self.omega_min, self.omega_max)
        )
        omega_rad = _rpm_to_rad_s(self.omega)

        metrics = self.case.simulate(omega_rad, self.step_duration)

        e_step = _step_energy_joules(metrics, omega_rad)
        e_norm = e_step / E_REF_JOULES

        final = metrics[-1]
        concentrations = np.array(
            [final[f"C{i}"] for i in range(N_BINS)], dtype=np.float64
        )
        self.mixing_index = _mixing_index(concentrations)
        self.energy_total += e_step
        self.step_count += 1

        reward = -(self.alpha * self.mixing_index + self.beta * e_norm)
        terminated = False
        truncated = self.step_count >= self.max_steps

        return self._obs(), float(reward), terminated, truncated, self._info()
