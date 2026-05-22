"""Taylor-Couette mixing Gymnasium environment backed by OpenFOAM."""

from gymnasium.envs.registration import register

from tc_gym.env import TaylorCouetteMixingEnv
from tc_gym.openfoam import OpenFOAMCase

__all__ = ["TaylorCouetteMixingEnv", "OpenFOAMCase"]

register(
    id="TaylorCouetteMixing-v0",
    entry_point="tc_gym.env:TaylorCouetteMixingEnv",
)
